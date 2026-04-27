#!/usr/bin/env python3
"""
Rule-Based Preprocessing Module for C-Unit Segmentation
Stage 1: Handle deterministic transformations before LLM processing

Author: Yugant Soni + AI Assistant
Date: December 2024
"""

import re
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import argparse

from avatar_inference import AvatarResponseInferrer
try:
    import spacy
except Exception:
    spacy = None


class RuleBasedProcessor:
    """
    Enhanced SALT formatting processor handling:
    - Timestamp conversion and pause timing
    - Speaker normalization and C-unit segmentation
    - Coordinating conjunction splitting
    - Basic morphological marking
    - Maze detection and filled pause formatting
    - Overlap detection patterns
    """
    
    def __init__(
        self,
        use_avatar_inference: bool = True,
        use_llm: bool = False,
        morph_mode: str = "regex",
        mark_verb_3sg: bool = False,
    ):
        # Rule 2: Avatar Response Inference
        self.use_avatar_inference = use_avatar_inference
        self.avatar_inferrer = AvatarResponseInferrer(use_llm=use_llm) if use_avatar_inference else None
        self.morph_mode = morph_mode
        self.mark_verb_3sg = mark_verb_3sg
        self._spacy_nlp = None

        # Filled pause patterns (simple cases)
        self.filled_pauses = {
            'uh', 'um', 'hm', 'hmm', 'er', 'ah', 'oh', 'uhoh', 'oops', 'ooh'
        }
        
        # Speaker normalization patterns
        self.speaker_patterns = {
            r'\bP:': 'P:',
            r'\bAv:': 'Av:',
            r'\bAvatar:': 'Av:',
            r'\bParticipant:': 'P:'
        }
        
        # Coordinating conjunctions that split C-units
        self.coordinating_conjunctions = {
            'and', 'or', 'but', 'so', 'then'
        }
        
        # Subordinating conjunctions that keep C-units together
        self.subordinating_conjunctions = {
            'because', 'that', 'when', 'who', 'after', 'before', 
            'which', 'although', 'if', 'unless', 'while', 'as', 
            'how', 'until', 'like', 'where', 'since'
        }
        
        # Common verbs that need morphological marking
        self.common_verbs = {
            'look': 'look/ed', 'get': 'get', 'go': 'go', 'come': 'come', 'help': 'help',
            'want': 'want', 'need': 'need', 'dress': 'dress/ed', 'ready': 'ready',
            'understand': 'understand', 'know': 'know', 'think': 'think',
            'feel': 'feel', 'wear': 'wear', 'put': 'put', 'stand': 'stand',
            'sit': 'sit', 'stay': 'stay', 'rest': 'rest'
        }
    
    def parse_timestamp(self, timestamp_str: str) -> Tuple[int, int, int]:
        """Parse [HH:MM:SS] format to (hours, minutes, seconds)"""
        # Remove brackets and split
        clean_time = timestamp_str.strip('[]')
        parts = clean_time.split(':')
        
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
        return 0, 0, 0
    
    def format_salt_timestamp(self, hours: int, minutes: int, seconds: int) -> str:
        """Convert to SALT format: -M:SS or -MM:SS.
        Rule 3 rounds to the nearest *even-minute* marker (every 2 minutes)
        when within 5 seconds (e.g., 3:58 -> 4:00, 6:02 -> 6:00).
        """
        total_seconds = hours * 3600 + minutes * 60 + seconds
        total_minutes = total_seconds // 60
        remaining_seconds = total_seconds % 60

        # Round to nearest even-minute boundary (every 120s) when very close.
        # This avoids snapping to odd-minute boundaries like 5:00.
        mod_even_minute = total_seconds % 120
        if mod_even_minute <= 5 and total_seconds >= 120:
            total_seconds -= mod_even_minute
            total_minutes = total_seconds // 60
            remaining_seconds = 0
        elif mod_even_minute >= 115:
            total_seconds += (120 - mod_even_minute)
            total_minutes = total_seconds // 60
            remaining_seconds = 0

        if total_minutes == 0 and remaining_seconds == 0:
            return "-0:00"
        return f"-{total_minutes}:{remaining_seconds:02d}"
    
    def calculate_pause_duration(self, prev_time: Tuple[int, int, int], 
                                curr_time: Tuple[int, int, int]) -> int:
        """Calculate pause duration in seconds between timestamps"""
        prev_total = prev_time[0] * 3600 + prev_time[1] * 60 + prev_time[2]
        curr_total = curr_time[0] * 3600 + curr_time[1] * 60 + curr_time[2]
        return curr_total - prev_total
    
    def format_pause_code(self, duration_seconds: int, is_inter_utterance: bool = True) -> str:
        """Format pause codes according to SALT rules"""
        if duration_seconds < 2:
            return "; " if is_inter_utterance else ":"
        else:
            # Round to nearest second, round up at half-second
            rounded = round(duration_seconds)
            if is_inter_utterance:
                return f"; :{rounded:02d}"
            else:
                return f":{rounded:02d}"
    
    def normalize_speakers(self, text: str) -> str:
        """Normalize speaker labels to consistent P: and Av: format"""
        for pattern, replacement in self.speaker_patterns.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def handle_redactions(self, text: str) -> str:
        """Standardize redaction format"""
        # Convert [redacted] to {redacted}
        text = re.sub(r'\[redacted\]', '{redacted}', text, flags=re.IGNORECASE)
        return text
    
    # Response words that can be standalone C-units (Rule 8)
    RESPONSE_WORDS = {'yes', 'no', 'yeah', 'okay', 'ok', 'nope', 'nah', 'yep', 'yup', 'sure'}

    # Pronouns/subjects that indicate a new clause after a response word
    CLAUSE_STARTERS = {
        'i', 'we', 'he', 'she', 'it', 'they', 'you', 'that', 'this',
        'there', 'my', 'the', 'a', 'but', 'and', 'let', "let's",
        "don't", "i'm", "i'll", "we'll", "we're", "it's", "that's",
    }

    # Short tag phrases that should NOT trigger splitting after a response word
    # e.g., "Okay, I guess." stays as one C-unit
    TAG_PHRASES = {
        'i guess', 'i think', 'i mean', 'i suppose', 'i reckon',
        'you know', 'right',
    }

    def _split_response_words(self, content: str, speaker: str) -> List[str]:
        """
        Rule 8: Split leading response words (yes, no, okay, etc.) into
        separate C-units when followed by a new clause.

        Examples:
          "No, I don't need help." -> ["No.", "I don't need help."]
          "Okay, I guess."        -> ["Okay, I guess."]  (keep together)
          "Yes, let's go."        -> ["Yes.", "Let's go."]

        The split only happens when the word after the comma looks like
        the start of a new independent clause (pronoun, subject, etc.).
        """
        # Match: response_word + comma + space + rest
        m = re.match(
            r'^(' + '|'.join(self.RESPONSE_WORDS) + r')[,.]?\s+(.+)$',
            content, re.IGNORECASE
        )
        if not m:
            return [f"{speaker}: {content}"]

        response_word = m.group(1)
        rest = m.group(2).strip()

        if not rest:
            return [f"{speaker}: {content}"]

        # Check if the rest is just a short tag phrase (keep together)
        rest_lower = rest.lower().rstrip('.,!?').strip()
        if rest_lower in self.TAG_PHRASES:
            return [f"{speaker}: {content}"]

        # Check if the rest starts with a clause starter (new independent clause)
        first_word_of_rest = rest.split()[0].lower().rstrip('.,!?')
        if first_word_of_rest in self.CLAUSE_STARTERS:
            # Split: response word becomes its own C-unit
            response_cu = response_word.capitalize() + '.'
            # Capitalize the rest
            rest_capitalized = rest[0].upper() + rest[1:] if rest else rest
            return [
                f"{speaker}: {response_cu}",
                f"{speaker}: {rest_capitalized}",
            ]

        # Not a new clause — keep together (e.g., "No, thanks.")
        return [f"{speaker}: {content}"]

    def _split_on_coordinating_conjunctions(self, content: str) -> List[str]:
        """
        Rules 5/6/7:
        - Split on coordinating conjunctions when they start a new clause.
        - Preserve subordinating structures.
        - Do not split "so that" constructions.
        """
        working = content.strip()
        if not working:
            return [content]

        pattern = re.compile(r',\s+(and|or|but|so|then)\s+', re.IGNORECASE)
        clause_starters = self.CLAUSE_STARTERS.union({"who", "what", "when", "where", "why", "how"})

        for m in pattern.finditer(working):
            conj = m.group(1).lower()
            before = working[:m.start()].strip()
            after = working[m.end():].strip()
            if not before or not after:
                continue

            after_first = re.sub(r"[^\w']", "", after.split()[0].lower())
            if after_first not in clause_starters:
                continue

            # Rule 7: preserve "so that" subordinate construction.
            if conj == "so" and after.lower().startswith("that "):
                continue

            # Rule 6: avoid splitting before explicit subordinating conjunction tails.
            before_last = re.sub(r"[^\w']", "", before.split()[-1].lower())
            if before_last in self.subordinating_conjunctions:
                continue

            first_unit = before if before.endswith(('.', '!', '?')) else f"{before}."
            second_unit = f"{conj.capitalize()} {after}"
            return [first_unit, second_unit]

        return [working]

    def _split_coordination_prefixed(self, line: str, speaker: str) -> List[str]:
        """Apply Rules 5/6/7 splitting to an existing speaker-prefixed line."""
        prefix = f"{speaker}: "
        if not line.startswith(prefix):
            return [line]
        content = line[len(prefix):].strip()
        parts = self._split_on_coordinating_conjunctions(content)
        return [f"{speaker}: {p}" for p in parts]

    def segment_cunits(self, text: str, speaker: str) -> List[str]:
        """
        Segment text into proper C-units.
        1. Split on sentence boundaries (., !, ?) outside parentheses/braces
        2. Rule 8: Split leading response words (yes/no/okay) into separate C-units
        Preserves {inferred} markers attached to the preceding C-unit.
        """
        if not text.strip():
            return [text]

        # Clean the text first
        clean_text = text.strip()

        # Remove speaker prefix for processing
        if clean_text.startswith(f"{speaker}:"):
            clean_text = clean_text[len(f"{speaker}:"):].strip()

        # Extract trailing {inferred} marker if present - we'll re-attach it later
        inferred_marker = ""
        inferred_match = re.search(r'\s*\{inferred\}\s*$', clean_text)
        if inferred_match:
            inferred_marker = " {inferred}"
            clean_text = clean_text[:inferred_match.start()].strip()

        # Step 1: Split on sentence boundaries, but not inside parentheses or braces
        sentence_splits = []
        current_sent = ""
        paren_depth = 0
        brace_depth = 0

        for char in clean_text:
            current_sent += char
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
            elif char in '.!?' and paren_depth == 0 and brace_depth == 0:
                # End of sentence outside parentheses/braces
                if current_sent.strip():
                    sentence_splits.append(current_sent.strip())
                current_sent = ""

        # Add any remaining content
        if current_sent.strip():
            sentence_splits.append(current_sent.strip())

        if not sentence_splits:
            return [text]

        # Step 2: Rule 8 - Split response words in each sentence fragment
        result = []
        for fragment in sentence_splits:
            split_units = self._split_response_words(fragment, speaker)
            result.extend(split_units)

        if not result:
            return [text]

        # Step 3: Rules 5/6/7 - Coordination-aware clause splitting
        coord_result = []
        for unit in result:
            coord_result.extend(self._split_coordination_prefixed(unit, speaker))
        result = coord_result if coord_result else result

        # Re-attach {inferred} marker to the last C-unit
        if inferred_marker:
            result[-1] = result[-1] + inferred_marker

        return result
    
    def detect_repetitions_and_mazes(self, text: str) -> str:
        """
        Detect repetitions and format as mazes: I, I do not -> (I) I do not
        """
        # Pattern for repetitions like "I, I" or "um, um"
        # This handles simple repetition cases
        
        # Pattern 1: Word, Word (same word repeated)
        text = re.sub(r'\b(\w+),\s+\1\b', r'(\1) \1', text)
        
        # Pattern 2: Single word repetitions without comma
        words = text.split()
        processed_words = []
        i = 0
        
        while i < len(words):
            if i + 1 < len(words):
                current_word = words[i].strip('.,!?').lower()
                next_word = words[i + 1].strip('.,!?').lower()
                
                # If same word repeated
                if current_word == next_word and current_word not in {'the', 'a', 'an', 'to'}:
                    processed_words.append(f"({words[i]})")
                    processed_words.append(words[i + 1])
                    i += 2
                    continue
            
            processed_words.append(words[i])
            i += 1
        
        return ' '.join(processed_words)
    
    def _is_standalone_acknowledgment(self, words: list, index: int) -> bool:
        """
        Check if a filled-pause-like word (HM, HUH, etc.) is actually a standalone
        acknowledgment/response rather than a hesitation filler within a longer utterance.

        Standalone responses like "HM." or "HUH?" should NOT be marked as [FP].
        But "Um, I need help" should mark "Um" as [FP].
        """
        # Words that are typically standalone responses, not fillers
        standalone_words = {'hm', 'hmm', 'huh'}

        clean_word = re.sub(r'[^\w]', '', words[index].lower())
        if clean_word not in standalone_words:
            return False

        # Filter out speaker labels and markers for content word count
        content_words = [w for w in words
                         if not re.match(r'^(P:|Av:|Avatar:|Participant:|\{[^}]*\})$', w)]

        # If this word (possibly with punctuation) is the only content, it's standalone
        if len(content_words) <= 1:
            return True

        # If it's at the start and followed by a question mark or period, likely standalone
        word = words[index]
        if word.endswith(('.', '?', '!')) and index == 0 and len(content_words) <= 2:
            return True

        return False

    def detect_filled_pauses_enhanced(self, text: str) -> str:
        """Enhanced filled pause detection with better formatting.
        Avoids marking standalone acknowledgments (HM., HUH?) as filled pauses."""
        words = text.split()
        processed_words = []

        for i, word in enumerate(words):
            # Clean word for comparison (remove all non-word characters)
            clean_word = re.sub(r'[^\w]', '', word.lower())

            if clean_word in self.filled_pauses:
                # Don't mark standalone acknowledgments as filled pauses
                if self._is_standalone_acknowledgment(words, i):
                    processed_words.append(word)
                # Check if word is already in parentheses (from maze detection)
                elif word.startswith('(') and word.endswith(')'):
                    base = word[1:-1]
                    processed_words.append(f"({base} [FP])")
                elif word.endswith((',', '.', '!', '?')):
                    base_word = word[:-1]
                    punct = word[-1]
                    processed_words.append(f"({base_word} [FP]){punct}")
                else:
                    processed_words.append(f"({word} [FP])")
            else:
                processed_words.append(word)

        return ' '.join(processed_words)
    
    def _get_spacy_nlp(self):
        """Load spaCy model lazily for Rule 14."""
        if self._spacy_nlp is not None:
            return self._spacy_nlp
        if spacy is None:
            return None
        try:
            self._spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            self._spacy_nlp = None
        return self._spacy_nlp

    def apply_morphological_marking_regex(self, text: str) -> str:
        """
        Apply basic regex-based morphological marking for common past tense verbs.
        """
        # Simple past tense patterns for common verbs
        morphological_patterns = {
            r'\bget dressed\b': 'get dress/ed',
            r'\bgot dressed\b': 'got dress/ed', 
            r'\bget ready\b': 'get ready',
            r'\bgot ready\b': 'got ready',
            r'\blooked\b': 'look/ed',
            r'\bhelped\b': 'help/ed',
            r'\bwanted\b': 'want/ed',
            r'\bneeded\b': 'need/ed',
            r'\bstarted\b': 'start/ed',
            r'\bfinished\b': 'finish/ed'
        }
        
        for pattern, replacement in morphological_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def apply_morphological_marking_spacy(self, text: str) -> str:
        """
        Rule 14: Apply spaCy-based morphological markings.
        Falls back to regex mode if spaCy/model is unavailable.
        """
        nlp = self._get_spacy_nlp()
        if nlp is None:
            return self.apply_morphological_marking_regex(text)

        speaker_match = re.match(r'^(P:|Av:)\s*(.*)$', text)
        speaker_prefix = ""
        content = text
        if speaker_match:
            speaker_prefix = f"{speaker_match.group(1)} "
            content = speaker_match.group(2)

        doc = nlp(content)
        out_tokens = []
        for token in doc:
            t = token.text
            low = t.lower()
            lemma = token.lemma_ if token.lemma_ else t

            if token.is_space:
                out_tokens.append(token.text)
                continue
            if token.is_punct or token.like_url or token.like_num:
                out_tokens.append(t + token.whitespace_)
                continue

            repl = t

            # /ing and /ed
            if low.endswith("ing") and lemma.lower() != low:
                repl = f"{lemma}/ing"
            elif low.endswith("ed") and lemma.lower() != low:
                repl = f"{lemma}/ed"
            # regular noun plurals
            elif token.pos_ == "NOUN" and token.morph.get("Number") == ["Plur"] and lemma.lower() != low:
                suffix = "/es" if low.endswith(("ches", "shes", "ses", "xes", "zes", "oes")) else "/s"
                repl = f"{lemma}{suffix}"
            # optional verb 3sg
            elif (
                self.mark_verb_3sg
                and token.pos_ in ("VERB", "AUX")
                and low.endswith("s")
                and "Person=3" in str(token.morph)
                and lemma.lower() != low
            ):
                suffix = "/es" if low.endswith("es") else "/s"
                repl = f"{lemma}{suffix}"

            # preserve capitalization on replacements
            if t and t[0].isupper() and repl:
                repl = repl[0].upper() + repl[1:]

            out_tokens.append(repl + token.whitespace_)

        return speaker_prefix + "".join(out_tokens).strip()

    def apply_morphological_marking(self, text: str) -> str:
        """Rule 14 dispatcher based on configured morphology mode."""
        if self.morph_mode == "off":
            return text
        if self.morph_mode == "spacy":
            return self.apply_morphological_marking_spacy(text)
        return self.apply_morphological_marking_regex(text)

    def apply_linked_words(self, text: str) -> str:
        """
        Rule 23: Link multi-word units with underscores.
        """
        linked_patterns = [
            (r'\bhealth care provider\b', 'health_care_provider'),
            (r'\bhealthcare provider\b', 'health_care_provider'),
            (r'\bfire truck\b', 'fire_truck'),
            (r'\bice cream\b', 'ice_cream'),
            (r'\bliving room\b', 'living_room'),
            (r'\bdining room\b', 'dining_room'),
            (r'\bbed room\b', 'bed_room'),
            (r'\bhealth care\b', 'health_care'),
            (r'\bmr\.?\s+([A-Z][a-z]+)\b', r'Mr_\1'),
            (r'\bmrs\.?\s+([A-Z][a-z]+)\b', r'Mrs_\1'),
            (r'\bdr\.?\s+([A-Z][a-z]+)\b', r'Dr_\1'),
        ]

        for pattern, replacement in linked_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def apply_lexical_normalization(self, text: str) -> str:
        """
        Rule 29: Normalize accepted spelling variants.
        """
        lexical_patterns = [
            (r'\bok\b', 'okay'),
            (r"\bain[’']?t\b", "ain't"),
            (r'\buh[\s-]+oh\b', 'uhoh'),
            (r'\bbet you\b', 'betcha'),
            (r'\bgonnaa+\b', 'gonna'),
            (r'\bgottaa+\b', 'gotta'),
            (r'\bwannaa+\b', 'wanna'),
            (r'\bhaftaa+\b', 'hafta'),
            (r'\boughtaa+\b', 'oughta'),
            (r'\bbetchaa+\b', 'betcha'),
            (r'\byeahh+\b', 'yeah'),
            (r'\bnahh+\b', 'nah'),
            (r'\bhmmm+\b', 'hmm'),
        ]

        for pattern, replacement in lexical_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def apply_conjunction_reduction_normalization(self, text: str) -> str:
        """
        Rule 9: Normalize reduced conjunction variants.
        """
        reductions = [
            (r"(?<!\w)'cause\b", "because"),
            (r"\bcuz\b", "because"),
            (r"\ban'(?=\s|$)", "and"),
            (r"(?<!\w)'n(?=\s|$)", "and"),
        ]
        for pattern, replacement in reductions:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def apply_tags_and_questions(self, text: str) -> str:
        """
        Rule 10: Normalize common tag-question endings to question form.
        """
        tag_patterns = [
            (r',\s*right\.', ', right?'),
            (r',\s*okay\.', ', okay?'),
            (r',\s*ok\.', ', okay?'),
            (r",\s*isn't it\.", ", isn't it?"),
            (r",\s*aren't you\.", ", aren't you?"),
            (r",\s*don't you\.", ", don't you?"),
        ]
        for pattern, replacement in tag_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        if re.match(r'^(P:|Av:)\s*(who|what|when|where|why|how|do|does|did|can|could|will|would|is|are)\b',
                    text, flags=re.IGNORECASE):
            if not text.rstrip().endswith('?'):
                text = text.rstrip('.!') + '?'
        return text

    def apply_interjections_in_mazes(self, text: str) -> str:
        """
        Rule 18: Wrap common interjection phrases in maze parentheses.
        Example: "The man, I think he's tall, is here" -> "The man (I think he's tall) is here"
        """
        interjection_patterns = [
            r'i think',
            r'i guess',
            r'i mean',
            r'you know',
            r'i suppose',
        ]
        pattern = re.compile(
            r',\s*((?:' + '|'.join(interjection_patterns) + r')[^,]*)\s*,',
            re.IGNORECASE
        )
        return pattern.sub(lambda m: f" ({m.group(1).strip()}) ", text)
    
    def improve_pause_timing(self, duration_seconds: int, context: str = "") -> str:
        """
        Improved pause timing based on context and SALT conventions
        """
        # Contextual pause adjustments based on gold standard patterns
        if duration_seconds < 2:
            return "; :02"  # Default short pause
        elif duration_seconds < 5:
            return f"; :0{min(duration_seconds, 3)}"  # Cap at :03 for short pauses
        elif duration_seconds < 10:
            return f"; :0{min(duration_seconds, 6)}"  # Medium pauses
        else:
            return f"; :{min(duration_seconds, 16):02d}"  # Longer pauses, cap at :16
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning and normalization with SALT formatting"""
        # Remove any remaining Descript timestamps from content
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle speaker formatting
        text = self.normalize_speakers(text)
        
        # Handle redactions
        text = self.handle_redactions(text)

        # Rule 29: lexical normalization
        text = self.apply_lexical_normalization(text)

        # Rule 9: conjunction-reduction normalization
        text = self.apply_conjunction_reduction_normalization(text)
        
        # Apply morphological marking
        text = self.apply_morphological_marking(text)

        # Rule 23: linked words
        text = self.apply_linked_words(text)

        # Rule 18: interjections in mazes
        text = self.apply_interjections_in_mazes(text)
        
        # Detect repetitions and mazes
        text = self.detect_repetitions_and_mazes(text)
        
        # Enhanced filled pause detection
        text = self.detect_filled_pauses_enhanced(text)

        # Rule 10: tags and question normalization
        text = self.apply_tags_and_questions(text)
        
        # Basic punctuation cleanup
        text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
        
        return text

    def add_overlap_markers(self, line: str) -> str:
        """Rule 19: Mark a speaker line as overlapping using <...>."""
        if not (line.startswith("P: ") or line.startswith("Av: ")):
            return line
        if "<" in line and ">" in line:
            return line  # already marked

        speaker, content = line.split(":", 1)
        content = content.strip()
        if not content:
            return line
        return f"{speaker}: <{content}>"

    def is_potentially_abandoned_utterance(self, line: str) -> bool:
        """Rule 20: Conservative detection for abandoned utterances."""
        if not (line.startswith("P: ") or line.startswith("Av: ")):
            return False

        speaker, content = line.split(":", 1)
        content = content.strip()
        if not content:
            return False
        if content.endswith(">"):
            return False  # already marked
        if content.endswith((".", "!", "?")):
            return False  # complete utterance
        if "<" in content and ">" in content:
            return False  # overlap marker, not abandonment

        word_count = len(content.split())
        if word_count < 2:
            return False

        # Conservative cues for incomplete thought endings.
        incomplete_endings = {
            "and", "or", "but", "so", "because", "if", "when", "that",
            "to", "the", "a", "an", "my", "your", "his", "her", "their",
            "this", "these", "those", "then",
        }
        last_token = re.sub(r"[^\w]", "", content.split()[-1].lower())
        return (content.endswith(",") or last_token in incomplete_endings)

    def add_abandoned_marker(self, line: str) -> str:
        """Append SALT abandoned marker `>` to a speaker line."""
        if not self.is_potentially_abandoned_utterance(line):
            return line

        speaker, content = line.split(":", 1)
        content = content.rstrip()
        return f"{speaker}:{content}>"
    
    def add_salt_header(self, filename: str) -> List[str]:
        """Generate SALT-compliant header"""
        # Extract base name without extension
        base_name = Path(filename).stem.replace('(Descript generated)', '').strip()
        
        header = [
            f"{base_name}",
            "",  # Empty line after title
            # Note: We're not adding full headers as they require manual input
            # The LLM stage can add these if needed
        ]
        return header
    
    def process_transcript(self, input_text: str, filename: str) -> str:
        """
        Main processing function for a single transcript

        Args:
            input_text: Raw Descript transcript content
            filename: Original filename for header generation

        Returns:
            Preprocessed transcript text
        """
        # Rule 2: Infer missing Avatar responses before main processing
        inference_details = []
        if self.use_avatar_inference and self.avatar_inferrer:
            input_text, inference_details = self.avatar_inferrer.infer_responses(input_text)
            if inference_details:
                print(f"  Rule 2: Inferred {len(inference_details)} missing Avatar responses")
                for detail in inference_details:
                    print(f"    - {detail['gap_type']}: \"{detail['response']}\" "
                          f"(conf={detail['confidence']:.0%}, after: \"{detail['preceding_content']}\")")

        lines = input_text.strip().split('\n')
        processed_lines = []
        
        # Add header
        processed_lines.extend(self.add_salt_header(filename))
        
        # Track timing for pause calculation and time markers
        prev_timestamp = None
        prev_speaker = None
        last_time_marker_seconds = None
        last_speaker_line_idx = None
        
        # Process each line
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Skip title lines (first line usually)
            if line_idx == 0 and not line.startswith('['):
                continue
            
            # Look for timestamp pattern [HH:MM:SS]
            timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
            
            if timestamp_match:
                # Extract timestamp and content
                timestamp_str = timestamp_match.group(1)
                content = line[timestamp_match.end():].strip()
                
                # Parse timestamp
                curr_timestamp = self.parse_timestamp(f"[{timestamp_str}]")
                curr_total_seconds = curr_timestamp[0] * 3600 + curr_timestamp[1] * 60 + curr_timestamp[2]
                pause_duration = None
                
                # Add time marker if at major intervals (every minute or significant gaps)
                if prev_timestamp is None:
                    # First timestamp - always add initial time marker
                    processed_lines.append(self.format_salt_timestamp(*curr_timestamp))
                    last_time_marker_seconds = curr_total_seconds
                else:
                    # Calculate pause duration
                    pause_duration = self.calculate_pause_duration(prev_timestamp, curr_timestamp)

                    # Rule 3: Time marker placement
                    # Gold standard places markers at scene transitions (detectable
                    # via large gaps in Descript timestamps, typically 30+ seconds)
                    # and at transcript boundaries. Timestamps are rounded to the
                    # nearest even minute when close (e.g., 6:02 -> 6:00).
                    should_add_time_marker = False
                    if pause_duration >= 30:  # Large gap — likely scene transition
                        should_add_time_marker = True

                    if should_add_time_marker:
                        processed_lines.append(self.format_salt_timestamp(*curr_timestamp))
                        last_time_marker_seconds = curr_total_seconds

                    # Use improved pause timing
                    if pause_duration >= 1.5:  # Significant pause
                        pause_code = self.improve_pause_timing(pause_duration)
                        processed_lines.append(pause_code)
                
                # Process the content
                if content:
                    # Clean and normalize the content
                    clean_content = self.clean_text(content)
                    
                    # Extract speaker
                    speaker_match = re.match(r'^(P:|Av:)', clean_content)
                    if speaker_match:
                        speaker = speaker_match.group(1).rstrip(':')
                        
                        # Apply C-unit segmentation
                        segmented_lines = self.segment_cunits(clean_content, speaker)
                        
                        # Rule 19: Overlapping speech approximation
                        # If speakers switch with near-simultaneous timestamps, mark overlap
                        is_overlap = (
                            pause_duration is not None
                            and pause_duration <= 1
                            and prev_speaker is not None
                            and prev_speaker != f"{speaker}:"
                        )
                        if is_overlap and last_speaker_line_idx is not None:
                            processed_lines[last_speaker_line_idx] = self.add_overlap_markers(
                                processed_lines[last_speaker_line_idx]
                            )
                            segmented_lines = [self.add_overlap_markers(s) for s in segmented_lines]
                        elif (
                            pause_duration is not None
                            and pause_duration >= 2
                            and prev_speaker is not None
                            and prev_speaker != f"{speaker}:"
                            and last_speaker_line_idx is not None
                        ):
                            # Rule 20: Mark likely abandoned utterance on prior speaker line
                            processed_lines[last_speaker_line_idx] = self.add_abandoned_marker(
                                processed_lines[last_speaker_line_idx]
                            )

                        # Add all segmented lines
                        start_idx = len(processed_lines)
                        processed_lines.extend(segmented_lines)
                        if segmented_lines:
                            last_speaker_line_idx = start_idx + len(segmented_lines) - 1
                    else:
                        # No speaker found, add as is
                        processed_lines.append(clean_content)
                
                # Update tracking variables
                prev_timestamp = curr_timestamp
                if content:
                    # Extract speaker from content
                    speaker_match = re.match(r'^(P:|Av:)', clean_content)
                    if speaker_match:
                        prev_speaker = speaker_match.group(1)
            else:
                # Line without timestamp - might be continuation or other content
                if line.strip():
                    clean_line = self.clean_text(line)
                    processed_lines.append(clean_line)
        
        # Add end-of-transcript time marker (avoid duplicate trailing marker)
        if prev_timestamp:
            end_marker = self.format_salt_timestamp(*prev_timestamp)
            if not processed_lines or processed_lines[-1] != end_marker:
                processed_lines.append(end_marker)
        
        return '\n'.join(processed_lines)
    
    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """Process a single transcript file"""
        try:
            # Read input file
            with open(input_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
            
            # Process the transcript
            processed_text = self.process_transcript(input_text, input_path.name)
            
            # Write output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            print(f"✅ Processed: {input_path.name} -> {output_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {input_path.name}: {e}")
            return False
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, bool]:
        """Process all Descript generated files in input directory"""
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Find all Descript generated files
        descript_files = list(input_dir.glob("*Descript generated*.txt"))
        
        if not descript_files:
            print("⚠️  No Descript generated files found in input directory")
            return results
        
        print(f"📁 Processing {len(descript_files)} files from {input_dir}")
        print(f"📁 Output directory: {output_dir}")
        print("-" * 60)
        
        for input_file in sorted(descript_files):
            # Generate output filename
            output_name = input_file.name.replace("(Descript generated)", "(Rule-Based Processed)")
            output_path = output_dir / output_name
            
            # Process the file
            success = self.process_file(input_file, output_path)
            results[input_file.name] = success
        
        # Print summary
        successful = sum(results.values())
        total = len(results)
        print("-" * 60)
        print(f"📊 Processing complete: {successful}/{total} files successful")
        
        if successful < total:
            print("❌ Failed files:")
            for filename, success in results.items():
                if not success:
                    print(f"   - {filename}")
        
        return results


def test_rule_3_time_markers():
    """Test Rule 3: Time Marker Conversion"""
    print("\n" + "="*60)
    print("TESTING RULE 3: TIME MARKER CONVERSION")
    print("="*60)

    processor = RuleBasedProcessor(use_avatar_inference=False)

    passed = 0
    failed = 0

    # --- Direct format_salt_timestamp tests ---
    format_tests = [
        ("Initial marker", (0, 0, 0), "-0:00"),
        ("Mid-transcript", (0, 2, 30), "-2:30"),
        ("10 minutes", (0, 10, 15), "-10:15"),
        ("With hours", (1, 5, 45), "-65:45"),
        ("Round down (6:02 -> 6:00)", (0, 6, 2), "-6:00"),
        ("Round up (3:58 -> 4:00)", (0, 3, 58), "-4:00"),
        ("No rounding (3:30)", (0, 3, 30), "-3:30"),
        ("Round down (8:04 -> 8:00)", (0, 8, 4), "-8:00"),
        ("No odd-minute snap (5:02 stays)", (0, 5, 2), "-5:02"),
        ("No odd-minute snap (4:58 stays)", (0, 4, 58), "-4:58"),
        ("Round up to even-minute (5:58 -> 6:00)", (0, 5, 58), "-6:00"),
        ("Don't round 0:03 (minute 0)", (0, 0, 3), "-0:03"),  # Don't round at minute 0
    ]

    for name, (h, m, s), expected in format_tests:
        result = processor.format_salt_timestamp(h, m, s)
        if result == expected:
            print(f"  PASS: {name} -> {result}")
            passed += 1
        else:
            print(f"  FAIL: {name} -> expected {expected}, got {result}")
            failed += 1

    # --- Integration test: initial marker ---
    result = processor.process_transcript("[00:00:00] P: Good morning.", "test.txt")
    if "-0:00" in result:
        print(f"  PASS: Initial marker -0:00 in output")
        passed += 1
    else:
        print(f"  FAIL: Initial marker -0:00 missing")
        failed += 1

    # --- Integration test: scene transition marker (30+ second gap) ---
    test_input = "[00:00:00] P: Hello.\n[00:00:35] P: New scene."
    result = processor.process_transcript(test_input, "test.txt")
    marker_count = len(re.findall(r'^-\d+:\d+', result, re.MULTILINE))
    if marker_count >= 3:  # start + transition + end
        print(f"  PASS: Scene transition marker added for 35s gap")
        passed += 1
    else:
        print(f"  FAIL: Scene transition marker not added for 35s gap (got {marker_count} markers)")
        failed += 1

    # --- Integration test: no marker for small gap ---
    test_input = "[00:00:00] P: Hello.\n[00:00:05] P: How are you?"
    result = processor.process_transcript(test_input, "test.txt")
    marker_count = len(re.findall(r'^-\d+:\d+', result, re.MULTILINE))
    if marker_count == 2:  # Only start + end marker
        print(f"  PASS: No extra marker for 5s gap (start + end only)")
        passed += 1
    else:
        print(f"  FAIL: Expected 2 markers (start+end), got {marker_count}")
        failed += 1

    # --- Integration test: end marker present ---
    test_input = "[00:00:00] P: Hello.\n[00:05:30] P: Goodbye."
    result = processor.process_transcript(test_input, "test.txt")
    lines = [l.strip() for l in result.strip().split('\n') if l.strip()]
    last_line = lines[-1]
    if last_line.startswith('-'):
        print(f"  PASS: End marker present: {last_line}")
        passed += 1
    else:
        print(f"  FAIL: End marker missing, last line: {last_line}")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rule 3 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


def test_rule_19_overlapping_speech():
    """Test Rule 19: Overlapping speech markers."""
    print("\n" + "="*60)
    print("TESTING RULE 19: OVERLAPPING SPEECH")
    print("="*60)

    processor = RuleBasedProcessor(use_avatar_inference=False)
    passed = 0
    failed = 0

    # Speaker switch with 1-second gap should be treated as overlap
    overlap_input = (
        "[00:00:10] P: Do you want coffee?\n"
        "[00:00:11] Av: Yes."
    )
    overlap_result = processor.process_transcript(overlap_input, "test.txt")
    if "P: <Do you want coffee?>" in overlap_result and "Av: <Yes.>" in overlap_result:
        print("  PASS: Overlap markers added for near-simultaneous speaker switch")
        passed += 1
    else:
        print("  FAIL: Expected overlap markers not found")
        failed += 1

    # Same speaker continuation should not add overlap markers
    no_overlap_input = (
        "[00:00:10] P: Hello.\n"
        "[00:00:11] P: I am here."
    )
    no_overlap_result = processor.process_transcript(no_overlap_input, "test.txt")
    if "<" not in no_overlap_result and ">" not in no_overlap_result:
        print("  PASS: No overlap markers for same-speaker continuation")
        passed += 1
    else:
        print("  FAIL: Unexpected overlap markers on same speaker")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rule 19 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    return failed == 0


def test_rule_20_abandoned_utterances():
    """Test Rule 20: Abandoned utterance marker (>) detection."""
    print("\n" + "="*60)
    print("TESTING RULE 20: ABANDONED UTTERANCES")
    print("="*60)

    processor = RuleBasedProcessor(use_avatar_inference=False)
    passed = 0
    failed = 0

    # Incomplete line followed by speaker switch (non-overlap) should get >
    abandoned_input = (
        "[00:00:10] P: And then she,\n"
        "[00:00:14] Av: I want dinner."
    )
    abandoned_result = processor.process_transcript(abandoned_input, "test.txt")
    if "P: And then she,>" in abandoned_result:
        print("  PASS: Abandoned marker added for incomplete interrupted thought")
        passed += 1
    else:
        print("  FAIL: Expected abandoned marker not found")
        failed += 1

    # Complete sentence should not get >
    complete_input = (
        "[00:00:10] P: I am ready.\n"
        "[00:00:14] Av: Okay."
    )
    complete_result = processor.process_transcript(complete_input, "test.txt")
    if "P: I am ready.>" not in complete_result:
        print("  PASS: No abandoned marker for complete sentence")
        passed += 1
    else:
        print("  FAIL: Unexpected abandoned marker for complete sentence")
        failed += 1

    # Near-simultaneous overlap should use Rule 19 markers, not Rule 20
    overlap_input = (
        "[00:00:10] P: And then she,\n"
        "[00:00:11] Av: Yes."
    )
    overlap_result = processor.process_transcript(overlap_input, "test.txt")
    if "P: <And then she,>" in overlap_result and "P: And then she,>" not in overlap_result:
        print("  PASS: Overlap case uses Rule 19 marker precedence")
        passed += 1
    else:
        print("  FAIL: Rule 20 should not override overlap behavior")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rule 20 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    return failed == 0


def test_rule_23_linked_words():
    """Test Rule 23: Linked words using underscores."""
    print("\n" + "="*60)
    print("TESTING RULE 23: LINKED WORDS")
    print("="*60)

    processor = RuleBasedProcessor(use_avatar_inference=False)
    passed = 0
    failed = 0

    direct_tests = [
        ("Compound noun", "P: I like fire truck toys.", "P: I like fire_truck toys."),
        ("Healthcare phrase", "P: I am your health care provider.", "P: I am your health_care_provider."),
        ("Title linking", "P: Mr Frog is here.", "P: Mr_Frog is here."),
        ("No false positive", "P: This is a normal sentence.", "P: This is a normal sentence."),
    ]

    for name, input_text, expected in direct_tests:
        result = processor.clean_text(input_text)
        if result == expected:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} -> expected '{expected}', got '{result}'")
            failed += 1

    # Integration check through transcript pipeline
    transcript_input = "[00:00:01] P: I work in health care."
    transcript_result = processor.process_transcript(transcript_input, "test.txt")
    if "P: I work in health_care." in transcript_result:
        print("  PASS: Pipeline applies linked words")
        passed += 1
    else:
        print("  FAIL: Pipeline did not apply linked words")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rule 23 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    return failed == 0


def test_rule_29_lexical_normalization():
    """Test Rule 29: Accepted spelling variant normalization."""
    print("\n" + "="*60)
    print("TESTING RULE 29: LEXICAL NORMALIZATION")
    print("="*60)

    processor = RuleBasedProcessor(use_avatar_inference=False)
    passed = 0
    failed = 0

    direct_tests = [
        ("ok -> okay", "P: ok, I can help.", "P: okay, I can help."),
        ("bet you -> betcha", "P: I bet you can do it.", "P: I betcha can do it."),
        ("ain’t normalization", "P: I ain’t ready.", "P: I ain't ready."),
        ("no false positive", "P: This is already fine.", "P: This is already fine."),
    ]

    for name, input_text, expected in direct_tests:
        result = processor.clean_text(input_text)
        if result == expected:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} -> expected '{expected}', got '{result}'")
            failed += 1

    # Direct lexical-only check to avoid interactions with filled-pause tagging
    lexical_only = processor.apply_lexical_normalization("Av: uh oh.")
    if lexical_only == "Av: uhoh.":
        print("  PASS: uh oh -> uhoh")
        passed += 1
    else:
        print(f"  FAIL: uh oh -> uhoh -> got '{lexical_only}'")
        failed += 1

    # Integration check through transcript pipeline
    transcript_input = "[00:00:01] P: It is ok to go."
    transcript_result = processor.process_transcript(transcript_input, "test.txt")
    if "P: It is okay to go." in transcript_result:
        print("  PASS: Pipeline applies lexical normalization")
        passed += 1
    else:
        print("  FAIL: Pipeline did not apply lexical normalization")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rule 29 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    return failed == 0


def test_rule_14_morphological_marking():
    """Test Rule 14: spaCy-based morphological marking."""
    print("\n" + "="*60)
    print("TESTING RULE 14: MORPHOLOGICAL MARKING")
    print("="*60)

    processor = RuleBasedProcessor(
        use_avatar_inference=False,
        morph_mode="spacy",
        mark_verb_3sg=True,
    )
    nlp = processor._get_spacy_nlp()
    if nlp is None:
        print("  SKIP: spaCy model 'en_core_web_sm' unavailable")
        print("  Install with: python -m spacy download en_core_web_sm")
        print("="*60 + "\n")
        return True

    passed = 0
    failed = 0

    r1 = processor.clean_text("P: She walked home.")
    if "walk/ed" in r1:
        print("  PASS: /ed marking")
        passed += 1
    else:
        print(f"  FAIL: Missing /ed marking -> {r1}")
        failed += 1

    r2 = processor.clean_text("P: They are running.")
    if "run/ing" in r2:
        print("  PASS: /ing marking")
        passed += 1
    else:
        print(f"  FAIL: Missing /ing marking -> {r2}")
        failed += 1

    r3 = processor.clean_text("P: The dogs bark.")
    if "dog/s" in r3:
        print("  PASS: plural noun marking")
        passed += 1
    else:
        print(f"  FAIL: Missing plural marking -> {r3}")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rule 14 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    return failed == 0


def test_rules_5_6_7_coordination():
    """Test Rules 5/6/7: conjunction splitting/preservation/disambiguation."""
    print("\n" + "="*60)
    print("TESTING RULES 5/6/7: COORDINATION LOGIC")
    print("="*60)

    processor = RuleBasedProcessor(use_avatar_inference=False)
    passed = 0
    failed = 0

    # Rule 5: split coordinating conjunction with new clause
    r5 = processor.segment_cunits("P: I am ready, and I can help.", "P")
    if len(r5) == 2 and r5[0].startswith("P: I am ready") and r5[1].startswith("P: And I can help"):
        print("  PASS: Rule 5 coordinating conjunction split")
        passed += 1
    else:
        print(f"  FAIL: Rule 5 split expected 2 units, got {r5}")
        failed += 1

    # Rule 6: preserve subordinating conjunction structure
    r6 = processor.segment_cunits("P: I stayed because it was raining.", "P")
    if len(r6) == 1:
        print("  PASS: Rule 6 subordinating conjunction preserved")
        passed += 1
    else:
        print(f"  FAIL: Rule 6 expected 1 unit, got {r6}")
        failed += 1

    # Rule 7: do not split "so that"
    r7 = processor.segment_cunits("P: I moved so that I could see.", "P")
    if len(r7) == 1:
        print("  PASS: Rule 7 so/so-that disambiguation preserved")
        passed += 1
    else:
        print(f"  FAIL: Rule 7 expected 1 unit, got {r7}")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rules 5/6/7 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    return failed == 0


def test_rule_10_tags_and_questions():
    """Test Rule 10: tags and question normalization."""
    print("\n" + "="*60)
    print("TESTING RULE 10: TAGS AND QUESTIONS")
    print("="*60)
    processor = RuleBasedProcessor(use_avatar_inference=False)
    passed = 0
    failed = 0

    t1 = processor.clean_text("P: It's cold, right.")
    if t1 == "P: It's cold, right?":
        print("  PASS: Tag question punctuation normalized")
        passed += 1
    else:
        print(f"  FAIL: Tag question normalization mismatch: {t1}")
        failed += 1

    t2 = processor.clean_text("P: Can you help me.")
    if t2 == "P: Can you help me?":
        print("  PASS: Direct question punctuation normalized")
        passed += 1
    else:
        print(f"  FAIL: Direct question normalization mismatch: {t2}")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rule 10 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    return failed == 0


def test_rule_9_conjunction_reduction():
    """Test Rule 9: conjunction reduction normalization."""
    print("\n" + "="*60)
    print("TESTING RULE 9: CONJUNCTION REDUCTION")
    print("="*60)
    processor = RuleBasedProcessor(use_avatar_inference=False)
    passed = 0
    failed = 0

    c1 = processor.clean_text("P: I left 'cause I was tired.")
    if "because I was tired." in c1:
        print("  PASS: 'cause normalized to because")
        passed += 1
    else:
        print(f"  FAIL: 'cause normalization mismatch: {c1}")
        failed += 1

    c2 = processor.clean_text("P: Fish an' chips.")
    if "Fish and chips." in c2:
        print("  PASS: an' normalized to and")
        passed += 1
    else:
        print(f"  FAIL: an' normalization mismatch: {c2}")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rule 9 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    return failed == 0


def test_rule_18_interjections_in_mazes():
    """Test Rule 18: interjections in mazes."""
    print("\n" + "="*60)
    print("TESTING RULE 18: INTERJECTIONS IN MAZES")
    print("="*60)
    processor = RuleBasedProcessor(use_avatar_inference=False)
    passed = 0
    failed = 0

    m1 = processor.clean_text("P: The man, I think he's tall, is here.")
    if "(I think he's tall)" in m1:
        print("  PASS: Interjection wrapped in maze parentheses")
        passed += 1
    else:
        print(f"  FAIL: Interjection maze mismatch: {m1}")
        failed += 1

    m2 = processor.clean_text("P: The man is here.")
    if "(" not in m2:
        print("  PASS: No false positive maze wrapping")
        passed += 1
    else:
        print(f"  FAIL: Unexpected maze wrapping: {m2}")
        failed += 1

    print("\n" + "-"*60)
    print(f"Rule 18 Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    return failed == 0


def main():
    """Command line interface for rule-based processing"""
    parser = argparse.ArgumentParser(
        description="Rule-based preprocessing for C-unit segmentation"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/yuganthareshsoni/CunitSegementation/extracted_text"),
        help="Directory containing Descript generated transcripts"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default=Path("/Users/yuganthareshsoni/CunitSegementation/rule_based_output"),
        help="Directory to save rule-based processed transcripts"
    )
    parser.add_argument(
        "--test-rule3",
        action="store_true",
        help="Run Rule 3 (Time Marker Conversion) unit tests"
    )
    parser.add_argument(
        "--test-rule19",
        action="store_true",
        help="Run Rule 19 (Overlapping Speech) unit tests"
    )
    parser.add_argument(
        "--test-rule20",
        action="store_true",
        help="Run Rule 20 (Abandoned Utterances) unit tests"
    )
    parser.add_argument(
        "--test-rule23",
        action="store_true",
        help="Run Rule 23 (Linked Words) unit tests"
    )
    parser.add_argument(
        "--test-rule29",
        action="store_true",
        help="Run Rule 29 (Lexical Normalization) unit tests"
    )
    parser.add_argument(
        "--test-rule14",
        action="store_true",
        help="Run Rule 14 (Morphological Marking) unit tests"
    )
    parser.add_argument(
        "--test-rule567",
        action="store_true",
        help="Run Rules 5/6/7 (Coordination) unit tests"
    )
    parser.add_argument(
        "--test-rule10",
        action="store_true",
        help="Run Rule 10 (Tags and Questions) unit tests"
    )
    parser.add_argument(
        "--test-rule9",
        action="store_true",
        help="Run Rule 9 (Conjunction Reduction) unit tests"
    )
    parser.add_argument(
        "--test-rule18",
        action="store_true",
        help="Run Rule 18 (Interjections in Mazes) unit tests"
    )
    parser.add_argument(
        "--no-avatar-inference",
        action="store_true",
        help="Disable Rule 2 (Avatar Response Inference)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM (FLAN-T5) for avatar response inference (default: template-based)"
    )
    parser.add_argument(
        "--morph-mode",
        type=str,
        choices=["regex", "spacy", "off"],
        default="regex",
        help="Rule 14 morphology mode: regex (default), spacy, or off"
    )
    parser.add_argument(
        "--mark-verb-3sg",
        action="store_true",
        help="Enable /s marking for 3rd person singular verbs in spacy mode"
    )

    args = parser.parse_args()

    # Run tests if requested
    if args.test_rule3:
        success = test_rule_3_time_markers()
        return 0 if success else 1
    if args.test_rule19:
        success = test_rule_19_overlapping_speech()
        return 0 if success else 1
    if args.test_rule20:
        success = test_rule_20_abandoned_utterances()
        return 0 if success else 1
    if args.test_rule23:
        success = test_rule_23_linked_words()
        return 0 if success else 1
    if args.test_rule29:
        success = test_rule_29_lexical_normalization()
        return 0 if success else 1
    if args.test_rule14:
        success = test_rule_14_morphological_marking()
        return 0 if success else 1
    if args.test_rule567:
        success = test_rules_5_6_7_coordination()
        return 0 if success else 1
    if args.test_rule10:
        success = test_rule_10_tags_and_questions()
        return 0 if success else 1
    if args.test_rule9:
        success = test_rule_9_conjunction_reduction()
        return 0 if success else 1
    if args.test_rule18:
        success = test_rule_18_interjections_in_mazes()
        return 0 if success else 1

    # Initialize processor
    processor = RuleBasedProcessor(
        use_avatar_inference=not args.no_avatar_inference,
        use_llm=args.use_llm,
        morph_mode=args.morph_mode,
        mark_verb_3sg=args.mark_verb_3sg,
    )
    
    # Process all files
    results = processor.process_directory(args.input_dir, args.output_dir)
    
    # Exit with appropriate code
    if all(results.values()):
        print("🎉 All files processed successfully!")
        return 0
    else:
        print("⚠️  Some files failed to process")
        return 1


if __name__ == "__main__":
    exit(main())
