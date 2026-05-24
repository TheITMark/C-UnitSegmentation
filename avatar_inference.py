#!/usr/bin/env python3
"""
Rule 2: Avatar Response Inference Module for C-Unit Segmentation
Detects gaps where Avatar responses are likely missing from Descript ASR output
and uses LLM-assisted generation to infer plausible responses.

The VR Avatar has a constrained vocabulary of responses. This module:
1. Parses the conversation into structured turns
2. Detects gaps where Avatar responses are likely missing
3. Classifies the conversational context (question type, greeting, etc.)
4. Uses FLAN-T5 (or template fallback) to generate plausible responses
5. Inserts inferred responses with {inferred} marking

Author: Mark McKenzie + AI Assistant
Date: April 2026
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: Tuple[int, int, int]  # (hours, minutes, seconds)
    speaker: str  # 'P', 'Av', or other
    content: str  # The utterance text
    is_inferred: bool = False
    total_seconds: int = 0

    def __post_init__(self):
        h, m, s = self.timestamp
        self.total_seconds = h * 3600 + m * 60 + s


@dataclass
class Gap:
    """Represents a detected gap where an Avatar response is likely missing."""
    position: int  # Index in the turns list where the response should be inserted
    preceding_turn: ConversationTurn  # The P turn before the gap
    following_turn: Optional[ConversationTurn]  # The next turn (if any)
    gap_type: str  # 'yes_no_question', 'open_question', 'greeting', 'statement', 'name_call'
    time_gap_seconds: int  # Time between surrounding turns
    confidence: float  # 0.0-1.0 confidence that a response is missing


class AvatarResponseInferrer:
    """
    Detects and infers missing Avatar responses in Descript transcripts.

    The VR Avatar has a constrained set of responses observed in gold standards:
    - Acknowledgments: HM, HM?, HUH?, HUH.
    - Binary: YES, NO
    - Confusion: "I do not understand", "I don't understand"
    - Contextual: domain-specific responses (about photos, clothes, etc.)
    """

    def __init__(self, use_llm: bool = True, llm_processor=None, mode: str = "standard"):
        self.use_llm = use_llm
        self.llm_processor = llm_processor
        self.mode = mode

        # Known Avatar response templates (from gold standard analysis)
        self.avatar_vocabulary = {
            'acknowledgment': ['HM.', 'HM?'],
            'confusion': ['HUH?', 'HUH.', 'I do not understand.', "I don't understand."],
            'affirmative': ['YES.', 'Okay.'],
            'negative': ['NO.', "I don't need help."],
            'no_response': ['{no audible or visual response observed}'],
        }

        # Patterns for detecting question types
        self.yes_no_starters = {
            'are', 'is', 'do', 'does', 'did', 'can', 'could', 'would', 'will',
            'shall', 'have', 'has', 'was', 'were', 'may', 'might'
        }

        # Greeting patterns
        self.greeting_patterns = [
            r'\bgood morning\b', r'\bhello\b', r'\bhi\b', r'\bhey\b',
            r'\bgood afternoon\b', r'\bgood evening\b',
        ]

        # Patterns that suggest P is echoing/responding to an unseen Avatar response
        self.echo_patterns = [
            # P starts with "No." or "Yes." - echoing Avatar's answer
            (r'^(?:P:\s*)?(?:No|Yes|Yeah|Okay|OK)[.,!]?\s', 'echo_response'),
            # P starts with "That's okay" - comforting after confusion
            (r'^(?:P:\s*)?That\'?s (?:okay|OK|alright)', 'comfort_after_confusion'),
            # P starts with "Well," - transitioning after Avatar response
            (r'^(?:P:\s*)?Well,', 'transition'),
        ]

        # Rule 2 tuning knobs (set by mode)
        self.min_gap_for_inference = 3
        self.min_confidence_threshold = 0.60
        self.max_inferred_per_file: Optional[int] = None
        self.allow_greeting_inference = True

        if self.mode == "conservative":
            self.min_gap_for_inference = 5
            self.min_confidence_threshold = 0.80
            self.max_inferred_per_file = 8
            self.allow_greeting_inference = False
        elif self.mode == "aggressive":
            self.min_gap_for_inference = 2
            self.min_confidence_threshold = 0.55
            self.max_inferred_per_file = None
            self.allow_greeting_inference = True

        # LLM prompt template for response generation
        self.llm_prompt_template = (
            "In a healthcare VR simulation, a Participant (P) is caring for a patient Avatar (Av) "
            "with dementia. The Avatar has limited responses: HM, HUH?, YES, NO, "
            "\"I do not understand\", or short contextual replies.\n\n"
            "Given this conversation context, what is the most likely Avatar response?\n\n"
            "Context:\n{context}\n\n"
            "The Avatar's response should be one of: {options}\n\n"
            "Avatar response:"
        )

    def parse_descript_to_turns(self, input_text: str) -> List[ConversationTurn]:
        """
        Parse raw Descript transcript into structured conversation turns.
        Handles multi-line continuations and merged speaker lines.
        """
        lines = input_text.strip().split('\n')
        turns: List[ConversationTurn] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match timestamp pattern [HH:MM:SS]
            timestamp_match = re.search(r'\[(\d{2}):(\d{2}):(\d{2})\]', line)
            if not timestamp_match:
                # Skip title lines and non-timestamped content
                continue

            h, m, s = int(timestamp_match.group(1)), int(timestamp_match.group(2)), int(timestamp_match.group(3))
            content = line[timestamp_match.end():].strip()

            # Remove any inline timestamps from content (e.g., "[00:01:00]" mid-line)
            content = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', content).strip()

            if not content:
                continue

            # Detect speaker
            speaker_match = re.match(r'^(P|Av|Avatar|Participant):\s*', content)
            if speaker_match:
                speaker_raw = speaker_match.group(1)
                speaker = 'P' if speaker_raw in ('P', 'Participant') else 'Av'
                content = content[speaker_match.end():].strip()
            else:
                # Continuation of previous speaker, or unknown
                speaker = turns[-1].speaker if turns else 'P'

            if content:
                turns.append(ConversationTurn(
                    timestamp=(h, m, s),
                    speaker=speaker,
                    content=content,
                ))

        return turns

    def _classify_utterance(self, content: str) -> str:
        """Classify a P utterance to determine expected Avatar response type."""
        content_lower = content.lower().strip()

        # Remove speaker prefix if present
        content_lower = re.sub(r'^p:\s*', '', content_lower)

        # Check if it's a name call (just the name, possibly with ?)
        if re.match(r'^[a-z]+[?.,!]?$', content_lower) and len(content_lower) < 15:
            return 'name_call'

        # Check if it ends with a question mark
        is_question = content.rstrip().endswith('?')

        if is_question:
            # Check if it's a yes/no question
            first_word = content_lower.split()[0] if content_lower.split() else ''
            if first_word in self.yes_no_starters:
                return 'yes_no_question'
            # Check for "who", "what", "where", "when", "how", "why"
            if first_word in {'who', 'what', 'where', 'when', 'how', 'why'}:
                return 'open_question'
            return 'open_question'

        # Check if it's a greeting
        for pattern in self.greeting_patterns:
            if re.search(pattern, content_lower):
                return 'greeting'

        # Check for directives/commands
        if any(content_lower.startswith(w) for w in ['let\'s', 'take', 'come', 'put']):
            return 'directive'

        return 'statement'

    def _check_echo_response(self, turn: ConversationTurn) -> Optional[str]:
        """
        Check if a P turn starts with an echo of an unseen Avatar response.
        For example: "No. Well, I think..." suggests Avatar said "No" before this.
        """
        content = turn.content.strip()
        for pattern, echo_type in self.echo_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return echo_type
        return None

    def merge_continuation_turns(self, turns: List[ConversationTurn]) -> List[ConversationTurn]:
        """Merge consecutive same-speaker turns that are clearly continuations.

        Descript often wraps lines mid-sentence (e.g., "Good morning. How" /
        "was your sleep?").  We merge them before gap detection to avoid
        false-positive avatar inference between the fragments.
        """
        if not turns:
            return turns

        merged: List[ConversationTurn] = [turns[0]]

        for turn in turns[1:]:
            prev = merged[-1]
            if (
                turn.speaker == prev.speaker
                and turn.content
                and turn.content[0].islower()
            ):
                # Continuation — merge content into previous turn
                prev.content = f"{prev.content} {turn.content}"
                continue

            # Also merge if previous content ends without terminal punctuation
            # and same speaker (e.g., "Good morning, James. How" / "was your sleep?")
            prev_content = prev.content.rstrip()
            if (
                turn.speaker == prev.speaker
                and prev_content
                and not prev_content[-1] in '.!?'
            ):
                prev.content = f"{prev.content} {turn.content}"
                continue

            merged.append(turn)

        return merged

    def detect_gaps(self, turns: List[ConversationTurn]) -> List[Gap]:
        """
        Detect positions where Avatar responses are likely missing.

        Heuristics:
        1. P asks a question and the next turn is also P (no Avatar response)
        2. P makes a greeting and the next turn is also P
        3. Significant time gap between consecutive P turns
        4. P echoes/responds to something Avatar didn't visibly say
        """
        gaps: List[Gap] = []

        for i, turn in enumerate(turns):
            if turn.speaker != 'P':
                continue

            next_turn = turns[i + 1] if i + 1 < len(turns) else None

            # Only look for gaps when next turn is also P (no Avatar response between)
            if next_turn is None or next_turn.speaker != 'P':
                continue

            # Skip if the next P turn looks like a mid-sentence continuation
            # (starts lowercase, or previous ends without terminal punctuation
            # and next is a short fragment)
            next_content = next_turn.content.strip()
            prev_content = turn.content.rstrip()
            prev_last_char = prev_content[-1] if prev_content else ''
            if next_content and next_content[0].islower() and prev_last_char not in '.!?':
                continue

            # Calculate time gap
            time_gap = next_turn.total_seconds - turn.total_seconds

            # Classify this P utterance
            utterance_type = self._classify_utterance(turn.content)

            # Determine confidence based on context
            confidence = 0.0

            # --- Heuristic 1: P asks a direct question, next is P ---
            if utterance_type in ('yes_no_question', 'open_question'):
                # Only infer if there's a time gap (Avatar needs processing time)
                if time_gap >= self.min_gap_for_inference:
                    confidence = 0.85
                # Short gap — likely P self-correcting or continuing, not a missed response

            # --- Heuristic 2: P greeting with time gap, next is P ---
            elif utterance_type == 'greeting' and self.allow_greeting_inference:
                if time_gap >= 4:
                    confidence = 0.70

            # --- Heuristic 3: P echoes unseen Avatar response ---
            echo_type = self._check_echo_response(next_turn)
            if echo_type == 'echo_response':
                # P is echoing Avatar's answer - high confidence
                confidence = max(confidence, 0.90)
                utterance_type = 'echo_detected'
            elif echo_type == 'comfort_after_confusion':
                confidence = max(confidence, 0.75)

            # Only add gaps above confidence threshold
            if confidence >= self.min_confidence_threshold:
                gaps.append(Gap(
                    position=i + 1,  # Insert after this P turn
                    preceding_turn=turn,
                    following_turn=next_turn,
                    gap_type=utterance_type,
                    time_gap_seconds=time_gap,
                    confidence=confidence,
                ))

        return gaps

    def _generate_response_template(self, gap: Gap) -> str:
        """
        Generate a response using templates based on gap type.
        Fallback when LLM is not available.
        """
        if gap.gap_type == 'yes_no_question':
            # Check if the next P turn echoes the answer
            if gap.following_turn:
                next_content = gap.following_turn.content.lower().strip()
                if next_content.startswith('no') or next_content.startswith('well,'):
                    return 'NO.'
                if next_content.startswith('yes') or next_content.startswith('great') or next_content.startswith('okay'):
                    return 'YES.'
            # Default for yes/no questions from confused Avatar
            return 'HUH?'

        elif gap.gap_type == 'open_question':
            return 'HUH?'

        elif gap.gap_type == 'greeting':
            return 'HM.'

        elif gap.gap_type == 'name_call':
            return 'HM?'

        elif gap.gap_type == 'echo_detected':
            # Extract what P echoed
            if gap.following_turn:
                next_content = gap.following_turn.content.strip()
                # P starts with "No." -> Avatar said "No"
                no_match = re.match(r'^No[.,!]?\s', next_content, re.IGNORECASE)
                if no_match:
                    return 'NO.'
                yes_match = re.match(r'^Yes[.,!]?\s', next_content, re.IGNORECASE)
                if yes_match:
                    return 'YES.'
            return 'HM.'

        elif gap.gap_type == 'directive':
            return 'HUH?'

        # Default
        return 'HUH?'

    def _generate_response_llm(self, gap: Gap, surrounding_turns: List[ConversationTurn]) -> str:
        """
        Generate a response using FLAN-T5 based on conversational context.
        Falls back to template if LLM is unavailable or produces poor output.
        """
        if not self.use_llm or self.llm_processor is None:
            return self._generate_response_template(gap)

        # Build context from surrounding turns (up to 4 turns for context)
        context_lines = []
        for turn in surrounding_turns[-4:]:
            prefix = f"{turn.speaker}:"
            context_lines.append(f"{prefix} {turn.content}")
        context_lines.append("Av: ???")
        context_str = "\n".join(context_lines)

        # Determine candidate options based on gap type
        if gap.gap_type == 'yes_no_question':
            options = "YES, NO, HUH?, I do not understand"
        elif gap.gap_type in ('open_question', 'directive'):
            options = "HUH?, HM?, I do not understand, I don't understand"
        elif gap.gap_type == 'greeting':
            options = "HM, HM?, HUH?"
        elif gap.gap_type == 'echo_detected':
            options = "YES, NO, HM, HUH?"
        else:
            options = "HM, HUH?, I do not understand"

        prompt = self.llm_prompt_template.format(context=context_str, options=options)

        try:
            llm_response = self.llm_processor.generate_response(prompt, max_length=30)
            llm_response = llm_response.strip().rstrip('.')

            # Validate LLM response against known Avatar vocabulary
            valid_responses = set()
            for responses in self.avatar_vocabulary.values():
                for r in responses:
                    valid_responses.add(r.lower().rstrip('.'))

            if llm_response.lower() in valid_responses:
                # Normalize capitalization for short responses
                if llm_response.upper() in ('HM', 'HUH', 'YES', 'NO'):
                    return llm_response.upper() + ('?' if '?' in llm_response else '.')
                return llm_response + '.'

            # LLM gave something unexpected - fall back to template
            return self._generate_response_template(gap)

        except Exception as e:
            print(f"  LLM inference failed: {e}, using template fallback")
            return self._generate_response_template(gap)

    def infer_responses(self, input_text: str) -> Tuple[str, List[Dict]]:
        """
        Main entry point: detect gaps and infer missing Avatar responses.

        Args:
            input_text: Raw Descript transcript text

        Returns:
            Tuple of (augmented transcript text, list of inference details)
        """
        # Step 1: Parse into structured turns
        turns = self.parse_descript_to_turns(input_text)

        if not turns:
            return input_text, []

        # No merging — gap detection handles continuations directly

        # Step 2: Detect gaps
        gaps = self.detect_gaps(turns)

        if not gaps:
            return input_text, []

        # Conservative cap to avoid over-insertion cascades.
        if self.max_inferred_per_file is not None and len(gaps) > self.max_inferred_per_file:
            gaps = sorted(gaps, key=lambda g: g.confidence, reverse=True)[:self.max_inferred_per_file]
            gaps = sorted(gaps, key=lambda g: g.position)

        # Step 3: Generate responses and build augmented turn list
        inferred_details = []
        # Process gaps in reverse order so insertion indices stay valid
        for gap in sorted(gaps, key=lambda g: g.position, reverse=True):
            # Get surrounding context for LLM
            start_ctx = max(0, gap.position - 3)
            surrounding = turns[start_ctx:gap.position]

            # Generate response
            response = self._generate_response_llm(gap, surrounding)

            # Create inferred turn with interpolated timestamp
            inferred_timestamp = gap.preceding_turn.timestamp
            inferred_turn = ConversationTurn(
                timestamp=inferred_timestamp,
                speaker='Av',
                content=response,
                is_inferred=True,
            )

            # Insert into turns list
            turns.insert(gap.position, inferred_turn)

            inferred_details.append({
                'position': gap.position,
                'gap_type': gap.gap_type,
                'confidence': gap.confidence,
                'response': response,
                'preceding_content': gap.preceding_turn.content[:60],
                'time_gap': gap.time_gap_seconds,
            })

        # Step 4: Reconstruct the transcript text with inferred responses
        output_lines = []
        # Preserve the title line
        first_line = input_text.strip().split('\n')[0].strip()
        if not first_line.startswith('['):
            output_lines.append(first_line)

        for turn in turns:
            ts = f"[{turn.timestamp[0]:02d}:{turn.timestamp[1]:02d}:{turn.timestamp[2]:02d}]"
            if turn.is_inferred:
                output_lines.append(f"{ts} Av: {turn.content} {{inferred}}")
            else:
                output_lines.append(f"{ts} {turn.speaker}: {turn.content}")

        # Reverse the details list to be in forward order
        inferred_details.reverse()

        return '\n'.join(output_lines), inferred_details


def test_avatar_inference():
    """Unit tests for Avatar Response Inference (Rule 2)."""
    print("\n" + "=" * 60)
    print("TESTING RULE 2: AVATAR RESPONSE INFERENCE")
    print("=" * 60)

    inferrer = AvatarResponseInferrer(use_llm=False)
    passed = 0
    failed = 0

    # --- Test 1: Greeting gap detection (needs 4+ second gap) ---
    test_input = """VR_Test
[00:00:00] P: Good morning, James.
[00:00:05] P: How was your sleep?
[00:00:10] Av: Um, okay, I guess."""

    augmented, details = inferrer.infer_responses(test_input)
    if any(d['gap_type'] == 'greeting' for d in details):
        print("  PASS: Detected greeting gap (Good morning, 5s gap)")
        passed += 1
    else:
        print("  FAIL: Did not detect greeting gap")
        failed += 1

    # --- Test 2: Yes/No question gap (needs 3+ second gap) ---
    test_input = """VR_Test
[00:00:00] P: Are you ready to get up?
[00:00:06] P: Well, I think they're making breakfast."""

    augmented, details = inferrer.infer_responses(test_input)
    if any(d['gap_type'] == 'yes_no_question' for d in details):
        print("  PASS: Detected yes/no question gap")
        passed += 1
    else:
        print("  FAIL: Did not detect yes/no question gap")
        failed += 1

    # --- Test 3: Echo detection (P starts with "No.") ---
    test_input = """VR_Test
[00:00:00] P: Are you ready to get up for the day?
[00:00:06] P: No. Well, I think they're making breakfast."""

    augmented, details = inferrer.infer_responses(test_input)
    if any(d['gap_type'] == 'echo_detected' for d in details):
        print("  PASS: Detected echo response (P echoing Avatar's 'No')")
        passed += 1
    else:
        print("  FAIL: Did not detect echo response")
        failed += 1

    # Verify the inferred response is NO.
    if any(d['response'] == 'NO.' for d in details):
        print("  PASS: Correctly inferred 'NO.' from echo")
        passed += 1
    else:
        print(f"  FAIL: Expected 'NO.' but got {[d['response'] for d in details]}")
        failed += 1

    # --- Test 4: No gap when Avatar responds ---
    test_input = """VR_Test
[00:00:00] P: How are you?
[00:00:04] Av: I do not understand.
[00:00:08] P: That's okay."""

    augmented, details = inferrer.infer_responses(test_input)
    if len(details) == 0:
        print("  PASS: No false positive when Avatar already responds")
        passed += 1
    else:
        print(f"  FAIL: False positive - detected {len(details)} gaps when none expected")
        failed += 1

    # --- Test 5: {inferred} marker present in output ---
    test_input = """VR_Test
[00:00:00] P: Good morning, James.
[00:00:06] P: How was your sleep?"""

    augmented, details = inferrer.infer_responses(test_input)
    if '{inferred}' in augmented:
        print("  PASS: {inferred} marker present in output")
        passed += 1
    else:
        print("  FAIL: {inferred} marker missing from output")
        failed += 1

    # --- Test 6: Open question generates HUH? ---
    test_input = """VR_Test
[00:00:00] P: What would you like to do today?
[00:00:06] P: How about breakfast?"""

    augmented, details = inferrer.infer_responses(test_input)
    if any(d['response'] == 'HUH?' for d in details):
        print("  PASS: Open question generates 'HUH?' response")
        passed += 1
    else:
        print(f"  FAIL: Expected 'HUH?' but got {[d['response'] for d in details]}")
        failed += 1

    # --- Test 7: Question with sufficient time gap ---
    test_input = """VR_Test
[00:00:00] P: Who's the little boy in the picture?
[00:00:06] P: He looks just like you."""

    augmented, details = inferrer.infer_responses(test_input)
    if any(d['gap_type'] == 'open_question' for d in details):
        print("  PASS: Detected open question gap with time gap")
        passed += 1
    else:
        print(f"  FAIL: Did not detect open question gap, got {[d['gap_type'] for d in details]}")
        failed += 1

    # --- Test 8: Multiple consecutive P turns ---
    test_input = """VR_Test
[00:00:00] P: Good morning, James.
[00:00:05] P: How was your sleep?
[00:00:10] P: Are you hungry?
[00:00:16] Av: Um, okay, I guess."""

    augmented, details = inferrer.infer_responses(test_input)
    if len(details) >= 2:
        print(f"  PASS: Detected {len(details)} gaps in consecutive P turns")
        passed += 1
    else:
        print(f"  FAIL: Expected 2+ gaps but detected {len(details)}")
        failed += 1

    # --- Test 9: GhostRider-like sequence ---
    test_input = """VR_GhostRider
[00:00:00] P: Good morning, James. How
[00:00:16] P: was your sleep?
[00:00:19] Av: Um, um, okay, I guess.
[00:00:26] P: Are you ready to get up for the day?
[00:00:32] P: No. Well, I think that they're making you some good breakfast out there."""

    augmented, details = inferrer.infer_responses(test_input)
    # Should detect: greeting gap before "How was your sleep", echo gap before "No. Well..."
    has_greeting_or_question = any(d['gap_type'] in ('greeting', 'open_question', 'statement') for d in details)
    has_echo = any(d['gap_type'] == 'echo_detected' for d in details)
    if len(details) >= 1:
        print(f"  PASS: GhostRider sequence - detected {len(details)} gaps")
        passed += 1
    else:
        print(f"  FAIL: GhostRider sequence - expected gaps but detected {len(details)}")
        failed += 1

    # --- Test 10: Short time gap - no inference even for questions ---
    test_input = """VR_Test
[00:00:00] P: Are you okay?
[00:00:02] P: Let's go."""

    augmented, details = inferrer.infer_responses(test_input)
    if len(details) == 0:
        print("  PASS: No inference for short gap (2 seconds, below threshold)")
        passed += 1
    else:
        print(f"  FAIL: Should not infer with 2-second gap, detected {len(details)}")
        failed += 1

    print("\n" + "-" * 60)
    print(f"Rule 2 Test Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        success = test_avatar_inference()
        exit(0 if success else 1)
    else:
        print("Usage: python avatar_inference.py --test")
        print("  Or import and use AvatarResponseInferrer class directly")
