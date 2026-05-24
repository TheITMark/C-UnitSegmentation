#!/usr/bin/env python3
"""
Alignment-aware postprocessor for SALT-style transcripts.

Goal:
- Repair likely speaker-turn drift after rule-based processing.
- Apply conservative, context-aware fixes only.
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple


SPEAKER_RE = re.compile(r"^(P:|Av:)\s*(.*)$")
PAUSE_RE = re.compile(r"^;\s*:?(?P<seconds>\d{2})?$")


class AlignmentPostprocessor:
    def __init__(self, insert_no_response: bool = True):
        self.insert_no_response = insert_no_response
        self.avatar_short = {"hm", "hmm", "huh", "uh", "yes", "no", "okay", "ok"}
        self.avatar_phrases = (
            "i do not understand",
            "i don't understand",
            "i dont understand",
            "i don't need help",
            "i dont need help",
            "i don't know you",
            "i dont know you",
            "no audible or visual response observed",
        )
        self.participant_cues = (
            "would you",
            "can you",
            "are you",
            "do you",
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "would you like",
            "my name",
            "good morning",
        )

    def _parse_speaker_line(self, line: str) -> Optional[Tuple[str, str]]:
        m = SPEAKER_RE.match(line.strip())
        if not m:
            return None
        return m.group(1)[:-1], m.group(2).strip()

    def _is_avatar_like(self, content: str) -> bool:
        low = content.lower().strip()
        low_word = re.sub(r"[^\w]", "", low)
        if low_word in self.avatar_short:
            return True
        return any(p in low for p in self.avatar_phrases)

    def _is_participant_like(self, content: str) -> bool:
        low = content.lower().strip()
        if low.endswith("?"):
            return True
        return any(c in low for c in self.participant_cues)

    def _is_prompt(self, speaker: str, content: str) -> bool:
        return speaker == "P" and self._is_participant_like(content)

    def _pause_seconds(self, line: str) -> int:
        m = PAUSE_RE.match(line.strip())
        if not m:
            return 0
        sec = m.group("seconds")
        if sec is None:
            return 2
        return int(sec)

    def _flip_obvious_speaker_mismatches(self, lines: List[str]) -> List[str]:
        out: List[str] = []
        for line in lines:
            parsed = self._parse_speaker_line(line)
            if parsed is None:
                out.append(line)
                continue
            speaker, content = parsed
            if speaker == "P" and self._is_avatar_like(content):
                out.append(f"Av: {content}")
            elif speaker == "Av" and self._is_participant_like(content):
                out.append(f"P: {content}")
            else:
                out.append(line)
        return out

    def _merge_fragmented_turns(self, lines: List[str]) -> List[str]:
        """
        Merge obvious same-speaker fragments:
        - previous speaker line is short and not terminal
        - current speaker line same speaker and starts with lowercase/continuation
        """
        out: List[str] = []
        continuation = {"and", "or", "but", "so", "then", "not", "to", "of", "for", "are", "is", "am"}

        for line in lines:
            parsed = self._parse_speaker_line(line)
            if parsed is None:
                out.append(line)
                continue
            speaker, content = parsed

            if out:
                prev_parsed = self._parse_speaker_line(out[-1])
                if prev_parsed is not None:
                    prev_speaker, prev_content = prev_parsed
                    first_word = re.sub(r"[^\w']", "", content.split()[0].lower()) if content.split() else ""
                    prev_words = prev_content.split()
                    prev_terminal = bool(re.search(r"[.!?]$", prev_content))
                    prev_short = len(prev_words) <= 3 and not prev_terminal
                    starts_lower = bool(content and content[:1].islower())
                    if (
                        speaker == prev_speaker
                        and (prev_short or starts_lower or first_word in continuation)
                    ):
                        out[-1] = f"{prev_speaker}: {prev_content} {content}".strip()
                        continue

            out.append(line)
        return out

    def _insert_missing_avatar_no_response(self, lines: List[str]) -> List[str]:
        if not self.insert_no_response:
            return lines

        out: List[str] = []
        for i, line in enumerate(lines):
            out.append(line)
            current = self._parse_speaker_line(line)
            if current is None:
                continue
            spk, content = current
            if not self._is_prompt(spk, content):
                continue

            # Scan forward until next speaker line; allow pause lines only.
            pause_total = 0
            j = i + 1
            saw_avatar = False
            next_speaker: Optional[Tuple[str, str]] = None
            while j < len(lines):
                p = self._parse_speaker_line(lines[j])
                if p is not None:
                    next_speaker = p
                    if p[0] == "Av":
                        saw_avatar = True
                    break
                pause_total += self._pause_seconds(lines[j])
                j += 1

            if (
                not saw_avatar
                and next_speaker is not None
                and self._is_prompt(next_speaker[0], next_speaker[1])
                and pause_total >= 4
            ):
                out.append("Av: {no audible or visual response observed}")

        return out

    def process_text(self, text: str) -> str:
        lines = text.splitlines()
        lines = self._flip_obvious_speaker_mismatches(lines)
        lines = self._merge_fragmented_turns(lines)
        lines = self._insert_missing_avatar_no_response(lines)
        return "\n".join(lines)

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        try:
            text = input_path.read_text(encoding="utf-8")
            processed = self.process_text(text)
            output_path.write_text(processed, encoding="utf-8")
            print(f"✅ Aligned: {input_path.name} -> {output_path.name}")
            return True
        except Exception as exc:
            print(f"❌ Error aligning {input_path.name}: {exc}")
            return False

    def process_directory(self, input_dir: Path, output_dir: Path) -> int:
        output_dir.mkdir(parents=True, exist_ok=True)
        files = list(input_dir.glob("*Rule-Based Processed*.txt"))
        if not files:
            print("⚠️ No Rule-Based Processed files found.")
            return 1

        ok = 0
        for fp in sorted(files):
            out_name = fp.name.replace("(Rule-Based Processed)", "(Aligned Processed)")
            if self.process_file(fp, output_dir / out_name):
                ok += 1

        print(f"📊 Alignment complete: {ok}/{len(files)} files successful")
        return 0 if ok == len(files) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Alignment-aware postprocessing pass")
    parser.add_argument("--input-dir", type=Path, required=True, help="Rule-based output directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Aligned output directory")
    parser.add_argument(
        "--no-insert-no-response",
        action="store_true",
        help="Disable conservative no-response insertion between prompts",
    )
    args = parser.parse_args()

    processor = AlignmentPostprocessor(insert_no_response=not args.no_insert_no_response)
    return processor.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

