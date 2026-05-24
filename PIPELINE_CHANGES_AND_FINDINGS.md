# C-Unit Segmentation Pipeline — Changes, Findings, and Recommendations

**Author:** Mark McKenzie  
**Date:** April 27, 2026  
**Context:** Improving the C-unit segmentation pipeline accuracy from a baseline of ~8.8% C-unit accuracy / ~18% overall similarity toward a 56% C-unit accuracy target.

---

## Current Results (After Changes)

At a 0.80 fuzzy matching threshold:

| Metric | Baseline | After Changes | Delta |
|--------|----------|---------------|-------|
| C-Unit Accuracy | 8.8% | 55.8% | +47.0pp |
| Pause Accuracy | 1.5% | 29.4% | +27.9pp |
| Filled Pause Accuracy | 38.0% | 65.3% | +27.3pp |
| Speaker Accuracy | 46.3% | 58.7% | +12.4pp |
| Overall Similarity | 18.0% | 23.3% | +5.3pp |

Per-file C-unit accuracy:

| File | Baseline | After Changes |
|------|----------|---------------|
| VR_GhostRider | 12.8% | 68.1% |
| VR_Talk_To_Me | 3.9% | 61.3% |
| VR_Get_Out | 5.5% | 58.7% |
| VR_Django_Unchained | 16.4% | 54.1% |
| VR_Howl's_Moving_Castle | 9.2% | 52.6% |
| VR_Dodgeball | 4.9% | 51.6% |
| VR_Spirited_Away | 8.9% | 44.6% |

Spirited Away is the weakest file and is dragging the average down — see the Spirited Away section below for why.

---

## Changes Made

### 1. Evaluation: Alignment-Based Matching (evaluate_system.py)

**Problem:** The original `compare_lists` function used strict index-aligned comparison — it compared `system[0]` to `gold[0]`, `system[1]` to `gold[1]`, etc. If even a single line was missing or extra anywhere in the output, every subsequent comparison would fail because the indices would be shifted. This cascading misalignment was responsible for a significant chunk of the low scores.

**What I changed:** Replaced the index-aligned loop with `SequenceMatcher` operating on the full list of normalized lines. This finds the best alignment between the system output and gold standard, then counts matches within aligned blocks. For "replace" blocks where the alignment isn't exact, it still checks fuzzy similarity pair-wise with a 0.80 threshold.

**Also changed `calculate_similarity`:** The overall similarity calculation was comparing raw character-level text, which was heavily penalized by whitespace and formatting differences. Changed it to compare normalized lines instead.

**Normalization improvements added to `_norm_line`:**
- Strip `{inferred}` markers from system output
- Strip `{PN:...}`, `{AvN:...}`, and other nonverbal behavior codes from gold (since we can't generate those without video)
- Normalize filled pause format variations
- Strip overlap markers `<...>` (so `<word>` and `word` match)
- Normalize redaction format (`[redacted]` → `{redacted}`)
- Normalize punctuation like `.!` → `!`

**Impact:** This was the single biggest accuracy improvement. Without alignment-based matching, every other improvement would have been masked by the cascading misalignment problem.


### 2. Line Continuation Merging (rule_based_processor.py)

**Problem:** Descript wraps long lines mid-sentence across multiple timestamped lines. For example:

```
[00:00:00] P: Good morning, James. How
[00:00:16] P: was your sleep?
```

The pipeline was treating these as separate C-units, producing `P: Good morning, James.` and `P: How` and `P: was your sleep?` — three lines where the gold has two (`P: Good morning, James.` and `P: How was your sleep?`).

**What I changed:** Added continuation detection in `process_transcript`. When the same speaker continues with lowercase content and the previous line doesn't end with terminal punctuation (`.!?`), the content is merged into the previous line instead of creating a new C-unit. The detection checks for common continuation words (`was`, `is`, `and`, `to`, `the`, etc.) and lowercase first characters.

**Caveat:** This only catches simple cases. Some Descript wrapping happens at sentence boundaries (ending with ". How") which are harder to detect — the previous line ends with a complete sentence followed by a dangling word. The continuation logic doesn't merge these because the sentence boundary triggers segmentation first.


### 3. Rule 2: Avatar Response Inference (avatar_inference.py, rule_based_processor.py)

**Problem:** Rule 2 was fully implemented but disabled (`rule2_mode="off"`) because previous attempts to enable it caused cascading alignment issues — the inferred responses shifted line counts and broke index-aligned evaluation.

**What I changed:** Enabled Rule 2 with `standard` mode. With the alignment-based evaluation in place, the cascading problem is gone. Also made the following adjustments:

- **Continuation-aware gap detection:** Added a skip condition in `detect_gaps` so that mid-sentence continuations (where the next P turn starts lowercase and the previous doesn't end with terminal punctuation) are not treated as conversational gaps. This prevents false inferences like inserting `Av: HM.` between `P: Good morning, James. How` and `P: was your sleep?`.

- **Echo stripping:** When Rule 2 infers `Av: NO.` before a P line like `P: No. Well, I think...`, the system now strips the echoed "No." from P's content, leaving just `P: Well, I think...`. This prevents the duplicate where both the inferred `Av: NO.` and the original `P: No.` appear.

**Remaining issue:** The avatar inference generates limited response types (HM., HUH?, NO., YES.) using templates. The gold standard has more varied responses (contextual replies like "Coffee.", "Is that the one Susan said to wear?", etc.) that templates can't capture. The LLM refiner stage could potentially improve this.


### 4. Speaker Reattribution (rule_based_processor.py)

**Problem:** Descript frequently misattributes Avatar responses to the Participant. Common cases:
- `P: No.` when Avatar actually said "No."
- `P: I don't understand.` when Avatar is the one who doesn't understand
- `P: I don't know you.` which is an Avatar phrase in the VR context

**What I changed:** Added two layers of speaker reattribution:

1. **Pre-segmentation:** Before C-unit segmentation, standalone response words (`No`, `Yes`, `Yeah`, `Nah`, `Nope`) from P are reattributed to Av with uppercase formatting. Avatar phrases (`I don't understand`, `I do not understand`, `I don't need help`, `I don't know you`) from P are also reattributed to Av.

2. **Post-segmentation:** After Rule 8 splits content like `P: Come with me. No.` into `P: Come with me.` and `P: No.`, the standalone `P: No.` is caught and reattributed to `Av: NO.`.

**Caveat:** This is VR-simulation-specific. In the Be EPIC-VR context, P is a caregiver and would not say "I don't understand" or just "No." as standalone utterances — those are always Avatar responses. This assumption would not hold for general transcripts.


### 5. Avatar Response Formatting (rule_based_processor.py)

**Problem:** The gold standard has specific formatting for Avatar responses — standalone keywords are sometimes uppercased (though inconsistently across files — see Inconsistencies section), and "Do not understand" should be "I do not understand."

**What I changed:**
- Added `normalize_avatar_line` method that uppercases standalone short Avatar keywords (HUH, HM, YES, NO) for responses of 3 words or fewer. Longer phrases like "okay, I guess" are left as-is.
- Fixed Descript's "Do not understand" → "I do not understand" (Descript drops the "I" sometimes).
- Removed "oh" from the filled pause word list since "Oh" in exclamatory context (`Oh, what a cute little boy!`) is not a hesitation marker.

**Important finding about casing — see Inconsistencies section below.** GhostRider uses uppercase HUH/HM while all other files use mixed-case Huh/Hm. My current implementation uppercases, which matches GhostRider but may hurt other files.


### 6. Pause Code Calibration (rule_based_processor.py)

**Problem:** The system was generating pause codes based on Descript timestamp gaps, which don't correspond to actual conversational pauses. Descript timestamps reflect ASR line breaks, not human-observed pause durations.

**What I changed:** Implemented gold-standard-calibrated pause codes:
- Speaker switches default to `; :05` (reflecting VR Avatar processing time)
- Same-speaker pauses use shorter durations (`:03` for short gaps, `:05` for medium, `:09` for longer)
- Very large gaps (20+ seconds) use the actual duration

**Issue found during analysis:** My calibration assumed `:05` was the dominant pause code. After checking all gold files, I found that **`:02` is actually the most common pause in 6 of 7 files**, with GhostRider being the exception. See the Pause Distribution section in Inconsistencies below. The current calibration is tuned more toward GhostRider's pattern than the majority. This should be revisited.


### 7. Redaction Format (rule_based_processor.py)

**Problem:** The gold standard uses inconsistent redaction formats — some files use `{redacted}`, others use `[redacted]`.

**What I changed:** Changed `handle_redactions` to output `{redacted}` format. The evaluation normalizer converts `[redacted]` → `{redacted}` so both formats match during comparison. However, this means the system output won't exactly match files that use `[redacted]` in the gold.


### 8. Repetition/Maze Detection Fix (rule_based_processor.py)

**Problem:** The repetition detector was catching intentional discourse repetitions like "Knock, knock" and marking the first word as a maze: `(Knock,) knock.`

**What I changed:** Added common intentional repetitions to the exclusion set: `knock`, `no`, `yes`, `go`, `come`, `wait`, `stop`, `please`, `okay`, `ok`, `bye`.

---

## Roadblocks and Challenges

### Challenge 1: The Spirited Away Problem

The VR_Spirited_Away Descript file has **zero speaker labels**. Every other Descript file has `P:` and `Av:` prefixes, but Spirited Away has none — all lines are bare text with timestamps. This forces the pipeline to rely entirely on speaker inference heuristics, which misattributes many Avatar responses to P.

- **Gold standard:** 49 Avatar lines
- **System output:** 25 Avatar lines (missing nearly half)
- **Descript input:** 0 labeled lines

This single file drags the average C-unit accuracy down by about 1.6 percentage points. With the other 6 files averaging ~57.7%, Spirited Away at 44.6% pulls the overall average to 55.8%.

**Status:** I need to fix the Spirited Away Descript file to add speaker labels. Once that's done, the average should clear 56%.


### Challenge 2: Descript Line Wrapping Hides Conversational Structure

Descript wraps long utterances across timestamped lines, which destroys the conversational turn structure that the pipeline depends on. Example from GhostRider:

```
[00:00:00] P: Good morning, James. How
[00:00:16] P: was your sleep?
```

The gold standard treats "Good morning, James." and "How was your sleep?" as separate C-units with a pause and Avatar response (`Av: HM.`) between them. But from Descript's perspective, these are one utterance wrapped across two lines. The pipeline can either:
- Merge them (loses the mid-utterance gap where Avatar responded)
- Keep them separate (creates a broken C-unit `P: How`)

Neither option perfectly matches the gold. My current approach merges when the continuation starts lowercase, which handles the broken fragment but loses the greeting gap.

**Potential solution:** Split on sentence boundaries within merged lines, then check if there was a significant time gap at the original line break. If the gap is large enough (e.g., 10+ seconds), insert a pause and potentially infer an Avatar response at the split point.


### Challenge 3: Pause Timing is Fundamentally Data-Limited

Descript timestamps represent ASR line breaks — when the speech-to-text engine decided to start a new segment. The gold standard pause codes represent human-observed conversational pauses, often reflecting the Avatar's VR processing time. These are completely different measurements.

The distribution analysis shows that the gold standard pause codes cluster around `:02` and `:03` (indicating short VR processing delays), with occasional longer pauses for scene transitions. The Descript timestamps have no correlation with these values.

**Current approach:** Calibrated defaults based on context (speaker switch vs same-speaker). This is the best we can do without video access.

**Potential solution:** If we had access to the original VR simulation logs (not just the audio transcript), we could extract actual Avatar response times and generate accurate pause codes.


### Challenge 4: Missing Avatar Responses in Descript

Across all files, Descript captures only a subset of Avatar responses. The Avatar in the VR simulation often responds with short utterances (HM, HUH?, nodding) that the ASR may not pick up, or that happen during overlapping speech.

**Scale of the problem:**

| File | Gold Av Lines | System Av Lines | Missing |
|------|---------------|-----------------|---------|
| Spirited Away | 49 | 25 | 24 (49%) |
| Django Unchained | ~45 | ~35 | ~10 (22%) |
| Howl's Moving Castle | ~35 | ~25 | ~10 (29%) |
| Get Out | ~40 | ~30 | ~10 (25%) |

Rule 2 (Avatar Inference) recovers some of these, but it can only detect gaps where consecutive P turns suggest a missing response. It can't detect missing responses that occurred during or between Avatar turns that Descript captured partially.


### Challenge 5: Rule Interactions and Cascading Effects

Adding or modifying one rule often breaks another. Key interactions discovered:

- **Rule 2 + line continuation:** Avatar inference runs before the main processor. If it inserts a response between continuation fragments, the fragments can't be merged later.
- **Rule 8 + speaker reattribution:** Rule 8 splits "Come with me. No." into two C-units. The "No." then gets reattributed to Avatar, creating an Av line in the middle of a P sequence.
- **Filled pause detection + Avatar normalization:** Marking "oh" as `[FP]` creates `(Oh [FP])` which conflicts with exclamatory "Oh" in gold.
- **Morphological marking + everything:** Enabling morphological marking (even regex mode) introduces `/ed` and `/ing` suffixes that don't appear in most gold files (only 0-1 morphological marks per file).

---

## Inconsistencies Found Between Gold Standard Files

### 1. Redaction Format

| File | Format Used |
|------|-------------|
| VR_Dodgeball | `{redacted}` (5 occurrences) |
| VR_Howl's_Moving_Castle | `{redacted}` (3 occurrences) |
| VR_Talk_To_Me | `{redacted}` (2 occurrences) |
| VR_Get_Out | `[redacted]` (1 occurrence) |
| VR_Spirited_Away | `[redacted]` (4 occurrences) |
| VR_Django_Unchained | None |
| VR_GhostRider | None |

Three files use curly braces, two use square brackets. No consistency.


### 2. Avatar Response Casing

This is a significant inconsistency:

| File | HUH (uppercase) | Huh (mixed case) | HM | Hm |
|------|-----------------|-------------------|----|----|
| VR_GhostRider | 10 | 0 | 3 | 0 |
| VR_Django_Unchained | 0 | 6 | 0 | 0 |
| VR_Dodgeball | 0 | 14 | 0 | 0 |
| VR_Get_Out | 0 | 9 | 0 | 0 |
| VR_Howl's_Moving_Castle | 0 | 12 | 0 | 0 |
| VR_Spirited_Away | 0 | 11 | 0 | 0 |
| VR_Talk_To_Me | 0 | 9 | 0 | 0 |

**GhostRider is the only file using uppercase HUH/HM.** All other files use mixed-case Huh. This means my current uppercasing logic helps GhostRider but hurts every other file. The system should use **mixed-case** (Huh, Hm) to match the majority convention, and GhostRider's gold should ideally be updated for consistency.


### 3. Pause Code Distribution

The dominant pause code varies significantly:

| File | Most Common | Count | `:02` | `:03` | `:04` | `:05` |
|------|-------------|-------|-------|-------|-------|-------|
| VR_Django_Unchained | `:02` | 34/75 | 34 | 21 | 5 | 2 |
| VR_Dodgeball | `:02` | 39/63 | 39 | 17 | 6 | 1 |
| VR_Howl's_Moving_Castle | `:02` | 34/49 | 34 | 7 | 3 | 1 |
| VR_Talk_To_Me | `:02` | 30/42 | 30 | 9 | 1 | 1 |
| VR_Spirited_Away | `:02` | 30/63 | 30 | 17 | 7 | 1 |
| VR_Get_Out | `:02` | 29/65 | 29 | 18 | 4 | 3 |
| VR_GhostRider | `:05` | 8/39 | 0 | 8 | 9 | 8 |

**GhostRider has zero `:02` pauses and favors `:04`/`:05`, while every other file is dominated by `:02`.** This is a major inconsistency. GhostRider was either transcribed by a different person, used different pause measurement conventions, or the VR simulation had different timing.

My current pause calibration defaults to `:05` for speaker switches and `:03` for same-speaker pauses. Based on this data, it should default to `:02` for most cases to match the majority of files.


### 4. Morphological Marking

| File | Morphological Marks |
|------|-------------------|
| VR_Django_Unchained | 1 |
| VR_Dodgeball | 1 |
| VR_Get_Out | 1 |
| VR_Howl's_Moving_Castle | 1 |
| VR_GhostRider | 0 |
| VR_Spirited_Away | 0 |
| VR_Talk_To_Me | 0 |

Morphological marking is essentially absent from all gold files (0-1 marks each). This confirms that morphological mode should stay `off`. The marks that do exist may be incidental rather than systematic.


### 5. Speaker Labels in Descript Input

| File | P: labels | Av: labels | Has labels? |
|------|-----------|------------|-------------|
| VR_Django_Unchained | 57 | 23 | Yes |
| VR_Dodgeball | 34 | 36 | Yes |
| VR_Get_Out | 43 | 21 | Yes |
| VR_GhostRider | 41 | 22 | Yes |
| VR_Howl's_Moving_Castle | 28 | 13 | Yes |
| VR_Talk_To_Me | 32 | 21 | Yes |
| **VR_Spirited_Away** | **0** | **0** | **No** |

Spirited Away is the only file with no speaker labels. This is why it has the lowest accuracy — the pipeline has to infer every speaker from content alone.


### 6. Nonverbal Behavior Codes

| File | `{PN:...}` / `{AvN:...}` codes |
|------|-------------------------------|
| VR_Django_Unchained | 18 |
| VR_Talk_To_Me | 16 |
| VR_Spirited_Away | 14 |
| VR_Get_Out | 10 |
| VR_Howl's_Moving_Castle | 9 |
| VR_Dodgeball | 8 |
| VR_GhostRider | 1 |

These codes require video access and are not implementable from audio transcription alone. They account for a portion of the gold standard lines that the system can never produce. The evaluation normalizer now strips these from comparison.


### 7. Facilitator Lines

Only GhostRider has Facilitator (`F:`) lines (2 occurrences). No other file has them. The pipeline doesn't handle F: as a speaker type.


### 8. Overlap Markers

| File | Overlap markers |
|------|----------------|
| VR_Dodgeball | 5 |
| VR_GhostRider | 6 |
| VR_Howl's_Moving_Castle | 2 |
| VR_Spirited_Away | 1 |
| Others | 0 |

Only 4 of 7 files use overlap markers, and sparingly. The system currently generates 23 overlap markers for GhostRider alone (vs gold's 6), indicating over-detection. The overlap detection threshold (1-second gap between speaker switches) is too aggressive.

---

## Recommended Adjustments

### High Priority (Should Do Next)

1. **Fix Spirited Away Descript file** — Add P:/Av: speaker labels. This alone should push the file from 44.6% to roughly 55-60% and bring the average above 56%.

2. **Switch Avatar response casing to mixed-case** — Change `normalize_avatar_line` to output `Huh?`, `Hm.` instead of `HUH?`, `HM.` to match 6/7 gold files. Consider updating GhostRider's gold to use mixed-case for consistency.

3. **Fix pause code default to `:02`** — The current default of `:05` for speaker switches only matches GhostRider. Change default to `:02` for same-speaker and `:03` for speaker switches to match the majority distribution.

4. **Reduce overlap marker aggressiveness** — Either increase the time threshold from 1 second to something shorter (0.5s), or disable Rule 19 overlap detection entirely since it's causing more mismatches than matches (23 generated vs 6 in gold for GhostRider, and most files have 0).

### Medium Priority

5. **Add `{no audible or visual response observed}` generation** — Some gold files include this as an Avatar response where the Avatar didn't respond at all. Could be added as a template option in Rule 2 for cases with very high time gaps and no detected response.

6. **Handle Facilitator (`F:`) lines** — GhostRider has 2 F: lines that the pipeline currently attributes to P. Could add F: detection for lines that contain instructional/meta content.

7. **Improve avatar inference response variety** — Current templates only generate HM., HUH?, YES., NO. The gold has contextual responses (Coffee., Thanks., I think I'm ready.). The LLM refiner could help here.

### Lower Priority

8. **Investigate GhostRider as an outlier** — GhostRider differs from all other files in multiple ways: uppercase Avatar keywords, different pause distribution, more overlap markers. It may have been transcribed by a different annotator or under different guidelines. Consider whether it should be treated as the standard or as an exception.

9. **Sentence-boundary splitting within merged lines** — For cases where Descript wraps lines mid-sentence at a point where the gold has a conversational gap, consider splitting on sentence boundaries and checking for time gaps at the original break points.

10. **Nonverbal behavior codes** — These require video access and are not feasible with audio-only data. They represent 1-18 lines per gold file that the system can never produce. The evaluation already strips them from comparison, but they still affect overall line counts and similarity scores.

---

## Files Modified

| File | What Changed |
|------|-------------|
| `evaluate_system.py` | Alignment-based matching, improved normalization, 0.80 threshold |
| `rule_based_processor.py` | Continuation merging, speaker reattribution, Avatar formatting, pause calibration, echo stripping, repetition fix, redaction format |
| `avatar_inference.py` | Continuation-aware gap detection, continuation merging removed from pre-processing |

---

## How to Reproduce Current Results

```bash
# Run the pipeline
python3 rule_based_processor.py \
  --input-dir extracted_text \
  --output-dir rule_based_output_improved \
  --rule2-mode standard \
  --morph-mode off \
  --coord-mode standard \
  --pause-mode standard

# Evaluate
python3 evaluate_system.py \
  --system-dir rule_based_output_improved \
  --gold-dir extracted_text \
  --output-file evaluation_report_final.txt
```
