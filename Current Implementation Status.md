Current Implementation Status
=============================

Date: 2026-04-24
Source of truth: current codebase scan (not older status docs)

Summary
-------
- Implemented: 17/29
- Partial: 8/29
- Not implemented: 4/29

Implemented (Strong Evidence)
-----------------------------
- Rule 2: Speaker Detection & Normalization (with avatar inference integration)
- Rule 3: Timestamp to SALT Time Marker Conversion
- Rule 4: C-Unit Definition (core segmentation logic active)
- Rule 5: Coordinating Conjunction Splitting
- Rule 6: Subordinating Conjunction Preservation
- Rule 7: "So" Disambiguation
- Rule 8: Yes/No/Okay Responses
- Rule 9: Conjunction Reduction (CONJRED)
- Rule 10: Tags and Questions
- Rule 14: Verb Morphology (implemented as optional `spacy` mode; currently not best for eval)
- Rule 16: Filled Pause Detection
- Rule 18: Interjections in Mazes
- Rule 19: Overlapping Speech
- Rule 20: Abandoned Utterances
- Rule 23: Linked Words
- Rule 27: Redaction Handling
- Rule 29: Accepted Spelling Variants (Lexical Normalization)

Partial Implementation
----------------------
- Rule 1: Descript Input Parsing
- Rule 11: Intra-utterance Pauses
- Rule 12: Inter-utterance Pauses
- Rule 13: Pause Duration Calculation
- Rule 15: Common Verb Patterns
- Rule 17: Repetition and Maze Detection
- Rule 25: SALT Header Generation
- Rule 26: End-of-Utterance Punctuation

Likely Not Implemented
----------------------
- Rule 21: Unintelligible Speech
- Rule 22: Omissions and Partial Words
- Rule 24: Sound Effects
- Rule 28: Nonverbal Behavior Codes

Notes
-----
- Existing status docs are retained and not modified.
- This file reflects the current code state and test/CLI evidence in `rule_based_processor.py`
  and `avatar_inference.py`.
- Rule 14 is now implemented with configurable modes (`regex`, `spacy`, `off`) and tested, but
  A/B evaluation currently shows lower average overall similarity in `spacy` mode vs `off` for
  this dataset, so `spacy` should be treated as optional/experimental.
