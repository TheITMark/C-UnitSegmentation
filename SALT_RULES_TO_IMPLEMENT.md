# SALT C-Unit Segmentation Rules - Implementation Checklist

## **CRITICAL INPUT PROCESSING RULES**

### **Rule 1: Descript Input Parsing**
- **Status**: 🔴 BROKEN - Over-splitting sentences
- **Fix**: Parse `[HH:MM:SS] Speaker: Content` format correctly
- **Current Issue**: Splits every sentence on periods instead of preserving natural C-units
- **Implementation**: Fix `process_transcript()` function lines 600-650

### **Rule 2: Speaker Detection & Normalization**
- **Status**: 🟢 WORKING - Missing Avatar responses
- **SALT Rule**: `P:` = Participant, `Av:` = Avatar
- **Fix**: Detect both P: and Av: speakers from Descript input
- **Implementation**: Enhance speaker detection in content parsing
- **Fix implemented**: avatar_inference.py — standalone module with gap detection, context classification, and template-based response generation (LLM-ready when we want to enable it with --use-llm)
Integrated into rule_based_processor.py as a pipeline stage (toggle with --no-avatar-inference)
11 unit tests, all passing
Bug Fixed: A bug where standalone acknowledgments (HM., HUH?) were incorrectly marked as filled pauses. 

### **Rule 3: Timestamp to SALT Time Marker Conversion**
- **Status**: 🟢 WORKING - Basic conversion works
- **SALT Rule**: Convert `[00:01:30]` → `-1:30` (time markers)
- **Implementation**: `format_salt_timestamp()` function works but needs integration
- **Fix implemented**: fix implemented to get the format and conversion right. Tested and passed 100%
---

## **C-UNIT SEGMENTATION RULES**

### **Rule 4: C-Unit Definition (CORE RULE)**
- **Status**: 🔴 BROKEN - Fundamental misunderstanding
- **SALT Rule**: C-unit = independent clause + all its modifiers + subordinate clauses
- **Current Issue**: Splits on every period instead of linguistic boundaries
- **Fix**: Complete rewrite of `segment_cunits()` function

### **Rule 5: Coordinating Conjunction Splitting**
- **Status**: 🟡 PARTIAL - Logic exists but not working
- **SALT Rule**: Split on `and`, `or`, `but`, `so` (not "so that"), `then`
- **Example**: `"I went home and I ate dinner"` → 2 C-units
- **Implementation**: Fix conjunction detection in `segment_cunits()`

### **Rule 6: Subordinating Conjunction Preservation**
- **Status**: 🟡 PARTIAL - List exists but not applied
- **SALT Rule**: Keep together: `because`, `that`, `when`, `who`, `after`, `before`, `which`, `although`, `if`, `unless`, `while`, `as`, `how`, `until`, `like`, `where`, `since`
- **Example**: `"I went home because I was tired"` → 1 C-unit
- **Implementation**: Enhance subordinating conjunction logic

### **Rule 7: "So" Disambiguation**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `"so"` = coordinating (split) UNLESS `"so that"` = subordinating (keep together)
- **Example**: `"I was tired so I went home"` → 2 C-units
- **Example**: `"I went home so that I could rest"` → 1 C-unit
- **Implementation**: Add "so that" detection logic

### **Rule 8: Yes/No/Okay Responses**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `yes`, `no`, `okay`, `yeah`, `nope`, `nah` = separate C-units
- **Example**: `P: Are you ready? Av: Yes.` → 2 C-units
- **Implementation**: Add response detection

### **Rule 9: Conjunction Reduction (CONJRED)**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `"And [verb]"` with missing subject = new C-unit
- **Example**: `"I went home. And ate dinner."` → 2 C-units
- **Implementation**: Add CONJRED detection (without *CONJRED tag per 2023 update)

### **Rule 10: Tags and Questions**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: Keep `"you know"`, `"I guess"`, `"I mean"`, `"right?"` within C-unit
- **Example**: `"She went home, you know."` → 1 C-unit
- **Implementation**: Add tag detection and preservation

---

## **PAUSE TIMING RULES**

### **Rule 11: Intra-utterance Pauses**
- **Status**: 🟡 PARTIAL - Basic logic exists
- **SALT Rule**: `:` for pauses within same speaker's C-unit
- **SALT Rule**: `≥2 seconds` = `:05` (with duration)
- **SALT Rule**: `1.5-1.9 seconds` = `:` (before next word)
- **Implementation**: Fix `format_pause_code()` function

### **Rule 12: Inter-utterance Pauses**
- **Status**: 🟡 PARTIAL - Basic logic exists
- **SALT Rule**: `;` for pauses between different speakers or C-units
- **SALT Rule**: `≥2 seconds` = `; :05` (with duration)
- **SALT Rule**: `1.5-1.9 seconds between same speaker` = `:` at start of next C-unit
- **Implementation**: Enhance pause timing logic

### **Rule 13: Pause Duration Calculation**
- **Status**: 🟡 PARTIAL - Function exists but needs fixing
- **SALT Rule**: Round to nearest second, round up at 0.5 seconds
- **Example**: `2.3s → :02`, `1.5s → :02`
- **Implementation**: Fix rounding in `calculate_pause_duration()`

---

## **MORPHOLOGICAL MARKING RULES**

### **Rule 14: Verb Morphology (spaCy Integration)**
- **Status**: 🟡 PARTIAL - spaCy code exists but not working
- **SALT Rule**: Mark bound morphemes with `/`
- **Examples**: `looked → look/ed`, `jumping → jump/ing`, `cars → car/s`
- **Implementation**: Debug `apply_morphological_marking_spacy()` function

### **Rule 15: Common Verb Patterns**
- **Status**: 🟡 PARTIAL - Fallback patterns exist
- **SALT Rule**: Past tense verbs need `/ed` marking
- **Examples**: `dressed → dress/ed`, `packed → pack/ed`, `crashed → crash/ed`
- **Implementation**: Expand fallback patterns in `apply_morphological_marking_fallback()`

---

## **MAZE AND FILLED PAUSE RULES**

### **Rule 16: Filled Pause Detection**
- **Status**: 🟡 PARTIAL - Basic detection exists
- **SALT Rule**: `(uh [FP])`, `(um [FP])`, `(hm [FP])`, `(er [FP])`, `(ah [FP])`
- **List**: `uh`, `um`, `hm`, `hmm`, `er`, `ah`, `oh`, `like`
- **Implementation**: Enhance `detect_filled_pauses_enhanced()` function

### **Rule 17: Repetition and Maze Detection**
- **Status**: 🟡 PARTIAL - Basic patterns exist
- **SALT Rule**: Repetitions and false starts in parentheses `()`
- **Examples**: `"I, I want"` → `"(I) I want"`, `"The (woma*) woman"`
- **Implementation**: Enhance `detect_repetitions_and_mazes()` function

### **Rule 18: Interjections in Mazes**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `"The man (I think it's a man) was reading"`
- **Implementation**: Add interjection detection within mazes

---

## **SPECIAL CODES AND FORMATTING RULES**

### **Rule 19: Overlapping Speech**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `< >` around overlapping segments
- **Example**: `P: Do you want <apples> Av: <yes>`
- **Implementation**: Add overlap detection

### **Rule 20: Abandoned Utterances**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `>` for incomplete thoughts (not interrupted)
- **Example**: `"And then she>" "I want dinner"`
- **Implementation**: Add abandonment detection

### **Rule 21: Unintelligible Speech**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `X` (word), `XX` (segment), `XXX` (utterance)
- **Implementation**: Add unintelligible detection

### **Rule 22: Omissions and Partial Words**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `*` after letters for unfinished words `(w*)`
- **SALT Rule**: `*` before words for omitted words `*to`
- **SALT Rule**: `/*` for omitted morphemes `go/*s`
- **Implementation**: Add omission detection

### **Rule 23: Linked Words**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `_` to link multi-word units
- **Examples**: `fire_truck`, `Mr_Frog`, `%woof_woof`
- **Implementation**: Add compound word detection

### **Rule 24: Sound Effects**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `%` prefix for sound effects
- **Example**: `"The dog said %woof_woof"`
- **Implementation**: Add sound effect detection

---

## **HEADER AND FORMATTING RULES**

### **Rule 25: SALT Header Generation**
- **Status**: 🟡 PARTIAL - Basic header exists
- **SALT Rule**: Speaker identification, metadata, time markers
- **Required**: `$Av= Avatar, $P=Participant`, `+ Language: English`, etc.
- **Implementation**: Enhance `add_salt_header()` function

### **Rule 26: End-of-Utterance Punctuation**
- **Status**: 🟡 PARTIAL - Basic punctuation exists
- **SALT Rule**: `.` (statement), `!` (exclamation), `?` (question)
- **Implementation**: Ensure proper punctuation assignment

### **Rule 27: Redaction Handling**
- **Status**: 🟢 WORKING - Function exists and works
- **SALT Rule**: `[redacted]` → `{redacted}`
- **Implementation**: `handle_redactions()` function works

### **Rule 28: Nonverbal Behavior Codes**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: `{PN: touches shoulder}`, `{AvN: smiles}`
- **SALT Rule**: Context without colon: `{Av laying in bed}`
- **Implementation**: Add nonverbal detection and formatting

---

## **SPELLING AND LEXICAL CONVENTIONS**

### **Rule 29: Accepted Spelling Variants**
- **Status**: 🔴 NOT IMPLEMENTED
- **SALT Rule**: Standardize variants
- **Yes words**: `okay`, `ok`, `yeah`, `yup`, `yes`
- **No words**: `no`, `nope`, `nah`
- **Contractions**: `gonna`, `gotta`, `wanna`, `hafta`, `oughta`, `betcha`
- **Other**: `ain't`, `oh`, `uhoh`, `oops`, `ooh`, `huh`, `hmm`
- **Implementation**: Add lexical normalization

---

## **PRIORITY IMPLEMENTATION ORDER**

### **Phase 1: Critical Fixes (Week 1)**
1. **Rule 1**: Fix Descript input parsing
2. **Rule 2**: Fix speaker detection
3. **Rule 4**: Rewrite C-unit segmentation core logic
4. **Rule 5**: Fix coordinating conjunction splitting

### **Phase 2: Core Segmentation (Week 2)**
5. **Rule 6**: Subordinating conjunction preservation
6. **Rule 7**: "So" vs "so that" disambiguation
7. **Rule 8**: Yes/No/Okay response handling
8. **Rule 14**: Enable spaCy morphological marking

### **Phase 3: Pause and Timing (Week 3)**
9. **Rule 11**: Intra-utterance pause timing
10. **Rule 12**: Inter-utterance pause timing
11. **Rule 16**: Enhanced filled pause detection
12. **Rule 17**: Maze and repetition detection

### **Phase 4: Advanced Features (Week 4)**
13. **Rule 9**: Conjunction reduction (CONJRED)
14. **Rule 19**: Overlapping speech detection
15. **Rule 25**: Complete SALT header generation
16. **Rule 29**: Lexical normalization

---

## **TESTING STRATEGY**

### **Test Each Rule Individually**
- Create unit tests for each rule
- Test with gold standard examples
- Measure accuracy improvement per rule

### **Integration Testing**
- Test rule combinations
- Ensure rules don't conflict
- Validate against full transcripts

### **Regression Testing**
- Ensure new rules don't break existing functionality
- Maintain accuracy benchmarks
- Test edge cases and corner cases
