#!/usr/bin/env python3
"""
System Evaluation Script for C-Unit Segmentation
Compares our hybrid system output against human-coded gold standard transcripts

Author: Yugant Soni + AI Assistant
Date: December 2024
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set
import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import json


@dataclass
class TranscriptMetrics:
    """Container for transcript comparison metrics"""
    file_name: str
    total_lines_system: int
    total_lines_gold: int
    c_unit_accuracy: float
    pause_accuracy: float
    filled_pause_accuracy: float
    morphological_accuracy: float
    speaker_accuracy: float
    overall_similarity: float
    detailed_comparison: Dict


class SALTEvaluator:
    """Evaluates SALT transcript quality against gold standard"""
    
    def __init__(self):
        self.results = []
    
    # --- Normalization helpers for robust matching ---
    def _canonical_stem(self, filename: str) -> str:
        """Create a canonical key from a filename for pairing.

        - Remove extension
        - Drop trailing parenthetical suffixes like (FLAN-T5 Refined), (Orthographic Segmented Transcript)
        - Remove trailing "_Transcript" or " Transcript"
        - Collapse whitespace and lowercase
        """
        s = filename
        s = re.sub(r"\.txt$", "", s, flags=re.IGNORECASE)
        # remove any trailing parenthetical block
        s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
        # remove trailing _Transcript or  Transcript
        s = re.sub(r"(?:[_ ]Transcript)\s*$", "", s, flags=re.IGNORECASE)
        # collapse spaces and lowercase
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def _norm_line(self, s: str) -> str:
        """Normalize a line for fuzzy comparison."""
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        # strip any inline timestamps like [HH:MM:SS]
        s = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", s)
        # strip {inferred} markers from system output
        s = re.sub(r"\s*\{inferred\}", "", s)
        # strip {PN:...} nonverbal behavior codes from gold
        s = re.sub(r"\s*\{[^}]*\}", "", s)
        # normalize filled pause format variations
        s = re.sub(r"\(\s*(\w+)\s*\[fp\]\s*\)", r"(\1 [fp])", s)
        # normalize redaction format
        s = re.sub(r"\[redacted\]", "{redacted}", s)
        # strip overlap markers <...>
        s = re.sub(r"<([^>]+)>", r"\1", s)
        # normalize commas in filled pauses: "(um [fp])," vs "(um [fp])"
        s = re.sub(r"\)\s*,", ")", s)
        # normalize punctuation: ".!" or ".?" -> just the last
        s = re.sub(r"\.([!?])", r"\1", s)
        # strip trailing whitespace within content
        s = re.sub(r"\s+([.!?])", r"\1", s)
        # normalize "i said," vs "i said"
        s = re.sub(r",\s*$", "", s)
        return s.strip()
        
    def extract_c_units(self, text: str) -> List[str]:
        """Extract C-units (lines starting with P: or Av:)"""
        lines = text.strip().split('\n')
        c_units = []
        for line in lines:
            line = line.strip()
            if line.startswith('P:') or line.startswith('Av:'):
                c_units.append(line)
        return c_units
    
    def extract_pause_codes(self, text: str) -> List[str]:
        """Extract pause codes (lines starting with ; or :)"""
        lines = text.strip().split('\n')
        pauses = []
        for line in lines:
            line = line.strip()
            if line.startswith(';') or line.startswith(':'):
                pauses.append(line)
        return pauses
    
    def extract_filled_pauses(self, text: str) -> List[str]:
        """Extract filled pause patterns (text with [FP])"""
        filled_pauses = re.findall(r'\([^)]*\[FP\][^)]*\)', text)
        return filled_pauses
    
    def extract_morphological_marks(self, text: str) -> List[str]:
        """Extract morphological markings (words with /)"""
        morph_marks = re.findall(r'\b\w+/\w+\b', text)
        return morph_marks
    
    def extract_speaker_lines(self, text: str) -> List[str]:
        """Extract all speaker-prefixed lines"""
        lines = text.strip().split('\n')
        speaker_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('P:') or line.startswith('Av:'):
                speaker_lines.append(line[:2])  # Just the speaker code
        return speaker_lines
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate overall text similarity using difflib on normalized lines."""
        lines1 = [self._norm_line(l) for l in text1.strip().split('\n') if l.strip()]
        lines2 = [self._norm_line(l) for l in text2.strip().split('\n') if l.strip()]
        matcher = SequenceMatcher(None, lines1, lines2)
        return matcher.ratio()
    
    def compare_lists(self, system_list: List[str], gold_list: List[str]) -> float:
        """Compare two lists using alignment-based matching.

        Uses SequenceMatcher on normalized lines to find the best alignment
        between system and gold lists, then counts matching pairs.  This avoids
        the cascading-mismatch problem of strict index-aligned comparison.
        """
        if not gold_list:
            return 1.0 if not system_list else 0.0
        if not system_list:
            return 0.0

        sys_norm = [self._norm_line(s) for s in system_list]
        gold_norm = [self._norm_line(g) for g in gold_list]

        # Use SequenceMatcher to align the two lists of normalized strings
        matcher = SequenceMatcher(None, sys_norm, gold_norm)
        matches = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                matches += (i2 - i1)
            elif tag == 'replace':
                # Within replaced blocks, count fuzzy matches pair-wise
                for k in range(min(i2 - i1, j2 - j1)):
                    if SequenceMatcher(None, sys_norm[i1 + k], gold_norm[j1 + k]).ratio() >= 0.80:
                        matches += 1

        max_len = max(len(system_list), len(gold_list))
        return matches / max_len if max_len > 0 else 0.0
    
    def analyze_c_unit_segmentation(self, system_text: str, gold_text: str) -> Dict:
        """Detailed analysis of C-unit segmentation quality"""
        system_cunits = self.extract_c_units(system_text)
        gold_cunits = self.extract_c_units(gold_text)
        
        analysis = {
            'system_count': len(system_cunits),
            'gold_count': len(gold_cunits),
            'count_difference': abs(len(system_cunits) - len(gold_cunits)),
            'sample_system': system_cunits[:3],
            'sample_gold': gold_cunits[:3],
        }
        
        return analysis

    def summarize_mismatches(self, system_text: str, gold_text: str) -> Dict:
        """Create compact diagnostics to explain major mismatch patterns."""
        system_lines = [l.strip() for l in system_text.split('\n') if l.strip()]
        gold_lines = [l.strip() for l in gold_text.split('\n') if l.strip()]
        system_cunits = self.extract_c_units(system_text)
        gold_cunits = self.extract_c_units(gold_text)
        system_pauses = self.extract_pause_codes(system_text)
        gold_pauses = self.extract_pause_codes(gold_text)

        # Speaker sequence mismatches at aligned indices
        speaker_mismatches = 0
        system_speakers = self.extract_speaker_lines(system_text)
        gold_speakers = self.extract_speaker_lines(gold_text)
        for idx in range(min(len(system_speakers), len(gold_speakers))):
            if system_speakers[idx] != gold_speakers[idx]:
                speaker_mismatches += 1

        # Find representative unmatched C-units from each side
        unmatched_system = []
        unmatched_gold = []
        gold_norm = [self._norm_line(x) for x in gold_cunits]
        system_norm = [self._norm_line(x) for x in system_cunits]

        for line in system_cunits:
            nl = self._norm_line(line)
            if nl not in gold_norm:
                unmatched_system.append(line)
            if len(unmatched_system) >= 5:
                break

        for line in gold_cunits:
            nl = self._norm_line(line)
            if nl not in system_norm:
                unmatched_gold.append(line)
            if len(unmatched_gold) >= 5:
                break

        return {
            "line_count_delta": len(system_lines) - len(gold_lines),
            "cunit_count_delta": len(system_cunits) - len(gold_cunits),
            "pause_count_delta": len(system_pauses) - len(gold_pauses),
            "speaker_mismatch_count": speaker_mismatches,
            "sample_unmatched_system_cunits": unmatched_system,
            "sample_unmatched_gold_cunits": unmatched_gold,
        }
    
    def evaluate_transcript(self, system_file: Path, gold_file: Path) -> TranscriptMetrics:
        """Evaluate a single transcript against gold standard"""
        
        # Read files
        with open(system_file, 'r', encoding='utf-8') as f:
            system_text = f.read()
        
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_text = f.read()
        
        # Extract components
        system_cunits = self.extract_c_units(system_text)
        gold_cunits = self.extract_c_units(gold_text)
        
        system_pauses = self.extract_pause_codes(system_text)
        gold_pauses = self.extract_pause_codes(gold_text)
        
        system_fp = self.extract_filled_pauses(system_text)
        gold_fp = self.extract_filled_pauses(gold_text)
        
        system_morph = self.extract_morphological_marks(system_text)
        gold_morph = self.extract_morphological_marks(gold_text)
        
        system_speakers = self.extract_speaker_lines(system_text)
        gold_speakers = self.extract_speaker_lines(gold_text)
        
        # Calculate accuracies
        c_unit_accuracy = self.compare_lists(system_cunits, gold_cunits)
        pause_accuracy = self.compare_lists(system_pauses, gold_pauses)
        fp_accuracy = self.compare_lists(system_fp, gold_fp)
        morph_accuracy = self.compare_lists(system_morph, gold_morph)
        speaker_accuracy = self.compare_lists(system_speakers, gold_speakers)
        overall_similarity = self.calculate_similarity(system_text, gold_text)
        
        # Detailed comparison
        detailed = {
            'c_unit_analysis': self.analyze_c_unit_segmentation(system_text, gold_text),
            'pause_counts': {
                'system': len(system_pauses),
                'gold': len(gold_pauses)
            },
            'filled_pause_counts': {
                'system': len(system_fp),
                'gold': len(gold_fp)
            },
            'morphological_counts': {
                'system': len(system_morph),
                'gold': len(gold_morph)
            },
            'mismatch_diagnostics': self.summarize_mismatches(system_text, gold_text),
        }
        
        return TranscriptMetrics(
            file_name=system_file.name,
            total_lines_system=len(system_text.split('\n')),
            total_lines_gold=len(gold_text.split('\n')),
            c_unit_accuracy=c_unit_accuracy,
            pause_accuracy=pause_accuracy,
            filled_pause_accuracy=fp_accuracy,
            morphological_accuracy=morph_accuracy,
            speaker_accuracy=speaker_accuracy,
            overall_similarity=overall_similarity,
            detailed_comparison=detailed
        )
    
    def find_matching_files(self, system_dir: Path, gold_dir: Path) -> List[Tuple[Path, Path]]:
        """Find matching system and gold standard files using canonical stems."""
        matches: List[Tuple[Path, Path]] = []
        
        # All system outputs - support both FLAN-T5 and Rule-Based files
        system_files = list(system_dir.glob("*FLAN-T5 Refined*.txt"))
        if not system_files:
            # Try Rule-Based Processed files
            system_files = list(system_dir.glob("*Rule-Based Processed*.txt"))
        if not system_files:
            # Try alignment postprocessor outputs
            system_files = list(system_dir.glob("*Aligned Processed*.txt"))
        
        # Index gold files by canonical stem
        gold_index: Dict[str, Path] = {}
        for gold_file in gold_dir.glob("*Orthographic Segmented Transcript*.txt"):
            key = self._canonical_stem(gold_file.name)
            gold_index[key] = gold_file
        
        for system_file in system_files:
            key = self._canonical_stem(system_file.name)
            gold_file = gold_index.get(key)
            if gold_file is not None:
                matches.append((system_file, gold_file))
                print(f"✅ Found match: {system_file.name} ↔ {gold_file.name}")
            else:
                print(f"⚠️  No gold standard found for: {system_file.name}")
        
        return matches
    
    def evaluate_system(self, system_dir: Path, gold_dir: Path) -> List[TranscriptMetrics]:
        """Evaluate entire system against gold standard"""
        
        print("🔍 Finding matching transcript pairs...")
        matches = self.find_matching_files(system_dir, gold_dir)
        
        if not matches:
            print("❌ No matching files found!")
            return []
        
        print(f"📊 Evaluating {len(matches)} transcript pairs...")
        print("-" * 80)
        
        results = []
        for system_file, gold_file in matches:
            print(f"📄 Evaluating: {system_file.name}")
            metrics = self.evaluate_transcript(system_file, gold_file)
            results.append(metrics)
            
            # Print summary for this file
            print(f"   C-Unit Accuracy: {metrics.c_unit_accuracy:.2%}")
            print(f"   Pause Accuracy: {metrics.pause_accuracy:.2%}")
            print(f"   Filled Pause Accuracy: {metrics.filled_pause_accuracy:.2%}")
            print(f"   Morphological Accuracy: {metrics.morphological_accuracy:.2%}")
            print(f"   Overall Similarity: {metrics.overall_similarity:.2%}")
            print()
        
        return results
    
    def generate_report(self, results: List[TranscriptMetrics]) -> str:
        """Generate comprehensive evaluation report"""
        if not results:
            return "No results to report."
        
        # Calculate averages
        avg_c_unit = sum(r.c_unit_accuracy for r in results) / len(results)
        avg_pause = sum(r.pause_accuracy for r in results) / len(results)
        avg_fp = sum(r.filled_pause_accuracy for r in results) / len(results)
        avg_morph = sum(r.morphological_accuracy for r in results) / len(results)
        avg_speaker = sum(r.speaker_accuracy for r in results) / len(results)
        avg_similarity = sum(r.overall_similarity for r in results) / len(results)
        
        report = f"""
═══════════════════════════════════════════════════════════════════════════════
                    C-UNIT SEGMENTATION SYSTEM EVALUATION REPORT
═══════════════════════════════════════════════════════════════════════════════

📊 OVERALL SYSTEM PERFORMANCE:
├─ Files Evaluated: {len(results)}
├─ Average C-Unit Accuracy: {avg_c_unit:.1%}
├─ Average Pause Accuracy: {avg_pause:.1%}
├─ Average Filled Pause Accuracy: {avg_fp:.1%}
├─ Average Morphological Accuracy: {avg_morph:.1%}
├─ Average Speaker Accuracy: {avg_speaker:.1%}
└─ Average Overall Similarity: {avg_similarity:.1%}

📈 PERFORMANCE GRADE: {self._get_performance_grade(avg_similarity)}

📋 DETAILED RESULTS BY FILE:
"""
        
        for i, result in enumerate(results, 1):
            report += f"""
{i}. {result.file_name}
   ├─ Lines: {result.total_lines_system} (System) vs {result.total_lines_gold} (Gold)
   ├─ C-Unit Accuracy: {result.c_unit_accuracy:.1%}
   ├─ Pause Accuracy: {result.pause_accuracy:.1%}
   ├─ Filled Pause Accuracy: {result.filled_pause_accuracy:.1%}
   ├─ Morphological Accuracy: {result.morphological_accuracy:.1%}
   └─ Overall Similarity: {result.overall_similarity:.1%}
"""
        
        report += f"""
🔍 SYSTEM ANALYSIS:
├─ Strengths: {self._identify_strengths(results)}
├─ Areas for Improvement: {self._identify_weaknesses(results)}
└─ Recommendation: {self._get_recommendation(avg_similarity)}

═══════════════════════════════════════════════════════════════════════════════
"""
        
        return report
    
    def _get_performance_grade(self, similarity: float) -> str:
        """Get performance grade based on similarity"""
        if similarity >= 0.90:
            return "A+ (Excellent)"
        elif similarity >= 0.80:
            return "A (Very Good)"
        elif similarity >= 0.70:
            return "B (Good)"
        elif similarity >= 0.60:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"
    
    def _identify_strengths(self, results: List[TranscriptMetrics]) -> str:
        """Identify system strengths"""
        avg_speaker = sum(r.speaker_accuracy for r in results) / len(results)
        avg_fp = sum(r.filled_pause_accuracy for r in results) / len(results)
        avg_c_unit = sum(r.c_unit_accuracy for r in results) / len(results)
        
        strengths = []
        if avg_speaker > 0.9:
            strengths.append("Speaker identification")
        if avg_fp > 0.8:
            strengths.append("Filled pause detection")
        if avg_c_unit > 0.7:
            strengths.append("C-unit segmentation")
        
        return ", ".join(strengths) if strengths else "Basic formatting"
    
    def _identify_weaknesses(self, results: List[TranscriptMetrics]) -> str:
        """Identify areas needing improvement"""
        avg_pause = sum(r.pause_accuracy for r in results) / len(results)
        avg_morph = sum(r.morphological_accuracy for r in results) / len(results)
        
        weaknesses = []
        if avg_pause < 0.7:
            weaknesses.append("Pause timing accuracy")
        if avg_morph < 0.7:
            weaknesses.append("Morphological marking")
        
        return ", ".join(weaknesses) if weaknesses else "Minor refinements needed"
    
    def _get_recommendation(self, similarity: float) -> str:
        """Get recommendation based on performance"""
        if similarity >= 0.85:
            return "System ready for production use with minimal supervision"
        elif similarity >= 0.75:
            return "System suitable for use with expert review"
        else:
            return "System needs further refinement before deployment"


def main():
    """Command line interface for system evaluation"""
    parser = argparse.ArgumentParser(
        description="Evaluate C-unit segmentation system against gold standard"
    )
    parser.add_argument(
        "--system-dir",
        type=Path,
        default=Path("/Users/yuganthareshsoni/CunitSegementation/llm_refined_output"),
        help="Directory containing system output files"
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("/Users/yuganthareshsoni/CunitSegementation/extracted_text"),
        help="Directory containing gold standard files"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("/Users/yuganthareshsoni/CunitSegementation/evaluation_report.txt"),
        help="Output file for evaluation report"
    )
    parser.add_argument(
        "--diagnostics-file",
        type=Path,
        default=None,
        help="Optional JSON file for compact mismatch diagnostics summary"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SALTEvaluator()
    
    # Run evaluation
    print("🚀 Starting System Evaluation...")
    print(f"📁 System Directory: {args.system_dir}")
    print(f"📁 Gold Standard Directory: {args.gold_dir}")
    print("=" * 80)
    
    results = evaluator.evaluate_system(args.system_dir, args.gold_dir)
    
    if results:
        # Generate report
        report = evaluator.generate_report(results)
        
        # Save report
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Display report
        print(report)
        print(f"📄 Full report saved to: {args.output_file}")
        
        # Save detailed results as JSON
        json_file = args.output_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([
                {
                    'file_name': r.file_name,
                    'c_unit_accuracy': r.c_unit_accuracy,
                    'pause_accuracy': r.pause_accuracy,
                    'filled_pause_accuracy': r.filled_pause_accuracy,
                    'morphological_accuracy': r.morphological_accuracy,
                    'speaker_accuracy': r.speaker_accuracy,
                    'overall_similarity': r.overall_similarity,
                    'detailed_comparison': r.detailed_comparison
                }
                for r in results
            ], f, indent=2)
        
        print(f"📊 Detailed data saved to: {json_file}")

        if args.diagnostics_file is not None:
            diagnostics = []
            for r in results:
                diagnostics.append({
                    "file_name": r.file_name,
                    "overall_similarity": r.overall_similarity,
                    "c_unit_accuracy": r.c_unit_accuracy,
                    "pause_accuracy": r.pause_accuracy,
                    "speaker_accuracy": r.speaker_accuracy,
                    "mismatch_diagnostics": r.detailed_comparison.get("mismatch_diagnostics", {}),
                })
            with open(args.diagnostics_file, 'w', encoding='utf-8') as f:
                json.dump(diagnostics, f, indent=2)
            print(f"🧭 Diagnostics saved to: {args.diagnostics_file}")
        
        return 0
    else:
        print("❌ No files could be evaluated")
        return 1


if __name__ == "__main__":
    exit(main())
