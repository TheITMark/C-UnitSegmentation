#!/usr/bin/env python3
"""
LLM Refinement Module for C-Unit Segmentation
Stage 2: Handle complex linguistic decisions using Gemma 2B

Author: Yugant Soni + AI Assistant
Date: December 2024
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import json
from dataclasses import dataclass

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    print("✅ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Error importing transformers: {e}")
    print("Please install with: pip install transformers torch")
    exit(1)


@dataclass
class SALTExample:
    """Container for SALT transformation examples"""
    input_text: str
    expected_output: str
    transformation_type: str


class SALTPromptBuilder:
    """Builds prompts for SALT formatting tasks"""
    
    def __init__(self):
        self.system_prompt = self._build_system_prompt()
        self.examples = self._load_examples()
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt with SALT rules"""
        return """You are a specialized linguistic assistant for SALT (Systematic Analysis of Language Transcripts) formatting. Your task is to refine partially processed transcripts according to SALT conventions.

KEY RULES:
1. C-UNIT SEGMENTATION:l fuc 
   - Split coordinating conjunctions (and, or, but, so, then) into separate c-units
   - Keep subordinating conjunctions (because, that, when, who, after, before, so that, which, although, if, unless, while, as, how, until, like, where, since) as one c-unit
   - Treat yes/no/okay responses as separate c-units

2. MAZE DETECTION:
   - Identify false starts, repetitions, reformulations
   - Format as: (maze content) final form
   - Use [FP] for filled pauses within mazes: (uh [FP])

3. MORPHOLOGICAL MARKING:
   - Add /ed, /s, /ing to verbs and words: look/ed, pack/ed, jump/s

4. PAUSE REFINEMENT:
   - Use : for intra-utterance pauses
   - Use ; for inter-utterance pauses
   - Add duration if ≥2 seconds: :03, ; :05

5. SPECIAL CODES:
   - Overlapping speech: <overlap>
   - Abandoned utterances: utterance>
   - Unintelligible: X (word), XX (segment), XXX (utterance)

OUTPUT FORMAT: Return only the refined transcript text, no explanations."""

    def _load_examples(self) -> List[SALTExample]:
        """Load few-shot examples for in-context learning"""
        return [
            SALTExample(
                input_text="P: Let's go and get dressed and we'll get some breakfast.",
                expected_output="P: Let's go and get dressed.\nP: And we'll get some breakfast.",
                transformation_type="coordinating_conjunction"
            ),
            SALTExample(
                input_text="Av: (Um, [FP]) I, (um, [FP]) (um, [FP]) I need to go.",
                expected_output="Av: (Um [FP]) (I) (um [FP]) (um [FP]) (I) I need to go.",
                transformation_type="maze_refinement"
            ),
            SALTExample(
                input_text="Av: When the boy look in the jar he saw that the frog was missing.",
                expected_output="Av: When the boy look/ed in the jar he saw that the frog was missing.",
                transformation_type="morphological_marking"
            ),
            SALTExample(
                input_text="P: Do you want apples Av: yes",
                expected_output="P: Do you want <apples>\nAv: <Yes>.",
                transformation_type="overlap_detection"
            )
        ]
    
    def build_prompt(self, input_text: str, task_type: str = "comprehensive") -> str:
        """Build prompt for specific SALT refinement task"""
        
        # Add few-shot examples
        examples_text = ""
        for example in self.examples[:2]:  # Use first 2 examples
            examples_text += f"\nInput: {example.input_text}\nOutput: {example.expected_output}\n"
        
        prompt = f"""{self.system_prompt}

EXAMPLES:{examples_text}

Now refine this transcript according to SALT conventions:

Input: {input_text}
Output:"""
        
        return prompt


class FlanT5Processor:
    """Handles FLAN-T5-Small model operations - lightweight and fast"""
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        local_only: bool = True,
    ):
        self.model_name = model_name
        self.local_only = local_only
        # Force CPU for stability with FLAN-T5-Small (it's lightweight enough)
        self.device = "cpu"
        print(f"🔧 Using device: {self.device} (forced CPU for stability)")
        if self.local_only:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            print("🔒 Local-only mode enabled (no cloud model access).")
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.available = False
        self._load_model()
    
    def _load_model(self):
        """Load FLAN-T5-Small model and tokenizer"""
        try:
            print(f"📥 Loading {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_only,
            )
            
            # Load model - T5 uses AutoModelForSeq2SeqLM
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Always use float32 for stability
                local_files_only=self.local_only,
            )
            
            # Use direct model.generate for compatibility across transformers versions.
            self.available = True
            print("✅ FLAN-T5-Small loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Trying with CPU fallback...")
            self._load_model_cpu_fallback()
    
    def _load_model_cpu_fallback(self):
        """Fallback to CPU-only loading"""
        try:
            self.device = "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_only,
            )
            
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                local_files_only=self.local_only,
            )
            
            # Use direct model.generate for compatibility across transformers versions.
            self.available = True
            print("✅ FLAN-T5-Small loaded on CPU!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("⚠️ Continuing without model-backed generation (pattern-only refinement).")
            self.available = False
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate response from FLAN-T5-Small model"""
        try:
            if not self.available or self.tokenizer is None or self.model is None:
                return ""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                num_return_sequences=1,
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return ""


class LLMRefiner:
    """Main LLM refinement processor using FLAN-T5-Small"""
    
    def __init__(self, model_name: str = "google/flan-t5-small", local_only: bool = True):
        self.prompt_builder = SALTPromptBuilder()
        print("🤖 Initializing FLAN-T5-Small processor...")
        self.flan_t5 = FlanT5Processor(model_name, local_only=local_only)
        
        # Linguistic processing patterns
        self.coordinating_conjunctions = {
            'and', 'or', 'but', 'so', 'then'
        }
        self.subordinating_conjunctions = {
            'because', 'that', 'when', 'who', 'after', 'before', 
            'which', 'although', 'if', 'unless', 'while', 'as', 
            'how', 'until', 'like', 'where', 'since'
        }
    
    def detect_overlaps(self, text: str) -> str:
        """Detect overlapping speech patterns using FLAN-T5"""
        # Simple pattern-based overlap detection (more reliable than LLM for this)
        if 'P:' in text and 'Av:' in text and len(text.split()) < 15:
            # Likely overlapping speech - add markers
            words = text.split()
            if len(words) >= 3:
                # Simple heuristic for overlap
                return text.replace('P:', 'P: <').replace('Av:', '> Av: <') + '>'
        return text
    
    def refine_mazes(self, text: str) -> str:
        """Refine maze detection using simple patterns"""
        # Enhanced maze detection for patterns like "I, I do not"
        if ', ' in text:
            # Pattern: word, word -> (word) word
            text = re.sub(r'\b(\w+), (\w+)\b', lambda m: f'({m.group(1)}) {m.group(2)}' if m.group(1).lower() == m.group(2).lower() else m.group(0), text)
        return text
    
    def refine_pause_timing(self, text: str) -> str:
        """Refine pause timing based on simple rules"""
        # Adjust very long pauses to more reasonable durations
        text = re.sub(r'; :(\d{2,})', lambda m: f'; :{min(int(m.group(1)), 16):02d}', text)
        return text
    
    def process_line(self, line: str) -> str:
        """Process a single line through focused pattern-based refinement"""
        if not line.strip() or line.startswith('#') or line.startswith('-') or line.startswith(';'):
            return line  # Skip structural lines
        
        # Apply focused refinements using rule-based patterns (more reliable than LLM for these tasks)
        refined = line
        
        # 1. Detect overlapping speech patterns
        if 'P:' in refined and 'Av:' in refined:
            refined = self.detect_overlaps(refined)
        
        # 2. Refine complex maze patterns  
        if ', ' in refined:
            refined = self.refine_mazes(refined)
        
        # 3. Refine pause timing if needed
        if '; :' in refined:
            refined = self.refine_pause_timing(refined)
        
        return refined if refined.strip() else line
    
    def process_transcript(self, input_text: str) -> str:
        """Process entire transcript through LLM refinement"""
        lines = input_text.strip().split('\n')
        processed_lines = []
        
        print(f"🔄 Processing {len(lines)} lines through pattern-based refinement...")
        
        for i, line in enumerate(lines):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(lines)} lines")
            
            processed_line = self.process_line(line)
            processed_lines.append(processed_line)
        
        print("✅ Pattern-based refinement complete!")
        return '\n'.join(processed_lines)
    
    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """Process a single file through LLM refinement"""
        try:
            print(f"📄 Processing: {input_path.name}")
            
            # Read input
            with open(input_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
            
            # Process through LLM
            refined_text = self.process_transcript(input_text)
            
            # Write output
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(refined_text)
            
            print(f"✅ Completed: {input_path.name} -> {output_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {input_path.name}: {e}")
            return False
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, bool]:
        """Process all rule-based files through LLM refinement"""
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Find all rule-based processed files
        processed_files = list(input_dir.glob("*Rule-Based Processed*.txt"))
        
        if not processed_files:
            print("⚠️  No rule-based processed files found")
            return results
        
        print(f"📁 Processing {len(processed_files)} files through FLAN-T5-Small")
        print(f"📁 Output directory: {output_dir}")
        print("-" * 60)
        
        for input_file in sorted(processed_files):
            # Generate output filename
            output_name = input_file.name.replace("(Rule-Based Processed)", "(FLAN-T5 Refined)")
            output_path = output_dir / output_name
            
            # Process the file
            success = self.process_file(input_file, output_path)
            results[input_file.name] = success
        
        # Print summary
        successful = sum(results.values())
        total = len(results)
        print("-" * 60)
        print(f"📊 FLAN-T5 processing complete: {successful}/{total} files successful")
        
        return results


def main():
    """Command line interface for LLM refinement using FLAN-T5-Small"""
    parser = argparse.ArgumentParser(
        description="LLM refinement for C-unit segmentation using FLAN-T5-Small"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/yuganthareshsoni/CunitSegementation/rule_based_output"),
        help="Directory containing rule-based processed transcripts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/yuganthareshsoni/CunitSegementation/llm_refined_output"),
        help="Directory to save FLAN-T5 refined transcripts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-small",
        help="Hugging Face model name to use"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Force local/offline model loading only (no cloud calls)"
    )
    
    args = parser.parse_args()
    
    # Initialize FLAN-T5 refiner
    print("🚀 Initializing FLAN-T5 Refiner...")
    refiner = LLMRefiner(args.model, local_only=args.local_only)
    
    # Process all files
    results = refiner.process_directory(args.input_dir, args.output_dir)
    
    # Exit with appropriate code
    if all(results.values()):
        print("🎉 All files processed successfully!")
        return 0
    else:
        print("⚠️  Some files failed to process")
        return 1


if __name__ == "__main__":
    exit(main())
