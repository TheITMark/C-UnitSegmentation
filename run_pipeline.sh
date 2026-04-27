#!/usr/bin/env bash
set -euo pipefail

# Full pipeline runner with incremental output names.
# Keeps all previous folders/reports untouched.
#
# Usage:
#   ./run_pipeline.sh
#   ./run_pipeline.sh "Correct - Be EPIC-VR transcript examplesV02 2"

INPUT_DIR="${1:-Correct - Be EPIC-VR transcript examplesV02 2}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

PYTHON_BIN="python3"
if [[ -x "venv/bin/python3" ]]; then
  PYTHON_BIN="venv/bin/python3"
fi

next_run_id() {
  local i=1
  while :; do
    if [[ ! -d "extracted_text_${i}" ]] \
      && [[ ! -d "rule_based_output_${i}" ]] \
      && [[ ! -d "llm_refined_output_${i}" ]] \
      && [[ ! -f "evaluation_report_${i}.txt" ]] \
      && [[ ! -f "evaluation_report_${i}.json" ]]; then
      echo "$i"
      return 0
    fi
    i=$((i + 1))
  done
}

RUN_ID="$(next_run_id)"
EXTRACTED_DIR="extracted_text_${RUN_ID}"
RULE_BASED_DIR="rule_based_output_${RUN_ID}"
LLM_REFINED_DIR="llm_refined_output_${RUN_ID}"
REPORT_FILE="evaluation_report_${RUN_ID}.txt"

echo "============================================================"
echo "Running pipeline with run ID: ${RUN_ID}"
echo "Input directory: ${INPUT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Output folders/files:"
echo "  - ${EXTRACTED_DIR}"
echo "  - ${RULE_BASED_DIR}"
echo "  - ${LLM_REFINED_DIR}"
echo "  - ${REPORT_FILE} (+ .json)"
echo "============================================================"

echo
echo "[1/4] Extract DOCX to TXT"
"${PYTHON_BIN}" extract_docx_to_txt.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${EXTRACTED_DIR}"

echo
echo "[2/4] Rule-based processing"
"${PYTHON_BIN}" rule_based_processor.py \
  --input-dir "${EXTRACTED_DIR}" \
  --output-dir "${RULE_BASED_DIR}"

echo
echo "[3/4] LLM refinement"
"${PYTHON_BIN}" llm_refiner.py \
  --input-dir "${RULE_BASED_DIR}" \
  --output-dir "${LLM_REFINED_DIR}"

echo
echo "[4/4] Evaluation"
"${PYTHON_BIN}" evaluate_system.py \
  --system-dir "${LLM_REFINED_DIR}" \
  --gold-dir "${EXTRACTED_DIR}" \
  --output-file "${REPORT_FILE}"

echo
echo "Done. New run artifacts:"
echo "  ${EXTRACTED_DIR}"
echo "  ${RULE_BASED_DIR}"
echo "  ${LLM_REFINED_DIR}"
echo "  ${REPORT_FILE}"
echo "  ${REPORT_FILE%.txt}.json"
