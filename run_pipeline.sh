#!/usr/bin/env bash
set -euo pipefail

# Full pipeline runner with incremental output names.
# Keeps all previous folders/reports untouched.
#
# Usage:
#   ./run_pipeline.sh
#   ./run_pipeline.sh "Correct - Be EPIC-VR transcript examplesV02 2"
#
# Profiles:
#   PIPELINE_PROFILE=accuracy  (default): best-scoring baseline from ablation
#     - rule2 off, morph off, skip llm stage
#   PIPELINE_PROFILE=full:
#     - rule2 standard, morph regex, include llm stage
#
# Overrides:
#   RUN_LLM=1 ./run_pipeline.sh   # force run llm stage
#   RUN_LLM=0 ./run_pipeline.sh   # force skip llm stage
#   RUN_ALIGN=1 ./run_pipeline.sh # run alignment postprocessor stage

INPUT_DIR="${1:-Correct - Be EPIC-VR transcript examplesV02 2}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

PYTHON_BIN="python3"
if [[ -x "venv/bin/python3" ]]; then
  PYTHON_BIN="venv/bin/python3"
fi

PIPELINE_PROFILE="${PIPELINE_PROFILE:-accuracy}"
RUN_LLM="${RUN_LLM:-}"
RUN_ALIGN="${RUN_ALIGN:-0}"

RULE2_MODE="off"
MORPH_MODE="off"
if [[ "$PIPELINE_PROFILE" == "full" ]]; then
  RULE2_MODE="standard"
  MORPH_MODE="regex"
  if [[ -z "$RUN_LLM" ]]; then RUN_LLM="1"; fi
else
  # accuracy profile
  RULE2_MODE="off"
  MORPH_MODE="off"
  if [[ -z "$RUN_LLM" ]]; then RUN_LLM="0"; fi
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
ALIGNED_DIR="aligned_output_${RUN_ID}"
REPORT_RULE_BASED="evaluation_report_${RUN_ID}_rulebased.txt"
REPORT_ALIGNED="evaluation_report_${RUN_ID}_aligned.txt"
REPORT_LLM="evaluation_report_${RUN_ID}_llm.txt"
REPORT_FILE="evaluation_report_${RUN_ID}.txt"

echo "============================================================"
echo "Running pipeline with run ID: ${RUN_ID}"
echo "Input directory: ${INPUT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Profile: ${PIPELINE_PROFILE}"
echo "Rule2 mode: ${RULE2_MODE}"
echo "Morph mode: ${MORPH_MODE}"
echo "Run LLM stage: ${RUN_LLM}"
echo "Run alignment stage: ${RUN_ALIGN}"
echo "Output folders/files:"
echo "  - ${EXTRACTED_DIR}"
echo "  - ${RULE_BASED_DIR}"
echo "  - ${ALIGNED_DIR}"
echo "  - ${LLM_REFINED_DIR}"
echo "  - ${REPORT_RULE_BASED} (+ .json)"
if [[ "${RUN_ALIGN}" == "1" ]]; then
  echo "  - ${REPORT_ALIGNED} (+ .json)"
fi
if [[ "${RUN_LLM}" == "1" ]]; then
  echo "  - ${REPORT_LLM} (+ .json)"
fi
echo "  - ${REPORT_FILE} (+ .json, primary)"
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
  --output-dir "${RULE_BASED_DIR}" \
  --rule2-mode "${RULE2_MODE}" \
  --morph-mode "${MORPH_MODE}"

echo
echo "[3/4] Evaluation (rule-based output)"
"${PYTHON_BIN}" evaluate_system.py \
  --system-dir "${RULE_BASED_DIR}" \
  --gold-dir "${EXTRACTED_DIR}" \
  --output-file "${REPORT_RULE_BASED}"

if [[ "${RUN_ALIGN}" == "1" ]]; then
  echo
  echo "[4/4] Alignment postprocessor + evaluation"
  "${PYTHON_BIN}" alignment_postprocessor.py \
    --input-dir "${RULE_BASED_DIR}" \
    --output-dir "${ALIGNED_DIR}"

  "${PYTHON_BIN}" evaluate_system.py \
    --system-dir "${ALIGNED_DIR}" \
    --gold-dir "${EXTRACTED_DIR}" \
    --output-file "${REPORT_ALIGNED}"
fi

if [[ "${RUN_LLM}" == "1" ]]; then
  echo
  echo "[5/5] LLM refinement + evaluation"
  "${PYTHON_BIN}" llm_refiner.py \
    --input-dir "${RULE_BASED_DIR}" \
    --output-dir "${LLM_REFINED_DIR}" \
    --local-only

  "${PYTHON_BIN}" evaluate_system.py \
    --system-dir "${LLM_REFINED_DIR}" \
    --gold-dir "${EXTRACTED_DIR}" \
    --output-file "${REPORT_LLM}"

  # Primary report points to final (LLM) stage in full mode.
  cp "${REPORT_LLM}" "${REPORT_FILE}"
  cp "${REPORT_LLM%.txt}.json" "${REPORT_FILE%.txt}.json"
elif [[ "${RUN_ALIGN}" == "1" ]]; then
  # If alignment is enabled and LLM is disabled, alignment report becomes primary.
  cp "${REPORT_ALIGNED}" "${REPORT_FILE}"
  cp "${REPORT_ALIGNED%.txt}.json" "${REPORT_FILE%.txt}.json"
else
  # Primary report points to best baseline stage in accuracy mode.
  cp "${REPORT_RULE_BASED}" "${REPORT_FILE}"
  cp "${REPORT_RULE_BASED%.txt}.json" "${REPORT_FILE%.txt}.json"
fi

echo
echo "Done. New run artifacts:"
echo "  ${EXTRACTED_DIR}"
echo "  ${RULE_BASED_DIR}"
if [[ "${RUN_LLM}" == "1" ]]; then
  echo "  ${LLM_REFINED_DIR}"
fi
if [[ "${RUN_ALIGN}" == "1" ]]; then
  echo "  ${ALIGNED_DIR}"
fi
echo "  ${REPORT_RULE_BASED}"
if [[ "${RUN_ALIGN}" == "1" ]]; then
  echo "  ${REPORT_ALIGNED}"
fi
if [[ "${RUN_LLM}" == "1" ]]; then
  echo "  ${REPORT_LLM}"
fi
echo "  ${REPORT_FILE}"
echo "  ${REPORT_FILE%.txt}.json"
