#!/usr/bin/env bash
set -euo pipefail

# Run configuration ablations and print a compact comparison table.
#
# Usage:
#   ./run_ablation.sh
#   ./run_ablation.sh extracted_text_3

INPUT_DIR="${1:-extracted_text}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

PYTHON_BIN="python3"
if [[ -x "venv/bin/python3" ]]; then
  PYTHON_BIN="venv/bin/python3"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
AB_ROOT="ablation_${timestamp}"
mkdir -p "$AB_ROOT"

names=("A" "B" "C" "D" "E" "F" "G" "H")
labels=(
  "rule2=standard morph=regex coord=standard pause=standard"
  "rule2=off morph=regex coord=standard pause=standard"
  "rule2=standard morph=off coord=standard pause=standard"
  "rule2=off morph=off coord=standard pause=standard"
  "rule2=conservative morph=regex coord=standard pause=standard"
  "rule2=off morph=off coord=conservative pause=conservative"
  "rule2=off morph=off coord=aggressive pause=aggressive"
  "rule2=off morph=off coord=off pause=off"
)
args_list=(
  "--rule2-mode standard --morph-mode regex --coord-mode standard --pause-mode standard"
  "--rule2-mode off --morph-mode regex --coord-mode standard --pause-mode standard"
  "--rule2-mode standard --morph-mode off --coord-mode standard --pause-mode standard"
  "--rule2-mode off --morph-mode off --coord-mode standard --pause-mode standard"
  "--rule2-mode conservative --morph-mode regex --coord-mode standard --pause-mode standard"
  "--rule2-mode off --morph-mode off --coord-mode conservative --pause-mode conservative"
  "--rule2-mode off --morph-mode off --coord-mode aggressive --pause-mode aggressive"
  "--rule2-mode off --morph-mode off --coord-mode off --pause-mode off"
)

echo "Running ablations into: $AB_ROOT"
echo

for i in "${!names[@]}"; do
  name="${names[$i]}"
  label="${labels[$i]}"
  args="${args_list[$i]}"
  out_dir="${AB_ROOT}/rule_based_${name}"
  report_txt="${AB_ROOT}/evaluation_${name}.txt"
  diag_json="${AB_ROOT}/diagnostics_${name}.json"

  echo "[$name] ${label}"
  # shellcheck disable=SC2086
  "$PYTHON_BIN" rule_based_processor.py --input-dir "$INPUT_DIR" --output-dir "$out_dir" $args
  "$PYTHON_BIN" evaluate_system.py --system-dir "$out_dir" --gold-dir "$INPUT_DIR" --output-file "$report_txt" --diagnostics-file "$diag_json"
done

echo
echo "=== Ablation Summary ==="
"$PYTHON_BIN" - <<'PY' "$AB_ROOT"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for jf in sorted(root.glob("evaluation_*.json")):
    data = json.loads(jf.read_text(encoding="utf-8"))
    if not data:
        continue
    avg = sum(x.get("overall_similarity", 0.0) for x in data) / len(data)
    cavg = sum(x.get("c_unit_accuracy", 0.0) for x in data) / len(data)
    pavg = sum(x.get("pause_accuracy", 0.0) for x in data) / len(data)
    rows.append((jf.stem.replace("evaluation_", ""), avg, cavg, pavg))

rows.sort(key=lambda r: r[1], reverse=True)
print(f"{'Config':<8} {'Overall%':>9} {'CUnit%':>9} {'Pause%':>9}")
for name, overall, cunit, pause in rows:
    print(f"{name:<8} {overall*100:9.2f} {cunit*100:9.2f} {pause*100:9.2f}")

if rows:
    best = rows[0]
    print(f"\nBest by overall similarity: {best[0]} ({best[1]*100:.2f}%)")
PY
