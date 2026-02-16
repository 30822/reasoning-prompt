#!/bin/bash
# Permission: chmod +x scripts/gate_runner_openrouter.sh
# Usage:
#   ./scripts/gate_runner_openrouter.sh <solo|team|medcobe> [--model ...]
#
# Gate defaults:
#   - Input: resources/data/experiments/gate_check_sample.json
#   - Experiments: resources/experiments_gate.yaml (temporarily swapped in)
#   - Output root: output_gate/

set -euo pipefail

# --------------------
# Gate-only constants
# --------------------
INPUT_DATA_FILE="resources/data/experiments/gate_check_sample_v2.json"

SIMULATOR_MODEL="openai/gpt-4o"
JUDGE_MODEL="openai/gpt-5.2"

MAX_CONCURRENT_REQUESTS=5
BATCH_SIZE=5

OUTPUT_ROOT="output_gate"
mkdir -p \
  "$OUTPUT_ROOT/solo/by_model" \
  "$OUTPUT_ROOT/simulation/by_model" \
  "$OUTPUT_ROOT/team/by_model" \
  "$OUTPUT_ROOT/evaluation" \
  "$OUTPUT_ROOT/medcobe"

SOLO_BY_MODEL_DIR="$OUTPUT_ROOT/solo/by_model"
SIM_BY_MODEL_DIR="$OUTPUT_ROOT/simulation/by_model"
TEAM_BY_MODEL_DIR="$OUTPUT_ROOT/team/by_model"

SOLO_OUTPUT="$OUTPUT_ROOT/solo/solo_performance_results.json"
SIMULATION_OUTPUT="$OUTPUT_ROOT/simulation/simulator_results.json"
TEAM_OUTPUT="$OUTPUT_ROOT/team/team_results.json"
EVALUATION_OUTPUT="$OUTPUT_ROOT/evaluation/annotated_results.json"
MEDCOBE_OUTPUT="$OUTPUT_ROOT/medcobe/final_medcobe_evaluation.csv"

MODE="${1:-medcobe}"
shift || true

# --------------------
# Parse args
# --------------------
TARGET_MODELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      TARGET_MODELS+=("$2")
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# --------------------
# Default target models (gate)
# --------------------
DEFAULT_TARGET_MODELS=("openai/o3" "deepseek/deepseek-r1-0528")

# --------------------
# Build model list
# --------------------
if [ ${#TARGET_MODELS[@]} -eq 0 ]; then
  TARGET_MODELS=("${DEFAULT_TARGET_MODELS[@]}")
fi

models_to_python_list() {
    local result="["
    local first=true
    for model in "${TARGET_MODELS[@]}"; do
        if [ "$first" = true ]; then first=false; else result+=", "; fi
        result+="\"${model//\"/\\\"}\""
    done
    result+="]"
    echo "$result"
}
MODELS_LIST=$(models_to_python_list)

run_python() {
  python - <<EOF
$1
EOF
}

# --------------------
# Swap experiments.yaml -> experiments_gate.yaml
# --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXP_MAIN="${PROJECT_ROOT}/resources/experiments.yaml"
EXP_GATE="${PROJECT_ROOT}/resources/experiments_gate.yaml"
TS="$(date +%Y%m%d_%H%M%S)"
EXP_BAK="${EXP_MAIN}.bak_gate_${TS}"

if [[ ! -f "${EXP_GATE}" ]]; then
  echo "❌ Missing: ${EXP_GATE}"
  echo "   Create resources/experiments_gate.yaml first."
  exit 1
fi
if [[ ! -f "${EXP_MAIN}" ]]; then
  echo "❌ Missing: ${EXP_MAIN}"
  exit 1
fi

cp "${EXP_MAIN}" "${EXP_BAK}"
cp "${EXP_GATE}" "${EXP_MAIN}"

cleanup() {
  echo ""
  echo "=== Restoring experiments.yaml ==="
  cp "${EXP_BAK}" "${EXP_MAIN}"
  echo "Restored: ${EXP_MAIN}"
}
trap cleanup EXIT

export MEDCOBE_OUTPUT_ROOT="output_gate"
export MEDCOBE_EXPERIMENT_IDS="o3-B__B,o3-B__B_CL_SR,o3-B_COT_CL__B_SR,o3-B_COT_CL__B_CL_SR,r1_0528-B__B,r1_0528-B__B_CL_SR,r1_0528-B_COT_CL__B_SR,r1_0528-B_COT_CL__B_CL_SR"

# --------------------
# Pipeline dirs (gate)
# --------------------
run_python "
from pathlib import Path
root = Path(r'$PROJECT_ROOT')
for rel in [
  '$OUTPUT_ROOT/solo/by_model',
  '$OUTPUT_ROOT/simulation/by_model',
  '$OUTPUT_ROOT/team/by_model',
]:
  p = root / rel
  p.mkdir(parents=True, exist_ok=True)
  print('OK dir:', p)
"

case "$MODE" in
  solo)
    echo "=== Gate Solo ==="
    run_python "
import asyncio, sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.solo_performance_openrouter import run_solo_performance
asyncio.run(run_solo_performance(
  input_file='$INPUT_DATA_FILE',
  output_file='$SOLO_OUTPUT',
  models_to_test=$MODELS_LIST,
  max_concurrent_requests=$MAX_CONCURRENT_REQUESTS
))
"
    ;;

  team)
    echo "=== Gate Team (by_model) ==="
    run_python "
import asyncio, sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.team_performance_openrouter import recalculate_final_decision_by_model_dir
asyncio.run(recalculate_final_decision_by_model_dir(
  dataset_file='$INPUT_DATA_FILE',
  simulator_model='$SIMULATOR_MODEL',
  max_concurrent_requests=$MAX_CONCURRENT_REQUESTS,
  resume=True
))
"
    ;;

  medcobe)
    echo "=========================================="
    echo "Running Gate MedCOBE Pipeline"
    echo "Models: ${TARGET_MODELS[*]}"
    echo "Input: $INPUT_DATA_FILE"
    echo "Output: $OUTPUT_ROOT"
    echo "=========================================="

    echo "[Step 1/5] Solo..."
    run_python "
import asyncio, sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.solo_performance_openrouter import run_solo_performance
asyncio.run(run_solo_performance(
  input_file='$INPUT_DATA_FILE',
  output_file='$SOLO_OUTPUT',
  models_to_test=$MODELS_LIST,
  max_concurrent_requests=$MAX_CONCURRENT_REQUESTS
))
"

echo "  - merge solo by_model -> combined json (compat for downstream)"
run_python "
import json
from pathlib import Path

src_dir = Path(r'$SOLO_BY_MODEL_DIR')
out_path = Path(r'$SOLO_OUTPUT')
out_path.parent.mkdir(parents=True, exist_ok=True)

d = {}
for p in sorted(src_dir.glob('*.json')):
    with open(p, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    model = obj.get('model_name') or p.stem
    d[model] = obj

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

print(f'Merged {len(d)} models -> {out_path}')
"

    echo "[Step 2/5] Simulation..."
    run_python "
import asyncio, sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.simulation_openrouter import run_simulation
asyncio.run(run_simulation(
  input_file='$INPUT_DATA_FILE',
  target_models=$MODELS_LIST,
  mode='medcobe',
  simulator_model='$SIMULATOR_MODEL',
  max_concurrent=$MAX_CONCURRENT_REQUESTS,
  max_retries=3,
  output_file=None
))
"

    echo "[Step 3/5] Team recalc (by_model)..."
    run_python "
import asyncio, sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.team_performance_openrouter import recalculate_final_decision_by_model_dir
asyncio.run(recalculate_final_decision_by_model_dir(
  dataset_file='$INPUT_DATA_FILE',
  simulator_model='$SIMULATOR_MODEL',
  max_concurrent_requests=$MAX_CONCURRENT_REQUESTS,
  resume=True
))
"

echo "  - merge team by_model -> combined json (compat for evaluation.py)"
run_python "
import json
from pathlib import Path

team_dir = Path(r'$TEAM_BY_MODEL_DIR')
out_path = Path(r'$TEAM_OUTPUT')
out_path.parent.mkdir(parents=True, exist_ok=True)

d = {}
for p in sorted(team_dir.glob('*.json')):
    with open(p, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    model = obj.get('model_name') or p.stem
    d[model] = obj

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

print(f'Merged {len(d)} models -> {out_path}')
"

    echo "[Step 4/5] Evaluation..."
    run_python "
import asyncio, sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.evaluation import run_evaluation
asyncio.run(run_evaluation(
  team_result_file='$OUTPUT_ROOT/team/team_results.json',
  output_file='$EVALUATION_OUTPUT',
  dataset_file='$INPUT_DATA_FILE',
  judge_model='$JUDGE_MODEL',
  batch_size=$BATCH_SIZE,
  solo_acc_file='$SOLO_OUTPUT'
))
"

    echo "[Step 5/5] MedCOBE..."
    run_python "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.medcobe import calculate_medcobe_scores
calculate_medcobe_scores(
  annotated_file='$EVALUATION_OUTPUT',
  output_csv='$MEDCOBE_OUTPUT',
  solo_file='$SOLO_OUTPUT'
)
"
    echo "DONE. Output: $MEDCOBE_OUTPUT"
    ;;

  *)
    echo "Usage: $0 <solo|team|medcobe> [--model ...]"
    exit 1
    ;;
esac
