#!/bin/bash
# Usage:
#   ./scripts/run_openrouter.sh [solo|team|pcollab] \
#       [--model "openai/o3"] [--input resources/data/sample_data.json] \
#       [--max-concurrent 10]
# Modes: solo, team, pcollab (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

check_packages() {
    local missing_packages=()

    if ! python -c "import yaml" 2>/dev/null; then missing_packages+=("pyyaml"); fi
    if ! python -c "import openai" 2>/dev/null; then missing_packages+=("openai"); fi
    if ! python -c "import tqdm" 2>/dev/null; then missing_packages+=("tqdm"); fi
    if ! python -c "import pandas" 2>/dev/null; then missing_packages+=("pandas"); fi
    if ! python -c "import httpx" 2>/dev/null; then missing_packages+=("httpx"); fi
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo "Error: Missing required packages: ${missing_packages[*]}"
        echo "Install: pip install -r requirements.txt"
        exit 1
    fi
}
check_packages

DEFAULT_TARGET_MODELS=(
    "openai/o3"
    "deepseek/deepseek-r1-0528"
)

INPUT_DATA_FILE="resources/data/sample_data.json"
SIMULATOR_MODEL="openai/gpt-4o"          # clinician simulator
JUDGE_MODEL="deepseek/deepseek-v3.2"     # turn-level judge
MAX_CONCURRENT_REQUESTS=10
SKIP_DIALOGUE_LEVEL_JUDGE="True"         # True: skip whole-dialogue judge
BATCH_SIZE=20

OUTPUT_ROOT="output"
mkdir -p \
  "$OUTPUT_ROOT/solo/by_model" \
  "$OUTPUT_ROOT/simulation/by_model" \
  "$OUTPUT_ROOT/team/by_model" \
  "$OUTPUT_ROOT/evaluation" \
  "$OUTPUT_ROOT/pcollab"

SOLO_BY_MODEL_DIR="$OUTPUT_ROOT/solo/by_model"
SIM_BY_MODEL_DIR="$OUTPUT_ROOT/simulation/by_model"
TEAM_BY_MODEL_DIR="$OUTPUT_ROOT/team/by_model"

SOLO_OUTPUT="$OUTPUT_ROOT/solo/solo_performance_results.json"
SIMULATION_OUTPUT="$OUTPUT_ROOT/simulation/simulator_results.json"
TEAM_OUTPUT="$OUTPUT_ROOT/team/team_results.json"
EVALUATION_OUTPUT="$OUTPUT_ROOT/evaluation/annotated_results.json"
PCOLLAB_OUTPUT="$OUTPUT_ROOT/pcollab/final_pcollab_evaluation.csv"

MODE="${1:-pcollab}"
shift || true

TARGET_MODELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      TARGET_MODELS+=("$2")
      shift 2
      ;;
    --input)
      INPUT_DATA_FILE="$2"
      shift 2
      ;;
    --max-concurrent)
      MAX_CONCURRENT_REQUESTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 <solo|team|pcollab> [--model ...] [--input ...] [--max-concurrent ...]"
      exit 1
      ;;
  esac
done

if [ ! -f "$INPUT_DATA_FILE" ]; then
  echo "Error: input file not found: $INPUT_DATA_FILE"
  echo "Place cases as JSON (see resources/data/sample_data.json) or pass --input <path>."
  exit 1
fi

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


case "$MODE" in
  solo)
    echo "=== Solo ==="
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
    echo "  - merge solo by_model -> combined json (compat)"
    run_python "
import json
from pathlib import Path
root = Path('$PROJECT_ROOT')
solo_dir = root / 'output' / 'solo' / 'by_model'
out_path = root / '$SOLO_OUTPUT'
out_path.parent.mkdir(parents=True, exist_ok=True)

d = {}
for p in sorted(solo_dir.glob('*.json')):
    with open(p, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    model = obj["model_name"]
    d[model] = obj

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

print(f'Merged {len(d)} models -> {out_path}')
"
    ;;

  team)
    echo "=== Team (by_model) ==="
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
root = Path('$PROJECT_ROOT')
team_dir = root / 'output' / 'team' / 'by_model'
out_path = root / '$TEAM_OUTPUT'
out_path.parent.mkdir(parents=True, exist_ok=True)

d = {}
for p in sorted(team_dir.glob('*.json')):
    with open(p, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    model = obj["model_name"]
    d[model] = obj

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

print(f'Merged {len(d)} models -> {out_path}')
"
    ;;

  pcollab)
    echo "=========================================="
    echo "Running Pcollab Full Pipeline"
    echo "Models: ${TARGET_MODELS[*]}"
    echo "Input: $INPUT_DATA_FILE"
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
    echo "  - merge solo by_model -> combined json (compat)"
    run_python "
import json
from pathlib import Path
solo_dir = Path(r'$SOLO_BY_MODEL_DIR')
out_path = Path(r'$SOLO_OUTPUT')
out_path.parent.mkdir(parents=True, exist_ok=True)

d = {}
for p in sorted(solo_dir.glob('*.json')):
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
  mode='pcollab',
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
  team_result_file='$TEAM_OUTPUT',
  output_file='$EVALUATION_OUTPUT',
  dataset_file='$INPUT_DATA_FILE',
  judge_model='$JUDGE_MODEL',
  batch_size=$BATCH_SIZE,
  solo_acc_file='$SOLO_OUTPUT',
  use_openai_judge=True,
  skip_dialogue_level_judge=$SKIP_DIALOGUE_LEVEL_JUDGE
))
"

    echo "[Step 5/5] Pcollab..."
    run_python "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.collaborative_performance import calculate_collaborative_performance
calculate_collaborative_performance(
  annotated_file='$EVALUATION_OUTPUT',
  output_csv='$PCOLLAB_OUTPUT',
  solo_file='$SOLO_OUTPUT'
)
"
    echo "DONE. Output: $PCOLLAB_OUTPUT"
    ;;

  *)
    echo "Usage: $0 <solo|team|pcollab> [--model ...] [--input ...] [--max-concurrent ...]"
    exit 1
    ;;
esac
