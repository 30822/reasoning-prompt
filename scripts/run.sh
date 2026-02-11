#!/bin/bash

# Usage: ./scripts/run.sh <mode> [options]
# Modes: solo, team, medcobe

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

check_packages() {
    local missing_packages=()
    
    if ! python3 -c "import yaml" 2>/dev/null; then
        missing_packages+=("pyyaml")
    fi
    
    if ! python3 -c "import openai" 2>/dev/null; then
        missing_packages+=("openai")
    fi
    
    if ! python3 -c "import tqdm" 2>/dev/null; then
        missing_packages+=("tqdm")
    fi
    
    if ! python3 -c "import pandas" 2>/dev/null; then
        missing_packages+=("pandas")
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo "Error: Missing required packages: ${missing_packages[*]}"
        echo ""
        echo "Please install them by running:"
        echo "  pip install -r requirements.txt"
        echo ""
        echo "Or install individually:"
        echo "  pip install ${missing_packages[*]}"
        exit 1
    fi
}

check_packages

MODE="${1:-medcobe}"

TARGET_MODELS=(
    "gpt-5"
    "gpt-4o"
    "gpt-4o-mini"
    "gpt-3.5-turbo"
    "claude-sonnet-4-5-20250929"
    "claude-haiku-4-5-20251001"
    "claude-opus-4-5-20251101"
)

INPUT_DATA_FILE="resources/data/sample_data.json"
SIMULATOR_MODEL="gpt-4o"
JUDGE_MODEL="gpt-5.2"
MAX_CONCURRENT_REQUESTS=10
BATCH_SIZE=20

SOLO_OUTPUT="solo_performance_results.json"
SIMULATION_OUTPUT="simulator_results.json"
EVALUATION_OUTPUT="annotated_results.json"
TEAM_OUTPUT="simulator_results.json"
MEDCOBE_OUTPUT="final_medcobe_evaluation.csv"

run_python() {
    python3 -c "$1"
}

models_to_python_list() {
    local result="["
    local first=true
    for model in "${TARGET_MODELS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            result+=", "
        fi
        result+="\"$model\""
    done
    result+="]"
    echo "$result"
}

MODELS_LIST=$(models_to_python_list)

case "$MODE" in
    solo)
        echo "=========================================="
        echo "Running Solo Performance Evaluation"
        echo "=========================================="
        echo "Models: ${TARGET_MODELS[*]}"
        echo "Input: $INPUT_DATA_FILE"
        echo "Output: $SOLO_OUTPUT"
        echo ""
        
        run_python "
import asyncio
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.solo_performance import run_solo_performance

asyncio.run(run_solo_performance(
    input_file='$INPUT_DATA_FILE',
    output_file='$SOLO_OUTPUT',
    models_to_test=$MODELS_LIST,
    max_concurrent_requests=$MAX_CONCURRENT_REQUESTS
))
"
        echo ""
        echo "Solo performance evaluation completed!"
        ;;
    
    team)
        echo "=========================================="
        echo "Running Team Performance Evaluation"
        echo "=========================================="
        echo "Team result file: $TEAM_OUTPUT"
        echo "Dataset file: $INPUT_DATA_FILE"
        echo "Simulator model: $SIMULATOR_MODEL"
        echo ""
        
        run_python "
import asyncio
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.team_performance import recalculate_final_decision

asyncio.run(recalculate_final_decision(
    team_result_file='$TEAM_OUTPUT',
    dataset_file='$INPUT_DATA_FILE',
    simulator_model='$SIMULATOR_MODEL',
    max_concurrent_requests=$MAX_CONCURRENT_REQUESTS
))
"
        echo ""
        echo "Team performance evaluation completed!"
        ;;
    
    medcobe)
        echo "=========================================="
        echo "Running MedCOBE Pipeline"
        echo "=========================================="
        echo "Step 1: Simulation"
        echo "Step 2: Evaluation"
        echo "Step 3: MedCOBE Score Calculation"
        echo ""
        echo "Models: ${TARGET_MODELS[*]}"
        echo "Input: $INPUT_DATA_FILE"
        echo ""
        
        echo "------------------------------------------"
        echo "[Step 1/3] Running Simulation..."
        echo "------------------------------------------"
        run_python "
import asyncio
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.simulation import run_simulation

asyncio.run(run_simulation(
    input_file='$INPUT_DATA_FILE',
    output_file='$SIMULATION_OUTPUT',
    target_models=$MODELS_LIST,
    simulator_model='$SIMULATOR_MODEL',
    max_concurrent_requests=$MAX_CONCURRENT_REQUESTS
))
"
        echo ""
        echo "Simulation completed!"
        echo ""
        
        echo "------------------------------------------"
        echo "[Step 2/3] Running Evaluation..."
        echo "------------------------------------------"
        run_python "
import asyncio
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.evaluation import run_evaluation

asyncio.run(run_evaluation(
    team_result_file='$SIMULATION_OUTPUT',
    output_file='$EVALUATION_OUTPUT',
    dataset_file='$INPUT_DATA_FILE',
    judge_model='$JUDGE_MODEL',
    batch_size=$BATCH_SIZE,
    solo_acc_file='$SOLO_OUTPUT'
))
"
        echo ""
        echo "Evaluation completed!"
        echo ""
        
        echo "------------------------------------------"
        echo "[Step 3/3] Calculating MedCOBE Scores..."
        echo "------------------------------------------"
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
        echo ""
        echo "=========================================="
        echo "MedCOBE Pipeline Completed!"
        echo "=========================================="
        echo "Results saved to: $MEDCOBE_OUTPUT"
        ;;
    
    *)
        echo "Error: Unknown mode '$MODE'"
        echo ""
        echo "Usage: $0 <mode>"
        echo ""
        echo "Modes:"
        echo "  solo    - Run solo performance evaluation"
        echo "  team    - Run team performance evaluation"
        echo "  medcobe - Run full MedCOBE pipeline (simulation -> evaluation -> medcobe)"
        echo ""
        echo "Example:"
        echo "  $0 solo"
        echo "  $0 team"
        echo "  $0 medcobe"
        exit 1
        ;;
esac

