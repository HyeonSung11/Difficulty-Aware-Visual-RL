#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# Train Qwen3-VL-8B-Instruct with GSPO + LLM Judge (AWQ 4-bit)
# Based on: run_gspo_qwen3vl_instruct_mix_all_llmjudge.sh
# Only change: VLLM_JUDGE_MODEL_PATH switched to Qwen/Qwen3-32B-AWQ
#              (resolves the original "Qwen3.5-27B" typo and fits 4090x2 memory)
set -x
set -o pipefail

DATE_TIME="$(date +%Y%m%d_%H%M%S)"
TRAINER_EXPERIMENT_SUFFIX="qwen3vl_8b_instruct_awq_judge"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT

ALGORITHM_TAG="${ALGORITHM_TAG:-GSPO}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${ALGORITHM_TAG}_${TRAINER_EXPERIMENT_SUFFIX}_${DATE_TIME}}"
export TRAINER_EXPERIMENT_NAME

# Data + checkpoint paths (override at run time if needed)
export TRAIN_FILES="${TRAIN_FILES:-${VERO_DATA}/vero-600k/vero_600k_train.fixed.verl.jsonl}"
export VAL_FILES="${VAL_FILES:-${VERO_DATA}/vero-600k/vero_600k_val.fixed.verl.jsonl}"
export IMAGE_ROOT="${IMAGE_ROOT:-${VERO_DATA}/vero-600k}"
export ROOT_PATH="${ROOT_PATH:-${VERO_DATA}/vero_runs}"

# LLM Judge server
export GPU_MEMORY_UTILIZATION_LLM_JUDGE=0.65 
NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
VLLM_JUDGE_TENSOR_PARALLEL_SIZE="$NUM_GPUS"
START_VLLM_JUDGE_SERVER="1"
VLLM_JUDGE_PORT="51001"
VLLM_JUDGE_API_KEY="qwen"
VLLM_JUDGE_MODEL_NAME="qwen_vllm"
# CHANGED: was "$HF_HOME/Qwen/Qwen3.5-27B" (typo + 64GB FP16).
# Now uses official 4-bit AWQ build of Qwen3-32B (~16GB total, fits 4090x2).
VLLM_JUDGE_MODEL_PATH="Qwen/Qwen3-32B-AWQ"
VLLM_JUDGE_CHAT_TEMPLATE="$REPO_ROOT/examples/prompts/chat_template_no_think.jinja"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/shared/llm_judge_server.sh"
llm_judge_setup_and_start

python3 -m verl.trainer.main_ppo \
  --config-path="$SCRIPT_DIR/config" \
  --config-name='gspo_qwen3vl'
