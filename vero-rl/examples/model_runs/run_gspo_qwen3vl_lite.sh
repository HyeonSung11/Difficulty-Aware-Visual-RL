#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 4090x2 lite training (LLM judge off, demo scale: 10 steps)
# Goal: verify the GSPO pipeline runs end-to-end on a 2x RTX 4090 box.
set -x
set -o pipefail

# Persist HF caches so the (expensive) overlong-prompt filtering is reused across runs.
# The overlong filter cache in rl_dataset.py is keyed off HF_DATASETS_CACHE specifically;
# if it is unset the filter is recomputed from scratch every launch (~2h on Vero-600k).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

DATE_TIME="$(date +%Y%m%d_%H%M%S)"
TRAINER_EXPERIMENT_SUFFIX="qwen3vl_8b_lite_4090x2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT

ALGORITHM_TAG="${ALGORITHM_TAG:-GSPO}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${ALGORITHM_TAG}_${TRAINER_EXPERIMENT_SUFFIX}_${DATE_TIME}}"
export TRAINER_EXPERIMENT_NAME

# Data + checkpoint paths (override at run time if needed)
export TRAIN_FILES="${TRAIN_FILES:-${VERO_DATA}/vero-600k/vero_600k_train.verl.jsonl}"
export VAL_FILES="${VAL_FILES:-${VERO_DATA}/vero-600k/vero_600k_val.verl.jsonl}"
export IMAGE_ROOT="${IMAGE_ROOT:-${VERO_DATA}/vero-600k}"
export ROOT_PATH="${ROOT_PATH:-${VERO_DATA}/vero_runs}"

# No LLM judge server on 4090x2.
# reward_manager: naive in the config -> llm_judge type falls back to _llm_judge_dummy_reward (0.0).
echo "[lite] Skipping LLM judge server (judge type -> 0.0 via dummy reward)"

# NaiveRewardManager only accepts (tokenizer, num_examine, compute_score, reward_fn_key).
# The shared config merges format_score / llm_judge_kwargs / overlong_buffer_cfg into
# reward_model.reward_kwargs, which main_ppo splats into the manager -> TypeError.
# The actual scoring weight (format_score=0.2) is sourced separately from
# custom_reward_function.reward_kwargs, so dropping reward_model.reward_kwargs is safe here.
python3 -m verl.trainer.main_ppo \
  --config-path="$SCRIPT_DIR/config" \
  --config-name='gspo_qwen3vl_lite' \
  '~reward_model.reward_kwargs'
