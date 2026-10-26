#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 4090x2 LITE run WITH the AWQ LLM judge (Qwen3-32B-AWQ).
# Goal: verify the judged GSPO pipeline produces non-zero rewards on 2x RTX 4090.
# Reuses every fix from run_gspo_qwen3vl_lite.sh via the gspo_qwen3vl_judge_lite config.
set -x
set -o pipefail

# Persist HF caches so the overlong-prompt filtering is reused (cache hit, no 2h refilter).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# NOTE: do NOT set PYTORCH_ALLOC_CONF=expandable_segments:True here — the AWQ
# judge's vLLM sleep mode uses a CUDA memory pool that asserts against it.
# The actor compute_log_prob OOM headroom comes from rollout gpu_memory_utilization
# 0.45 instead (frees ~1.2GB/GPU once the KV cache is released).

DATE_TIME="$(date +%Y%m%d_%H%M%S)"
TRAINER_EXPERIMENT_SUFFIX="qwen3vl_8b_judge_lite_4090x2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT

ALGORITHM_TAG="${ALGORITHM_TAG:-GSPO}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${ALGORITHM_TAG}_${TRAINER_EXPERIMENT_SUFFIX}_${DATE_TIME}}"
export TRAINER_EXPERIMENT_NAME

export TRAIN_FILES="${TRAIN_FILES:-${VERO_DATA}/vero-600k/vero_600k_train.fixed.verl.jsonl}"
export VAL_FILES="${VAL_FILES:-${VERO_DATA}/vero-600k/vero_600k_val.fixed.verl.jsonl}"
export IMAGE_ROOT="${IMAGE_ROOT:-${VERO_DATA}/vero-600k}"
export ROOT_PATH="${ROOT_PATH:-${VERO_DATA}/vero_runs}"

# LLM judge server: Qwen3-32B-AWQ (~16GB total, TP=2 -> ~8GB/GPU).
# 0.4 util (~9.6GB/GPU when awake) is plenty for the AWQ weights + small KV;
# the server is put to sleep after startup (sleep level 1 frees the GPU), and
# verl wakes it only during the reward step.
export GPU_MEMORY_UTILIZATION_LLM_JUDGE=0.5
# The judge only scores (question <=2048 + response <=1024 + template); 32768 is
# huge and made KV-cache allocation fail. AWQ 32B weights take ~9.8GB/GPU (TP=2),
# so at util 0.45 only 0.99GB KV was left (1% short for 8192). util 0.5 + 6144
# gives comfortable KV headroom. The server sleeps after startup (frees GPU).
export VLLM_JUDGE_MAX_MODEL_LEN=6144
# Sleep level 2 (discard weights) so the asleep judge holds ~0 host RAM instead
# of offloading ~24GB to CPU (which OOM-killed the node alongside training).
export VLLM_JUDGE_SLEEP_LEVEL=2
# [4090-fix] Force the POLICY rollout vLLM to sleep at LEVEL 1 (offload weights to
# CPU, restore on wake) instead of the vLLM>=0.8.5 default of LEVEL 2 (discard).
# With LoRA + load_format=dummy, level-2 sleep discards the base weights and wake
# reloads RANDOM weights from the dummy loader; the tiny LoRA on top of random base
# produced pure-garbage rollouts and reward=-1.0 every step. Level 1 preserves the
# synced base across sleeps. Only affects the policy rollout, not the judge above.
export VERL_VLLM_SLEEP_LEVEL=1
# [think-prefill] Force the assistant response to begin with "<think>" so the
# strict format reward (<think>...</think><answer>...</answer>) can be met. The
# base Qwen3-VL-8B-Instruct (non-thinking) never emits <think> on its own (0/80
# rollouts) and substitutes <tool_call>, so every valid answer scored 0 and there
# was no learning signal. The dataset appends this to the prompt; the vero judge
# reward manager restores it and fixes the model's </answer>-for-</think> tag
# confusion before scoring. Keeps the paper's think/answer format faithful.
export VERL_RESPONSE_PREFILL='<think>'
NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
VLLM_JUDGE_TENSOR_PARALLEL_SIZE="$NUM_GPUS"
START_VLLM_JUDGE_SERVER="1"
VLLM_JUDGE_PORT="51001"
VLLM_JUDGE_API_KEY="qwen"
VLLM_JUDGE_MODEL_NAME="qwen_vllm"
VLLM_JUDGE_MODEL_PATH="Qwen/Qwen3-32B-AWQ"
VLLM_JUDGE_CHAT_TEMPLATE="$REPO_ROOT/examples/prompts/chat_template_no_think.jinja"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/shared/llm_judge_server.sh"
llm_judge_setup_and_start

# NOTE: unlike the judge-off lite script, do NOT drop reward_model.reward_kwargs
# here -- the vero_vllm_judge manager needs format_score / llm_judge_kwargs.
# [log-preservation] rollout 덤프를 런별 하위 폴더로 분리 -> 스텝번호로 덮어써지지 않고
# 각 실행의 rollout이 보존됨 (나중 자료 정리용). 기본 config 경로를 experiment name으로 override.
ROLLOUT_DUMP_DIR="${ROOT_PATH}/rollout_dumps/${TRAINER_EXPERIMENT_NAME}"
mkdir -p "$ROLLOUT_DUMP_DIR"

# 추가 hydra 오버라이드를 인자로 전달받는다 (인자가 없으면 기존 동작과 동일).
python3 -m verl.trainer.main_ppo \
  --config-path="$SCRIPT_DIR/config" \
  --config-name='gspo_qwen3vl_judge_lite' \
  trainer.rollout_data_dir="$ROLLOUT_DUMP_DIR" "$@"
