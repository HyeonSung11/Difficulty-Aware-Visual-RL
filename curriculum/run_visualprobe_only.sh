#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# VisualProbe 단독 평가 — judge를 먼저 띄운 뒤 생성+집계를 한 번에 한다.
#
# 왜 필요한가:
#   visual_probe_easy_reasoning 의 집계 함수(visual_probe_aggregate_judge_results)는
#   judge 엔진을 요구한다. judge 없이 그룹A(TP=2)에서 돌리면 집계 단계에서
#   RuntimeError("Unable to resolve judge model path")로 죽고, 그 예외가
#   샘플 저장보다 먼저 터져서 samples jsonl 자체가 안 남는다.
#   -> 오프라인 재채점도 불가능해지고 결국 N/A가 된다.
#   따라서 VisualProbe는 반드시 judge가 떠 있는 상태에서 돌려야 한다.
#
#   사용: bash run_visualprobe_only.sh <EXP_NAME> [<EXP_NAME> ...]
set -o pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm

REPO=${VERO_REPO}/vero-rl
EVAL_DIR=${VERO_REPO}/vero-eval
LOGDIR=$REPO/_logs
mkdir -p "$LOGDIR"
export LMMS_SYSTEM_PROMPT_FILE=$REPO/examples/prompts/system_prompt_concise.txt

JUDGE_PID=""
cleanup () { [ -n "$JUDGE_PID" ] && kill -9 "$JUDGE_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "########## judge 상시-켜짐 기동 (GPU1) :: $(date '+%F %T') ##########"
JLOG=$LOGDIR/judge_visualprobe_$(date +%Y%m%d_%H%M%S).log
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-32B-AWQ \
  --port 51001 --served-model-name qwen_vllm \
  --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --max-model-len 8192 --enforce-eager --dtype bfloat16 \
  --chat-template $REPO/examples/prompts/chat_template_no_think.jinja \
  --api-key qwen > "$JLOG" 2>&1 &
JUDGE_PID=$!
tries=0
until curl -s -m 3 http://127.0.0.1:51001/v1/models -H "Authorization: Bearer qwen" 2>/dev/null | grep -q qwen_vllm; do
  sleep 10; tries=$((tries+1))
  kill -0 $JUDGE_PID 2>/dev/null || { echo "!!! judge 사망 — $JLOG"; exit 1; }
  [ $tries -ge 90 ] && { echo "!!! judge 15분 미기동"; exit 1; }
done
echo ">>> judge 준비됨"

cd "$EVAL_DIR"
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_URL=http://127.0.0.1:51001/v1/chat/completions
export OPENAI_API_KEY=qwen; export GPT_API_KEY=qwen
export MODEL_VERSION=qwen_vllm
export JUDGE_MODEL_PATH=qwen_vllm
export CHARXIV_JUDGE_MODEL_PATH=qwen_vllm   # 집계 함수가 이 이름으로도 조회한다

for EXP in "$@"; do
  MERGED=${VERO_DATA}/vero_runs/merged_models/${EXP}
  RES=${VERO_REPO}/investigation_2x4090/eval_results_${EXP}
  if [ ! -f "$MERGED/config.json" ]; then echo "!!! merged 모델 없음, 건너뜀: $MERGED"; continue; fi
  mkdir -p "$RES"
  log=$LOGDIR/eval_${EXP}_VP_$(date +%Y%m%d_%H%M%S).log
  echo "########## START VisualProbe  $EXP :: $(date '+%F %T') ##########" | tee "$log"
  rm -rf /dev/shm/psm_* /dev/shm/vllm* 2>/dev/null
  timeout 25200 python3 -m lmms_eval --model vllm \
    --model_args "model=$MERGED,max_pixels=262144,enforce_eager=True,tensor_parallel_size=1,gpu_memory_utilization=0.93,max_model_len=18432" \
    --tasks visual_probe_easy_reasoning --batch_size 1 --log_samples --output_path "$RES" \
    --gen_kwargs "max_new_tokens=512" 2>&1 | tee -a "$log"
  echo "########## DONE VisualProbe  $EXP :: $(date '+%F %T') ##########" | tee -a "$log"
done
echo "########## VisualProbe 전체 완료 :: $(date '+%F %T') ##########"
