#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 논문 표의 두 구멍을 메우는 보충 평가.
#
#   [1단계] ScreenSpot 재측정 — curr 4k
#           curr 4k 만 좌표 형식 보정(rel1000) 이전 값이라 다른 열과 비교가 안 된다.
#           다른 모델이 측정된 것과 동일한 그룹A 설정(TP=2)으로 다시 돌린다.
#   [2단계] VisualProbe 측정 — 측정 base / curr 4k / curr+uns0
#           집계 함수가 judge 엔진을 요구하므로 judge 상시 기동 상태에서만 돌려야 한다.
#           (judge 없이 돌리면 집계에서 죽고 samples jsonl 조차 안 남는다)
#
# 실측 기준 예상 소요: ScreenSpot 52분 + judge 기동 4분 + VisualProbe 33분×3 ≒ 2시간 50분
set -o pipefail
# conda activate.d 의 cuda-nvcc 스크립트가 미정의 변수를 참조한다. 미리 채워둔다.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm

REPO=${VERO_REPO}/vero-rl
EVAL_DIR=${VERO_REPO}/vero-eval
RESROOT=${VERO_REPO}/investigation_2x4090
MERGEROOT=${VERO_DATA}/vero_runs/merged_models
LOGDIR=$REPO/_logs
mkdir -p "$LOGDIR"
export LMMS_SYSTEM_PROMPT_FILE=$REPO/examples/prompts/system_prompt_concise.txt
COMMON_MODEL="max_pixels=262144,enforce_eager=True"
GEN="max_new_tokens=512"

banner () { echo; echo "########## $* :: $(date '+%F %T') ##########"; }

JUDGE_PID=""
# judge 종료 시 부모만 죽이면 자식 VLLM::EngineCore 가 고아로 남아 GPU를 계속 점유한다.
cleanup () {
  [ -n "$JUDGE_PID" ] && kill -9 -"$JUDGE_PID" 2>/dev/null
  [ -n "$JUDGE_PID" ] && kill -9  "$JUDGE_PID" 2>/dev/null
  pkill -9 -f "vllm serve Qwen/Qwen3-32B-AWQ" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for i in $(seq 10); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null || echo 0)
    [ "${u:-0}" -lt 1000 ] && break
    sleep 2
  done
}
trap cleanup EXIT INT TERM

# 실험명 -> 모델 경로 (hf: 접두사는 병합 없이 HF 모델을 그대로 쓴다)
model_of () {
  case "$1" in
    base_qwen3vl_ourproto) echo "Qwen/Qwen3-VL-8B-Instruct" ;;
    *)                     echo "$MERGEROOT/$1" ;;
  esac
}

run_task () {  # $1=EXP  $2=task  $3=margs
  local exp="$1" task="$2" margs="$3"
  local model res log
  model=$(model_of "$exp"); res=$RESROOT/eval_results_${exp}
  mkdir -p "$res"
  log=$LOGDIR/gapfill_${exp}_${task}_$(date +%Y%m%d_%H%M%S).log
  banner "START  $exp / $task"
  rm -rf /dev/shm/psm_* /dev/shm/vllm* 2>/dev/null
  timeout 25200 python3 -m lmms_eval --model vllm \
    --model_args "model=$model,$COMMON_MODEL,$margs" \
    --tasks "$task" --batch_size 1 --log_samples --output_path "$res" \
    --gen_kwargs "$GEN" 2>&1 | tee -a "$log"
  banner "DONE   $exp / $task"
}

cd "$EVAL_DIR"

# ─────────── [1/2] ScreenSpot 재측정 (judge 불필요, 그룹A 설정 TP=2) ───────────
banner "[1/2] ScreenSpot 재측정 (좌표 보정 반영)"
export CUDA_VISIBLE_DEVICES=0,1
run_task curriculum_dyn_step4000 screenspot_point_in_box_reasoning \
  "tensor_parallel_size=2,gpu_memory_utilization=0.85,max_model_len=32768"

# ─────────── [2/2] VisualProbe (judge 상시 기동, GPU0 정책 + GPU1 judge) ───────────
banner "[2/2] judge 상시-켜짐 기동 (GPU1)"
JLOG=$LOGDIR/judge_gapfill_$(date +%Y%m%d_%H%M%S).log
CUDA_VISIBLE_DEVICES=1 setsid vllm serve Qwen/Qwen3-32B-AWQ \
  --port 51001 --served-model-name qwen_vllm \
  --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --max-model-len 8192 --enforce-eager --dtype bfloat16 \
  --chat-template $REPO/examples/prompts/chat_template_no_think.jinja \
  --api-key qwen > "$JLOG" 2>&1 &
JUDGE_PID=$!
echo "judge PID=$JUDGE_PID  log=$JLOG"
tries=0
until curl -s -m 3 http://127.0.0.1:51001/v1/models -H "Authorization: Bearer qwen" 2>/dev/null | grep -q qwen_vllm; do
  sleep 10; tries=$((tries+1))
  kill -0 $JUDGE_PID 2>/dev/null || { echo "!!! judge 사망 — 로그: $JLOG"; exit 1; }
  [ $tries -ge 90 ] && { echo "!!! judge 15분 미기동"; exit 1; }
done
echo ">>> judge 준비됨"

export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_URL=http://127.0.0.1:51001/v1/chat/completions
export OPENAI_API_KEY=qwen; export GPT_API_KEY=qwen
export MODEL_VERSION=qwen_vllm
export JUDGE_MODEL_PATH=qwen_vllm
export CHARXIV_JUDGE_MODEL_PATH=qwen_vllm   # 집계 함수가 이 이름으로도 조회한다

for EXP in base_qwen3vl_ourproto curriculum_dyn_step4000 curriculum_ablate_uns0_step6000; do
  run_task "$EXP" visual_probe_easy_reasoning \
    "tensor_parallel_size=1,gpu_memory_utilization=0.93,max_model_len=18432"
done

banner "보충 평가 전체 완료"
