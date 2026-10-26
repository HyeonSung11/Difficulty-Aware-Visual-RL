#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 그룹B judge 채점 재실행 — judge를 sleep 없이 "항상 켜짐"으로 직접 띄운다.
# (기존 실패 원인: sleep_level=2 judge를 채점 때마다 wake 실패 -> HTTPConnection 오류로 hang)
# judge는 GPU1 독차지라 상시 로드 OK. mathvision/mmmu는 이미 저장돼 제외.
set -o pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm 2>/dev/null

MERGED=${VERO_DATA}/vero_runs/merged_models/curriculum_dyn_step4000
EXP=curriculum_dyn_step4000
EVAL_DIR=${VERO_REPO}/vero-eval
REPO=${VERO_REPO}/vero-rl
RES=${VERO_REPO}/investigation_2x4090/eval_results_${EXP}
LOGDIR=$REPO/_logs
mkdir -p "$RES" "$LOGDIR"
export LMMS_SYSTEM_PROMPT_FILE=$REPO/examples/prompts/system_prompt_concise.txt
COMMON_MODEL="max_pixels=262144,enforce_eager=True"
GEN="max_new_tokens=512"

# ── judge 서버: sleep 모드 없이 직접 기동 (GPU1) ──
JLOG=$LOGDIR/judge_alwayson_$(date +%Y%m%d_%H%M%S).log
echo ">>> judge 상시-켜짐 기동 (GPU1, Qwen3-32B-AWQ, no sleep)"
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-32B-AWQ \
  --port 51001 --served-model-name qwen_vllm \
  --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
  --max-model-len 8192 --enforce-eager --dtype bfloat16 \
  --chat-template $REPO/examples/prompts/chat_template_no_think.jinja \
  --api-key qwen > "$JLOG" 2>&1 &
JUDGE_PID=$!
echo "judge PID=$JUDGE_PID, log=$JLOG"

# 준비 대기 (최대 15분)
tries=0
until curl -s -m 3 http://127.0.0.1:51001/v1/models -H "Authorization: Bearer qwen" 2>/dev/null | grep -q qwen_vllm; do
  sleep 10; tries=$((tries+1))
  if ! kill -0 $JUDGE_PID 2>/dev/null; then echo "!!! judge 사망 (로그 확인: $JLOG)"; exit 1; fi
  if [ $tries -ge 90 ]; then echo "!!! judge 15분 미기동"; kill -9 $JUDGE_PID; exit 1; fi
done
echo ">>> judge 준비됨 (상시 활성, sleep 안 함)"

cd "$EVAL_DIR"
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_URL=http://127.0.0.1:51001/v1/chat/completions
export OPENAI_API_KEY=qwen; export GPT_API_KEY=qwen
export MODEL_VERSION=qwen_vllm; export JUDGE_MODEL_PATH=qwen_vllm
B_MARGS="tensor_parallel_size=1,gpu_memory_utilization=0.93,max_model_len=18432"

run_task () {
  local tasks="$1" tag="$2" extra="${3:-}"
  local log=$LOGDIR/eval_${EXP}_${tag}_$(date +%Y%m%d_%H%M%S).log
  echo "########## START $tag ($tasks) $(date) ##########" | tee "$log"
  rm -rf /dev/shm/psm_* /dev/shm/vllm* 2>/dev/null
  timeout 25200 python3 -m lmms_eval --model vllm \
    --model_args "model=$MERGED,$COMMON_MODEL,$B_MARGS" \
    --tasks "$tasks" --batch_size 1 --log_samples --output_path "$RES" \
    --gen_kwargs "$GEN" $extra 2>&1 | tee -a "$log"
  echo "########## DONE $tag $(date) ##########" | tee -a "$log"
}

# judge 필요한 7개만 재실행 (mathvision/mmmu 제외)
run_task "charxiv_reasoning" "B_charxiv"
run_task "mmvetv2_group_img_reasoning" "B_mmvetv2"
run_task "simplevqa_en_reasoning" "B_simplevqa"
run_task "mm_mt_bench_reasoning" "B_mm_mt"
run_task "mia_bench_reasoning" "B_mia"
run_task "mmifeval_reasoning" "B_mmifeval"
run_task "chartmuseum_reasoning" "B_chartmuseum" "--limit 514"

echo "########## 그룹B judge 재실행 완료 $(date) ##########"
kill -9 $JUDGE_PID 2>/dev/null
