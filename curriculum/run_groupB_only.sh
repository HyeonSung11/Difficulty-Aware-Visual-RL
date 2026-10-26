#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 그룹 A는 이미 완료 → 남은 그룹 B(judge 9개) + VisualProbe 재채점만 실행.
# judge 서버 필수 변수(START_VLLM_JUDGE_SERVER 등) 전부 export 하는 버전.
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
cd "$EVAL_DIR"

run_task () {
  local tasks="$1" margs="$2" tag="$3" extra="${4:-}"
  local log=$LOGDIR/eval_${EXP}_${tag}_$(date +%Y%m%d_%H%M%S).log
  echo "########## START $tag ($tasks) $(date) ##########" | tee "$log"
  rm -rf /dev/shm/psm_* /dev/shm/vllm* 2>/dev/null
  timeout 25200 python3 -m lmms_eval --model vllm \
    --model_args "model=$MERGED,$COMMON_MODEL,$margs" \
    --tasks "$tasks" --batch_size 1 --log_samples --output_path "$RES" \
    --gen_kwargs "$GEN" $extra 2>&1 | tee -a "$log"
  echo "########## DONE $tag $(date) ##########" | tee -a "$log"
}

# ---- judge 서버 기동 (GPU1) ----
# [핵심] llm_judge_server.sh는 함수 정의만 하는 파일 → source 후 함수를 직접 호출해야 한다.
# `bash ... &`로 실행하면 함수가 정의만 되고 아무것도 안 뜬 채 종료됨(이전 실패 원인).
# 함수는 서버를 백그라운드로 띄우고 sleep 모드로 재운 뒤 리턴하며, 서버는 계속 살아있음.
echo ">>> judge 서버 기동 (GPU1, Qwen3-32B-AWQ)"
export REPO_ROOT=$REPO
# [fix] judge 기본 util 0.35는 32B-AWQ(~19GB) 가중치 올리면 KV 캐시 메모리가 없어
# "No available memory for the cache blocks"로 죽는다. judge가 GPU1을 독차지하므로 0.88로.
export GPU_MEMORY_UTILIZATION_LLM_JUDGE=0.88
export START_VLLM_JUDGE_SERVER=1
export DATE_TIME="$(date +%Y%m%d_%H%M%S)"
export TRAINER_EXPERIMENT_SUFFIX=eval_groupB
export VLLM_JUDGE_PORT=51001
export VLLM_JUDGE_API_KEY=qwen
export VLLM_JUDGE_MODEL_NAME=qwen_vllm
export VLLM_JUDGE_TENSOR_PARALLEL_SIZE=1
export VLLM_JUDGE_MODEL_PATH=Qwen/Qwen3-32B-AWQ
export VLLM_JUDGE_MAX_MODEL_LEN=8192
export VLLM_JUDGE_SLEEP_LEVEL=2
export VLLM_JUDGE_CHAT_TEMPLATE=$REPO/examples/prompts/chat_template_no_think.jinja
export VLLM_JUDGE_LOG_DIR=$REPO/vllm_logs
export CUDA_VISIBLE_DEVICES=1
source "$REPO/examples/model_runs/shared/llm_judge_server.sh"
llm_judge_setup_and_start   # 서버 뜰 때까지 대기 → sleep 재우고 리턴 (서버는 백그라운드 유지)
# 살아있는지 확인
if ! curl -s -m 5 http://127.0.0.1:51001/v1/models -H "Authorization: Bearer qwen" 2>/dev/null | grep -q qwen_vllm; then
  echo "!!! judge 서버 기동 실패 — 중단"; exit 1
fi
echo ">>> judge 서버 준비됨 (sleep 모드, 요청 시 자동 wake)"

export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_URL=http://127.0.0.1:51001/v1/chat/completions
export OPENAI_API_KEY=qwen; export GPT_API_KEY=qwen
export MODEL_VERSION=qwen_vllm; export JUDGE_MODEL_PATH=qwen_vllm
# [fix] fair 성공값과 동일하게 eval util 0.93 (0.9는 8B 가중치 후 KV캐시 부족 -> max_len 18432 실패)
B_MARGS="tensor_parallel_size=1,gpu_memory_utilization=0.93,max_model_len=18432"

# ---- 그룹 B: judge 9개 (vero judge manager가 judge wake/sleep 관리) ----
for T in charxiv_reasoning mmvetv2_group_img_reasoning simplevqa_en_reasoning \
         mmmu_pro_reasoning mathvision_test_reasoning \
         mm_mt_bench_reasoning mia_bench_reasoning mmifeval_reasoning; do
  run_task "$T" "$B_MARGS" "B_${T%%_*}"
done
run_task "chartmuseum_reasoning" "$B_MARGS" "B_chartmuseum" "--limit 514"

# ---- VisualProbe 오프라인 재채점 (맨 마지막, judge 명시적 wake 후) ----
# score_visualprobe.py는 judge wake 로직이 없어 sleep 상태면 무한 대기 -> 먼저 깨운다.
echo ">>> judge wake (VisualProbe 채점용)"
curl -s -m 15 -X POST "http://127.0.0.1:51001/wake_up" -H "Authorization: Bearer qwen" 2>/dev/null
sleep 5
echo ">>> VisualProbe 오프라인 재채점"
python3 ${VERO_REPO}/investigation_2x4090/score_visualprobe.py 2>&1 | tee -a $LOGDIR/eval_${EXP}_visualprobe_score.log || true

echo "########## 그룹B 평가 완료 $(date) ##########"
kill $JUDGE_PID 2>/dev/null
