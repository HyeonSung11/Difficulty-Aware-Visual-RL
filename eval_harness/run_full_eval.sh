#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 개선 모델 전체 평가를 "우리가 쓴 그대로" 한 번에 재현.
#   사용: bash run_full_eval.sh <MERGED_MODEL_PATH> [EXP_NAME]
# 그룹 A(no-judge, TP2) → 그룹 B(judge, GPU-split) 순차 실행.
set -uo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm

MERGED="${1:?사용: run_full_eval.sh <MERGED_MODEL_PATH> [EXP_NAME]}"
EXP="${2:-$(basename "$MERGED")}"
EVAL_DIR=${VERO_REPO}/vero-eval
REPO=${VERO_REPO}/vero-rl
RES=${VERO_REPO}/investigation_2x4090/eval_results_${EXP}
LOGDIR=$REPO/_logs
mkdir -p "$RES"

# ===== 전 태스크 공통 (채점 파라미터 고정) =====
export LMMS_SYSTEM_PROMPT_FILE=$REPO/examples/prompts/system_prompt_concise.txt
COMMON_MODEL="max_pixels=262144,enforce_eager=True"
GEN="max_new_tokens=512"
cd "$EVAL_DIR"

run_task () {  # $1=tasks $2=extra_model_args $3=tag $4=extra_cli
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

# ============ 그룹 A: no-judge (TP=2, max_model_len 32768, GPU0+1) ============
export CUDA_VISIBLE_DEVICES=0,1
A_TASKS="chartqa_reasoning,chartqa_pro_reasoning,infovqa_val_reasoning,evochart_reasoning,\
countbenchqa_reasoning,countqa_reasoning,mme_realworld_lite_reasoning,vstar_bench_reasoning,\
screenspot_point_in_box_reasoning,visual_probe_easy_reasoning,mathvista_testmini_reasoning,\
cv_bench_reasoning,blink_multi_view_reasoning,erqa_reasoning,embspatial_reasoning,\
game_qa_lite_reasoning,realworldqa_reasoning"
# masked-scatter 격리를 위해 태스크를 개별 실행(한 태스크 크래시가 전체를 안 죽이게)
for T in ${A_TASKS//,/ }; do
  run_task "$T" "tensor_parallel_size=2,gpu_memory_utilization=0.85,max_model_len=32768" "A_${T%%_*}"
done

# visual_probe judge 오프라인 채점 (TP2엔 judge 없음)
echo ">>> VisualProbe 오프라인 채점"
python3 ${VERO_REPO}/investigation_2x4090/score_visualprobe.py 2>&1 | tee -a $LOGDIR/eval_${EXP}_visualprobe_score.log || true

# ============ 그룹 B: judge 필요 (judge 서버 GPU1 + eval GPU0) ============
echo ">>> judge 서버 기동 (GPU1, Qwen3-32B-AWQ, non-think, max-model-len 8192)"
# [fix] llm_judge_server.sh는 START_VLLM_JUDGE_SERVER=1 이 없으면 서버를 안 띄우고
# 조용히 끝난다 -> until curl 루프가 무한 대기. 필수 변수 전부 export.
export START_VLLM_JUDGE_SERVER=1
export DATE_TIME="$(date +%Y%m%d_%H%M%S)"
export VLLM_JUDGE_PORT=51001
export VLLM_JUDGE_API_KEY=qwen
export VLLM_JUDGE_MODEL_NAME=qwen_vllm
export VLLM_JUDGE_TENSOR_PARALLEL_SIZE=1
export VLLM_JUDGE_MODEL_PATH=Qwen/Qwen3-32B-AWQ
export VLLM_JUDGE_MAX_MODEL_LEN=8192
export VLLM_JUDGE_CHAT_TEMPLATE=$REPO/examples/prompts/chat_template_no_think.jinja
CUDA_VISIBLE_DEVICES=1 bash $REPO/examples/model_runs/shared/llm_judge_server.sh &
JUDGE_PID=$!
# 서버 뜰 때까지 대기
until curl -s -m 3 http://127.0.0.1:51001/v1/models -H "Authorization: Bearer qwen" 2>/dev/null | grep -q qwen_vllm; do
  sleep 10; echo "  judge 서버 대기중..."
done
echo ">>> judge 서버 준비됨"

export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_URL=http://127.0.0.1:51001/v1/chat/completions
export OPENAI_API_KEY=qwen; export GPT_API_KEY=qwen
export MODEL_VERSION=qwen_vllm; export JUDGE_MODEL_PATH=qwen_vllm
B_MARGS="tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=18432"

# 개별 실행(크래시 격리). chartmuseum만 --limit 514.
for T in charxiv_reasoning mmvetv2_group_img_reasoning simplevqa_en_reasoning \
         mmmu_pro_reasoning mathvision_test_reasoning \
         mm_mt_bench_reasoning mia_bench_reasoning mmifeval_reasoning; do
  run_task "$T" "$B_MARGS" "B_${T%%_*}"
done
run_task "chartmuseum_reasoning" "$B_MARGS" "B_chartmuseum" "--limit 514"

kill $JUDGE_PID 2>/dev/null || true
echo "########## ALL DONE. 결과: $RES ##########"
