#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 체크포인트 1개 전체 평가: LoRA 병합 -> 그룹A(no-judge, TP2) -> judge 상시기동
#                            -> VisualProbe 오프라인 채점 -> 그룹B(judge) -> 정리
#
#   사용: bash run_eval_one.sh <EXP_NAME> <CHECKPOINT_DIR>
#   예:   bash run_eval_one.sh curriculum_uns1_step6000 \
#           ${VERO_DATA}/vero_runs/checkpoints/verl/curriculum_uns1_scratch/global_step_6000
#
# judge는 sleep 모드를 쓰지 않는다. (sleep_level=2 wake 실패로 채점 hang 났던 이력)
set -o pipefail
# conda activate.d 의 cuda-nvcc 스크립트가 미정의 변수를 참조한다. 미리 채워둔다.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm

EXP="${1:?사용: run_eval_one.sh <EXP_NAME> <CHECKPOINT_DIR | hf:MODEL_ID>}"
CK="${2:?사용: run_eval_one.sh <EXP_NAME> <CHECKPOINT_DIR | hf:MODEL_ID>}"
# 2번째 인자가 hf:로 시작하면 LoRA 병합을 건너뛰고 해당 모델을 그대로 평가한다
# (예: base 모델 평가 -> hf:Qwen/Qwen3-VL-8B-Instruct)
if [[ "$CK" == hf:* ]]; then
  SKIP_MERGE=1
  MERGED="${CK#hf:}"
else
  SKIP_MERGE=0
  ADAPTER=$CK/actor/lora_adapter
  MERGED=${VERO_DATA}/vero_runs/merged_models/${EXP}
fi

EVAL_DIR=${VERO_REPO}/vero-eval
REPO=${VERO_REPO}/vero-rl
RES=${VERO_REPO}/investigation_2x4090/eval_results_${EXP}
LOGDIR=$REPO/_logs
mkdir -p "$RES" "$LOGDIR"

export LMMS_SYSTEM_PROMPT_FILE=$REPO/examples/prompts/system_prompt_concise.txt
COMMON_MODEL="max_pixels=262144,enforce_eager=True"
GEN="max_new_tokens=512"

JUDGE_PID=""
# judge 종료: 부모만 죽이면 자식 VLLM::EngineCore가 고아로 남아 GPU를 계속 점유한다.
# (실제로 uns1/uns2 두 번 모두 22GB를 붙잡아 다음 작업을 실패시켰음)
cleanup () {
  [ -n "$JUDGE_PID" ] && kill -9 -"$JUDGE_PID" 2>/dev/null   # 프로세스 그룹
  [ -n "$JUDGE_PID" ] && kill -9  "$JUDGE_PID" 2>/dev/null
  pkill -9 -f "vllm serve Qwen/Qwen3-32B-AWQ" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null || echo 0)
    [ "${u:-0}" -lt 1000 ] && break
    sleep 2
  done
}
trap cleanup EXIT INT TERM

banner () { echo; echo "########## $* :: $(date '+%F %T') ##########"; }

# ─────────────────── [1/4] LoRA 병합 ───────────────────
banner "[1/4] LoRA 병합  $EXP"
if [ "$SKIP_MERGE" = "1" ]; then
  echo ">>> 병합 건너뜀 (HF 모델 직접 평가): $MERGED"
elif [ -f "$MERGED/config.json" ]; then
  echo ">>> 이미 병합됨, 건너뜀: $MERGED"
else
  [ -d "$ADAPTER" ] || { echo "!!! 어댑터 없음: $ADAPTER"; exit 1; }
  python3 ${VERO_REPO}/investigation_2x4090/merge_lora.py "$ADAPTER" "$MERGED" || exit 1
  [ -f "$MERGED/config.json" ] || { echo "!!! 병합 실패 (config.json 없음)"; exit 1; }
fi

run_task () {  # $1=task  $2=tag  $3=margs  $4=extra_cli
  local task="$1" tag="$2" margs="$3" extra="${4:-}"
  local log=$LOGDIR/eval_${EXP}_${tag}_$(date +%Y%m%d_%H%M%S).log
  banner "START $tag ($task)"
  rm -rf /dev/shm/psm_* /dev/shm/vllm* 2>/dev/null
  timeout 25200 python3 -m lmms_eval --model vllm \
    --model_args "model=$MERGED,$COMMON_MODEL,$margs" \
    --tasks "$task" --batch_size 1 --log_samples --output_path "$RES" \
    --gen_kwargs "$GEN" $extra 2>&1 | tee -a "$log"
  banner "DONE $tag"
}

# ─────────────── [2/4] 그룹 A: judge 불필요 (TP=2) ───────────────
banner "[2/4] 그룹 A 시작 (17 tasks, TP=2)"
export CUDA_VISIBLE_DEVICES=0,1
A_MARGS="tensor_parallel_size=2,gpu_memory_utilization=0.85,max_model_len=32768"
cd "$EVAL_DIR"
for T in chartqa_reasoning chartqa_pro_reasoning infovqa_val_reasoning evochart_reasoning \
         countbenchqa_reasoning countqa_reasoning mme_realworld_lite_reasoning \
         vstar_bench_reasoning screenspot_point_in_box_reasoning visual_probe_easy_reasoning \
         mathvista_testmini_reasoning cv_bench_reasoning blink_multi_view_reasoning \
         erqa_reasoning embspatial_reasoning game_qa_lite_reasoning realworldqa_reasoning; do
  run_task "$T" "A_${T%%_*}" "$A_MARGS"
done

# ─────────────── [3/4] judge 상시 기동 (GPU1) ───────────────
banner "[3/4] judge 상시-켜짐 기동 (GPU1, Qwen3-32B-AWQ, no sleep)"
JLOG=$LOGDIR/judge_alwayson_${EXP}_$(date +%Y%m%d_%H%M%S).log
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
echo ">>> judge 준비됨 (상시 활성)"

# VisualProbe 오프라인 채점 (judge 켜진 지금 해야 함)
banner "VisualProbe 오프라인 채점"
VP_RESULTS_DIR="$RES" python3 ${VERO_REPO}/investigation_2x4090/score_visualprobe.py \
  2>&1 | tee "$LOGDIR/eval_${EXP}_visualprobe_score.log" || echo "!!! VisualProbe 채점 실패(계속 진행)"

# ─────────────── [4/4] 그룹 B: judge 필요 (GPU0) ───────────────
banner "[4/4] 그룹 B 시작 (9 tasks, GPU0 + judge GPU1)"
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_URL=http://127.0.0.1:51001/v1/chat/completions
export OPENAI_API_KEY=qwen; export GPT_API_KEY=qwen
export MODEL_VERSION=qwen_vllm; export JUDGE_MODEL_PATH=qwen_vllm
B_MARGS="tensor_parallel_size=1,gpu_memory_utilization=0.93,max_model_len=18432"
for T in charxiv_reasoning mmvetv2_group_img_reasoning simplevqa_en_reasoning \
         mmmu_pro_reasoning mathvision_test_reasoning \
         mm_mt_bench_reasoning mia_bench_reasoning mmifeval_reasoning; do
  run_task "$T" "B_${T%%_*}" "$B_MARGS"
done
run_task "chartmuseum_reasoning" "B_chartmuseum" "$B_MARGS" "--limit 514"

banner "완료  $EXP  ->  $RES"
