#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# uns2 보충 실행 — GPU1을 점유하던 고아 judge(VLLM::EngineCore) 때문에
# 즉시 실패한 그룹A 2개(ChartQA, ChartQA-Pro)를 재실행한다.
#   실행 전 조건: 체인 평가가 완전히 끝나 GPU가 비어 있을 것.
set -o pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm

EXP=curriculum_uns2_step6000
MERGED=${VERO_DATA}/vero_runs/merged_models/${EXP}
REPO=${VERO_REPO}/vero-rl
EVAL_DIR=${VERO_REPO}/vero-eval
RES=${VERO_REPO}/investigation_2x4090/eval_results_${EXP}
LOGDIR=$REPO/_logs
mkdir -p "$RES" "$LOGDIR"
export LMMS_SYSTEM_PROMPT_FILE=$REPO/examples/prompts/system_prompt_concise.txt

# GPU가 정말 비었는지 먼저 확인 (같은 사고 재발 방지)
for g in 0 1; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$used" -gt 2000 ]; then
    echo "!!! GPU$g 에 ${used}MiB 사용 중 — 점유 프로세스를 먼저 정리하세요"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    exit 1
  fi
done
echo ">>> GPU 비어 있음 확인"

cd "$EVAL_DIR"
export CUDA_VISIBLE_DEVICES=0,1
A_MARGS="tensor_parallel_size=2,gpu_memory_utilization=0.85,max_model_len=32768"
for T in chartqa_reasoning chartqa_pro_reasoning; do
  log=$LOGDIR/eval_${EXP}_catchup_${T%%_*}_$(date +%Y%m%d_%H%M%S).log
  echo "########## START $T :: $(date '+%F %T') ##########" | tee "$log"
  rm -rf /dev/shm/psm_* /dev/shm/vllm* 2>/dev/null
  timeout 25200 python3 -m lmms_eval --model vllm \
    --model_args "model=$MERGED,max_pixels=262144,enforce_eager=True,$A_MARGS" \
    --tasks "$T" --batch_size 1 --log_samples --output_path "$RES" \
    --gen_kwargs "max_new_tokens=512" 2>&1 | tee -a "$log"
  echo "########## DONE $T :: $(date '+%F %T') ##########" | tee -a "$log"
done
echo "########## uns2 보충 완료 :: $(date '+%F %T') ##########"
