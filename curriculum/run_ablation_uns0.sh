#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# P2 절제 실험: 커리큘럼 4k에서 출발점·스텝을 고정한 채 min_unsolved만 0으로 바꿔 +2,000스텝.
#   → uns1/curr+uns2의 교락(출발점 × 편입 비율) 해소 + 동일 스텝(6,000) 비탐색 대조군 확보.
#
# [중요] main_ppo를 직접 호출하지 않고 원 실행 스크립트를 재사용한다.
#   원 스크립트가 REPO_ROOT, VERL_VLLM_SLEEP_LEVEL=1, VERL_RESPONSE_PREFILL='<think>',
#   judge 메모리 설정 등 학습 성패를 좌우하는 환경변수를 세팅하기 때문.
#   (직접 호출했다가 REPO_ROOT 누락으로 즉시 실패한 이력 있음)
set -o pipefail
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm

CKROOT=${VERO_DATA}/vero_runs/checkpoints/verl
SRC=$CKROOT/curriculum_dyn_0731_2127/global_step_4000
EXP=curriculum_ablate_uns0
REPO=${VERO_REPO}/vero-rl
LOG=${VERO_REPO}/curriculum/train_${EXP}_$(date +%Y%m%d_%H%M%S).log

[ -d "$SRC" ] || { echo "!!! 출발 체크포인트 없음: $SRC"; exit 1; }

# GPU/잔여 프로세스 정리
pkill -9 -f "main_ppo|vllm serve|VLLM::EngineCore|ray::|raylet" 2>/dev/null
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done
ray stop --force >/dev/null 2>&1
sleep 8

export TRAIN_FILES=${VERO_DATA}/vero-600k/vero_600k_train.curriculum.jsonl
export TRAINER_EXPERIMENT_NAME=$EXP

echo "########## P2 절제 학습 시작 :: $(date '+%F %T') ##########" | tee "$LOG"
echo "  출발: $SRC (step 4000) → 6000,  min_unsolved=0,  실험명=$EXP" | tee -a "$LOG"
cd "$REPO"
bash examples/model_runs/run_gspo_qwen3vl_judge_lite.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$SRC" \
  data.curriculum.min_unsolved=0 \
  trainer.total_training_steps=6000 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "########## P2 절제 학습 종료 rc=$rc :: $(date '+%F %T') ##########" | tee -a "$LOG"
exit $rc
