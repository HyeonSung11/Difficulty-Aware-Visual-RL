#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 6000 스텝 체크포인트 2개를 순차 평가 (GPU 2장을 다 쓰므로 병렬 불가).
#   1) curriculum_uns1_step6000   : scratch, min_unsolved=1
#   2) curriculum_uns2_step6000   : 4000 resume, min_unsolved=2
set -o pipefail
CKROOT=${VERO_DATA}/vero_runs/checkpoints/verl
S=${VERO_REPO}/curriculum/run_eval_one.sh

echo "======== 체인 평가 시작 $(date '+%F %T') ========"
bash "$S" curriculum_uns1_step6000 "$CKROOT/curriculum_uns1_scratch/global_step_6000"
rc1=$?
echo "======== [1/2] uns1 종료 rc=$rc1  $(date '+%F %T') ========"

sleep 60   # GPU 메모리 완전 반납 대기
bash "$S" curriculum_uns2_step6000 "$CKROOT/curriculum_dyn_0731_2127/global_step_6000"
rc2=$?
echo "======== [2/2] uns2 종료 rc=$rc2  $(date '+%F %T') ========"
echo "======== 체인 평가 완료 $(date '+%F %T') ========"
