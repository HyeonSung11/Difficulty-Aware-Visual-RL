#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 커리큘럼 step_4000 → LoRA 병합 → 전체 평가(그룹 A→B) 자동 연결
set -o pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm 2>/dev/null

CK=${VERO_DATA}/vero_runs/checkpoints/verl/curriculum_dyn_0731_2127/global_step_4000
ADAPTER=$CK/actor/lora_adapter
MERGED=${VERO_DATA}/vero_runs/merged_models/curriculum_dyn_step4000
EXP=curriculum_dyn_step4000
MLOG=${VERO_REPO}/curriculum/eval_${EXP}.log

echo "########## [1/2] LoRA 병합 시작 $(date) ##########" | tee "$MLOG"
python3 ${VERO_REPO}/investigation_2x4090/merge_lora.py "$ADAPTER" "$MERGED" 2>&1 | tee -a "$MLOG"
if [ ! -f "$MERGED/config.json" ]; then
  echo "!!! 병합 실패 (config.json 없음) — 중단" | tee -a "$MLOG"; exit 1
fi
echo "########## [2/2] 전체 평가 시작 $(date) ##########" | tee -a "$MLOG"
bash ${VERO_REPO}/eval_harness/run_full_eval.sh "$MERGED" "$EXP" 2>&1 | tee -a "$MLOG"
echo "########## 평가 완료 $(date) ##########" | tee -a "$MLOG"
