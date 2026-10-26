#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 체인 평가 후 잔여 보충: (1) uns2 ChartQA/ChartQA-Pro 재실행  (2) 양쪽 VisualProbe
set -o pipefail
C=${VERO_REPO}/curriculum
echo "======== 보충 시작 $(date '+%F %T') ========"
bash "$C/run_catchup_uns2.sh"; echo "---- catchup_uns2 rc=$? ----"
sleep 30
bash "$C/run_visualprobe_only.sh" curriculum_uns1_step6000 curriculum_uns2_step6000
echo "---- visualprobe rc=$? ----"
echo "======== 보충 완료 $(date '+%F %T') ========"
