#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# ============================================================
#  정전/중단 후 커리큘럼 학습 재개 스크립트
#  핵심: 같은 실험명으로 실행해야 최신 체크포인트에서 이어짐
#        (resume_mode=auto 라서 latest_checkpointed_iteration.txt 읽고 자동 재개)
# ============================================================
source ${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
conda activate verovlm
cd ${VERO_REPO}/vero-rl

# GPU/잔여 프로세스 정리 (정전 후 깨끗하게)
pkill -9 -f "main_ppo|vllm serve|VLLM::|TaskRunner|ray::|raylet" 2>/dev/null
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done
ray stop --force 2>/dev/null
sleep 6

# ★ 같은 실험명 = 재개의 핵심 ★
export TRAIN_FILES=${VERO_DATA}/vero-600k/vero_600k_train.curriculum.jsonl
export TRAINER_EXPERIMENT_NAME=curriculum_dyn_0731_2127

LOG=${VERO_REPO}/curriculum/dyn_train_resume_$(date +%Y%m%d_%H%M%S).log
echo "재개: $TRAINER_EXPERIMENT_NAME (최신 저장 step 확인 -> 그 다음부터)"
cat ${VERO_DATA}/vero_runs/checkpoints/verl/${TRAINER_EXPERIMENT_NAME}/latest_checkpointed_iteration.txt 2>/dev/null
nohup bash examples/model_runs/run_gspo_qwen3vl_judge_lite.sh >> "$LOG" 2>&1 &
echo "PID=$!  로그=$LOG"
