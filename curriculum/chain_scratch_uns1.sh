#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 체인 실험: 현재 실험(min_unsolved=2, 4000->6000)이 6000 완료되면
#   -> config를 min_unsolved=1로 바꾸고
#   -> 처음부터(scratch, base 모델) 6000 step 새 실험 시작.
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"; conda activate verovlm 2>/dev/null
REPO=${VERO_REPO}/vero-rl
CFG=$REPO/examples/model_runs/config/gspo_qwen3vl_judge_lite.yaml
CUR_CK=${VERO_DATA}/vero_runs/checkpoints/verl/curriculum_dyn_0731_2127
NEW_NAME=curriculum_uns1_scratch
DATA=${VERO_DATA}/vero-600k/vero_600k_train.curriculum.jsonl

echo "[chain] 현재 실험(6000) 완료 대기 시작 $(date)"
# 최대 60시간 대기: latest==6000 이고 main_ppo 종료되면 완료로 판정
for i in $(seq 1 720); do
  latest=$(cat "$CUR_CK/latest_checkpointed_iteration.txt" 2>/dev/null)
  if [ "$latest" = "6000" ] && ! pgrep -f "main_ppo" >/dev/null 2>&1; then
    echo "[chain] 현재 실험 6000 완료 확인 $(date)"; break
  fi
  sleep 300
done

# GPU/잔여 정리
echo "[chain] 잔여 정리"
pkill -9 -f "main_ppo" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "TaskRunner" 2>/dev/null
ray stop --force >/dev/null 2>&1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
sleep 8

# config: min_unsolved 2 -> 1
echo "[chain] config min_unsolved 2 -> 1"
sed -i 's/^\(\s*min_unsolved:\s*\)2/\11/' "$CFG"
grep -n "min_unsolved" "$CFG"

# 새 실험 scratch 시작 (새 실험명 -> 체크포인트 없음 -> base 모델부터 0->6000)
cd $REPO
export TRAIN_FILES=$DATA
export TRAINER_EXPERIMENT_NAME=$NEW_NAME
NLOG=${VERO_REPO}/curriculum/dyn_train_${NEW_NAME}_$(date +%Y%m%d_%H%M%S).log
echo "$NLOG" > ${VERO_REPO}/curriculum/.current_log
echo "[chain] 새 실험 시작: $NEW_NAME (scratch, min_unsolved=1, 6000 step) -> $NLOG"
nohup setsid bash examples/model_runs/run_gspo_qwen3vl_judge_lite.sh > "$NLOG" 2>&1 < /dev/null &
echo "[chain] launched PID=$!"
