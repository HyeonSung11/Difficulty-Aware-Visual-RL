#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# P2 재시도: 절제 학습(min_unsolved=0) → 절제 모델 평가
set -o pipefail
C=${VERO_REPO}/curriculum
say(){ echo; echo "════════ $* :: $(date '+%F %T') ════════"; echo; }
gpu_wait_free () {
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "vllm serve Qwen" 2>/dev/null
  for i in $(seq 1 60); do
    a=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "${a:-9999}" -lt 1500 ] && [ "${b:-9999}" -lt 1500 ]; then echo ">>> GPU 여유 (${a}/${b} MiB)"; return 0; fi
    sleep 10
  done
  echo "!!! GPU 10분간 미해제"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
}
say "P2a  min_unsolved=0 절제 학습 (4000 → 6000)"
gpu_wait_free
bash "$C/run_ablation_uns0.sh"; echo "---- P2a rc=$? ----"

say "P2b  절제 모델 평가"
gpu_wait_free
CK=${VERO_DATA}/vero_runs/checkpoints/verl/curriculum_ablate_uns0/global_step_6000
if [ -d "$CK/actor/lora_adapter" ]; then
  bash "$C/run_eval_one.sh" curriculum_ablate_uns0_step6000 "$CK"; echo "---- P2b rc=$? ----"
else
  echo "!!! 절제 체크포인트 없음, P2b 건너뜀: $CK"
fi
gpu_wait_free
say "P2 완료"
