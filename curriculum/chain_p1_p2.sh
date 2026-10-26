#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

# 논문 완성을 위한 잔여 실험 전체 체인 (GPU 2장, 순차)
#   P1  base 모델을 본 프로토콜로 평가            (학습 0, 평가 ~2.5일)
#   P2a min_unsolved=0 절제 학습 (4k→6k)          (~35시간)
#   P2b 그 결과 모델 평가                          (~2.5일)
set -o pipefail
C=${VERO_REPO}/curriculum
say(){ echo; echo "════════ $* :: $(date '+%F %T') ════════"; echo; }

gpu_wait_free () {   # 이전 단계의 잔여 프로세스로 다음 단계가 죽는 것을 방지
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -f "vllm serve Qwen"  2>/dev/null
  for i in $(seq 1 60); do
    a=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "${a:-9999}" -lt 1500 ] && [ "${b:-9999}" -lt 1500 ]; then echo ">>> GPU 여유 확인 (${a}/${b} MiB)"; return 0; fi
    sleep 10
  done
  echo "!!! GPU가 10분간 비지 않음 — 계속 진행"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
}

say "체인 시작"

# ─────────── P1: base 평가 ───────────
say "P1  base 모델 평가 (본 프로토콜)"
gpu_wait_free
bash "$C/run_eval_one.sh" base_qwen3vl_ourproto hf:Qwen/Qwen3-VL-8B-Instruct
echo "---- P1 rc=$? ----"

# ─────────── P2a: 절제 학습 ───────────
say "P2a  min_unsolved=0 절제 학습 (4000 → 6000)"
gpu_wait_free
bash "$C/run_ablation_uns0.sh"
echo "---- P2a rc=$? ----"

# ─────────── P2b: 절제 모델 평가 ───────────
say "P2b  절제 모델 평가"
gpu_wait_free
CK=${VERO_DATA}/vero_runs/checkpoints/verl/curriculum_ablate_uns0/global_step_6000
if [ -d "$CK/actor/lora_adapter" ]; then
  bash "$C/run_eval_one.sh" curriculum_ablate_uns0_step6000 "$CK"
  echo "---- P2b rc=$? ----"
else
  echo "!!! 절제 학습 체크포인트 없음, P2b 건너뜀: $CK"
fi

gpu_wait_free
say "체인 완료"
