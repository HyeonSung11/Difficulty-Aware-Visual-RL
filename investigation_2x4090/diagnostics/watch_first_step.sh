#!/bin/bash
# --- repo/data locations (override via environment) ---
VERO_REPO="${VERO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"
VERO_DATA="${VERO_DATA:-$VERO_REPO/data}"   # Vero-600k images/jsonl + runs live here
export VERO_REPO VERO_DATA

LOG=$(cat /tmp/claude-1002/-mnt-local-storage-sung-code/c0731b06-f8ed-46ef-ae8a-93559994963d/scratchpad/current_log_path.txt)
RESTART=$(cat /tmp/claude-1002/-mnt-local-storage-sung-code/c0731b06-f8ed-46ef-ae8a-93559994963d/scratchpad/restart_epoch.txt)
DUMP=${VERO_DATA}/vero_runs/rollout_dumps
# 최대 45분 대기 (judge 로드 + init + 첫 generation)
for i in $(seq 1 270); do
  # 첫 step 메트릭 라인?
  if grep -qE "critic/rewards/mean" "$LOG" 2>/dev/null; then
    echo "=== 첫 step 메트릭 발견 ==="
    grep -oE "training/global_step:[0-9]+|critic/score/mean:[^ ]+|critic/rewards/mean:[^ ]+|response_length/clip_ratio:[^ ]+|response_length/mean:[^ ]+" "$LOG" | tail -6
    # 재시작 후 생성된 새 dump 중 가장 최근 것 coherence 판정
    NEWDUMP=$(find "$DUMP" -name '*.jsonl' -newermt "@$RESTART" 2>/dev/null | sort | head -1)
    echo "새 dump: $NEWDUMP"
    if [ -n "$NEWDUMP" ]; then
      python3 -c "
import json,sys
p='$NEWDUMP'
tot=0;good=0;boxed=0;n=0
for ln in open(p):
    o=json.loads(ln);out=o.get('output','');n+=1
    if out:
        tot+=len(out);good+=sum(1 for c in out if ord(c)<128)
    if 'boxed' in out: boxed+=1
print(f'  samples={n} ascii_frac={good/max(tot,1):.2f} boxed={boxed}/{n}')
o=json.loads(open(p).readline())
print('  sample output[:300]:', repr(o.get('output','')[:300]))
"
    fi
    exit 0
  fi
  sleep 10
done
echo "=== 45분 내 첫 step 미도달 - 로그 tail ==="
tail -8 "$LOG"
