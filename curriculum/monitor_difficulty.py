#!/usr/bin/env python3
"""커리큘럼 난이도 분포 모니터 — 200 step마다 (live_p, 샘플링 무게) 변화를 표로.

import os
로그의 "[curriculum] weights 갱신 live_p={...} -> w={...}" 줄과
"training/global_step:N ... critic/score/mean:X" 줄을 파싱한다.
사용: python monitor_difficulty.py [logfile]   (없으면 .current_log 사용)
"""
import re, sys, glob, os

TIERS = ["easy", "medium", "hard", "unsolved"]
CUR = os.path.expandvars("${VERO_REPO}/curriculum")


def find_log():
    if len(sys.argv) > 1:
        return sys.argv[1]
    p = os.path.join(CUR, ".current_log")
    if os.path.exists(p):
        cur = open(p).read().strip()
        if os.path.exists(cur):
            return cur
    # 폴백: 가장 최근 dyn_train 로그
    logs = sorted(glob.glob(os.path.join(CUR, "dyn_train_*.log")), key=os.path.getmtime)
    return logs[-1] if logs else None


def parse_dict(s):
    """{'easy': None, 'medium': 0.576, ...} → dict"""
    out = {}
    for k, v in re.findall(r"'(\w+)':\s*(None|[-\d.eE]+)", s):
        out[k] = None if v == "None" else float(v)
    return out


def main():
    log = find_log()
    if not log or not os.path.exists(log):
        print("로그를 찾을 수 없음"); return
    text = open(log, errors="ignore").read()

    # 1) 커리큘럼 무게 갱신 이벤트 (200 step마다)
    events = []
    for m in re.finditer(r"weights 갱신 live_p=(\{[^}]*\})\s*->\s*w=(\{[^}]*\})", text):
        events.append((parse_dict(m.group(1)), parse_dict(m.group(2))))

    # 2) step별 전체 score (정답률) — 이벤트에 붙일 참고값
    steps = re.findall(r"training/global_step:(\d+)", text)
    cur_step = int(steps[-1]) if steps else 0

    print(f"\n{'='*72}")
    print(f" 커리큘럼 난이도 분포 추이   (현재 global_step ≈ {cur_step})")
    print(f" 로그: {os.path.basename(log)}")
    print(f"{'='*72}")
    if not events:
        print(" 아직 첫 200 step 갱신 전 — 무게 갱신 이벤트 없음.")
        print(" (init 무게: easy 0.05 / medium 0.75 / hard 0.20 / unsolved 0.0)")
        return

    # live_p 추이 표
    print("\n[ tier별 라이브 정답률 p  (측정 안된 tier=·) ]")
    print(f"{'갱신#':>5} | " + " | ".join(f"{t:>8}" for t in TIERS))
    print("-" * 52)
    for i, (lp, _) in enumerate(events, 1):
        cells = []
        for t in TIERS:
            v = lp.get(t)
            cells.append(f"{v:8.3f}" if isinstance(v, float) else f"{'·':>8}")
        print(f"{i:>5} | " + " | ".join(cells))

    # 샘플링 무게(=난이도 분포) 추이 표
    print("\n[ 샘플링 무게 = 다음 200 step 동안 각 난이도에서 뽑을 비율 ]")
    print(f"{'갱신#':>5} | " + " | ".join(f"{t:>8}" for t in TIERS))
    print("-" * 52)
    for i, (_, w) in enumerate(events, 1):
        cells = [f"{w.get(t,0):8.3f}" for t in TIERS]
        print(f"{i:>5} | " + " | ".join(cells))

    # 해석 (마지막 이벤트 기준)
    lp, w = events[-1]
    print(f"\n[ 해석 — 최신 갱신 기준 ]")
    measured = {t: lp[t] for t in TIERS if isinstance(lp.get(t), float)}
    if measured:
        hardest = min(measured, key=measured.get)
        heaviest = max(TIERS, key=lambda t: w.get(t, 0))
        print(f"  · 가장 어려운(측정) tier: {hardest} (p={measured[hardest]:.3f})")
        print(f"  · 무게 최대 tier: {heaviest} ({w[heaviest]:.3f}) ← 지금 집중 학습 중")
    if len(events) >= 2:
        lp0 = events[0][0]
        for t in ["medium", "hard"]:
            a, b = lp0.get(t), lp.get(t)
            if isinstance(a, float) and isinstance(b, float):
                arrow = "↑상승" if b > a else "↓하락" if b < a else "→유지"
                print(f"  · {t} 정답률: {a:.3f} → {b:.3f} {arrow}")
    print()


if __name__ == "__main__":
    main()
