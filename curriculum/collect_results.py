#!/usr/bin/env python3
"""각 모델의 eval_results_* 디렉터리에서 벤치마크 점수를 수집한다.

import os
lmms-eval 은 태스크 실행마다 <타임스탬프>_results.json 을 하나씩 남긴다.
같은 태스크가 여러 번 실행된 경우(재측정) 가장 최근 파일을 채택한다.
"""
import glob, json, os, re, sys

RESROOT = os.path.expandvars("${VERO_REPO}/investigation_2x4090")
MODELS = {                       # 표 열 이름 -> eval_results_ 디렉터리
    "mbase": "base_qwen3vl_ourproto",
    "terse": "fair",
    "curr":  "curriculum_dyn_step4000",
    "uns0":  "curriculum_ablate_uns0_step6000",
    "uns2":  "curriculum_uns2_step6000",
    "uns1":  "curriculum_uns1_step6000",
}
# 점수 키는 태스크마다 이름이 다르다. 숫자 하나짜리 지표를 골라낸다.
SKIP_KEYS = ("alias", "_stderr")


def score_of(metrics):
    cands = {k: v for k, v in metrics.items()
             if isinstance(v, (int, float))
             and not any(s in k for s in SKIP_KEYS)}
    if not cands:
        return None
    # 'judge_accuracy' / '*_ACC' / 'acc' 류를 우선하고, 없으면 유일값을 쓴다
    for pref in ("judge_accuracy", "_ACC", "acc", "score", "relaxed"):
        for k, v in cands.items():
            if pref.lower() in k.lower():
                return k, v
    k, v = next(iter(cands.items()))
    return k, v


def collect(exp):
    """태스크 -> (점수, 출처파일). 같은 태스크는 최신 실행을 채택."""
    out = {}
    d = os.path.join(RESROOT, f"eval_results_{exp}")
    for f in sorted(glob.glob(os.path.join(d, "*_results.json"))):
        stamp = os.path.basename(f).split("_results")[0]
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for task, metrics in data.get("results", {}).items():
            got = score_of(metrics)
            if got is None:
                continue
            key, val = got
            prev = out.get(task)
            if prev is None or stamp >= prev[2]:
                out[task] = (val, key, stamp, os.path.basename(f))
    return out


if __name__ == "__main__":
    per = {col: collect(exp) for col, exp in MODELS.items()}
    table = json.load(open(os.path.expandvars("${VERO_REPO}/curriculum/final_table_6way.json")))
    print(f"{'benchmark':<34} " + " ".join(f"{c:>8}" for c in MODELS))
    for row in table:
        cells = []
        for col in MODELS:
            hit = per[col].get(row["key"])
            cells.append(f"{hit[0]*100:8.2f}" if hit else f"{'-':>8}")
        print(f"{row['key']:<34} " + " ".join(cells))
