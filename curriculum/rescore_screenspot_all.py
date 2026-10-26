#!/usr/bin/env python3
"""저장된 ScreenSpot 출력 전체를 두 좌표 규약으로 재채점한다.

import os
배경: terse 열의 89.1 은 오프라인 재채점(abs_pixel 가정)에서 나왔고,
      나머지 열은 평가 실행(relative_1000 가정)에서 나왔다.
      두 가정이 양립하지 않으므로 같은 기준으로 다시 계산해 비교 가능성을 확인한다.
GPU를 쓰지 않는다 (저장된 응답만 파싱).
"""
import glob, json, os, sys
sys.path.insert(0, os.path.expandvars("${VERO_REPO}/vero-eval"))
from datasets import load_dataset
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks.screenspot_point_in_box.utils_point import (
    _parse_pred_point, _normalize_point, compute_point_in_box,
)

RESROOT = os.path.expandvars("${VERO_REPO}/investigation_2x4090")
MODELS = [
    ("mbase", "base_qwen3vl_ourproto"),
    ("terse", "fair"),
    ("curr",  "curriculum_dyn_step4000"),
    ("uns0",  "curriculum_ablate_uns0_step6000"),
    ("uns2",  "curriculum_uns2_step6000"),
    ("uns1",  "curriculum_uns1_step6000"),
]

ds = load_dataset("rootsautomation/ScreenSpot", split="test")
sizes = {i: ds[i]["image"].size for i in range(len(ds))}
ws = sorted(w for w, _ in sizes.values())
print(f"ScreenSpot {len(ds)}문항 | 이미지 폭 중앙값 {ws[len(ws)//2]}, 범위 {ws[0]}~{ws[-1]}\n")

print(f"{'모델':<7} {'파싱':>9} {'abs_pixel':>10} {'rel_1000':>10}   출처")
for col, exp in MODELS:
    fs = sorted(glob.glob(f"{RESROOT}/eval_results_{exp}/*samples_screenspot*.jsonl"))
    if not fs:
        print(f"{col:<7} {'-':>9} {'-':>10} {'-':>10}   (없음)")
        continue
    rows = [json.loads(l) for l in open(fs[-1])]
    hit = {"abs_pixel": 0, "relative_1000": 0}
    parsed = 0
    for o in rows:
        w, h = sizes[o["doc_id"]]
        gt = o["screenspot_PointInBox_ACC"]["bbox"]          # 이미 정규화된 GT
        pt = _parse_pred_point(extract_final_answer(o["filtered_resps"][0]))
        if pt is None:
            continue
        parsed += 1
        for fmt in hit:
            hit[fmt] += int(compute_point_in_box(gt, _normalize_point(pt, w, h, fmt)))
    n = len(rows)
    print(f"{col:<7} {parsed}/{n:<5} {100*hit['abs_pixel']/n:9.2f} "
          f"{100*hit['relative_1000']/n:10.2f}   {os.path.basename(fs[-1])[:24]}")
