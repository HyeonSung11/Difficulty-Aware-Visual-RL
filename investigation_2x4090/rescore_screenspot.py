"""ScreenSpot 오프라인 재채점.

import os
버그: 채점 시점에 doc['image']가 사라져 _get_image_size가 (None,None) 반환 →
abs_pixel 정규화가 모든 pred를 [0,0]으로 만들어 정확도 0.2%.

여기서는 재생성 없이, 저장된 모델 출력(filtered_resps)과 데이터셋의 실제
이미지 크기를 결합해 원래 채점 로직을 그대로 재현한다.
"""
import glob, json, sys
sys.path.insert(0, os.path.expandvars("${VERO_REPO}/vero-eval"))
from datasets import load_dataset
from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks.screenspot_point_in_box.utils_point import (
    _parse_pred_point, _normalize_point, _normalize_bbox, compute_point_in_box,
)

SAMPLES = sorted(glob.glob(
    os.path.expandvars("${VERO_REPO}/investigation_2x4090/eval_results_fair/*samples_screenspot_point_in_box_reasoning.jsonl")))[-1]
print("samples:", SAMPLES)

ds = load_dataset("rootsautomation/ScreenSpot", split="test")
# doc_id -> (width, height)  (샘플의 doc_id는 데이터셋 인덱스와 일치)
sizes = {i: ds[i]["image"].size for i in range(len(ds))}

rows = [json.loads(l) for l in open(SAMPLES)]
assert len(rows) == len(ds), f"len mismatch {len(rows)} vs {len(ds)}"

correct = 0
by_type = {}
parsed = 0
for o in rows:
    did = o["doc_id"]
    w, h = sizes[did]
    acc = o["screenspot_PointInBox_ACC"]
    gt_bbox = acc["bbox"]                       # 이미 normalized (dataset gt)
    dtype = acc["data_type"]; src = acc["data_source"]
    resp = o["filtered_resps"][0]
    pt = _parse_pred_point(extract_final_answer(resp))
    if pt is None:
        pred = [0.0, 0.0]
    else:
        parsed += 1
        pred = _normalize_point(pt, w, h, "abs_pixel")   # [x/w, y/h] clamp
    ok = compute_point_in_box(gt_bbox, pred)
    correct += int(ok)
    key = ("mobile" if ("ios" in src or "android" in src) else
           "desktop" if ("macos" in src or "windows" in src) else "web") + "_" + dtype
    by_type.setdefault(key, []).append(int(ok))

n = len(rows)
print(f"\n파싱 성공: {parsed}/{n}")
print(f"=== ScreenSpot PointInBox_ACC = {correct/n:.4f}  ({correct}/{n}) = {100*correct/n:.1f} ===")
print("--- 서브카테고리 ---")
for k in sorted(by_type):
    v = by_type[k]
    print(f"  {k:16s} {sum(v)/len(v)*100:5.1f}  ({sum(v)}/{len(v)})")

with open(os.path.expandvars("${VERO_REPO}/weekly_report_0720_0724/results/screenspot_rescore_0720.txt"), "w") as f:
    f.write(f"ScreenSpot PointInBox_ACC (offline rescore, image-size fix)\n")
    f.write(f"accuracy = {correct/n:.4f} ({correct}/{n}) = {100*correct/n:.1f}\n")
    f.write(f"parsed = {parsed}/{n}\n")
    for k in sorted(by_type):
        v = by_type[k]
        f.write(f"  {k:16s} {sum(v)/len(v)*100:5.1f} ({sum(v)}/{len(v)})\n")
