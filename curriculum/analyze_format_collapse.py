#!/usr/bin/env python3
"""MIABench 400문항 출력 형식 분석 — 표 5의 재현 가능한 근거.

import os
기존 표 5 수치(23.0/12.5/13.0/19.5/54.8%)를 만든 붕괴 판정 기준이 기록되지
않아 재현이 불가능했다. 여기서는 기준을 코드로 고정하고, 기준 선택에 결과가
얼마나 민감한지도 함께 출력한다.

정의
  답변 블록  : <answer>...</answer> 안쪽 문자열 (strip)
  앞 본문    : <answer> 앞의 문자열 (strip)
  붕괴       : 답변 블록 길이 <= COLLAPSE_CHARS 문자
               (임계값은 보고 관행일 뿐 적합된 값이 아니다. 20~50 구간의
                민감도를 아래 표로 함께 보고한다.)
"""
import glob, json, math, re, statistics as st
from scipy import stats

RESROOT = os.path.expandvars("${VERO_REPO}/investigation_2x4090")
COLLAPSE_CHARS = 40
MODELS = [("terse 4k", "fair", "—"),
          ("curr. 4k", "curriculum_dyn_step4000", "—"),
          ("curr+uns0", "curriculum_ablate_uns0_step6000", "0%"),
          ("curr+uns2", "curriculum_uns2_step6000", "50%"),
          ("uns1 (scratch)", "curriculum_uns1_step6000", "25%")]
ANS = re.compile(r"<answer>(.*?)</answer>", re.S)


def load(exp):
    fs = sorted(glob.glob(f"{RESROOT}/eval_results_{exp}/*samples_mia_bench*.jsonl"))
    if not fs:
        return None
    ans, pre = [], []
    for line in open(fs[-1]):
        o = json.loads(line)
        r = o["filtered_resps"]
        r = r[0] if isinstance(r, list) else str(r)
        m = ANS.search(r)
        ans.append(m.group(1).strip() if m else "")
        pre.append((r[:m.start()] if m else r).strip())
    return ans, pre


def ztest(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 1.0 if se == 0 else 2 * (1 - stats.norm.cdf(abs(k1 / n1 - k2 / n2) / se))


if __name__ == "__main__":
    D = {name: load(exp) for name, exp, _ in MODELS}
    n = len(next(iter(D.values()))[0])
    print(f"MIABench {n}문항 | 붕괴 기준: 답변 블록 <= {COLLAPSE_CHARS}자\n")
    print(f"{'모델':<16}{'편입':>6}{'답변 중앙(자)':>14}{'앞 본문 중앙(자)':>17}{'붕괴 비율':>11}")
    cnt = {}
    for name, _, ratio in MODELS:
        a, p = D[name]
        cnt[name] = sum(1 for x in a if len(x) <= COLLAPSE_CHARS)
        print(f"{name:<16}{ratio:>6}{st.median(map(len,a)):>14.0f}"
              f"{st.median(map(len,p)):>17.0f}{100*cnt[name]/n:>10.1f}%")

    print("\n[비율검정] 양측 p")
    for a, b in [("curr. 4k","curr+uns0"), ("curr+uns0","curr+uns2"), ("curr+uns2","uns1 (scratch)")]:
        p = ztest(cnt[a], n, cnt[b], n)
        print(f"  {a:<16} vs {b:<16} p={p:.4f}  {'유의' if p<0.05 else '비유의'}")

    print(f"\n[민감도] 임계값별 붕괴 비율(%) 및 uns2 vs uns1 p값")
    print(f"{'임계값':>7}" + "".join(f"{m:>12}" for m,_,_ in MODELS) + f"{'uns2:uns1 p':>13}")
    for th in (20, 25, 30, 35, 40, 45, 50):
        c = {m: sum(1 for x in D[m][0] if len(x) <= th) for m,_,_ in MODELS}
        p = ztest(c["curr+uns2"], n, c["uns1 (scratch)"], n)
        print(f"{th:>7}" + "".join(f"{100*c[m]/n:>11.1f}%" for m,_,_ in MODELS) + f"{p:>13.1e}")
