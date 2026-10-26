#!/usr/bin/env python3
"""수정본.docx용 컬러 버전. 색으로 계열을 구분한다(흑백 인쇄 규정 미적용)."""
import os
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

FONT = os.path.expandvars("${VERO_REPO}/curriculum/NanumGothic.ttf")
fp = fm.FontProperties(fname=FONT); fm.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": fp.get_name(), "axes.unicode_minus": False,
                     "axes.edgecolor": "black", "text.color": "black",
                     "axes.labelcolor": "black", "xtick.color": "black",
                     "ytick.color": "black"})
OUT = os.path.expandvars("${VERO_REPO}/papers/ICROS/")

# Okabe-Ito 색맹 친화 팔레트
BLUE, ORANGE, GREEN, VERMIL, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"

# ── 그림 2 : tier별 실시간 정답률 추이
traj = json.load(open(os.path.expandvars("${VERO_REPO}/curriculum/uns1_tier_traj.json")))
steps = [r["step"] for r in traj]
fig, ax = plt.subplots(figsize=(3.07, 1.55), dpi=600)
for key, color, mk in (("medium", BLUE, "o"), ("hard", ORANGE, "s"), ("unsolved", GREEN, "^")):
    ax.plot(steps, [r[key] for r in traj], color=color, lw=1.1, marker=mk,
            ms=2.2, markerfacecolor=color, markeredgecolor=color,
            markeredgewidth=0.6, label=key)
ax.set_xlabel("training step", fontproperties=fp, fontsize=7)
ax.set_ylabel("live pass rate", fontproperties=fp, fontsize=7)
ax.set_ylim(0, 1.0); ax.set_xlim(0, 6100); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
ax.grid(color="0.85", lw=0.4); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(labelsize=6, length=2, pad=1.5)
ax.legend(prop=fm.FontProperties(fname=FONT, size=6.5), ncol=3, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.03), handlelength=2.0,
          columnspacing=1.0, handletextpad=0.4)
fig.tight_layout()
fig.savefig(OUT + "fig_frontier_color.png", bbox_inches="tight", pad_inches=0.02,
            facecolor="white")

# ── 그림 3 : 도메인별 비교 (색으로 구분)
rows = [r for r in json.load(open(os.path.expandvars("${VERO_REPO}/curriculum/final_table_6way.json")))
        if r["curr"] is not None and r.get("uns0") is not None]
order = ["Chart\\&OCR", "Ground\\&Cnt", "STEM", "Spatial", "Knowledge", "Caption\\&IF"]
lab = {"Chart\\&OCR": "Chart", "Ground\\&Cnt": "Ground", "STEM": "STEM",
       "Spatial": "Spatial", "Knowledge": "Knowl.", "Caption\\&IF": "Caption"}
keys = ["terse", "curr", "uns0", "uns2", "uns1"]
agg = {d: {k: [] for k in keys} for d in order}
for r in rows:
    for k in keys: agg[r["dom"]][k].append(r[k])
m = lambda d, k: sum(agg[d][k]) / len(agg[d][k])

fig, ax = plt.subplots(figsize=(3.07, 1.58), dpi=600)
x = np.arange(len(order)); w = 0.16
series = [("terse (4k)", BLUE), ("curriculum (4k)", ORANGE),
          ("curr+uns0 (mix 0%)", GREEN), ("curr+uns2 (mix 50%)", VERMIL),
          ("uns1 scratch (mix 25%)", PURPLE)]
for i, ((l, color), k) in enumerate(zip(series, keys)):
    ax.bar(x + (i - 2) * w, [m(d, k) for d in order], w, label=l,
           facecolor=color, edgecolor="black", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels([lab[d] for d in order],
                                     fontproperties=fp, fontsize=6)
ax.set_ylabel("domain average score", fontproperties=fp, fontsize=7)
ax.set_ylim(35, 75); ax.grid(axis="y", color="0.85", lw=0.4); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(labelsize=6, length=2, pad=1.5)
ax.legend(prop=fm.FontProperties(fname=FONT, size=5.5), ncol=2, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.26), handlelength=1.8,
          columnspacing=0.8, handletextpad=0.4)
fig.tight_layout()
fig.savefig(OUT + "fig_domain_compare_color.png", bbox_inches="tight", pad_inches=0.02,
            facecolor="white")
print("컬러 그림 2개 저장 완료")
