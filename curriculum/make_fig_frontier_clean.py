#!/usr/bin/env python3
# 논문 그림 1: uns1(scratch) 학습 중 tier별 실시간 정답률 추이 — 그래프만, 주석 없음
import os
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT=os.path.expandvars("${VERO_REPO}/curriculum/NanumGothic.ttf")
fp=fm.FontProperties(fname=FONT); fm.fontManager.addfont(FONT)
plt.rcParams["font.family"]=fp.get_name(); plt.rcParams["axes.unicode_minus"]=False

traj=json.load(open(os.path.expandvars("${VERO_REPO}/curriculum/uns1_tier_traj.json")))
steps=[r["step"] for r in traj]
fig,ax=plt.subplots(figsize=(7.0,3.0),dpi=300)
style=[("medium","#2f6fb5","o","-"),("hard","#e08a3c","s","--"),("unsolved","#7c4dbe","^","-.")]
for key,col,mk,ls in style:
    ys=[r[key] for r in traj]
    ax.plot(steps, ys, color=col, lw=1.6, ls=ls, marker=mk, ms=3.2,
            markerfacecolor="white", markeredgewidth=1.0, label=key)
ax.set_xlabel("training step", fontproperties=fp, fontsize=9)
ax.set_ylabel("live pass rate", fontproperties=fp, fontsize=9)
ax.set_ylim(0, 0.8); ax.set_xlim(0, 6100)
ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax.grid(color="#d8dee9", lw=0.6); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8)
ax.legend(prop=fm.FontProperties(fname=FONT,size=8), ncol=3, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5,1.14))
fig.tight_layout()
out=os.path.expandvars("${VERO_REPO}/papers/ICROS/fig_frontier.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("저장:", out)
