#!/usr/bin/env python3
# 논문 그림 2: 도메인별 terse / curr4k / uns1 / uns2 비교 (흑백 인쇄 대비 고려)
import os
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

FONT=os.path.expandvars("${VERO_REPO}/curriculum/NanumGothic.ttf")
fp=fm.FontProperties(fname=FONT); fm.fontManager.addfont(FONT)
plt.rcParams["font.family"]=fp.get_name(); plt.rcParams["axes.unicode_minus"]=False

rows=json.load(open(os.path.expandvars("${VERO_REPO}/curriculum/final_table_6way.json")))
rows=[r for r in rows if r["curr"] is not None and r.get("uns0") is not None]
order=["Chart\\&OCR","Ground\\&Cnt","STEM","Spatial","Knowledge","Caption\\&IF"]
label={"Chart\\&OCR":"Chart&OCR","Ground\\&Cnt":"Ground&Cnt","STEM":"STEM",
       "Spatial":"Spatial","Knowledge":"Knowledge","Caption\\&IF":"Caption&IF"}
agg={d:{"terse":[], "curr":[], "uns0":[], "uns2":[], "uns1":[]} for d in order}
for r in rows:
    for k in ("terse","curr","uns0","uns2","uns1"): agg[r["dom"]][k].append(r[k])
m=lambda d,k: sum(agg[d][k])/len(agg[d][k])

fig,ax=plt.subplots(figsize=(7.0,3.4),dpi=300)
x=np.arange(len(order)); w=0.16
series=[("terse (4k)","#b9c0cc","//"),("커리큘럼 (4k)","#7f8ea3",""),
        ("curr+uns0 (편입 0%)","#2f9e5f",""),("curr+uns2 (편입 50%)","#2f6fb5",""),
        ("uns1 scratch (25%)","#e08a3c","xx")]
keys=["terse","curr","uns0","uns2","uns1"]
for i,((lab,col,hat),k) in enumerate(zip(series,keys)):
    ax.bar(x+(i-2)*w,[m(d,k) for d in order],w,label=lab,
           color=col,hatch=hat,edgecolor="#33383f",linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels([label[d] for d in order],fontproperties=fp,fontsize=8)
ax.set_ylabel("도메인 평균 점수",fontproperties=fp,fontsize=9)
ax.set_ylim(35,75); ax.grid(axis="y",color="#d8dee9",lw=0.6)
ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8)
ax.legend(prop=fm.FontProperties(fname=FONT,size=7),ncol=3,frameon=False,
          loc="upper center",bbox_to_anchor=(0.5,1.22))
fig.tight_layout()
fig.savefig(os.path.expandvars("${VERO_REPO}/papers/ICROS/fig_domain_compare.png"),
            bbox_inches="tight",pad_inches=0.02,facecolor="white")
print("저장: papers/ICROS/fig_domain_compare.png")
for d in order:
    print(f"  {label[d]:12s} terse {m(d,'terse'):6.2f}  curr {m(d,'curr'):6.2f}  uns1 {m(d,'uns1'):6.2f}  uns2 {m(d,'uns2'):6.2f}")
