#!/usr/bin/env python3
# 벤치마크 비교표 PNG — 인용(base,Vero) + 측정(base,terse,curr,curr+uns2,uns1)
import os
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT=os.path.expandvars("${VERO_REPO}/curriculum/NanumGothic.ttf")
fp=fm.FontProperties(fname=FONT); fm.fontManager.addfont(FONT)
plt.rcParams["font.family"]=fp.get_name(); plt.rcParams["axes.unicode_minus"]=False

rows=json.load(open(os.path.expandvars("${VERO_REPO}/curriculum/final_table_6way.json")))
com=[r for r in rows if r["mbase"] is not None and r["uns0"] is not None and r["curr"] is not None]; n=len(com)
av=lambda f: sum(x[f] for x in com)/n

GOOD="#cfe9d4"; WARN="#f7f0c8"; BAD="#f6cccc"; NA="#dcdcdc"
HDR="#e8edf5"; Z0="#fbfcfe"; Z1="#f2f5f9"; INK="#141a24"; SUB="#5b6779"; AVGC="#d6e4f7"
def col(v, mb):
    if v is None: return NA
    if mb is None: return None
    d=v-mb
    return GOOD if d>=-2 else (WARN if d>=-6 else BAD)

hdr=["Domain","Benchmark","base\n(인용)","Vero\n(인용)","base\n측정","terse 4k","curr. 4k","curr+uns0\n(편입 0%)","curr+uns2\n(편입 50%)","uns1 6k\n(scratch)"]
cw =[0.112,0.165,0.068,0.068,0.081,0.096,0.096,0.106,0.106,0.102]
nrow=len(rows)+3
fig=plt.figure(figsize=(12.8,0.335*nrow+1.7),dpi=200); fig.patch.set_facecolor("white")
fig.text(0.5,0.978,"난이도 커리큘럼 RL — 벤치마크 비교 (동일 프로토콜 측정 6개 모델 + 원논문 인용 2열)",
         ha="center",fontproperties=fp,fontsize=14,color=INK,weight="bold")
fig.text(0.5,0.948,"Qwen3-VL-8B · LoRA r32 · GSPO · 2×RTX 4090 · concise, 512²px, 512 tok · 셀 색상은 측정 base 대비",
         ha="center",fontproperties=fp,fontsize=8.6,color=SUB)
ax=fig.add_axes([0.03,0.075,0.945,0.85]); ax.axis("off")
rh=1.0/nrow; y=1.0
for j,h in enumerate(hdr):
    x=sum(cw[:j])
    ax.add_patch(plt.Rectangle((x,y-rh),cw[j],rh,transform=ax.transAxes,facecolor=HDR,edgecolor="white",lw=1.1))
    ax.text(x+cw[j]/2,y-rh/2,h,transform=ax.transAxes,ha="center",va="center",
            fontproperties=fp,fontsize=8.2,color=INK,weight="bold")
y-=rh; prev=None
for i,r in enumerate(rows):
    dom=r["dom"].replace("\\&","&"); dshow=dom if dom!=prev else ""; prev=dom
    nm=r["name"]+(" ※" if r["name"]=="ScreenSpot" else "")
    mb=r["mbase"]
    def fmt(v): return "N/A" if v is None else f"{v:.1f}"
    vals=[dshow,nm,f'{r["base"]:.1f}',f'{r["vero"]:.1f}',fmt(mb),fmt(r["terse"]),fmt(r["curr"]),fmt(r["uns0"]),fmt(r["uns2"]),fmt(r["uns1"])]
    cols=[None,None,None,None,(NA if mb is None else None),
          col(r["terse"],mb),col(r["curr"],mb),col(r["uns0"],mb),col(r["uns2"],mb),col(r["uns1"],mb)]
    for j,v in enumerate(vals):
        x=sum(cw[:j])
        fc=cols[j] if cols[j] else (Z0 if i%2==0 else Z1)
        ax.add_patch(plt.Rectangle((x,y-rh),cw[j],rh,transform=ax.transAxes,facecolor=fc,edgecolor="white",lw=1.1))
        ha="left" if j<2 else "center"; tx=x+0.006 if j<2 else x+cw[j]/2
        ax.text(tx,y-rh/2,v,transform=ax.transAxes,ha=ha,va="center",
                fontproperties=fp,fontsize=8.2,color=INK,weight=("bold" if j>=4 else "normal"))
    y-=rh
avg=["","측정 평균 (26개 공통)",f'{av("base"):.1f}',f'{av("vero"):.1f}',f'{av("mbase"):.1f}',
     f'{av("terse"):.1f}',f'{av("curr"):.1f}',f'{av("uns0"):.1f}',f'{av("uns2"):.1f}',f'{av("uns1"):.1f}']
avv=[av("terse"),av("curr"),av("uns0"),av("uns2"),av("uns1")]; best=max(avv)
for j,v in enumerate(avg):
    x=sum(cw[:j])
    isbest = j>=5 and abs(avv[j-5]-best)<1e-9
    fc=AVGC if isbest else HDR
    ax.add_patch(plt.Rectangle((x,y-rh),cw[j],rh,transform=ax.transAxes,facecolor=fc,edgecolor="white",lw=1.1))
    ha="left" if j<2 else "center"; tx=x+0.006 if j<2 else x+cw[j]/2
    ax.text(tx,y-rh/2,v,transform=ax.transAxes,ha=ha,va="center",
            fontproperties=fp,fontsize=8.6,color=INK,weight="bold")
fig.text(0.03,0.042,"셀 색상(측정 base 대비):",fontproperties=fp,fontsize=8.3,color=SUB)
for k,(c,lab) in enumerate([(GOOD,"base-2 이상"),(WARN,"약간 하락"),(BAD,"크게 하락"),(NA,"N/A")]):
    x=0.185+k*0.105
    fig.patches.append(plt.Rectangle((x,0.038),0.016,0.014,transform=fig.transFigure,facecolor=c,edgecolor="#c3cad6"))
    fig.text(x+0.02,0.042,lab,fontproperties=fp,fontsize=8.3,color=SUB)
fig.text(0.03,0.014,"※ ScreenSpot: 측정 6열 중 curr.4k만 좌표보정 이전 raw 점수. 인용 2열은 원논문 프로토콜(16,384 tok, 4096²px)이라 직접 비교 불가. "
                    "평균은 6개 모델 모두 값이 있는 26개 기준(VisualProbe 제외).",
         fontproperties=fp,fontsize=7.4,color=SUB)
out=os.path.expandvars("${VERO_REPO}/curriculum/result_table_6models.png")
fig.savefig(out,facecolor="white",bbox_inches="tight",pad_inches=0.14); plt.close(fig)
print("저장:",out)
