"""
Vero 2x4090 학습 건강도 종합 그래프.
로그(_logs/judge_concise_dump_*.log)와 rollout 덤프(런별 폴더)에서 지표를 뽑아
6패널 PNG로 저장. reward 하나가 아니라 여러 각도로 "진짜 학습되는지"를 본다.

usage: python plot_training_health.py <log_path> <dump_dir> [out_png]
"""
import sys, re, json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path = sys.argv[1]
dump_dir = sys.argv[2]
out_png = sys.argv[3] if len(sys.argv) > 3 else "training_health.png"

# ---- 1) 로그 파싱 (스텝별 스칼라) ----
KEYS = {
    "step": r"training/global_step:(\d+)",
    "reward": r"critic/rewards/mean:([-\d.eE]+)",
    "reward_max": r"critic/rewards/max:([-\d.eE]+)",
    "reward_min": r"critic/rewards/min:([-\d.eE]+)",
    "grad": r"actor/grad_norm:np\.float64\(([-\d.eE]+|nan)\)",
    "adv_mean": r"critic/advantages/mean:([-\d.eE]+)",
    "adv_max": r"critic/advantages/max:([-\d.eE]+)",
    "resp_len": r"response_length/mean:([-\d.eE]+)",
    "resp_clip": r"response_length/clip_ratio:([-\d.eE]+)",
    "pg_loss": r"actor/pg_loss:np\.float64\(([-\d.eE]+|nan)\)",
}
rows = []
for line in open(log_path):
    if "training/global_step" not in line:
        continue
    rec = {}
    ok = True
    for k, pat in KEYS.items():
        m = re.search(pat, line)
        if not m:
            ok = False
            break
        v = m.group(1)
        rec[k] = float("nan") if v == "nan" else float(v)
    if ok:
        rows.append(rec)
rows.sort(key=lambda r: r["step"])
steps = np.array([r["step"] for r in rows])
reward = np.array([r["reward"] for r in rows])
reward_max = np.array([r["reward_max"] for r in rows])
reward_min = np.array([r["reward_min"] for r in rows])
grad = np.array([r["grad"] for r in rows])
adv_mean = np.array([r["adv_mean"] for r in rows])
adv_max = np.array([r["adv_max"] for r in rows])
resp_len = np.array([r["resp_len"] for r in rows])
resp_clip = np.array([r["resp_clip"] for r in rows])

# ---- 2) 덤프 파싱 (스텝별 accuracy/format/boxed) ----
d_step, d_acc, d_fmt, d_box = [], [], [], []
for f in sorted(glob.glob(os.path.join(dump_dir, "*.jsonl")), key=lambda x: int(os.path.basename(x)[:-6])):
    st = int(os.path.basename(f)[:-6])
    acc = fmt = box = n = 0
    for ln in open(f):
        o = json.loads(ln); out = o.get("output", ""); n += 1
        acc += float(o.get("accuracy") or 0); fmt += float(o.get("format") or 0)
        if "boxed" in out:
            box += 1
    if n:
        d_step.append(st); d_acc.append(acc/n); d_fmt.append(fmt/n); d_box.append(box/n)
d_step = np.array(d_step); d_acc = np.array(d_acc); d_fmt = np.array(d_fmt); d_box = np.array(d_box)

def smooth(y, w=25):
    # 경계 보정 이동평균: 끝/시작에서 window가 데이터 밖으로 나가도 0으로 채우지 않고
    # 실제로 겹친 점 개수로만 나눔 -> 곡선 끝이 인위적으로 0쪽으로 처지는 아티팩트 제거.
    if len(y) < 3:
        return y
    w = min(w, max(3, len(y)//3))
    k = np.ones(w)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y, dtype=float), k, mode="same")
    return num/den

def grad0_frac(g, w=25):
    z = (np.nan_to_num(g, nan=0.0) == 0.0).astype(float)
    return smooth(z, w)

# ---- 3) 그림 ----
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle(f"Vero 2x4090 Training Health — {os.path.basename(log_path)}  (steps={len(steps)})", fontsize=13)

# (1) reward
a = ax[0, 0]
a.fill_between(steps, reward_min, reward_max, alpha=0.15, color="tab:blue", label="min~max")
a.plot(steps, reward, color="tab:blue", alpha=0.3, lw=0.6)
a.plot(steps, smooth(reward), color="tab:blue", lw=2, label="reward (smoothed)")
a.set_title("(1) Reward (total) - necessary, NOT sufficient")
a.set_xlabel("step"); a.set_ylabel("reward"); a.legend(fontsize=8); a.grid(alpha=0.3)

# (2) accuracy vs format  ← 핵심
a = ax[0, 1]
a.plot(d_step, smooth(d_acc), color="tab:red", lw=2, label="accuracy (real correctness)")
a.plot(d_step, smooth(d_fmt), color="tab:green", lw=2, label="format")
a.plot(d_step, smooth(d_box), color="tab:orange", lw=1.5, ls="--", label="boxed rate")
a.set_ylim(0, 1.05)
a.set_title("(2) Accuracy vs Format [KEY] - real skill vs format")
a.set_xlabel("step"); a.set_ylabel("ratio"); a.legend(fontsize=8); a.grid(alpha=0.3)

# (3) response length
a = ax[0, 2]
a.plot(steps, resp_len, color="tab:purple", alpha=0.3, lw=0.6)
a.plot(steps, smooth(resp_len), color="tab:purple", lw=2, label="mean len")
a.set_title("(3) Response Length - collapse/rambling check")
a.set_xlabel("step"); a.set_ylabel("tokens"); a.legend(fontsize=8); a.grid(alpha=0.3)

# (4) grad_norm + grad0 fraction
a = ax[1, 0]
a.plot(steps, np.nan_to_num(grad, nan=0.0), color="tab:gray", alpha=0.3, lw=0.6)
a.plot(steps, smooth(np.nan_to_num(grad, nan=0.0)), color="black", lw=2, label="grad_norm (smoothed)")
a2 = a.twinx()
a2.plot(steps, grad0_frac(grad), color="tab:red", lw=1.5, ls=":", label="grad=0 fraction")
a2.set_ylabel("grad=0 fraction", color="tab:red"); a2.set_ylim(0, 1)
a.set_title("(4) grad_norm & grad=0 fraction - learning efficiency")
a.set_xlabel("step"); a.set_ylabel("grad_norm"); a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)

# (5) advantages
a = ax[1, 1]
a.plot(steps, smooth(adv_max), color="tab:cyan", lw=2, label="|adv| max (signal strength)")
a.plot(steps, smooth(np.abs(adv_mean)), color="tab:blue", lw=1.5, label="|adv mean|")
a.axhline(0, color="k", lw=0.5)
a.set_title("(5) Advantages - learning signal present?")
a.set_xlabel("step"); a.set_ylabel("advantage"); a.legend(fontsize=8); a.grid(alpha=0.3)

# (6) response clip ratio (예산 초과 잘림)
a = ax[1, 2]
a.plot(steps, smooth(resp_clip), color="tab:brown", lw=2, label="clip_ratio")
a.set_ylim(-0.02, 1.02)
a.set_title("(6) Response Clip ratio - truncated at budget")
a.set_xlabel("step"); a.set_ylabel("clip ratio"); a.legend(fontsize=8); a.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(out_png, dpi=110)
print(f"saved: {out_png}")

# 요약 수치
def last(y, n=50):
    y = y[~np.isnan(y)]
    return float(np.mean(y[-n:])) if len(y) else float("nan")
print(f"[요약] steps={len(steps)}  reward(최근50)={last(reward):.3f}  accuracy(최근)={last(d_acc):.3f}  "
      f"format(최근)={last(d_fmt):.3f}  resp_len(최근)={last(resp_len):.0f}  grad0율(최근)={last(grad0_frac(grad)):.2f}")
