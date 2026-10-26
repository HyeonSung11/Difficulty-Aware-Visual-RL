"""난이도 라벨(jsonl) → (ability/data_source) × tier 교차표 생성 (markdown + png)."""
import os
import argparse, json
from collections import defaultdict, Counter

TIERS = ["easy", "medium", "hard", "unsolved"]


def load(path):
    return [json.loads(l) for l in open(path)]


def crosstab(rows, key):
    tab = defaultdict(lambda: Counter())
    for r in rows:
        tab[r[key]][r["tier"]] += 1
    return tab


def to_md(tab, key, sort_by_total=True):
    keys = list(tab.keys())
    if sort_by_total:
        keys.sort(key=lambda k: -sum(tab[k].values()))
    lines = [f"| {key} | " + " | ".join(TIERS) + " | total | mean_p |",
             "|" + "---|" * (len(TIERS) + 3)]
    tot = Counter()
    for k in keys:
        c = tab[k]; n = sum(c.values())
        for t in TIERS:
            tot[t] += c.get(t, 0)
        # mean_p 근사: easy=0.875, medium=0.5, hard=0.125, unsolved=0
        mp = (c.get("easy",0)*0.875 + c.get("medium",0)*0.5 + c.get("hard",0)*0.125) / n if n else 0
        lines.append(f"| {k} | " + " | ".join(str(c.get(t,0)) for t in TIERS) + f" | {n} | {mp:.2f} |")
    N = sum(tot.values())
    lines.append(f"| **합계** | " + " | ".join(f"**{tot[t]}**" for t in TIERS) + f" | **{N}** | |")
    return "\n".join(lines)


def to_png(tab, key, out, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(tab.keys(), key=lambda k: -sum(tab[k].values()))
    cell = []
    for k in keys:
        c = tab[k]; n = sum(c.values())
        row = [k] + [str(c.get(t, 0)) for t in TIERS] + [str(n)]
        cell.append(row)
    tot = [sum(tab[k].get(t,0) for k in keys) for t in TIERS]
    cell.append(["Total"] + [str(x) for x in tot] + [str(sum(tot))])
    cols = [key] + TIERS + ["total"]
    tcolor = {"easy": "#b6e3b6", "medium": "#fdf3c8", "hard": "#f4b6b6", "unsolved": "#dddddd"}
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * len(cell))); ax.axis("off")
    cc = []
    for i, row in enumerate(cell):
        rc = ["white"]
        for j, t in enumerate(TIERS):
            rc.append(tcolor[t] if i < len(cell)-1 else "#cccccc")
        rc.append("white")
        cc.append(rc)
    tb = ax.table(cellText=cell, colLabels=cols, cellColours=cc,
                  colColours=["#d0d0d0"]*len(cols), loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.3)
    for j in range(len(cols)): tb[0, j].set_text_props(weight="bold")
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout(); plt.savefig(out, dpi=140, bbox_inches="tight")
    print("saved:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", default=os.path.expandvars("${VERO_REPO}/curriculum"))
    args = ap.parse_args()
    rows = load(args.labels)
    print(f"라벨 {len(rows)}개 로드\n")
    md_all = [f"# 난이도 교차표 (n={len(rows)})\n"]
    for key, title in [("ability", "Ability x Difficulty Tier"),
                       ("data_source", "Data source x Difficulty Tier")]:
        tab = crosstab(rows, key)
        md = to_md(tab, key)
        md_all.append(f"## {title}\n\n{md}\n")
        print(f"## {title}\n{md}\n")
        to_png(tab, key, f"{args.outdir}/difficulty_by_{key}.png", title)
    open(f"{args.outdir}/difficulty_crosstab.md", "w").write("\n".join(md_all))
    print(f"\nmarkdown 저장: {args.outdir}/difficulty_crosstab.md")


if __name__ == "__main__":
    main()
