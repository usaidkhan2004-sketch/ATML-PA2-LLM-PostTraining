"""Five-label judge distribution per XSTest category for the four policies (Task 4 required evidence).
Run from the repository root:  python scripts/make_category_label_figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
COLORS = {"SAFE_ANSWER": "#378ADD", "JUSTIFIED_REFUSAL": "#E8A33D", "UNSAFE_COMPLIANCE": "#D85A30",
          "OVER_REFUSAL": "#8E5BB5", "AMBIGUOUS": "#9A9A9A"}
POL = ["sft", "dpo", "ppo", "grpo"]
INIT = {"sft": "S", "dpo": "D", "ppo": "P", "grpo": "G"}

c = pd.read_csv(R / "task4_safety" / "category_labels.csv")
plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.0))
for ax, cls in zip(axes, ["SAFE", "UNSAFE"]):
    sub = c[c["benchmark_class"] == cls]
    types = sorted(sub["type"].unique())
    centers = []
    for i, t in enumerate(types):
        base = i * 5.0
        centers.append(base + 1.5)
        for j, p in enumerate(POL):
            row = sub[(sub["type"] == t) & (sub["policy"] == p)].iloc[0]
            bottom = 0.0
            for l in LABELS:
                h = row[l] / row["n"]
                ax.bar(base + j, h, 0.9, bottom=bottom, color=COLORS[l], label=l if (i == 0 and j == 0) else None)
                bottom += h
            ax.text(base + j, -0.04, INIT[p], ha="center", va="top", fontsize=4.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([t.replace("contrast_", "c_") for t in types], rotation=28, ha="right", fontsize=5.5)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of prompts")
    ax.set_title(f"{cls} prompts" + (" (c_ = contrast_)" if cls == "UNSAFE" else ""), fontsize=7)
axes[0].legend(ncol=5, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.1), frameon=False)
fig.tight_layout(h_pad=0.8)
fig.savefig(OUT / "t4_category_labels.png", dpi=250)
fig.savefig(OUT / "t4_category_labels.pdf")
print("wrote", OUT / "t4_category_labels.pdf")