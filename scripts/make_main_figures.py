"""Main-text figures drawn at the printed width (5.5 in, the NeurIPS text width) so fonts are readable.
Run from the repository root:  python scripts/make_main_figures.py
Writes report/figures/t2_standard_run.pdf, t3_standard_run.pdf and t4_category_labels.pdf (plus .png)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

R = Path("results")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
W = 5.5
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 0.9, "lines.markersize": 2.2})


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def trajectory(rows, panels, name, ncols, height):
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(W, height), sharex=True)
    axes = axes.ravel()
    x = [r["update"] for r in rows]
    for i, (ax, (key, title, log)) in enumerate(zip(axes, panels)):
        y = [r[key] for r in rows]
        if log:
            pts = [(a, b) for a, b in zip(x, y) if b > 0]
            ax.plot([a for a, _ in pts], [b for _, b in pts], "o-", color="#2F6DB5")
            ax.set_yscale("log")
        elif max(y) == min(y) == 0:
            ax.plot(x, y, "o-", color="#2F6DB5")
            ax.set_ylim(-0.01, 0.2)
            ax.set_yticks([0, 0.1, 0.2])
        else:
            ax.plot(x, y, "o-", color="#2F6DB5")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            ax.yaxis.get_offset_text().set_fontsize(6)
        ax.set_title(title, pad=2)
        ax.set_xticks([1, 10, 20])
        if i >= len(panels) - ncols:
            ax.set_xlabel("update", labelpad=1)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    save(fig, name)


ppo = jl(R / "task2_ppo" / "standard_train_log.jsonl")
trajectory(ppo, [("raw_reward", "learned reward", False), ("kl_token_mean", "token KL", False),
                 ("entropy", "entropy", False), ("response_tokens", "length (tokens)", False),
                 ("clip_fraction", "clip fraction", False), ("policy_loss", "policy loss", False),
                 ("value_loss", "value loss", False), ("grad_norm_policy", "policy grad norm", False),
                 ("critic_explained_variance", "critic expl. variance", False),
                 ("mean_abs_ratio_dev_last_epoch", r"mean $|\rho-1|$", False)], "t2_standard_run", 5, 2.55)

grpo = jl(R / "task3_grpo" / "standard_train_log.jsonl")
trajectory(grpo, [("reward_mean", "mean group reward", False), ("reward_std_in_group", "within-group std", False),
                  ("kl_token_mean", "token KL", False), ("policy_loss", "policy loss", False),
                  ("grad_norm", "grad norm (log)", True), ("entropy", "entropy", False),
                  ("length_mean", "mean length (tokens)", False), ("masked_token_fraction", "masked fraction", False)],
           "t3_standard_run", 4, 2.45)

# ---------- Task 4: judge-label shares per XSTest category ----------
LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
NAMES = {"SAFE_ANSWER": "safe answer", "JUSTIFIED_REFUSAL": "justified refusal", "UNSAFE_COMPLIANCE": "unsafe compliance",
         "OVER_REFUSAL": "over-refusal", "AMBIGUOUS": "ambiguous"}
COLORS = {"SAFE_ANSWER": "#378ADD", "JUSTIFIED_REFUSAL": "#E8A33D", "UNSAFE_COMPLIANCE": "#D85A30",
          "OVER_REFUSAL": "#8E5BB5", "AMBIGUOUS": "#9A9A9A"}
POL = ["sft", "dpo", "ppo", "grpo"]
c = pd.read_csv(R / "task4_safety" / "category_labels.csv")
fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [10, 8]}, sharey=True)
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
            for lab in LABELS:
                h = row[lab] / row["n"]
                ax.bar(base + j, h, 0.85, bottom=bottom, color=COLORS[lab],
                       label=NAMES[lab] if (cls == "SAFE" and i == 0 and j == 0) else None)
                bottom += h
    ax.set_xticks(centers)
    ax.set_xticklabels([t.replace("contrast_", "").replace("_", " ") for t in types], rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{cls.lower()} prompts" + (" (contrast categories)" if cls == "UNSAFE" else ""), pad=2)
axes[0].set_ylabel("share of prompts")
fig.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.1), frameon=False, handlelength=1.0, columnspacing=1.0)
fig.tight_layout(pad=0.3, w_pad=0.6)
save(fig, "t4_category_labels")