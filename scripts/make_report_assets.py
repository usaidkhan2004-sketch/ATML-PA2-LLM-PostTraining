from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, FIG, TAB = Path("results"), Path("report/figures"), Path("report/tables")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False})
MD, STEPS = [], []
POLICIES4 = ["sft", "dpo", "ppo", "grpo"]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def js(p):
    return json.load(open(p))


def step(fn):
    STEPS.append(fn)
    return fn


def save(fig, name, headroom=True):
    if headroom:   # leave empty space above bar charts so an inside legend never covers data
        for ax in fig.axes:
            if ax.get_legend() is not None and ax.get_ylim()[0] >= 0:
                ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("figure", name)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if math.isnan(v) else f"{v:.4g}"
    return str(v)


def table(name, df, title):
    df.to_csv(TAB / f"{name}.csv", index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    MD.append(f"### {title}  (`{name}.csv`)\n\n" + "\n".join(lines) + "\n")
    print("table", name)


def traj(rows, panels, name, ncols=4, xkey="update"):
    n = len(panels)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (k, title, scale) in zip(axes, panels):
        ax.plot([r[xkey] for r in rows], [r[k] for r in rows], "o-", ms=3, lw=1)
        ax.set_title(title)
        ax.set_xlabel(xkey)
        if scale:
            ax.set_yscale(scale)
    for ax in axes[n:]:
        ax.axis("off")
    save(fig, name, headroom=False)


def forest(df, label_col, d, lo, hi, title, name):
    df = df[df[d].notna()].reset_index(drop=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(df) + 1.2))
    ax.errorbar(df[d], y, xerr=[df[d] - df[lo], df[hi] - df[d]], fmt="o", capsize=3)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df[label_col], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("held-out reward change vs midpoint (paired, 95% bootstrap CI)")
    ax.set_title(title)
    save(fig, name, headroom=False)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return (c - h) / d, (c + h) / d


# --------------------------------------------------------------------------- Task 1
@step
def task1_table():
    names = [("SFT base", "sft"), ("DPO standard (1 epoch)", "standard"), ("short fork beta=0.03", "beta_0.03"),
             ("short fork beta=0.10", "beta_0.1"), ("short fork beta=0.30", "beta_0.3"), ("length-balanced DPO", "length_balanced")]
    rows = []
    for label, n in names:
        e = js(R / "task1_dpo" / f"eval_{n}.json")
        tp = R / "task1_dpo" / f"{n}_train_summary.json"
        t = js(tp) if tp.exists() else {}
        s, g, w = e["pairs"]["standard"], e["generation"], e["word_limit"]
        base = n == "sft"
        rows.append({"condition": label, "train_pairs": t.get("examples", 0), "optimizer_steps": t.get("optimizer_steps", 0),
                     "beta": t.get("beta", np.nan),
                     "heldout_pref_acc": np.nan if base else s["preference_accuracy"],
                     "heldout_dpo_loss(own beta)": np.nan if base else s["dpo_loss"], "mean_margin": np.nan if base else s["mean_margin"],
                     "KL_token": g["kl_token_mean"], "RM_score_mean": g["rm_score"]["mean"], "RM_score_std": g["rm_score"]["std"],
                     "len_mean": g["response_tokens"]["mean"], "len_std": g["response_tokens"]["std"],
                     "wordlimit_compliance": w["compliance_rate"]})
    table("t1_dpo_summary", pd.DataFrame(rows), "Task 1: DPO summary (standard run and short forks use different budgets)")


@step
def task1_beta_figure():
    forks = [("0.03", "beta_0.03"), ("0.10", "beta_0.1"), ("0.30", "beta_0.3")]
    ev = [js(R / "task1_dpo" / f"eval_{n}.json") for _, n in forks]
    sft, std = js(R / "task1_dpo" / "eval_sft.json"), js(R / "task1_dpo" / "eval_standard.json")
    panels = [("held-out preference accuracy", lambda e: e["pairs"]["standard"]["preference_accuracy"], False),
              ("mean implicit margin", lambda e: e["pairs"]["standard"]["mean_margin"], False),
              ("KL from reference (token mean)", lambda e: e["generation"]["kl_token_mean"], True),
              ("reward-model score", lambda e: e["generation"]["rm_score"]["mean"], True),
              ("mean response length (tokens)", lambda e: e["generation"]["response_tokens"]["mean"], True)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.8))
    for ax, (title, f, with_sft) in zip(axes, panels):
        ax.plot([x for x, _ in forks], [f(e) for e in ev], "o-", label="short forks (600 pairs)")
        ax.axhline(f(std), color="gray", ls=":", label="standard run (1446 pairs)")
        if with_sft:
            ax.axhline(f(sft), color="k", ls="--", lw=0.8, label="SFT base")
        ax.set_title(title)
        ax.set_xlabel("beta")
    axes[0].legend(fontsize=7)
    save(fig, "t1_beta_study", headroom=False)


@step
def task1_length_figure_and_table():
    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    ev = {"standard DPO": js(R / "task1_dpo" / "eval_standard.json"), "length-balanced DPO": js(R / "task1_dpo" / "eval_length_balanced.json")}
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, (label, e) in enumerate(ev.items()):
        s = e["pairs"]["stratified"]["by_stratum"]
        acc = [s[k]["preference_accuracy"] for k in strata]
        ci = [1.96 * math.sqrt(a * (1 - a) / s[k]["n"]) for a, k in zip(acc, strata)]
        ax.bar(np.arange(3) + i * 0.38, acc, 0.38, yerr=ci, capsize=3, label=label)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(np.arange(3) + 0.19)
    ax.set_xticklabels(strata, fontsize=8)
    ax.set_ylabel("held-out preference accuracy (95% CI)")
    ax.legend(fontsize=8)
    save(fig, "t1_length_strata")
    lc = js(R / "task1_dpo" / "length_comparison.json")
    rows = []
    for key, label in [("standard_dpo", "standard DPO"), ("length_balanced_dpo", "length-balanced DPO")]:
        m = lc["models"][key]
        rows.append({"model": label, **{f"acc_{k}": m["by_stratum"][k]["preference_accuracy"] for k in strata},
                     "n_per_stratum": "/".join(str(m["by_stratum"][k]["n"]) for k in strata), "acc_overall_stratified": m["overall_accuracy"],
                     "corr(margin, chosen-rejected length)": m["corr_margin_vs_lengthdiff"], "gen_len_mean": m["generation_length_tokens"]["mean"],
                     "gen_len_std": m["generation_length_tokens"]["std"], "wordlimit_compliance": m["wordlimit_compliance"],
                     "RM_score": m["rm_score_mean"], "KL_token": m["kl_token_mean"]})
    table("t1_length_confounding", pd.DataFrame(rows), "Task 1: length-confounding diagnostics")
    ds = lc["dataset_length_stats"]
    table("t1_dataset_length_stats", pd.DataFrame([{"train_set": k, **v} for k, v in ds.items()]), "Task 1: length structure of the training sets (a property of the data)")


@step
def task1_training_curves():
    f, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for name, label in [("standard", "standard"), ("length_balanced", "length-balanced")]:
        rows = jl(R / "task1_dpo" / f"{name}_train_log.jsonl")
        for a, k in zip(ax, ["loss", "margin", "grad_norm"]):
            a.plot([r["step"] for r in rows], [r[k] for r in rows], lw=1, label=label)
    for a, t in zip(ax, ["DPO loss", "implicit reward margin", "gradient norm (pre-clip)"]):
        a.set_title(t)
        a.set_xlabel("optimizer step")
    ax[0].legend(fontsize=8)
    save(f, "t1_training_curves", headroom=False)


# --------------------------------------------------------------------------- Task 2
def _run_summary(run_dir, name, rows, cols):
    s = js(R / run_dir / f"{name}_train_summary.json")
    out = {"updates": len(rows), "wall_clock_s": s["wall_clock_s"], "peak_mps_driver_GB(sampled)": s["peak_mps_driver_gb_sampled"],
           "total_generated_tokens": s["total_generated_tokens"]}
    for k, how in cols:
        v = np.array([r[k] for r in rows], dtype=float)
        out[f"{k}_{how}"] = float(np.nanmean(v) if how == "mean" else np.nanmedian(v) if how == "median" else np.nanmax(v))
    return out


@step
def task2_standard_run():
    rows = jl(R / "task2_ppo" / "standard_train_log.jsonl")
    traj(rows, [("raw_reward", "learned reward", None), ("kl_token_mean", "sampled KL (token mean)", None), ("entropy", "token entropy", None),
                ("response_tokens", "response length", None), ("policy_loss", "policy loss", None), ("value_loss", "value loss", None),
                ("clip_fraction", "clip fraction", None), ("grad_norm_policy", "policy grad norm", None),
                ("grad_norm_value", "critic grad norm", "log"), ("critic_explained_variance", "critic explained variance", None),
                ("mean_abs_ratio_dev_last_epoch", "mean |rho-1| (2nd epoch)", None)], "t2_standard_run")
    table("t2_standard_run_summary", pd.DataFrame([_run_summary("task2_ppo", "standard", rows, [
        ("raw_reward", "mean"), ("kl_token_mean", "mean"), ("entropy", "mean"), ("response_tokens", "mean"), ("policy_loss", "mean"),
        ("value_loss", "mean"), ("clip_fraction", "mean"), ("grad_norm_policy", "median"), ("grad_norm_value", "median"),
        ("critic_explained_variance", "median")])]), "Task 2: standard PPO continuation, summary over 20 updates (+ time and memory)")


@step
def task2_clipping():
    d = js(R / "task2_ppo" / "clipping_cached_batch.json")
    upd, noise = pd.DataFrame(d["updates"]), pd.DataFrame(d["noise_floor"])
    scales = sorted(upd["lr_scale"].unique())
    fig, ax = plt.subplots(1, len(scales), figsize=(4.4 * len(scales), 3.1), squeeze=False)
    for a, s in zip(ax[0], scales):
        for eps in sorted(upd["eps"].unique()):
            g = upd[(upd["lr_scale"] == s) & (upd["eps"] == eps)].sort_values("step")
            a.plot(g["step"], 100 * g["outside_fraction"], "o-", label=f"eps={eps:g}")
        a.set_title(f"policy learning rate x{s:g}")
        a.set_xlabel("optimizer step on the cached batch")
        a.set_ylabel("% tokens outside [1-eps, 1+eps]")
    ax[0][0].legend(fontsize=8)
    save(fig, "t2_clipping_cached_batch", headroom=False)
    rows = []
    for (s, eps), g in upd.groupby(["lr_scale", "eps"]):
        g = g.sort_values("step")
        a0, a1 = g.iloc[0], g.iloc[-1]
        rows.append({"setting": f"lr x{s:g}", "eps": eps, "outside_step0": a0["outside_fraction"], "outside_last": a1["outside_fraction"],
                     "binding_last": a1["binding_fraction"], "mean_abs_rho_dev_last": a1["mean_abs_ratio_dev"],
                     "surrogate_clipped_last": a1["surrogate_clipped"], "surrogate_unclipped_last": a1["surrogate_unclipped"], "last_step": int(a1["step"])})
    for _, r in noise.iterrows():
        rows.append({"setting": "noise floor: recomputed vs CACHED old log-probs (no update)", "eps": r["eps"], "outside_step0": r["outside_fraction"],
                     "binding_last": r["binding_fraction"], "mean_abs_rho_dev_last": r["mean_abs_ratio_dev"]})
    table("t2_clipping_cached_batch", pd.DataFrame(rows), "Task 2: cached-batch clipping study (affected-token fractions)")


@step
def task2_forks():
    keep = ["condition", "reward_raw", "delta_raw_vs_midpoint", "delta_raw_ci_lo", "delta_raw_ci_hi", "kl_token_mean", "entropy_token_mean",
            "len_mean", "len_std", "frac_truncated", "stab_mean_clip_fraction", "stab_max_ratio_dev", "stab_median_policy_gradnorm", "stab_max_policy_gradnorm"]
    c, k = pd.read_csv(R / "task2_ppo" / "clipping_summary.csv"), pd.read_csv(R / "task2_ppo" / "kl_summary.csv")
    table("t2_clipping_forks", c[[x for x in keep if x in c.columns]], "Task 2: epsilon forks (8 updates each; stability = mean clip fraction, max |rho-1|, grad norm)")
    table("t2_kl_forks", k[[x for x in keep if x in k.columns]], "Task 2: KL-pressure forks (beta_KL = 0 / 0.10 / 0.20)")
    forest(c[c["condition"] != "midpoint (start)"], "condition", "delta_raw_vs_midpoint", "delta_raw_ci_lo", "delta_raw_ci_hi", "PPO clipping forks", "t2_clipping_forks")
    forest(k[k["condition"] != "midpoint (start)"], "condition", "delta_raw_vs_midpoint", "delta_raw_ci_lo", "delta_raw_ci_hi", "PPO KL-pressure forks", "t2_kl_forks")


# --------------------------------------------------------------------------- Task 3
@step
def task3_standard_run():
    rows = jl(R / "task3_grpo" / "standard_train_log.jsonl")
    traj(rows, [("reward_mean", "mean group reward", None), ("reward_std_in_group", "within-group reward std", None), ("kl_token_mean", "sampled KL (token mean)", None),
                ("policy_loss", "policy loss", None), ("grad_norm", "grad norm", "log"), ("entropy", "token entropy", None),
                ("length_mean", "mean completion length", None), ("frac_truncated", "fraction truncated", None),
                ("masked_token_fraction", "masked token fraction", None)], "t3_standard_run", ncols=3)
    s = js(R / "task3_grpo" / "standard_train_summary.json")
    row = _run_summary("task3_grpo", "standard", rows, [("reward_mean", "mean"), ("reward_std_in_group", "mean"), ("kl_token_mean", "mean"),
                                                         ("policy_loss", "mean"), ("grad_norm", "median"), ("grad_norm", "max"), ("entropy", "mean"), ("length_mean", "mean")])
    row["uninformative_group_fraction"] = s["uninformative_group_fraction"]
    table("t3_standard_run_summary", pd.DataFrame([row]), "Task 3: standard GRPO continuation, summary over 20 updates (+ time and memory)")


@step
def task3_groupsize():
    d = pd.DataFrame(js(R / "task3_grpo" / "group_size_study.json")["rows"])
    d["informative_rate_ci"] = d["informative_rate_ci"].apply(lambda x: f"{x[0]:.2f}-{x[1]:.2f}")
    d["mean_group_std_ci"] = d["mean_group_std_ci"].apply(lambda x: f"{x[0]:.2f}-{x[1]:.2f}")
    table("t3_group_size", d[["K", "bin", "n_groups", "informative_rate", "informative_rate_ci", "mean_group_std", "mean_group_std_ci",
                              "var_relative_signal", "weak_signal_rate"]], "Task 3: equal-generation group-size study (bins = terciles of prompt mean reward)")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    ks = sorted(d["K"].unique())
    for a, (col, t) in zip(ax, [("informative_rate", "informative-group rate"), ("mean_group_std", "mean within-group reward std"),
                                ("var_relative_signal", "variance of relative signal")]):
        for i, b in enumerate(["all", "hard", "medium", "easy"]):
            g = d[d["bin"] == b].sort_values("K")
            a.bar(np.arange(len(ks)) + i * 0.2, g[col], 0.2, label=b)
        a.set_xticks(np.arange(len(ks)) + 0.3)
        a.set_xticklabels([f"K={k}" for k in ks])
        a.set_title(t)
    ax[0].legend(fontsize=8, ncol=2)
    save(fig, "t3_group_size")


@step
def task3_normalization():
    n = pd.read_csv(R / "task3_grpo" / "normalization_summary.csv")
    keep = ["condition", "heldout_reward_raw", "delta_vs_midpoint", "delta_ci_lo", "delta_ci_hi", "kl_token_mean", "entropy_token_mean", "len_mean", "len_std",
            "frac_truncated", "train_mean_reward", "train_mean_grad_norm", "diag_gradnorm_per_adv_short", "diag_gradnorm_per_adv_long",
            "diag_ratio_long_over_short_median", "diag_mean_len_short", "diag_mean_len_long", "token_weight_short", "token_weight_long"]
    table("t3_normalization", n[[x for x in keep if x in n.columns]], "Task 3: canonical GRPO (1/T) vs Dr. GRPO (1/Lmax), held-out and length-conditioned statistics")
    f = n[n["condition"].str.startswith("fork")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(f))
    ax[0].bar(x - 0.2, f["diag_gradnorm_per_adv_short"], 0.4, label="shortest completion")
    ax[0].bar(x + 0.2, f["diag_gradnorm_per_adv_long"], 0.4, label="longest completion")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(f["condition"], fontsize=7)
    ax[0].set_ylabel("grad norm per unit |advantage|")
    ax[0].legend(fontsize=8)
    g = n[n["condition"] != "midpoint (start)"].reset_index(drop=True)
    y = np.arange(len(g))
    ax[1].errorbar(g["delta_vs_midpoint"], y, xerr=[g["delta_vs_midpoint"] - g["delta_ci_lo"], g["delta_ci_hi"] - g["delta_vs_midpoint"]], fmt="o", capsize=3)
    ax[1].axvline(0, color="gray", ls="--", lw=0.8)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(g["condition"], fontsize=7)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("held-out reward change vs midpoint (paired 95% CI)")
    save(fig, "t3_normalization")


# --------------------------------------------------------------------------- Task 4
@step
def task4_policy_table_and_rates():
    t = pd.read_csv(R / "task4_safety" / "safety_summary.csv")
    cols = ["policy", "n_safe", "n_unsafe", "safe_answer_rate", "safe_over_refusal_rate", "safe_refusal_any_rate", "unsafe_compliance_rate",
            "unsafe_justified_refusal_rate", "unsafe_comply_any_rate", "ambiguous_rate", "judge_inconsistent_safe", "judge_inconsistent_unsafe", "len_mean", "len_std"]
    table("t4_safety_summary", t[[c for c in cols if c in t.columns]], "Task 4: safety calibration per policy (judge labels, plus class-grounded refusal-any / comply-any views)")
    metrics = [("safe_answer_rate", "safe prompts: answered"), ("safe_refusal_any_rate", "safe prompts: refused (class-grounded)"),
               ("unsafe_justified_refusal_rate", "unsafe prompts: justified refusal"), ("unsafe_comply_any_rate", "unsafe prompts: complied (class-grounded)")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, (m, label) in enumerate(metrics):
        v = t[m].to_numpy()
        ax.bar(np.arange(len(t)) + i * 0.2, v, 0.2, yerr=[v - t[m + "_ci_lo"].to_numpy(), t[m + "_ci_hi"].to_numpy() - v], capsize=2, label=label)
    ax.set_xticks(np.arange(len(t)) + 0.3)
    ax.set_xticklabels(t["policy"])
    ax.set_ylabel("rate (Wilson 95% CI)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save(fig, "t4_rates", headroom=False)


@step
def task4_label_distribution():
    c = js(R / "task4_safety" / "safety_summary.json")["judge_label_counts_by_class"]
    labels = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a, cls in zip(ax, ["SAFE", "UNSAFE"]):
        bottom = np.zeros(len(c))
        for l in labels:
            v = np.array([c[p][cls][l] / sum(c[p][cls].values()) for p in c])
            a.bar(list(c), v, bottom=bottom, label=l)
            bottom += v
        a.set_title(f"{cls} prompts: judge-label distribution")
    ax[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, "t4_label_distribution", headroom=False)


@step
def task4_categories():
    c = pd.read_csv(R / "task4_safety" / "category_labels.csv")
    c["refusal_any"] = (c["JUSTIFIED_REFUSAL"] + c["OVER_REFUSAL"]) / c["n"]
    c["comply_any"] = (c["SAFE_ANSWER"] + c["UNSAFE_COMPLIANCE"]) / c["n"]
    table("t4_category_labels", c, "Task 4: judge-label counts per XSTest category and policy (full table)")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5))
    for a, cls, col, t in [(ax[0], "SAFE", "refusal_any", "safe prompts: share refused (over-refusal, class-grounded)"),
                           (ax[1], "UNSAFE", "comply_any", "unsafe prompts: share complied with")]:
        sub = c[c["benchmark_class"] == cls]
        types = sorted(sub["type"].unique())
        for i, p in enumerate(POLICIES4):
            g = sub[sub["policy"] == p].set_index("type").reindex(types)
            a.bar(np.arange(len(types)) + i * 0.2, g[col], 0.2, label=p)
        a.set_xticks(np.arange(len(types)) + 0.3)
        a.set_xticklabels(types, rotation=30, ha="right", fontsize=7)
        a.set_title(t)
    ax[0].legend(fontsize=8, ncol=4)
    save(fig, "t4_categories", headroom=False)


@step
def task4_audit():
    d = js(R / "task4_safety" / "safety_summary.json")
    if "manual_audit" not in d:
        print("[pending] no manual audit labels yet: rerun evaluate_safety after saving manual_audit_labels.csv, then rerun this script")
        return
    m = d["manual_audit"]
    labels = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
    conf = pd.DataFrame(m["confusion_manual_rows_by_judge_cols"]).reindex(index=labels, columns=labels).fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("judge label")
    ax.set_ylabel("manual label")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=8, color="white" if conf[i, j] > conf.max() / 2 else "black")
    fig.colorbar(im, fraction=0.04)
    save(fig, "t4_audit_confusion", headroom=False)
    rows = [{"scope": "overall", **m["overall"]}] + [{"scope": p, **v} for p, v in m["by_policy"].items()]
    table("t4_audit_agreement", pd.DataFrame(rows), "Task 4: judge vs manual agreement on the audited subset")


# --------------------------------------------------------------------------- Task 5
def _lenient(recs):
    import re
    num = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
    ok = 0
    for r in recs:
        m = num.findall(r["response"])
        try:
            ok += bool(m) and abs(float(m[-1].replace(",", "")) - float(r["gold_final"])) < 1e-9
        except ValueError:
            pass
    return ok / len(recs)


@step
def task5_tables_and_figures():
    names = ["sft", "rlvr", "rlaif"]
    summ = {d: js(R / "task5_feedback" / f"{d}_summary.json") for d in ("gsm", "transfer")}
    rows = []
    for d, label in [("gsm", "GSM8K (in-domain)"), ("transfer", "SVAMP (out-of-domain)")]:
        for n in names:
            p = summ[d]["policies"][n]
            pw = summ[d]["pairwise_vs_sft"].get(n, {})
            rec = jl(R / "task5_feedback" / f"{d}_generated_{n}.jsonl")
            rows.append({"dataset": label, "policy": n, "n": p["n"], "exact_accuracy": p["accuracy"], "acc_ci_lo": p["accuracy_ci"][0], "acc_ci_hi": p["accuracy_ci"][1],
                         "format_compliance": p["format_compliance"], "last_number_accuracy(diagnostic)": _lenient(rec), "len_mean": p["len_tokens_mean"],
                         "len_std": p["len_tokens_std"], "frac_truncated": p["frac_truncated"], "win_rate_vs_sft": pw.get("win_rate_vs_sft", np.nan),
                         "wins/ties/losses": f"{pw.get('wins')}/{pw.get('ties')}/{pw.get('losses')}" if pw else "",
                         "verifier_judge_agreement": pw.get("verifier_judge_agreement_3way", np.nan), "n_verifier_discordant": pw.get("n_verifier_discordant", np.nan),
                         "failure_types": str(p["failure_types"])})
    t = pd.DataFrame(rows)
    table("t5_math_eval", t, "Task 5: in-domain and out-of-domain evaluation (exact verifier, pairwise judge vs SFT)")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, (d, label) in enumerate([("GSM8K (in-domain)", "GSM8K"), ("SVAMP (out-of-domain)", "SVAMP")]):
        g = t[t["dataset"] == d]
        ax[0].bar(np.arange(3) + i * 0.38, g["exact_accuracy"], 0.38, yerr=[g["exact_accuracy"] - g["acc_ci_lo"], g["acc_ci_hi"] - g["exact_accuracy"]], capsize=3, label=label)
    ax[0].set_xticks(np.arange(3) + 0.19)
    ax[0].set_xticklabels(names)
    ax[0].set_ylabel("exact-answer accuracy (95% CI)")
    ax[0].legend(fontsize=8)
    g = t[t["dataset"].str.startswith("GSM")]
    for i, (col, lab) in enumerate([("exact_accuracy", "strict (verifier)"), ("last_number_accuracy(diagnostic)", "last-number (diagnostic)"), ("format_compliance", "format compliance")]):
        ax[1].bar(np.arange(3) + i * 0.27, g[col], 0.27, label=lab)
    ax[1].set_xticks(np.arange(3) + 0.27)
    ax[1].set_xticklabels(names)
    ax[1].set_title("GSM8K: how much is the answer format?")
    ax[1].legend(fontsize=7)
    for i, (d, label) in enumerate([("GSM8K (in-domain)", "GSM8K"), ("SVAMP (out-of-domain)", "SVAMP")]):
        h = t[(t["dataset"] == d) & (t["policy"] != "sft")]
        pw = [summ["gsm" if label == "GSM8K" else "transfer"]["pairwise_vs_sft"][p] for p in h["policy"]]
        w = np.array([x["win_rate_vs_sft"] for x in pw])
        ci = np.array([x["win_rate_ci"] for x in pw])
        ax[2].bar(np.arange(2) + i * 0.38, w, 0.38, yerr=[w - ci[:, 0], ci[:, 1] - w], capsize=3, label=label)
    ax[2].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[2].set_xticks(np.arange(2) + 0.19)
    ax[2].set_xticklabels(["rlvr", "rlaif"])
    ax[2].set_ylabel("AI pairwise win rate vs SFT (95% CI)")
    ax[2].legend(fontsize=8)
    save(fig, "t5_math_eval")


@step
def task5_diagnostics():
    s = js(R / "task5_feedback" / "diagnostic_summary.json")
    types = ["reasoning_corruption", "outcome_wrong_final", "outcome_gold_distractor", "persuasive_filler", "cross_corrupt_vs_wrongfinal"]
    rows = []
    for mech in ("verifier", "judge"):
        for tp in types:
            r = s[mech][tp]
            rows.append({"mechanism": mech, "pair_type": tp, "n": r["n"], "better_rate": r["better_rate"], "tie_rate": r["tie_rate"], "wrong_rate": r["wrong_rate"]})
        o = s[mech]["S_outcome"]
        rows.append({"mechanism": mech, "pair_type": "outcome_pooled (S_outcome)", "n": o["n"], "better_rate": o["better_rate"], "tie_rate": o["tie_rate"], "wrong_rate": o["wrong_rate"]})
        rows.append({"mechanism": mech, "pair_type": "S_reason", "n": s[mech]["reasoning_corruption"]["n"], "better_rate": s[mech]["S_reason"]})
    table("t5_diagnostics", pd.DataFrame(rows), "Task 5: controlled diagnostics (better / tie / wrong preference; S_reason, S_outcome)")
    table("t5_diagnostic_mean_words", pd.DataFrame([s["variant_mean_words"]]), "Task 5: mean words per diagnostic variant (length confound)")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    for a, mech in zip(ax, ["verifier", "judge"]):
        bottom = np.zeros(len(types))
        for k, lab in [("better_rate", "prefers better"), ("tie_rate", "tie"), ("wrong_rate", "prefers worse")]:
            v = np.array([s[mech][t][k] for t in types])
            a.bar(range(len(types)), v, bottom=bottom, label=lab)
            bottom += v
        a.set_xticks(range(len(types)))
        a.set_xticklabels(types, rotation=30, ha="right", fontsize=7)
        a.set_title(mech)
    ax[0].legend(fontsize=8, ncol=3, loc="upper center")
    save(fig, "t5_diagnostics")


# --------------------------------------------------------------------------- Task 6 (cross-task)
@step
def cross_task():
    rows = []

    def add(task, cond, e, kind):
        if kind == "dpo":
            g = e["generation"]
            rows.append({"task": task, "condition": cond, "reward": g["rm_score"]["mean"], "kl": g["kl_token_mean"], "entropy": np.nan, "len": g["response_tokens"]["mean"]})
        else:
            rows.append({"task": task, "condition": cond, "reward": e["reward_raw"]["mean"], "kl": e["kl_token_mean"], "entropy": e["entropy_token_mean"], "len": e["response_tokens"]["mean"]})

    for lab, n in [("sft", "sft"), ("standard", "standard"), ("beta 0.03", "beta_0.03"), ("beta 0.10", "beta_0.1"), ("beta 0.30", "beta_0.3"), ("length-balanced", "length_balanced")]:
        add("DPO", lab, js(R / "task1_dpo" / f"eval_{n}.json"), "dpo")
    for lab, n in [("sft", "sft"), ("midpoint", "midpoint"), ("standard", "standard"), ("eps 0.05", "fork_eps0.05_kl0.1"), ("eps 0.2/KL 0.1", "fork_eps0.2_kl0.1"),
                   ("eps 0.5", "fork_eps0.5_kl0.1"), ("KL 0", "fork_eps0.2_kl0.0"), ("KL 0.2", "fork_eps0.2_kl0.2")]:
        add("PPO", lab, js(R / "task2_ppo" / f"eval_{n}.json"), "rl")
    for lab, n in [("sft", "sft"), ("midpoint", "midpoint"), ("standard", "standard"), ("canonical", "fork_grpo"), ("Dr. GRPO", "fork_dr_grpo")]:
        add("GRPO", lab, js(R / "task3_grpo" / f"eval_{n}.json"), "rl")
    d = pd.DataFrame(rows)
    d["reward_minus_task_sft"] = d["reward"] - d.groupby("task")["reward"].transform(lambda s: s.iloc[0])
    table("t6_cross_task", d, "Task 6: cross-task summary (reward = reward-model score on each task's own held-out prompts; compare only within a task)")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for t, g in d.groupby("task"):
        ax.scatter(g["kl"], g["reward_minus_task_sft"], label=t, s=28)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.margins(x=0.12)
    ax.set_xlabel("KL from reference (token mean, symlog)")
    ax.set_ylabel("reward-model score minus the same task's SFT")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "t6_kl_vs_reward", headroom=False)


if __name__ == "__main__":
    for fn in STEPS:
        try:
            fn()
        except Exception as e:
            print(f"[skip] {fn.__name__}: {type(e).__name__}: {e}")
    (TAB / "all_tables.md").write_text("\n".join(MD), encoding="utf-8")
    print("wrote", TAB / "all_tables.md")