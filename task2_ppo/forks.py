from __future__ import annotations

import csv
import gc
import json

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu
from common.rl_eval import evaluate_rl_policy
from task2_ppo.continue_train import run_ppo


def fork_name(eps, beta, prefix=""):
    return f"{prefix}fork_eps{eps}_kl{beta}"


def run_fork(config_path, eps, beta, updates=None, eval_limit=None, prefix=""):
    """Short continuation from the identical midpoint (same prompts/seeds), then held-out evaluation."""
    cfg = load_yaml(config_path)
    rdir = repo_path(cfg["results_dir"])
    name = fork_name(eps, beta, prefix)
    adapter = f"outputs/task2_ppo/{name}"
    n_up = int(updates or cfg["fork_updates"])
    if (rdir / f"{name}_train_summary.json").exists():
        print(f"[skip] training for {name} already done")
    else:
        run_ppo(config_path, adapter, n_up, eps, beta, name)
        gc.collect()
        clear_gpu()
    if (rdir / f"eval_{name}.json").exists():
        print(f"[skip] evaluation for {name} already done")
    else:
        evaluate_rl_policy(config_path, adapter, name, limit=eval_limit)
        gc.collect()
        clear_gpu()
    return name


def _bootstrap_mean(diffs, n=2000, seed=0):
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(config_path, rows_spec, stem, ref_name="midpoint"):
    """Table over conditions: held-out metrics, paired reward change vs the midpoint, training stability stats."""
    cfg = load_yaml(config_path)
    rdir = repo_path(cfg["results_dir"])
    ref_path = rdir / f"eval_{ref_name}_generations.jsonl"
    ref = {r["prompt_id"]: r for r in read_jsonl(ref_path)} if ref_path.exists() else {}
    rows = []
    for label, name in rows_spec:
        p = rdir / f"eval_{name}.json"
        if not p.exists():
            continue
        e = json.load(open(p))
        gens = read_jsonl(rdir / f"eval_{name}_generations.jsonl")
        d_raw = [g["reward_raw"] - ref[g["prompt_id"]]["reward_raw"] for g in gens if g["prompt_id"] in ref]
        d_eff = [g["reward_effective"] - ref[g["prompt_id"]]["reward_effective"] for g in gens if g["prompt_id"] in ref]
        m1, lo1, hi1 = _bootstrap_mean(d_raw)
        m2, lo2, hi2 = _bootstrap_mean(d_eff)
        row = {
            "condition": label, "name": name, "n_prompts": e["n_prompts"],
            "reward_raw": round(e["reward_raw"]["mean"], 4), "reward_effective": round(e["reward_effective"]["mean"], 4),
            "delta_raw_vs_midpoint": round(m1, 4), "delta_raw_ci_lo": round(lo1, 4), "delta_raw_ci_hi": round(hi1, 4),
            "delta_eff_vs_midpoint": round(m2, 4), "delta_eff_ci_lo": round(lo2, 4), "delta_eff_ci_hi": round(hi2, 4),
            "kl_token_mean": round(e["kl_token_mean"], 6), "entropy_token_mean": round(e["entropy_token_mean"], 4),
            "len_mean": round(e["response_tokens"]["mean"], 1), "len_std": round(e["response_tokens"]["std"], 1),
            "frac_truncated": round(e["frac_truncated"], 4),
        }
        tl = rdir / f"{name}_train_log.jsonl"
        if tl.exists():
            logs = read_jsonl(tl)
            row.update({
                "train_updates": len(logs),
                "stab_mean_clip_fraction": round(float(np.mean([l["clip_fraction"] for l in logs])), 5),
                "stab_max_ratio_dev": round(float(np.max([l["mean_abs_ratio_dev_last_epoch"] for l in logs])), 5),
                "stab_median_policy_gradnorm": round(float(np.median([l["grad_norm_policy"] for l in logs])), 4),
                "stab_max_policy_gradnorm": round(float(np.max([l["grad_norm_policy"] for l in logs])), 4),
                "train_mean_kl_token": round(float(np.mean([l["kl_token_mean"] for l in logs])), 6),
                "train_median_critic_ev": round(float(np.nanmedian([l["critic_explained_variance"] for l in logs])), 3),
            })
        rows.append(row)
    if not rows:
        print("no evaluated conditions found")
        return rows
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 999)
    with (rdir / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    save_json(rdir / f"{stem}.json", rows)
    print(f"\n{stem} (reward change is paired over prompts vs the midpoint, with 95% bootstrap CI):")
    for r in rows:
        print(f"  {r['condition']:<26} reward {r['reward_raw']:.3f} (eff {r['reward_effective']:.3f}) "
              f"d_raw {r['delta_raw_vs_midpoint']:+.3f} [{r['delta_raw_ci_lo']:+.3f},{r['delta_raw_ci_hi']:+.3f}] "
              f"KL {r['kl_token_mean']:.5f} ent {r['entropy_token_mean']:.3f} len {r['len_mean']:.0f}+-{r['len_std']:.0f} "
              f"trunc {r['frac_truncated']:.3f}"
              + (f" | clip {r['stab_mean_clip_fraction']:.4f} maxdev {r['stab_max_ratio_dev']:.4f} gnorm_max {r['stab_max_policy_gradnorm']:.2f}" if "stab_mean_clip_fraction" in r else ""))
    return rows