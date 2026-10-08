from __future__ import annotations

import argparse
import csv
import gc
import json

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu
from common.rl_eval import evaluate_rl_policy
from task2_ppo.forks import _bootstrap_mean
from task3_grpo.continue_train import run_grpo


def run_condition(config_path, loss_type, updates=None, eval_limit=None, prefix=""):
    """Short continuation from the identical midpoint (same prompts/seeds/budget), then held-out evaluation."""
    cfg = load_yaml(config_path)
    rdir = repo_path(cfg["results_dir"])
    name = f"{prefix}fork_{loss_type}"
    adapter = f"outputs/task3_grpo/{name}"
    n = int(updates or cfg["fork_updates"])
    if (rdir / f"{name}_train_summary.json").exists():
        print(f"[skip] training for {name} already done")
    else:
        run_grpo(config_path, adapter, n, loss_type, name, True)
        gc.collect()
        clear_gpu()
    if (rdir / f"eval_{name}.json").exists():
        print(f"[skip] evaluation for {name} already done")
    else:
        evaluate_rl_policy(config_path, adapter, name, limit=eval_limit)
        gc.collect()
        clear_gpu()
    return name


def summarize_grpo(config_path, rows_spec, stem, ref_name="midpoint"):
    cfg = load_yaml(config_path)
    rdir = repo_path(cfg["results_dir"])
    max_comp = int(cfg["max_completion_length"])
    ref_path = rdir / f"eval_{ref_name}_generations.jsonl"
    ref = {r["prompt_id"]: r for r in read_jsonl(ref_path)} if ref_path.exists() else {}
    rows = []
    for label, name in rows_spec:
        p = rdir / f"eval_{name}.json"
        if not p.exists():
            continue
        e = json.load(open(p))
        gens = read_jsonl(rdir / f"eval_{name}_generations.jsonl")
        m, lo, hi = _bootstrap_mean([g["reward_raw"] - ref[g["prompt_id"]]["reward_raw"] for g in gens if g["prompt_id"] in ref])
        row = {
            "condition": label, "name": name, "n_prompts": e["n_prompts"],
            "heldout_reward_raw": round(e["reward_raw"]["mean"], 4),
            "delta_vs_midpoint": round(m, 4), "delta_ci_lo": round(lo, 4), "delta_ci_hi": round(hi, 4),
            "kl_token_mean": round(e["kl_token_mean"], 6), "entropy_token_mean": round(e["entropy_token_mean"], 4),
            "len_mean": round(e["response_tokens"]["mean"], 1), "len_std": round(e["response_tokens"]["std"], 1),
            "frac_truncated": round(e["frac_truncated"], 4),
        }
        tl = rdir / f"{name}_train_log.jsonl"
        if tl.exists():
            logs = read_jsonl(tl)
            row.update({
                "train_updates": len(logs),
                "train_mean_reward": round(float(np.mean([l["reward_mean"] for l in logs])), 4),
                "train_mean_group_std": round(float(np.mean([l["reward_std_in_group"] for l in logs])), 4),
                "train_uninformative_frac": round(float(np.mean([not l["informative_group"] for l in logs])), 4),
                "train_mean_grad_norm": round(float(np.mean([l["grad_norm"] for l in logs])), 4),
                "train_mean_clip_fraction": round(float(np.mean([l["clip_fraction"] for l in logs])), 5),
                "train_mean_completion_len": round(float(np.mean([l["length_mean"] for l in logs])), 1),
            })
            dg = [l for l in logs if "gnorm_short" in l and abs(l["adv_short"]) > 1e-3 and abs(l["adv_long"]) > 1e-3]
            if dg:
                gs = [l["gnorm_short"] / abs(l["adv_short"]) for l in dg]
                gl = [l["gnorm_long"] / abs(l["adv_long"]) for l in dg]
                lt = logs[0].get("loss_type", "grpo")
                row.update({
                    "diag_updates": len(dg),
                    "diag_gradnorm_per_adv_short": round(float(np.mean(gs)), 5),
                    "diag_gradnorm_per_adv_long": round(float(np.mean(gl)), 5),
                    "diag_ratio_long_over_short_median": round(float(np.median([b / a for a, b in zip(gs, gl)])), 4),
                    "diag_mean_len_short": round(float(np.mean([l["len_short"] for l in dg])), 1),
                    "diag_mean_len_long": round(float(np.mean([l["len_long"] for l in dg])), 1),
                    "token_weight_short": round(float(np.mean([1.0 / l["len_short"] if lt == "grpo" else 1.0 / max_comp for l in dg])), 6),
                    "token_weight_long": round(float(np.mean([1.0 / l["len_long"] if lt == "grpo" else 1.0 / max_comp for l in dg])), 6),
                })
        rows.append(row)
    if not rows:
        print("no evaluated conditions found")
        return rows
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (rdir / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    save_json(rdir / f"{stem}.json", rows)
    print(f"\n{stem} (reward change is paired over prompts vs the midpoint, with 95% bootstrap CI):")
    for r in rows:
        line = (f"  {r['condition']:<26} reward {r['heldout_reward_raw']:.3f} d {r['delta_vs_midpoint']:+.3f} "
                f"[{r['delta_ci_lo']:+.3f},{r['delta_ci_hi']:+.3f}] KL {r['kl_token_mean']:.5f} ent {r['entropy_token_mean']:.3f} "
                f"len {r['len_mean']:.0f}+-{r['len_std']:.0f} trunc {r['frac_truncated']:.3f}")
        if "diag_ratio_long_over_short_median" in r:
            line += (f" | grad-norm per unit advantage short/long {r['diag_gradnorm_per_adv_short']:.4f}/{r['diag_gradnorm_per_adv_long']:.4f} "
                     f"(median long/short {r['diag_ratio_long_over_short_median']:.2f}, lens {r['diag_mean_len_short']:.0f}/{r['diag_mean_len_long']:.0f})")
        print(line)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Comparing loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    prefix = "smoke_" if args.smoke else ""
    names = [run_condition(args.config, lt, updates=1 if args.smoke else None,
                           eval_limit=4 if args.smoke else None, prefix=prefix) for lt in ("grpo", "dr_grpo")]
    if not args.smoke:
        summarize_grpo(args.config, [("midpoint (start)", "midpoint"), ("SFT base", "sft"), ("standard 20 updates (GRPO)", "standard"),
                                     ("fork canonical GRPO (1/T)", names[0]), ("fork Dr. GRPO (1/Lmax)", names[1])],
                       "normalization_summary")


if __name__ == "__main__":
    main()