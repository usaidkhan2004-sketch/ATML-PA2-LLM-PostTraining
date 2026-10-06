from __future__ import annotations

import argparse
import csv
import gc

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate
from task1_dpo.train import run_training


def _exists(cfg, rel):
    return (repo_path(cfg["results_dir"]) / rel).exists()


def write_summary(cfg, rows_spec):
    rdir = repo_path(cfg["results_dir"])
    rows = []
    for label, name in rows_spec:
        p = rdir / f"eval_{name}.json"
        if not p.exists():
            continue
        e = load_json(p)
        tp = rdir / f"{name}_train_summary.json"
        t = load_json(tp) if tp.exists() else {}
        s, g, w = e["pairs"]["standard"], e["generation"], e["word_limit"]
        rows.append({
            "condition": label, "name": name,
            "train_pairs": t.get("examples", 0), "optimizer_steps": t.get("optimizer_steps", 0),
            "beta": t.get("beta", ""),
            "heldout_pref_acc": round(s["preference_accuracy"], 4),
            "heldout_dpo_loss_at_own_beta": round(s["dpo_loss"], 4),
            "mean_margin": round(s["mean_margin"], 4),
            "kl_token_mean": round(g["kl_token_mean"], 5),
            "rm_score_mean": round(g["rm_score"]["mean"], 4),
            "rm_score_std": round(g["rm_score"]["std"], 4),
            "len_mean_tokens": round(g["response_tokens"]["mean"], 1),
            "len_std_tokens": round(g["response_tokens"]["std"], 1),
            "wordlimit_compliance": w["compliance_rate"],
        })
    if not rows:
        return
    with (rdir / "dpo_summary_table.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    save_json(rdir / "dpo_summary_table.json", rows)
    print("\nDPO summary (held-out DPO loss uses each model's own beta, so compare accuracy/margin across beta):")
    for r in rows:
        print(f"  {r['condition']:<28} pairs={r['train_pairs']:<5} acc={r['heldout_pref_acc']:.3f} "
              f"margin={r['mean_margin']:.3f} KL={r['kl_token_mean']:.5f} RM={r['rm_score_mean']:.3f} "
              f"len={r['len_mean_tokens']:.0f}+-{r['len_std_tokens']:.0f} wl={r['wordlimit_compliance']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test (16 pairs, 4 eval items)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    betas = cfg["betas"][:1] if args.smoke else cfg["betas"]
    n_train = 16 if args.smoke else int(cfg["short_ablation_examples"])
    limit = 4 if args.smoke else None
    wl = 1 if args.smoke else 5
    tag = "smoke_" if args.smoke else ""
    print("beta values:", betas, "| short-run examples:", n_train)

    names = []
    for b in betas:
        name = f"{tag}beta_{b}"
        names.append(name)
        adapter = f"outputs/task1_dpo/{name}"
        if _exists(cfg, f"{name}_train_summary.json"):
            print(f"[skip] training for {name} already done")
        else:
            run_training(args.config, name, None, adapter, b, n_train)
            gc.collect()
            clear_gpu()
        if _exists(cfg, f"eval_{name}.json"):
            print(f"[skip] evaluation for {name} already done")
        else:
            evaluate(args.config, adapter, name, beta=b, sets=("standard",), limit=limit, wl_samples=wl)
            gc.collect()
            clear_gpu()

    if not args.smoke:
        write_summary(cfg, [("SFT base (no DPO)", "sft"), ("DPO standard (1 epoch)", "standard")]
                      + [(f"DPO short fork beta={b}", f"beta_{b}") for b in betas])


if __name__ == "__main__":
    main()