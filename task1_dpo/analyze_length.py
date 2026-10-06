from __future__ import annotations

import argparse
import gc

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from common.models import clear_gpu, load_tokenizer
from task1_dpo.evaluate import evaluate
from task1_dpo.train import run_training

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def _clean(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def dataset_length_stats(tokenizer, rows):
    """Dataset property: how response length relates to preference in a training set."""
    diffs, gaps = [], []
    for r in rows:
        yc, yr = preference_responses(r)
        diffs.append(len(tokenizer(yc, add_special_tokens=False)["input_ids"])
                     - len(tokenizer(yr, add_special_tokens=False)["input_ids"]))
        sc, sr = r.get("score_chosen"), r.get("score_rejected")
        gaps.append(float(sc) - float(sr) if sc is not None and sr is not None else float("nan"))
    d, g = np.asarray(diffs, float), np.asarray(gaps, float)
    ok = ~np.isnan(g)
    return {
        "n": len(rows),
        "frac_chosen_longer": float((d > 0).mean()),
        "frac_rejected_longer": float((d < 0).mean()),
        "mean_token_diff_chosen_minus_rejected": float(d.mean()),
        "median_token_diff": float(np.median(d)),
        "corr_lengthdiff_vs_scoregap": _clean(safe_corr(d[ok], g[ok])) if ok.any() else None,
    }


def model_length_view(cfg, name, length_diff_by_id):
    """Policy property: how a trained model's margins relate to length, per stratum."""
    recs = read_jsonl(f"{cfg['results_dir']}/eval_{name}_pairs_stratified.jsonl")
    out = {"n": len(recs), "overall_accuracy": float(np.mean([r["correct"] for r in recs])), "by_stratum": {}}
    for s in STRATA:
        sub = [r for r in recs if r.get("length_stratum") == s]
        if sub:
            acc = float(np.mean([r["correct"] for r in sub]))
            out["by_stratum"][s] = {"n": len(sub), "preference_accuracy": acc,
                                    "std_error": float(np.sqrt(acc * (1 - acc) / len(sub))),
                                    "mean_margin": float(np.mean([r["margin"] for r in sub]))}
    pairs = [(r["margin"], length_diff_by_id[r["prompt_id"]]) for r in recs if r["prompt_id"] in length_diff_by_id]
    out["corr_margin_vs_lengthdiff"] = _clean(safe_corr([p[0] for p in pairs], [p[1] for p in pairs]))
    e = load_json(f"{cfg['results_dir']}/eval_{name}.json")
    out["generation_length_tokens"] = e["generation"]["response_tokens"]
    out["wordlimit_compliance"] = e["word_limit"]["compliance_rate"]
    out["rm_score_mean"] = e["generation"]["rm_score"]["mean"]
    out["kl_token_mean"] = e["generation"]["kl_token_mean"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test (16 pairs, 4 eval items)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rdir = repo_path(cfg["results_dir"])

    tag = "smoke_" if args.smoke else ""
    name = f"{tag}length_balanced"
    adapter = "outputs/task1_dpo/smoke_length_balanced" if args.smoke else cfg["length_output"]
    limit = 4 if args.smoke else None

    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced), "| stratified eval rows:", len(stratified))

    if (rdir / f"{name}_train_summary.json").exists():
        print(f"[skip] training for {name} already done")
    else:
        run_training(args.config, name, cfg["paths"]["dpo_length_train"], adapter, None, 16 if args.smoke else None)
        gc.collect()
        clear_gpu()
    if (rdir / f"eval_{name}.json").exists():
        print(f"[skip] evaluation for {name} already done")
    else:
        evaluate(args.config, adapter, name, None, ("standard", "stratified"), limit, 1 if args.smoke else 5)
        gc.collect()
        clear_gpu()

    tok = load_tokenizer(cfg["base_model"])
    report = {
        "dataset_length_stats": {
            "standard_train": dataset_length_stats(tok, read_jsonl(cfg["paths"]["dpo_standard_train"])),
            "length_balanced_train": dataset_length_stats(tok, balanced),
        },
        "models": {},
    }
    length_diff_by_id = {r["prompt_id"]: r["length_difference"] for r in stratified}
    for label, n in [("standard_dpo", "standard"), ("length_balanced_dpo", name)]:
        report["models"][label] = model_length_view(cfg, n, length_diff_by_id)
    save_json(rdir / f"length_comparison{'_smoke' if args.smoke else ''}.json", report)

    print("\nDataset length structure (a property of the data):")
    for k, v in report["dataset_length_stats"].items():
        print(f"  {k}: chosen longer {v['frac_chosen_longer']:.3f}, rejected longer {v['frac_rejected_longer']:.3f}, "
              f"mean diff {v['mean_token_diff_chosen_minus_rejected']:.1f} tok, corr(lengthdiff, scoregap) {v['corr_lengthdiff_vs_scoregap']}")
    print("\nPreference accuracy on the length-stratified held-out set (a property of the policy):")
    for label, m in report["models"].items():
        parts = [f"{s}: {m['by_stratum'][s]['preference_accuracy']:.3f} (n={m['by_stratum'][s]['n']})"
                 for s in STRATA if s in m["by_stratum"]]
        gl = m["generation_length_tokens"]
        print(f"  {label}: overall {m['overall_accuracy']:.3f} | " + " | ".join(parts))
        print(f"      corr(margin, chosen-rejected length) {m['corr_margin_vs_lengthdiff']} | "
              f"gen len {gl['mean']:.1f}+-{gl['std']:.1f} | word-limit {m['wordlimit_compliance']} | RM {m['rm_score_mean']:.3f}")


if __name__ == "__main__":
    main()