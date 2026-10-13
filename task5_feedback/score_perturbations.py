from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

# (pair type, diagnostically better response, worse response)
PAIR_SPECS = [
    ("reasoning_corruption", "clean_correct", "corrupt_reasoning_correct_final"),
    ("outcome_wrong_final", "clean_correct", "good_reasoning_wrong_final"),
    ("outcome_gold_distractor", "clean_correct", "gold_distractor_wrong_final"),
    ("persuasive_filler", "clean_correct", "persuasive_filler_correct"),
    ("cross_corrupt_vs_wrongfinal", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final"),
]
OUTCOME_TYPES = ("outcome_wrong_final", "outcome_gold_distractor")


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def rates(recs, mech):
    n = len(recs)
    c = {o: sum(r[f"{mech}_outcome"] == o for r in recs) for o in ("better", "tie", "worse")}
    out = {"n": n, "better_rate": c["better"] / n, "tie_rate": c["tie"] / n, "wrong_rate": c["worse"] / n, "counts": c}
    out["better_ci"], out["wrong_ci"] = list(wilson(c["better"], n)), list(wilson(c["worse"], n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--limit-problems", type=int)
    ap.add_argument("--outdir", default="task5_feedback")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / args.outdir
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    pids = sorted(groups, key=lambda x: int(x))
    if args.limit_problems:
        pids = pids[: args.limit_problems]
    print("Diagnostic problems:", len(pids), "| pair types:", [p[0] for p in PAIR_SPECS])
    judge = PairwiseAIJudge(cfg, f"{cfg['results_dir']}/{args.outdir}/judge_cache.json")

    recs = []
    for k, pid in enumerate(pids, start=1):
        v = groups[pid]
        q = v["clean_correct"]["question"]
        for ptype, better, worse in PAIR_SPECS:
            rb = exact_reward(v[better]["response"], str(v[better]["gold_final"]))
            rw = exact_reward(v[worse]["response"], str(v[worse]["gold_final"]))
            raw = judge.compare(q, v[better]["response"], v[worse]["response"])   # A = diagnostically better response
            recs.append({
                "problem_id": pid, "pair_type": ptype, "better_variant": better, "worse_variant": worse,
                "verifier_reward_better": rb, "verifier_reward_worse": rw,
                "verifier_outcome": "better" if rb > rw else ("tie" if rb == rw else "worse"),
                "judge_raw": raw, "judge_outcome": {"A": "better", "TIE": "tie", "B": "worse"}[raw],
            })
        if k % 5 == 0:
            print(f"  scored {k}/{len(pids)} problems", flush=True)
    write_jsonl(outdir / "diagnostic_pairs.jsonl", recs)

    summary = {"n_problems": len(pids), "pair_types": {}, "variant_mean_words": {}}
    for mech in ("verifier", "judge"):
        summary[mech] = {p[0]: rates([r for r in recs if r["pair_type"] == p[0]], mech) for p in PAIR_SPECS}
        pooled = [r for r in recs if r["pair_type"] in OUTCOME_TYPES]
        summary[mech]["S_reason"] = summary[mech]["reasoning_corruption"]["better_rate"]
        summary[mech]["S_outcome"] = rates(pooled, mech)
    for var in EXPECTED_VARIANTS:
        summary["variant_mean_words"][var] = float(np.mean([len(groups[p][var]["response"].split()) for p in pids]))
    save_json(outdir / "diagnostic_summary.json", summary)
    with (outdir / "diagnostic_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mechanism", "pair_type", "n", "better_rate", "tie_rate", "wrong_rate"])
        for mech in ("verifier", "judge"):
            for p in PAIR_SPECS:
                r = summary[mech][p[0]]
                w.writerow([mech, p[0], r["n"], round(r["better_rate"], 4), round(r["tie_rate"], 4), round(r["wrong_rate"], 4)])
            r = summary[mech]["S_outcome"]
            w.writerow([mech, "outcome_pooled", r["n"], round(r["better_rate"], 4), round(r["tie_rate"], 4), round(r["wrong_rate"], 4)])

    dis = []
    for r in recs:
        if r["verifier_outcome"] != r["judge_outcome"]:
            v = groups[r["problem_id"]]
            dis.append({**r, "question": v["clean_correct"]["question"][:300],
                        "better_response": v[r["better_variant"]]["response"][:900], "worse_response": v[r["worse_variant"]]["response"][:900]})
    write_jsonl(outdir / "diagnostic_disagreements.jsonl", dis)

    print("\nControlled diagnostics (better / tie / wrong preference):")
    for mech in ("verifier", "judge"):
        for p in PAIR_SPECS:
            r = summary[mech][p[0]]
            print(f"  {mech:<8} {p[0]:<28} n={r['n']} better {r['better_rate']:.2f} tie {r['tie_rate']:.2f} wrong {r['wrong_rate']:.2f}")
        so = summary[mech]["S_outcome"]
        print(f"  {mech:<8} S_reason={summary[mech]['S_reason']:.2f} | S_outcome={so['better_rate']:.2f} (tie {so['tie_rate']:.2f} wrong {so['wrong_rate']:.2f}, n={so['n']})")
    print("  mean words per variant:", {k: round(v, 1) for k, v in summary["variant_mean_words"].items()})
    print(f"  verifier-vs-judge disagreements: {len(dis)} of {len(recs)} pairs -> diagnostic_disagreements.jsonl")


if __name__ == "__main__":
    main()