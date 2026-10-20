"""Task 5 statistics and example candidates (reads saved files only; no model is run).
Run from the repository root:  python -m task5_feedback.stats
Writes results/task5_feedback/task5_intervals.json and results/task5_feedback/example_candidates.txt
"""
import json
import math
from pathlib import Path

import numpy as np

R = Path("results/task5_feedback")
B = 10000
rng = np.random.default_rng(0)
out = {"bootstrap_resamples": B, "seed": 0}


def jl(name):
    with open(R / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n))


def boot_ci(d):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ---------- A. policy comparisons on GSM8K and SVAMP ----------
for ds in ["gsm", "transfer"]:
    G = {p: {str(r["id"]): r for r in jl(f"{ds}_generated_{p}.jsonl")} for p in ["sft", "rlvr", "rlaif"]}
    out[ds] = {}
    print(f"== {ds}")
    for p in ["rlvr", "rlaif"]:
        ids = sorted(set(G[p]) & set(G["sft"]))
        a = np.array([float(G[p][i]["correct"]) for i in ids])
        s = np.array([float(G["sft"][i]["correct"]) for i in ids])
        fa = np.array([float(G[p][i]["format_ok"]) for i in ids])
        fs = np.array([float(G["sft"][i]["format_ok"]) for i in ids])
        b, c = int(((a == 1) & (s == 0)).sum()), int(((a == 0) & (s == 1)).sum())
        fb, fc = int(((fa == 1) & (fs == 0)).sum()), int(((fa == 0) & (fs == 1)).sum())
        ident = float(np.mean([G[p][i]["response"] == G["sft"][i]["response"] for i in ids]))
        P = jl(f"{ds}_pairwise_{p}_vs_sft.jsonl")
        score = np.array([float(r["score"]) for r in P])
        vt = [r for r in P if r["verifier_diff"] == 0]
        judge_decides_when_verifier_ties = float(np.mean([r["score"] != 0.5 for r in vt])) if vt else float("nan")
        longer = []
        for r in P:
            i = str(r["id"])
            if r["score"] == 0.5 or i not in G[p]:
                continue
            lp, ls = G[p][i]["response_tokens"], G["sft"][i]["response_tokens"]
            if lp != ls:
                longer.append((lp > ls) == (r["score"] == 1.0))
        rec = {"n": len(ids), "acc_policy": float(a.mean()), "acc_sft": float(s.mean()),
               "acc_diff": float((a - s).mean()), "acc_diff_ci95": boot_ci(a - s),
               "only_policy_correct": b, "only_sft_correct": c, "acc_mcnemar_p": mcnemar_p(b, c),
               "format_only_policy": fb, "format_only_sft": fc, "format_mcnemar_p": mcnemar_p(fb, fc),
               "identical_response_rate": ident, "judge_tie_rate": float(np.mean(score == 0.5)),
               "n_verifier_tied_pairs": len(vt), "judge_decides_when_verifier_ties": judge_decides_when_verifier_ties,
               "judge_winner_is_longer_rate": float(np.mean(longer)) if longer else float("nan"),
               "n_decided_pairs_with_length_difference": len(longer)}
        out[ds][p] = rec
        print(f"  {p}: acc {rec['acc_policy']:.3f} vs sft {rec['acc_sft']:.3f}, diff {rec['acc_diff']:+.3f} {[round(x, 3) for x in rec['acc_diff_ci95']]} "
              f"(only policy {b}, only sft {c}, McNemar p={rec['acc_mcnemar_p']:.3f}) | format only policy {fb} only sft {fc} (p={rec['format_mcnemar_p']:.3f})")
        print(f"     identical to SFT {ident:.3f} | judge tie rate {rec['judge_tie_rate']:.3f} | verifier-tied pairs {len(vt)}, judge picks a side in "
              f"{judge_decides_when_verifier_ties:.3f} | judge winner is the longer response in {rec['judge_winner_is_longer_rate']:.3f} of {len(longer)} decided pairs")

# ---------- B. diagnostic set ----------
D = jl("diagnostic_pairs.jsonl")
out["diagnostic_judge_tie_rate_all_pairs"] = float(np.mean([r["judge_outcome"] == "tie" for r in D]))
print(f"\n== diagnostics: judge tie rate over all {len(D)} controlled pairs {out['diagnostic_judge_tie_rate_all_pairs']:.3f}")
(R / "task5_intervals.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"saved {R / 'task5_intervals.json'}")

# ---------- C. example candidates from verifier-judge disagreements ----------
ex = []


def flat(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " ..."


DIS = jl("diagnostic_disagreements.jsonl")
for t in ["reasoning_corruption", "persuasive_filler", "outcome_wrong_final", "outcome_gold_distractor"]:
    rows = [r for r in DIS if r["pair_type"] == t]
    rows = sorted(rows, key=lambda r: r["judge_outcome"] != "worse")[:2]
    ex.append("=" * 100)
    ex.append(f"{t}: {len([r for r in DIS if r['pair_type'] == t])} disagreements; showing {len(rows)}")
    for r in rows:
        ex.append(f"\n[problem {r['problem_id']}] verifier {r['verifier_outcome']} | judge {r['judge_outcome']} (better={r['better_variant']}, worse={r['worse_variant']})")
        ex.append(f"QUESTION: {flat(r['question'], 220)}")
        ex.append(f"BETTER  : {flat(r['better_response'], 420)}")
        ex.append(f"WORSE   : {flat(r['worse_response'], 420)}")
(R / "example_candidates.txt").write_text("\n".join(ex) + "\n", encoding="utf-8")
print("\n".join(ex))
print(f"\nsaved {R / 'example_candidates.txt'}")