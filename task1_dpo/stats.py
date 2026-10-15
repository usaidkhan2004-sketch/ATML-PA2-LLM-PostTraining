"""Confidence intervals and paired tests for the Task 1 claims.
Run from the repository root:  python -m task1_dpo.stats
Reads results/task1_dpo/eval_*_{pairs_*,generations,wordlimit}.jsonl
Writes results/task1_dpo/task1_intervals.json
"""
import json
import math
from pathlib import Path

import numpy as np

R = Path("results/task1_dpo")
B = 10000
rng = np.random.default_rng(0)
MODELS = ["sft", "standard", "beta_0.03", "beta_0.1", "beta_0.3", "length_balanced"]
out = {"bootstrap_resamples": B, "seed": 0}


def load(name):
    with open(R / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id(recs, field):
    d = {}
    for r in recs:
        d[r["prompt_id"]] = float(r[field])
    return d


def aligned(da, db):
    ids = sorted(set(da) & set(db))
    return ids, np.array([da[i] for i in ids]), np.array([db[i] for i in ids])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def boot_mean_ci(x):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired_binary(a, b):
    d = a - b
    return {"n": int(len(a)), "acc_a": float(a.mean()), "acc_b": float(b.mean()), "diff_a_minus_b": float(d.mean()),
            "diff_ci95": boot_mean_ci(d), "a_only_correct": int(((a == 1) & (b == 0)).sum()),
            "b_only_correct": int(((a == 0) & (b == 1)).sum()),
            "mcnemar_exact_p": mcnemar_p(int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))}


# ---------- 1. held-out pair accuracy (standard eval) ----------
print("== 1. standard held-out preference accuracy (Wilson 95%)")
pairs = {m: load(f"eval_{m}_pairs_standard.jsonl") for m in MODELS if m != "sft"}
corr = {m: by_id(r, "correct") for m, r in pairs.items()}
out["accuracy_standard_eval"] = {}
for m, d in corr.items():
    k, n = int(sum(d.values())), len(d)
    out["accuracy_standard_eval"][m] = {"correct": k, "n": n, "acc": k / n, "wilson95": wilson(k, n)}
    print(f"{m:16s} {k}/{n} = {k / n:.4f}  {wilson(k, n)[0]:.3f}-{wilson(k, n)[1]:.3f}")

print("\n   paired accuracy differences (A - B)")
out["accuracy_standard_paired"] = {}
for a, b in [("beta_0.03", "beta_0.1"), ("beta_0.1", "beta_0.3"), ("beta_0.03", "beta_0.3"),
             ("standard", "beta_0.03"), ("standard", "length_balanced")]:
    ids, xa, xb = aligned(corr[a], corr[b])
    res = paired_binary(xa, xb)
    out["accuracy_standard_paired"][f"{a}_minus_{b}"] = res
    print(f"{a} - {b}: diff {res['diff_a_minus_b']:+.4f} CI {res['diff_ci95'][0]:+.3f}..{res['diff_ci95'][1]:+.3f}"
          f"  (only A {res['a_only_correct']}, only B {res['b_only_correct']}, McNemar p={res['mcnemar_exact_p']:.3f})")

# ---------- 2. length-stratified accuracy ----------
print("\n== 2. length-stratified accuracy (Wilson 95%) and standard vs length-balanced (paired)")
strat = {m: load(f"eval_{m}_pairs_stratified.jsonl") for m in ["standard", "length_balanced"]}
out["stratified"] = {"per_model": {}, "paired_balanced_minus_standard": {}}
for m, recs in strat.items():
    out["stratified"]["per_model"][m] = {}
    for s in ["preferred_longer", "length_matched", "rejected_longer", "ALL"]:
        sel = [r for r in recs if s == "ALL" or r["length_stratum"] == s]
        k, n = int(sum(r["correct"] for r in sel)), len(sel)
        out["stratified"]["per_model"][m][s] = {"correct": k, "n": n, "acc": k / n, "wilson95": wilson(k, n)}
        print(f"{m:16s} {s:17s} {k}/{n} = {k / n:.4f}  {wilson(k, n)[0]:.3f}-{wilson(k, n)[1]:.3f}")
for s in ["preferred_longer", "length_matched", "rejected_longer", "ALL"]:
    da = {r["prompt_id"]: float(r["correct"]) for r in strat["length_balanced"] if s == "ALL" or r["length_stratum"] == s}
    db = {r["prompt_id"]: float(r["correct"]) for r in strat["standard"] if s == "ALL" or r["length_stratum"] == s}
    ids, xa, xb = aligned(da, db)
    res = paired_binary(xa, xb)
    out["stratified"]["paired_balanced_minus_standard"][s] = res
    print(f"balanced - standard, {s}: diff {res['diff_a_minus_b']:+.4f} CI {res['diff_ci95'][0]:+.3f}..{res['diff_ci95'][1]:+.3f}"
          f"  (only balanced {res['a_only_correct']}, only standard {res['b_only_correct']}, McNemar p={res['mcnemar_exact_p']:.3f})")

# ---------- 3. generation metrics ----------
print("\n== 3. generation metrics (mean, bootstrap 95% over prompts)")
gens = {m: load(f"eval_{m}_generations.jsonl") for m in MODELS}
fields = ["rm_score", "response_tokens", "kl_seq_sum", "kl_token_mean", "truncated"]
G = {m: {f: by_id(r, f) for f in fields} for m, r in gens.items()}
out["generation"] = {"per_model": {}, "paired_a_minus_b": {}}
for m in MODELS:
    out["generation"]["per_model"][m] = {}
    row = []
    for f in fields:
        x = np.array(list(G[m][f].values()))
        out["generation"]["per_model"][m][f] = {"mean": float(x.mean()), "ci95": boot_mean_ci(x), "n": int(len(x))}
        row.append(f"{f}={x.mean():.4g}")
    print(f"{m:16s} " + "  ".join(row))

print("\n   paired differences A - B (same prompts)")
pairs_ab = [("standard", "sft"), ("beta_0.03", "sft"), ("beta_0.1", "sft"), ("beta_0.3", "sft"), ("length_balanced", "sft"),
            ("beta_0.03", "beta_0.3"), ("beta_0.03", "beta_0.1"), ("beta_0.1", "beta_0.3"), ("standard", "length_balanced")]
for a, b in pairs_ab:
    key = f"{a}_minus_{b}"
    out["generation"]["paired_a_minus_b"][key] = {}
    row = []
    for f in ["rm_score", "response_tokens", "kl_seq_sum", "truncated"]:
        ids, xa, xb = aligned(G[a][f], G[b][f])
        d = xa - xb
        ci = boot_mean_ci(d)
        out["generation"]["paired_a_minus_b"][key][f] = {"diff": float(d.mean()), "ci95": ci, "n": int(len(d))}
        row.append(f"{f} {d.mean():+.4g} [{ci[0]:+.3g},{ci[1]:+.3g}]")
    print(f"{key:28s} " + "  ".join(row))

# ---------- 4. word-limit compliance (cluster bootstrap over prompts) ----------
print("\n== 4. word-limit compliance (10 prompts x 5 samples; intervals resample PROMPTS)")
wl = {}
for m in MODELS:
    d = {}
    for r in load(f"eval_{m}_wordlimit.jsonl"):
        d.setdefault(r["prompt_id"], []).append(float(r["compliant"]))
    wl[m] = {k: float(np.mean(v)) for k, v in d.items()}
out["wordlimit"] = {"per_model": {}, "paired_a_minus_b": {}}
for m in MODELS:
    x = np.array(list(wl[m].values()))
    k, n = int(round(x.mean() * 50)), 50
    out["wordlimit"]["per_model"][m] = {"mean": float(x.mean()), "n_prompts": int(len(x)),
                                        "cluster_bootstrap95": boot_mean_ci(x), "naive_wilson95_n50": wilson(k, n)}
    print(f"{m:16s} {x.mean():.3f}  prompt-bootstrap {boot_mean_ci(x)[0]:.3f}-{boot_mean_ci(x)[1]:.3f}  "
          f"(naive Wilson n=50: {wilson(k, n)[0]:.3f}-{wilson(k, n)[1]:.3f})")
for a, b in [("standard", "length_balanced"), ("standard", "sft"), ("length_balanced", "sft")]:
    ids, xa, xb = aligned(wl[a], wl[b])
    d = xa - xb
    out["wordlimit"]["paired_a_minus_b"][f"{a}_minus_{b}"] = {"diff": float(d.mean()), "ci95": boot_mean_ci(d), "n_prompts": int(len(d))}
    print(f"{a} - {b}: diff {d.mean():+.3f}  CI {boot_mean_ci(d)[0]:+.3f}..{boot_mean_ci(d)[1]:+.3f}")

dest = R / "task1_intervals.json"
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nsaved {dest}")