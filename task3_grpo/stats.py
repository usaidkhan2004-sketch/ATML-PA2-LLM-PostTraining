"""Task 3 statistics and example candidates (reads saved files only; no model is run).
Run from the repository root:  python -m task3_grpo.stats
Writes results/task3_grpo/task3_intervals.json and results/task3_grpo/example_candidates.txt
"""
import json
from pathlib import Path

import numpy as np

from common.data import prompt_messages, read_jsonl
from task3_grpo.analyze_group_size import TOL, WEAK, difficulty_bins, load_k8_cache

R = Path("results/task3_grpo")
B = 10000
rng = np.random.default_rng(0)
out = {"bootstrap_resamples": B, "seed": 0}
MAXC = 512  # max_completion_length in configs/grpo.yaml


def log(s=""):
    print(s)


def ranks(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def spearman_perm(x, y=None):
    y = np.arange(len(x), dtype=float) if y is None else ranks(y)
    rx = ranks(x)
    rho = float(np.corrcoef(rx, y)[0, 1])
    perm = np.array([np.corrcoef(rng.permutation(rx), y)[0, 1] for _ in range(B)])
    return rho, float(np.mean(np.abs(perm) >= abs(rho)))


def boot_ci(d):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ---------- A. standard 20-update continuation ----------
log("== A. standard GRPO run (20 updates, K=4, one prompt each)")
L = read_jsonl(str(R / "standard_train_log.jsonl"))
keys = ["reward_mean", "reward_std_in_group", "kl_k3", "kl_token_mean", "entropy", "length_mean", "frac_truncated",
        "masked_token_fraction", "policy_loss", "grad_norm", "clip_fraction"]
out["standard_trajectory"] = {}
for k in keys:
    x = np.array([r[k] for r in L], float)
    rho, p = spearman_perm(x)
    rec = {"mean": float(x.mean()), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
           "first10_mean": float(x[:10].mean()), "last10_mean": float(x[10:].mean()), "spearman_vs_update": rho, "spearman_perm_p": p}
    out["standard_trajectory"][k] = rec
    log(f"{k:24s} mean {rec['mean']:.4g}  median {rec['median']:.4g}  range [{rec['min']:.4g}, {rec['max']:.4g}]  "
        f"first10 {rec['first10_mean']:.4g} -> last10 {rec['last10_mean']:.4g}  spearman {rho:+.2f} (p={p:.3f})")
comp = [c for r in L for c in r["completion_tokens"]]
masked_tokens = sum(r["masked_token_fraction"] * sum(r["completion_tokens"]) for r in L)
active = [r for r in L if r["masked_token_fraction"] < 1]
spike = max(L, key=lambda r: r["grad_norm"])
rho_g, p_g = spearman_perm([r["grad_norm"] for r in active], [min(r["completion_tokens"]) for r in active])
out["standard_extra"] = {
    "completions": len(comp), "completions_truncated_at_cap": int(sum(c >= MAXC for c in comp)),
    "updates_fully_masked": [r["update"] for r in L if r["masked_token_fraction"] >= 1],
    "generated_tokens": int(sum(comp)), "masked_tokens": float(masked_tokens), "masked_token_share": float(masked_tokens / sum(comp)),
    "uninformative_groups": int(sum(not r["informative_group"] for r in L)),
    "updates_grad_norm_above_1": [r["update"] for r in L if r["grad_norm"] > 1.0],
    "spike_update": spike["update"], "spike_grad_norm": spike["grad_norm"], "spike_completion_tokens": spike["completion_tokens"],
    "spike_rewards": spike["rewards"], "spike_advantages": spike["advantages"],
    "median_grad_norm_active_updates": float(np.median([r["grad_norm"] for r in active])),
    "spearman_gradnorm_vs_shortest_completion_active": rho_g, "spearman_perm_p": p_g,
}
log(f"extra: {json.dumps(out['standard_extra'])}")

# ---------- B. equal-generation group-size study, with the (K-1)/K correction ----------
log("\n== B. group-size study (192 cached completions; consecutive regrouping; terciles of prompt mean reward)")
by_prompt = load_k8_cache("cached/grpo_k_cache.jsonl")
bins, _ = difficulty_bins(by_prompt)


def groups_for(k, perm=False):
    gs = []
    for pid, rows in by_prompt.items():
        rows = list(rows[:8])
        if perm:
            rows = [rows[i] for i in rng.permutation(8)]
        for gi in range(0, 8, k):
            gs.append((pid, rows[gi:gi + k]))
    return gs


def per_prompt(k, perm=False):
    d = {}
    for pid, chunk in groups_for(k, perm):
        r = np.array([float(x["reward"]) for x in chunk])
        sd = float(r.std())
        v = d.setdefault(pid, {"inf": [], "weak": [], "sd": [], "sq": 0.0, "n": 0})
        v["inf"].append(sd > TOL); v["weak"].append(sd < WEAK); v["sd"].append(sd)
        v["sq"] += float(((r - r.mean()) ** 2).sum()); v["n"] += len(r)
    return d


out["group_size"] = {}
P = {k: per_prompt(k) for k in (2, 4, 8)}
for b in ["all", "hard", "medium", "easy"]:
    pids = [p for p in by_prompt if b == "all" or bins[p] == b]
    for k in (2, 4, 8):
        d = P[k]
        inf = np.mean([x for p in pids for x in d[p]["inf"]]); weak = np.mean([x for p in pids for x in d[p]["weak"]])
        var = sum(d[p]["sq"] for p in pids) / sum(d[p]["n"] for p in pids)
        rnd = [per_prompt(k, perm=True) for _ in range(200)]
        rnd_weak = float(np.mean([np.mean([x for p in pids for x in q[p]["weak"]]) for q in rnd]))
        rec = {"informative_rate": float(inf), "weak_signal_rate": float(weak), "var_relative_signal": float(var),
               "var_corrected_K_over_Km1": float(var * k / (k - 1)), "random_partition_weak_rate": rnd_weak}
        out["group_size"][f"K{k}_{b}"] = rec
        log(f"K={k} {b:6s} informative {inf:.3f}  weak(<{WEAK}) {weak:.3f} (random-partition {rnd_weak:.3f})  "
            f"var {var:.3f}  var*K/(K-1) {rec['var_corrected_K_over_Km1']:.3f}")
    d8 = [np.mean(P[8][p]["weak"]) - np.mean(P[2][p]["weak"]) for p in pids]
    out["group_size"][f"weak_K8_minus_K2_{b}"] = {"diff": float(np.mean(d8)), "ci95": boot_ci(d8)}
    log(f"   weak-signal rate K=8 minus K=2 ({b}): {np.mean(d8):+.3f}  CI {boot_ci(d8)}")
unin = [(pid, chunk) for pid, chunk in groups_for(2) if float(np.std([float(x['reward']) for x in chunk])) <= TOL]
out["group_size"]["uninformative_K2_groups"] = [
    {"prompt": pid, "bin": bins[pid], "rewards": [float(x["reward"]) for x in c],
     "identical_text": c[0]["completion"] == c[1]["completion"], "tokens": [x["completion_tokens"] for x in c]} for pid, c in unin]
log(f"uninformative K=2 groups: {json.dumps(out['group_size']['uninformative_K2_groups'])}")

# ---------- C. canonical GRPO vs Dr. GRPO ----------
log("\n== C. normalization forks (8 updates each)")
out["normalization_train"] = {}
for name in ["fork_grpo", "fork_dr_grpo"]:
    T = read_jsonl(str(R / f"{name}_train_log.jsonl"))
    S = json.load(open(R / f"{name}_train_summary.json"))
    ok = [r for r in T if r["masked_token_fraction"] < 1 and abs(r["adv_short"]) > 1e-6 and abs(r["adv_long"]) > 1e-6]
    gs = [r["gnorm_short"] / abs(r["adv_short"]) for r in ok]
    gl = [r["gnorm_long"] / abs(r["adv_long"]) for r in ok]
    rec = {"generated_tokens": S["total_generated_tokens"], "updates_fully_masked": [r["update"] for r in T if r["masked_token_fraction"] >= 1],
           "mean_grad_norm": float(np.mean([r["grad_norm"] for r in T])), "max_grad_norm": float(max(r["grad_norm"] for r in T)),
           "gnorm_per_adv_short_mean": float(np.mean(gs)), "gnorm_per_adv_long_mean": float(np.mean(gl)),
           "median_long_over_short": float(np.median(np.array(gl) / np.array(gs))), "n_updates_used": len(ok),
           "first_update_rewards_identical_check": T[0]["rewards"]}
    out["normalization_train"][name] = rec
    log(f"{name:13s} tokens {rec['generated_tokens']}  masked updates {rec['updates_fully_masked']}  mean grad norm {rec['mean_grad_norm']:.3f} "
        f"(max {rec['max_grad_norm']:.3f})  gnorm/|A| short {rec['gnorm_per_adv_short_mean']:.3f} long {rec['gnorm_per_adv_long_mean']:.3f}  "
        f"median long/short {rec['median_long_over_short']:.2f}")

names = ["midpoint", "sft", "standard", "fork_grpo", "fork_dr_grpo"]
G = {n: {r["prompt_id"]: r for r in read_jsonl(str(R / f"eval_{n}_generations.jsonl"))} for n in names}
out["heldout_paired"] = {}
for a, b in [("fork_dr_grpo", "fork_grpo"), ("fork_grpo", "midpoint"), ("fork_dr_grpo", "midpoint"), ("standard", "midpoint")]:
    ids = sorted(set(G[a]) & set(G[b]))
    key = f"{a}_minus_{b}"
    out["heldout_paired"][key] = {}
    row = []
    for f in ["reward_raw", "response_tokens", "entropy_mean", "kl_seq_sum", "truncated"]:
        d = np.array([float(G[a][i][f]) - float(G[b][i][f]) for i in ids])
        ci = boot_ci(d)
        out["heldout_paired"][key][f] = {"diff": float(d.mean()), "ci95": ci, "n": len(ids)}
        row.append(f"{f} {d.mean():+.4g} [{ci[0]:+.3g},{ci[1]:+.3g}]")
    log(f"{key:28s} " + "  ".join(row))
ids = sorted(set(G["fork_dr_grpo"]) & set(G["fork_grpo"]))
dl = np.array([G["fork_dr_grpo"][i]["response_tokens"] - G["fork_grpo"][i]["response_tokens"] for i in ids], float)
dr = np.array([G["fork_dr_grpo"][i]["reward_raw"] - G["fork_grpo"][i]["reward_raw"] for i in ids], float)
out["heldout_paired"]["corr_length_diff_vs_reward_diff_dr_minus_grpo"] = float(np.corrcoef(dl, dr)[0, 1])
log(f"corr(length diff, reward diff), Dr minus canonical: {np.corrcoef(dl, dr)[0, 1]:+.3f}")

(R / "task3_intervals.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
log(f"\nsaved {R / 'task3_intervals.json'}")

# ---------- D. example candidates ----------
ex = []


def flat(s, n, end=False):
    s = " ".join(str(s).split())
    return s if len(s) <= n else (("... " + s[-n:]) if end else (s[:n] + " ..."))


train_p = {r["prompt_id"]: r for r in read_jsonl("data/rl_prompt_pool_train.jsonl")}
eval_p = {r["prompt_id"]: r for r in read_jsonl("data/rl_prompt_pool_eval.jsonl")}
ex.append("=" * 100)
ex.append(f"D1. GRADIENT SPIKE: standard update {spike['update']} (grad norm {spike['grad_norm']:.2f}); tokens {spike['completion_tokens']}, "
          f"rewards {[round(x, 2) for x in spike['rewards']]}, advantages {[round(x, 2) for x in spike['advantages']]}")
if spike["prompt_id"] in train_p:
    ex.append(f"PROMPT: {flat(prompt_messages(train_p[spike['prompt_id']])[-1]['content'], 400)}")
for r in L:
    if r["masked_token_fraction"] >= 1:
        ex.append(f"FULLY MASKED update {r['update']}: tokens {r['completion_tokens']}, rewards {[round(x, 2) for x in r['rewards']]}; PROMPT: "
                  f"{flat(prompt_messages(train_p[r['prompt_id']])[-1]['content'], 200) if r['prompt_id'] in train_p else '?'}")
ex.append("\n" + "=" * 100)
ex.append("D2. UNINFORMATIVE K=2 GROUPS IN THE CACHE (identical rewards)")
for pid, c in unin:
    ex.append(f"[bin {bins[pid]}] rewards {[round(float(x['reward']), 3) for x in c]}  tokens {[x['completion_tokens'] for x in c]}  identical text: {c[0]['completion'] == c[1]['completion']}")
    ex.append(f"   A: {flat(c[0]['completion'], 200)}")
    ex.append(f"   B: {flat(c[1]['completion'], 200)}")
ex.append("\n" + "=" * 100)
ex.append("D3. HELD-OUT: largest length differences, Dr. GRPO minus canonical GRPO (same prompt)")
order = sorted(ids, key=lambda i: G["fork_dr_grpo"][i]["response_tokens"] - G["fork_grpo"][i]["response_tokens"])
for tag, sel in [("Dr much SHORTER", order[:3]), ("Dr much LONGER", order[-3:][::-1])]:
    for i in sel:
        a, b = G["fork_grpo"][i], G["fork_dr_grpo"][i]
        ex.append(f"\n[{tag}] prompt_id={i[:12]}  canonical: {a['response_tokens']} tok, reward {a['reward_raw']:+.2f}, trunc {a['truncated']} | "
                  f"Dr: {b['response_tokens']} tok, reward {b['reward_raw']:+.2f}, trunc {b['truncated']}")
        ex.append(f"PROMPT   : {flat(prompt_messages(eval_p[i])[-1]['content'], 250) if i in eval_p else '?'}")
        ex.append(f"CANON END: {flat(a['response'], 160, end=True)}")
        ex.append(f"DR END   : {flat(b['response'], 160, end=True)}")
(R / "example_candidates.txt").write_text("\n".join(ex) + "\n", encoding="utf-8")
print("\n".join(ex))
print(f"\nsaved {R / 'example_candidates.txt'}")