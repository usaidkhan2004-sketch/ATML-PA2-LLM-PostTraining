"""Task 2 statistics and example candidates (reads saved files only; no model is run).
Run from the repository root:  python -m task2_ppo.stats
Writes results/task2_ppo/task2_intervals.json and results/task2_ppo/example_candidates.txt
"""
import json
from pathlib import Path

import numpy as np

from common.data import prompt_messages, read_jsonl

R = Path("results/task2_ppo")
B = 10000
rng = np.random.default_rng(0)
out = {"bootstrap_resamples": B, "seed": 0}
lines = []


def log(s=""):
    print(s)
    lines.append(s)


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman_perm(x):
    t = np.arange(len(x), dtype=float)
    rx = ranks(np.asarray(x, float))
    rho = float(np.corrcoef(rx, t)[0, 1])
    perm = np.array([np.corrcoef(rng.permutation(rx), t)[0, 1] for _ in range(B)])
    return rho, float(np.mean(np.abs(perm) >= abs(rho)))


def boot_ci(d):
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ---------- A. standard 20-update trajectory ----------
log("== A. standard run trajectory (20 updates, one prompt each)")
L = read_jsonl(str(R / "standard_train_log.jsonl"))
keys = ["raw_reward", "kl_token_mean", "kl_seq_sum", "entropy", "policy_loss", "value_loss", "grad_norm_policy",
        "grad_norm_value", "clip_fraction", "response_tokens", "critic_explained_variance", "critic_corr",
        "mean_abs_ratio_dev_last_epoch"]
out["standard_trajectory"] = {}
for k in keys:
    x = np.array([r[k] for r in L], float)
    rho, p = spearman_perm(x)
    rec = {"mean": float(x.mean()), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
           "first10_mean": float(x[:10].mean()), "last10_mean": float(x[10:].mean()),
           "spearman_vs_update": rho, "spearman_perm_p": p}
    out["standard_trajectory"][k] = rec
    log(f"{k:30s} mean {rec['mean']:.4g}  median {rec['median']:.4g}  range [{rec['min']:.4g}, {rec['max']:.4g}]  "
        f"first10 {rec['first10_mean']:.4g} -> last10 {rec['last10_mean']:.4g}  spearman {rho:+.2f} (p={p:.3f})")
ev = np.array([r["critic_explained_variance"] for r in L])
out["standard_extra"] = {
    "updates_with_negative_explained_variance": int((ev < 0).sum()), "n_updates": len(L),
    "frac_terminated_with_eos": float(np.mean([r["terminated_with_eos"] for r in L])),
    "pearson_reward_vs_length": float(np.corrcoef([r["raw_reward"] for r in L], [r["response_tokens"] for r in L])[0, 1]),
    "updates_with_any_clipping": int(sum(r["clip_fraction"] > 0 for r in L)),
}
log(f"extra: {out['standard_extra']}")

# ---------- B. size of the KL penalty relative to the reward, per training run ----------
log("\n== B. KL penalty magnitude in training (beta_KL * |sequence KL| vs |reward|)")
runs = ["standard", "fork_eps0.05_kl0.1", "fork_eps0.2_kl0.1", "fork_eps0.5_kl0.1", "fork_eps0.2_kl0.0", "fork_eps0.2_kl0.2"]
out["train_runs"] = {}
for name in runs:
    T = read_jsonl(str(R / f"{name}_train_log.jsonl"))
    beta = float(T[0]["beta_kl"])
    pen = np.array([beta * abs(r["kl_seq_sum"]) for r in T])
    rew = np.array([abs(r["raw_reward"]) for r in T])
    rec = {"beta_kl": beta, "eps": float(T[0]["eps"]), "updates": len(T),
           "mean_kl_token": float(np.mean([r["kl_token_mean"] for r in T])),
           "kl_token_per_update": [float(r["kl_token_mean"]) for r in T],
           "mean_abs_penalty": float(pen.mean()), "mean_abs_reward": float(rew.mean()),
           "penalty_to_reward_ratio": float(pen.mean() / rew.mean()),
           "max_abs_ratio_dev": float(max(r["mean_abs_ratio_dev_last_epoch"] for r in T)),
           "mean_clip_fraction": float(np.mean([r["clip_fraction"] for r in T])),
           "mean_entropy": float(np.mean([r["entropy"] for r in T])),
           "mean_response_tokens": float(np.mean([r["response_tokens"] for r in T]))}
    out["train_runs"][name] = rec
    log(f"{name:20s} beta={beta} eps={rec['eps']}  mean KL/token {rec['mean_kl_token']:+.2e}  "
        f"|penalty| {rec['mean_abs_penalty']:.2e} vs |reward| {rec['mean_abs_reward']:.3f}  "
        f"(ratio {rec['penalty_to_reward_ratio']:.1e})  max|rho-1| {rec['max_abs_ratio_dev']:.4f}  clip {rec['mean_clip_fraction']}")

# ---------- C. held-out paired differences (200 prompts) ----------
log("\n== C. held-out paired differences A - B (bootstrap 95% over the same 200 prompts)")
names = ["midpoint", "sft", "standard"] + runs[1:]
G = {n: {r["prompt_id"]: r for r in read_jsonl(str(R / f"eval_{n}_generations.jsonl"))} for n in names}
fields = ["reward_raw", "response_tokens", "entropy_mean", "kl_seq_sum"]
comparisons = [("standard", "midpoint"), ("fork_eps0.2_kl0.0", "fork_eps0.2_kl0.1"),
               ("fork_eps0.2_kl0.0", "fork_eps0.2_kl0.2"), ("fork_eps0.2_kl0.2", "fork_eps0.2_kl0.1"),
               ("fork_eps0.2_kl0.0", "midpoint"), ("fork_eps0.2_kl0.2", "midpoint")]
out["heldout_paired"] = {}
for a, b in comparisons:
    ids = sorted(set(G[a]) & set(G[b]))
    key = f"{a}_minus_{b}"
    out["heldout_paired"][key] = {}
    row = []
    for f in fields:
        d = np.array([G[a][i][f] - G[b][i][f] for i in ids], float)
        ci = boot_ci(d)
        out["heldout_paired"][key][f] = {"diff": float(d.mean()), "ci95": ci, "n": len(ids)}
        row.append(f"{f} {d.mean():+.4g} [{ci[0]:+.3g},{ci[1]:+.3g}]")
    log(f"{key:40s} " + "  ".join(row))
same = sum(G["fork_eps0.05_kl0.1"][i]["response"] == G["fork_eps0.5_kl0.1"][i]["response"] for i in G["fork_eps0.05_kl0.1"])
out["eps_forks_identical_responses"] = {"identical": int(same), "n": len(G["fork_eps0.05_kl0.1"])}
log(f"eps=0.05 vs eps=0.5 forks: identical responses on {same}/{len(G['fork_eps0.05_kl0.1'])} prompts")

(R / "task2_intervals.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
log(f"\nsaved {R / 'task2_intervals.json'}")

# ---------- D. example candidates: standard (20 updates) vs midpoint, same prompt ----------
ex = []


def say(s=""):
    ex.append(s)


def flat(s, n, end=False):
    s = " ".join(str(s).split())
    if len(s) <= n:
        return s
    return ("... " + s[-n:]) if end else (s[:n] + " ...")


P = {r["prompt_id"]: r for r in read_jsonl("data/rl_prompt_pool_eval.jsonl")}
ids = [i for i in G["standard"] if i in G["midpoint"] and i in P]
gain = sorted(ids, key=lambda i: G["standard"][i]["reward_raw"] - G["midpoint"][i]["reward_raw"], reverse=True)
for title, sel in [("LARGEST REWARD GAINS (standard minus midpoint)", gain[:6]),
                   ("LARGEST REWARD DROPS (standard minus midpoint)", gain[-4:][::-1])]:
    say("=" * 100)
    say(title)
    for k, i in enumerate(sel, 1):
        s, m = G["standard"][i], G["midpoint"][i]
        say(f"\n[{title.split()[1][0]}{k}] prompt_id={i[:12]}  reward midpoint={m['reward_raw']:+.2f} -> standard={s['reward_raw']:+.2f}  "
            f"tokens {m['response_tokens']} -> {s['response_tokens']}  truncated {m['truncated']} -> {s['truncated']}")
        say(f"PROMPT        : {flat(prompt_messages(P[i])[-1]['content'], 300)}")
        say(f"MIDPOINT START: {flat(m['response'], 300)}")
        say(f"MIDPOINT END  : {flat(m['response'], 150, end=True)}")
        say(f"STANDARD START: {flat(s['response'], 300)}")
        say(f"STANDARD END  : {flat(s['response'], 150, end=True)}")
(R / "example_candidates.txt").write_text("\n".join(ex) + "\n", encoding="utf-8")
print("\n".join(ex))
print(f"\nsaved {R / 'example_candidates.txt'}")