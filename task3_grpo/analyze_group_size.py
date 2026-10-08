from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

TOL = 1e-6      # numerical tolerance for "zero standard deviation" (as in the released advantage helper)
WEAK = 0.05     # a group with reward std below this is a practical near-tie


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, rng=None):
    """Split each prompt's 8 completions into consecutive groups of size k (total generations stay 192).
    With rng given, the 8 completions are randomly permuted first (robustness check)."""
    if 8 % k:
        raise ValueError("K must divide 8")
    groups = []
    for pid, rows in by_prompt.items():
        rows = list(rows[:8])
        if rng is not None:
            rows = [rows[i] for i in rng.permutation(8)]
        for gi in range(0, 8, k):
            chunk = rows[gi:gi + k]
            groups.append({"prompt_id": pid, "rewards": [float(r["reward"]) for r in chunk]})
    return groups


def difficulty_bins(by_prompt):
    """Prompt difficulty = mean reward over its 8 cached completions; thirds of the sorted prompts."""
    means = {pid: float(np.mean([float(r["reward"]) for r in rows[:8]])) for pid, rows in by_prompt.items()}
    order = sorted(means, key=lambda p: (means[p], p))
    third = len(order) // 3
    return {pid: ("hard" if i < third else ("easy" if i >= len(order) - third else "medium")) for i, pid in enumerate(order)}, means


def per_prompt_sums(groups):
    s = defaultdict(lambda: np.zeros(5))   # n_groups, informative, std_sum, centered_sq_sum, weak
    n_comp = defaultdict(int)
    for g in groups:
        r = np.asarray(g["rewards"])
        sd = float(r.std())
        v = s[g["prompt_id"]]
        v += np.array([1, sd > TOL, sd, ((r - r.mean()) ** 2).sum(), sd < WEAK])
        n_comp[g["prompt_id"]] += len(r)
    return s, n_comp


def metrics_from(sums, n_comp, pids):
    tot = sum((sums[p] for p in pids), np.zeros(5))
    comps = sum(n_comp[p] for p in pids)
    return {"n_groups": int(tot[0]), "informative_rate": tot[1] / tot[0], "mean_group_std": tot[2] / tot[0],
            "var_relative_signal": tot[3] / comps, "weak_signal_rate": tot[4] / tot[0]}


def bootstrap(sums, n_comp, pids, n=2000, seed=6304):
    rng = np.random.default_rng(seed)
    pids = list(pids)
    inf, sd = [], []
    for _ in range(n):
        m = metrics_from(sums, n_comp, [pids[i] for i in rng.integers(0, len(pids), len(pids))])
        inf.append(m["informative_rate"]); sd.append(m["mean_group_std"])
    q = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return {"informative_rate_ci": q(inf), "mean_group_std_ci": q(sd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    bins, means = difficulty_bins(by_prompt)
    total = sum(len(v[:8]) for v in by_prompt.values())
    print(f"Cached prompts: {len(by_prompt)} | total generations: {total} | group sizes: {cfg['group_sizes']}")
    print("difficulty bins:", {b: sum(1 for x in bins.values() if x == b) for b in ("hard", "medium", "easy")},
          "| prompt mean reward by bin:", {b: round(float(np.mean([means[p] for p in bins if bins[p] == b])), 3) for b in ("hard", "medium", "easy")})

    rows = []
    for k in [int(x) for x in cfg["group_sizes"]]:
        sums, n_comp = per_prompt_sums(regroup_equal_generation_budget(by_prompt, k))
        # robustness: expectation over random partitions of each prompt's completions
        rng = np.random.default_rng(6304)
        rnd = [metrics_from(*per_prompt_sums(regroup_equal_generation_budget(by_prompt, k, rng)), list(by_prompt)) for _ in range(200)]
        for label, pids in [("all", list(by_prompt))] + [(b, [p for p in bins if bins[p] == b]) for b in ("hard", "medium", "easy")]:
            m = metrics_from(sums, n_comp, pids)
            m.update(bootstrap(sums, n_comp, pids))
            m.update({"K": k, "bin": label, "n_prompts": len(pids), "n_generations": len(pids) * 8})
            if label == "all":
                m["random_partition_informative_rate_mean"] = float(np.mean([x["informative_rate"] for x in rnd]))
                m["random_partition_mean_group_std_mean"] = float(np.mean([x["mean_group_std"] for x in rnd]))
                m["random_partition_mean_group_std_sd"] = float(np.std([x["mean_group_std"] for x in rnd]))
            rows.append(m)
            print(f"K={k} {label:<7} groups={m['n_groups']:<3} informative {m['informative_rate']:.3f} "
                  f"[{m['informative_rate_ci'][0]:.3f},{m['informative_rate_ci'][1]:.3f}] | mean group std {m['mean_group_std']:.3f} "
                  f"[{m['mean_group_std_ci'][0]:.3f},{m['mean_group_std_ci'][1]:.3f}] | var(rel. signal) {m['var_relative_signal']:.3f} "
                  f"| weak-signal {m['weak_signal_rate']:.3f}"
                  + (f" | random-partition mean std {m['random_partition_mean_group_std_mean']:.3f}+-{m['random_partition_mean_group_std_sd']:.3f}" if label == "all" else ""))

    rdir = repo_path(cfg["results_dir"])
    flat = [{**r, "informative_rate_ci": f"{r['informative_rate_ci'][0]:.4f}..{r['informative_rate_ci'][1]:.4f}",
             "mean_group_std_ci": f"{r['mean_group_std_ci'][0]:.4f}..{r['mean_group_std_ci'][1]:.4f}"} for r in rows]
    keys = []
    for r in flat:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with (rdir / "group_size_study.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    save_json(rdir / "group_size_study.json", {"rows": rows, "difficulty_bins": bins, "prompt_mean_reward": means,
                                               "rules": {"regrouping": "consecutive groups of K over generation_index order, per prompt",
                                                         "difficulty": "terciles of per-prompt mean reward over the 8 cached completions",
                                                         "tolerance": TOL, "weak_signal_threshold": WEAK}})
    print(f"saved {rdir / 'group_size_study.csv'}")


if __name__ == "__main__":
    main()