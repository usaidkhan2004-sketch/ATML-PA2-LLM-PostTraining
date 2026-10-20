"""Across-update check of the PPO critic: does its average value predict the reward the response receives?
Run from the repository root:  python -m task2_ppo.critic_check
Writes results/task2_ppo/critic_check.json
"""
import json
from pathlib import Path

import numpy as np

from common.data import read_jsonl

R = Path("results/task2_ppo")
B = 10000
rng = np.random.default_rng(0)
L = read_jsonl(str(R / "standard_train_log.jsonl"))


def perm_corr(a, b):
    r = float(np.corrcoef(a, b)[0, 1])
    p = float(np.mean([abs(np.corrcoef(rng.permutation(a), b)[0, 1]) >= abs(r) for _ in range(B)]))
    return r, p


value = np.array([x["mean_value"] for x in L])
raw = np.array([x["raw_reward"] for x in L])
eff = np.array([x["effective_reward"] for x in L])
r_raw, p_raw = perm_corr(value, raw)
r_eff, p_eff = perm_corr(value, eff)
out = {"n_updates": len(L), "note": "explained variance in the log is computed over the tokens of one response per update; this "
       "checks instead whether the critic's mean value tracks the reward across updates",
       "corr_mean_value_vs_raw_reward": r_raw, "perm_p_raw": p_raw,
       "corr_mean_value_vs_effective_reward": r_eff, "perm_p_effective": p_eff}
(R / "critic_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))