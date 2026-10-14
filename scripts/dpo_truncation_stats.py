"""Count DPO pairs whose response is truncated by encode_prompt_response (prompt + response > max_sequence_length).
Run from the repository root:  python -m scripts.dpo_truncation_stats
Writes results/task1_dpo/truncation_stats.json
"""
import argparse

import numpy as np

from common.data import load_yaml, preference_responses, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer
from task1_dpo.train import filter_fitting_rows

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/dpo.yaml")
args = parser.parse_args()

cfg = load_yaml(args.config)
tok = load_tokenizer(cfg["base_model"])
max_len = int(cfg["max_sequence_length"])

out = {"max_sequence_length": max_len, "datasets": {}}
for key, path in cfg["paths"].items():
    if not key.startswith("dpo_"):
        continue
    rows = read_jsonl(path)
    kept, skipped = filter_fitting_rows(tok, rows, max_len)
    n_trunc_c = n_trunc_r = n_either = 0
    lost = []
    for row in kept:
        p = len(tok.apply_chat_template(prompt_messages(row), tokenize=True, add_generation_prompt=True))
        yc, yr = preference_responses(row)
        lc = len(tok(yc, add_special_tokens=False).input_ids)
        lr = len(tok(yr, add_special_tokens=False).input_ids)
        tc, tr = p + lc > max_len, p + lr > max_len
        n_trunc_c += tc
        n_trunc_r += tr
        n_either += tc or tr
        if tc:
            lost.append(p + lc - max_len)
        if tr:
            lost.append(p + lr - max_len)
    rec = {
        "path": path, "total": len(rows), "kept": len(kept), "skipped_prompt_too_long": len(skipped),
        "kept_with_chosen_truncated": int(n_trunc_c), "kept_with_rejected_truncated": int(n_trunc_r),
        "kept_with_either_truncated": int(n_either),
        "mean_tokens_cut_when_truncated": float(np.mean(lost)) if lost else 0.0,
        "max_tokens_cut": int(max(lost)) if lost else 0,
    }
    out["datasets"][key] = rec
    print(key, rec, flush=True)

dest = repo_path("results/task1_dpo/truncation_stats.json")
save_json(dest, out)
print("saved", dest)