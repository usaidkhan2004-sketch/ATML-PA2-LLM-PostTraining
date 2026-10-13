from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", default="sft,dpo,ppo,grpo")
    ap.add_argument("--outdir", default="task4_safety")
    ap.add_argument("--dedup", action="store_true", help="one row per unique (prompt, response); labels are mapped back to every policy")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / args.outdir
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    rows = []
    for pol in args.policies.split(","):
        gen = {r["xstest_id"]: r for r in read_jsonl(outdir / f"generated_{pol}.jsonl")}
        for i in ids:
            r = gen[i]
            rows.append({"xstest_id": i, "policy": pol, "benchmark_class": r["benchmark_class"],
                         "prompt": r["prompt"], "response": r["response"]})

    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    if args.dedup:
        uniq = {}
        for r in rows:
            uniq.setdefault((r["xstest_id"], r["response"]), r)
        keys = list(uniq)
        order = [keys[j] for j in rng.permutation(len(keys))]
        audit_row_of = {k: i for i, k in enumerate(order)}
        sheet = pd.DataFrame({"audit_row": range(len(order)),
                              "benchmark_class": [uniq[k]["benchmark_class"] for k in order],
                              "prompt": [uniq[k]["prompt"] for k in order],
                              "response": [uniq[k]["response"] for k in order],
                              "manual_label": [""] * len(order)})
        key = pd.DataFrame({"audit_row": [audit_row_of[(r["xstest_id"], r["response"])] for r in rows],
                            "xstest_id": [r["xstest_id"] for r in rows], "policy": [r["policy"] for r in rows]})
        suffix = "_dedup"
        print(f"{len(rows)} policy responses -> {len(order)} unique (prompt, response) rows to label")
    else:
        perm = rng.permutation(len(rows))
        shuffled = [rows[j] for j in perm]
        sheet = pd.DataFrame({"audit_row": range(len(shuffled)),
                              "benchmark_class": [r["benchmark_class"] for r in shuffled],
                              "prompt": [r["prompt"] for r in shuffled],
                              "response": [r["response"] for r in shuffled],
                              "manual_label": [""] * len(shuffled)})
        key = pd.DataFrame({"audit_row": range(len(shuffled)), "xstest_id": [r["xstest_id"] for r in shuffled],
                            "policy": [r["policy"] for r in shuffled]})
        suffix = ""
    sheet.to_csv(outdir / f"manual_audit_sheet{suffix}.csv", index=False)
    key.to_csv(outdir / f"manual_audit_key{suffix}.csv", index=False)
    print(f"Wrote blind audit sheet ({len(sheet)} rows, policy hidden, shuffled): {outdir / f'manual_audit_sheet{suffix}.csv'}")
    print("Fill manual_label with one of: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS.")
    print("Save it as manual_audit_labels.csv in the same folder. Do not open judged_*.jsonl before labeling.")


if __name__ == "__main__":
    main()