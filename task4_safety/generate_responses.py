from __future__ import annotations

import argparse
import gc
import time

import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4, limit: int | None = None):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    if limit:
        df = df.iloc[:limit]
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "xstest_label": str(row["label"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
        if (start // batch_size) % 5 == 0:
            print(f"  [{policy_name}] {min(start + batch_size, len(df))}/{len(df)}", flush=True)
        clear_gpu()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", default="sft,dpo,ppo,grpo")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="quick test: only the first N prompts")
    ap.add_argument("--outdir", default="task4_safety")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / args.outdir
    print("Policies:", args.policies, "| XSTest rows:", len(load_xstest(cfg)))
    for name in args.policies.split(","):
        out = outdir / f"generated_{name}.jsonl"
        if out.exists() and not args.force:
            print(f"[skip] {out.name} already exists")
            continue
        t0 = time.perf_counter()
        records = generate_for_policy(cfg, name, args.batch_size, args.limit)
        write_jsonl(out, records)
        print(f"[{name}] saved {len(records)} responses to {out} in {time.perf_counter() - t0:.0f}s", flush=True)
        gc.collect()
        clear_gpu()


if __name__ == "__main__":
    main()