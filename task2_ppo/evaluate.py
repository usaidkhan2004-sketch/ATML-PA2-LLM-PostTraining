from __future__ import annotations

import argparse

from common.rl_eval import evaluate_rl_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the plain base model")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--limit", type=int, help="quick test: only the first N prompts")
    ap.add_argument("--gen-batch", type=int, default=16)
    args = ap.parse_args()
    evaluate_rl_policy(args.config, args.adapter, args.name, gen_batch=args.gen_batch, limit=args.limit)


if __name__ == "__main__":
    main()