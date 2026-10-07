from __future__ import annotations

import argparse

from common.data import load_yaml
from task2_ppo.forks import fork_name, run_fork, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    prefix = "smoke_" if args.smoke else ""
    betas = cfg["kl_values"][:1] if args.smoke else cfg["kl_values"]
    eps = float(cfg["clip_epsilon"])
    for b in betas:
        run_fork(args.config, eps, float(b), updates=1 if args.smoke else None,
                 eval_limit=4 if args.smoke else None, prefix=prefix)
    if not args.smoke:
        summarize(args.config, [("midpoint (start)", "midpoint"), ("SFT base", "sft"), ("standard 20 updates", "standard")]
                  + [(f"fork beta_KL={b}", fork_name(eps, float(b))) for b in betas], "kl_summary")


if __name__ == "__main__":
    main()