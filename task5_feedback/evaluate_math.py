from __future__ import annotations

import argparse
import gc
import re
import time
from collections import Counter

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
POLICIES = ["sft", "rlvr", "rlaif"]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def boot_mean(x, n=2000, seed=0):
    a = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = [a[rng.integers(0, a.size, a.size)].mean() for _ in range(n)]
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def generate_policy(cfg, rows, tokenizer, name, batch_size, max_new):
    """Greedy, fixed prompt (the row's own `messages`), same decoding for every policy."""
    model = load_frozen_policy(cfg, name)
    recs = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        gen = batch_generate(model, tokenizer, [r["messages"] for r in chunk], max_prompt_length=512,
                             max_new_tokens=max_new, temperature=0.0, top_p=1.0, do_sample=False)
        for j, r in enumerate(chunk):
            resp, gold = gen["responses"][j], str(r["gold_final"])
            pred = extract_designated_final(resp)
            nums = {n.replace(",", "") for n in NUM_RE.findall(resp)}
            recs.append({
                "index": s + j, "id": str(r.get("prompt_id", r.get("source_index"))), "question": r["question"],
                "gold_final": gold, "response": resp, "response_tokens": int(gen["response_lengths"][j]),
                "truncated": bool(gen["truncated"][j]), "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                "pred": pred, "format_ok": pred is not None, "correct": bool(exact_reward(resp, gold)),
                "gold_in_text": any(numerically_equal(n, gold) for n in nums),
            })
        print(f"  [{name}] {min(s + batch_size, len(rows))}/{len(rows)}", flush=True)
        clear_gpu()
    del model
    return recs


def failure_type(r):
    if r["correct"]:
        return "correct"
    if not r["format_ok"]:
        return "truncated_no_final" if r["truncated"] else "no_designated_final"
    return "wrong_final_gold_in_text" if r["gold_in_text"] else "wrong_final"


def summarize_policy(recs):
    n, k = len(recs), sum(r["correct"] for r in recs)
    lo, hi = wilson(k, n)
    toks = [r["response_tokens"] for r in recs]
    return {"n": n, "accuracy": k / n, "accuracy_ci": [lo, hi], "format_compliance": float(np.mean([r["format_ok"] for r in recs])),
            "len_tokens_mean": float(np.mean(toks)), "len_tokens_std": float(np.std(toks)),
            "frac_truncated": float(np.mean([r["truncated"] for r in recs])),
            "failure_types": dict(Counter(failure_type(r) for r in recs))}


def pairwise_vs_sft(judge, sft, other):
    out = []
    for i, (s, o) in enumerate(zip(sft, other)):
        res = judge.compare(s["question"], o["response"], s["response"])   # A = the trained policy
        out.append({"index": i, "id": s["id"], "judge": res, "score": {"A": 1.0, "TIE": 0.5, "B": 0.0}[res],
                    "verifier_diff": int(o["correct"]) - int(s["correct"]),
                    "policy_correct": o["correct"], "sft_correct": s["correct"]})
        if (i + 1) % 25 == 0:
            print(f"    judged {i + 1}/{len(sft)}", flush=True)
    return out


def summarize_pairwise(recs):
    m, lo, hi = boot_mean([r["score"] for r in recs])
    vp = np.sign([r["verifier_diff"] for r in recs])
    jp = np.array([1 if r["judge"] == "A" else (-1 if r["judge"] == "B" else 0) for r in recs])
    disc = vp != 0
    out = {"n": len(recs), "win_rate_vs_sft": m, "win_rate_ci": [lo, hi], "wins": int((jp == 1).sum()),
           "ties": int((jp == 0).sum()), "losses": int((jp == -1).sum()),
           "verifier_judge_agreement_3way": float((vp == jp).mean()), "n_verifier_discordant": int(disc.sum())}
    if disc.any():
        out.update({"discordant_judge_picks_verifier_choice": float((jp[disc] == vp[disc]).mean()),
                    "discordant_judge_tie": float((jp[disc] == 0).mean()),
                    "discordant_judge_opposes_verifier": float((jp[disc] == -vp[disc]).mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer", "both"], default="both")
    ap.add_argument("--stage", choices=["generate", "judge", "all"], default="all")
    ap.add_argument("--limit", type=int, help="quick test: only the first N problems")
    ap.add_argument("--outdir", default="task5_feedback")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    datasets = ["gsm", "transfer"] if args.dataset == "both" else [args.dataset]
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / args.outdir
    tok = load_tokenizer(cfg["base_model"])
    rows_by = {d: read_jsonl(dataset_path(cfg, d))[: args.limit] if args.limit else read_jsonl(dataset_path(cfg, d)) for d in datasets}
    print("Policies:", POLICIES, "| rows:", {d: len(r) for d, r in rows_by.items()})

    if args.stage in ("generate", "all"):
        for d in datasets:
            for name in POLICIES:
                out = outdir / f"{d}_generated_{name}.jsonl"
                if out.exists() and not args.force:
                    print(f"[skip] {out.name} exists")
                    continue
                t0 = time.perf_counter()
                recs = generate_policy(cfg, rows_by[d], tok, name, args.batch_size, int(cfg["math_max_new_tokens"]))
                write_jsonl(out, recs)
                print(f"[{d}/{name}] saved {len(recs)} responses in {time.perf_counter() - t0:.0f}s", flush=True)
                gc.collect()
                clear_gpu()

    if args.stage in ("judge", "all"):
        judge = PairwiseAIJudge(cfg, f"{cfg['results_dir']}/{args.outdir}/judge_cache.json")
        print("Fixed pairwise judge loaded", flush=True)
        for d in datasets:
            sft = read_jsonl(outdir / f"{d}_generated_sft.jsonl")
            for name in ("rlvr", "rlaif"):
                out = outdir / f"{d}_pairwise_{name}_vs_sft.jsonl"
                if out.exists() and not args.force:
                    print(f"[skip] {out.name} exists")
                    continue
                t0 = time.perf_counter()
                recs = pairwise_vs_sft(judge, sft, read_jsonl(outdir / f"{d}_generated_{name}.jsonl"))
                write_jsonl(out, recs)
                print(f"[{d}/{name} vs sft] {len(recs)} comparisons in {time.perf_counter() - t0:.0f}s", flush=True)

    for d in datasets:
        summary = {"dataset": d, "policies": {}, "pairwise_vs_sft": {}}
        for name in POLICIES:
            p = outdir / f"{d}_generated_{name}.jsonl"
            if p.exists():
                summary["policies"][name] = summarize_policy(read_jsonl(p))
        for name in ("rlvr", "rlaif"):
            p = outdir / f"{d}_pairwise_{name}_vs_sft.jsonl"
            if p.exists():
                summary["pairwise_vs_sft"][name] = summarize_pairwise(read_jsonl(p))
        save_json(outdir / f"{d}_summary.json", summary)
        print(f"\n== {d} ==")
        for name, s in summary["policies"].items():
            print(f"  {name:<6} acc {s['accuracy']:.3f} [{s['accuracy_ci'][0]:.3f},{s['accuracy_ci'][1]:.3f}] format {s['format_compliance']:.3f} "
                  f"len {s['len_tokens_mean']:.0f}+-{s['len_tokens_std']:.0f} trunc {s['frac_truncated']:.3f} failures {s['failure_types']}")
        for name, s in summary["pairwise_vs_sft"].items():
            print(f"  {name:<6} vs SFT: win rate {s['win_rate_vs_sft']:.3f} [{s['win_rate_ci'][0]:.3f},{s['win_rate_ci'][1]:.3f}] "
                  f"(W/T/L {s['wins']}/{s['ties']}/{s['losses']}) agreement(3-way) {s['verifier_judge_agreement_3way']:.3f} "
                  f"discordant n={s['n_verifier_discordant']}")


if __name__ == "__main__":
    main()