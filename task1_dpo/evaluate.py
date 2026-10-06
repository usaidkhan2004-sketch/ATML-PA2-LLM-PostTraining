from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
    prompt_messages_from_preference,
    write_jsonl,
)
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import clear_gpu, get_device, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import filter_fitting_rows, make_collate


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _stats(x):
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "median": float(med),
            "iqr": float(q3 - q1), "min": float(a.min()), "max": float(a.max())}


@torch.no_grad()
def score_pairs(model, tokenizer, rows, max_len, beta, device, batch_size=2):
    """Per-pair DPO margin m = [logpi(y+)-logref(y+)] - [logpi(y-)-logref(y-)] (response tokens only)."""
    collate = make_collate(tokenizer, max_len)
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        chosen, rejected = collate(chunk)
        chosen, rejected = _to_device(chosen, device), _to_device(rejected, device)
        pol_c, _, _ = response_sequence_logprobs(model, chosen)
        pol_r, _, _ = response_sequence_logprobs(model, rejected)
        with reference_mode(model):
            ref_c, _, _ = response_sequence_logprobs(model, chosen)
            ref_r, _, _ = response_sequence_logprobs(model, rejected)
        for j, row in enumerate(chunk):
            margin = ((pol_c[j] - ref_c[j]) - (pol_r[j] - ref_r[j])).item()
            rec = {
                "prompt_id": row.get("prompt_id"), "source_index": row.get("source_index"),
                "margin": margin, "correct": bool(margin > 0),
                "loss": float(F.softplus(torch.tensor(-beta * margin)).item()),
                "policy_chosen_logp": pol_c[j].item(), "policy_rejected_logp": pol_r[j].item(),
                "ref_chosen_logp": ref_c[j].item(), "ref_rejected_logp": ref_r[j].item(),
                "score_chosen": row.get("score_chosen"), "score_rejected": row.get("score_rejected"),
            }
            if "length_stratum" in row:
                rec["length_stratum"] = row["length_stratum"]
            out.append(rec)
        if (i // batch_size) % 25 == 0:
            print(f"  pairs {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)
        clear_gpu()
    return out


def summarize_pairs(recs):
    out = {
        "n": len(recs),
        "preference_accuracy": float(np.mean([r["correct"] for r in recs])),
        "dpo_loss": float(np.mean([r["loss"] for r in recs])),
        "mean_margin": float(np.mean([r["margin"] for r in recs])),
    }
    strata = sorted({r["length_stratum"] for r in recs if "length_stratum" in r})
    if strata:
        out["by_stratum"] = {}
        for s in strata:
            sub = [r for r in recs if r.get("length_stratum") == s]
            out["by_stratum"][s] = {
                "n": len(sub),
                "preference_accuracy": float(np.mean([r["correct"] for r in sub])),
                "mean_margin": float(np.mean([r["margin"] for r in sub])),
            }
    return out


@torch.no_grad()
def generate_and_measure(model, tokenizer, rm, rm_tok, items, cfg, max_new, batch_size, max_prompt_len):
    """Sample one response per prompt; measure sampled-token KL vs reference, RM score, length."""
    g = cfg["generation"]
    recs, kl_sum, tok_cnt = [], 0.0, 0.0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        msgs = [m for _, m in chunk]
        b = batch_generate(model, tokenizer, msgs, max_prompt_len, max_new,
                           temperature=float(g["temperature"]), top_p=float(g["top_p"]),
                           do_sample=bool(g["do_sample"]))
        pol_lp, _ = response_token_logprobs(model, b["sequences"], b["attention_mask"], b["prompt_width"], b["response_ids"])
        with reference_mode(model):
            ref_lp, _ = response_token_logprobs(model, b["sequences"], b["attention_mask"], b["prompt_width"], b["response_ids"])
        mask = b["response_mask"]
        diff = (pol_lp - ref_lp) * mask
        seq_kl = diff.sum(-1)
        rewards = score_reward_pairs(rm, rm_tok, msgs, b["responses"], max_length=1024)
        kl_sum += diff.sum().item()
        tok_cnt += mask.sum().item()
        for j, (pid, _) in enumerate(chunk):
            n = b["response_lengths"][j]
            recs.append({
                "prompt_id": pid, "response": b["responses"][j], "response_tokens": n,
                "rm_score": rewards[j].item(), "kl_seq_sum": seq_kl[j].item(),
                "kl_token_mean": seq_kl[j].item() / max(n, 1),
                "terminated_with_eos": b["terminated_with_eos"][j], "truncated": b["truncated"][j],
            })
        print(f"  generated {min(i + batch_size, len(items))}/{len(items)}", flush=True)
        del pol_lp, ref_lp, diff, b
        clear_gpu()
    return recs, kl_sum / max(tok_cnt, 1.0)


def word_limit_eval(model, tokenizer, wl_rows, cfg, max_new, n_samples):
    g = cfg["generation"]
    prompts = [r["messages"] for r in wl_rows]
    recs = []
    for rep in range(n_samples):
        b = batch_generate(model, tokenizer, prompts, 512, max_new,
                           temperature=float(g["temperature"]), top_p=float(g["top_p"]),
                           do_sample=bool(g["do_sample"]))
        for r, resp, n in zip(wl_rows, b["responses"], b["response_lengths"]):
            text = r["messages"][-1]["content"]
            recs.append({
                "prompt_id": r["prompt_id"], "rep": rep, "response": resp, "tokens": n,
                "words": word_count(resp), "limit": parse_word_limit(text),
                "compliant": word_limit_compliance(text, resp),
            })
        clear_gpu()
    return recs


def evaluate(config_path, adapter, name, beta=None, sets=("standard",), limit=None,
             wl_samples=5, gen_batch=8, do_generation=True):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    device = get_device()
    beta = float(cfg["beta"] if beta is None else beta)
    max_len = int(cfg["max_sequence_length"])
    results_dir = repo_path(cfg["results_dir"])
    adapter_arg = None if str(adapter).lower() == "none" else adapter
    t0 = time.perf_counter()

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter_arg, trainable=False)
    summary = {"name": name, "adapter": adapter, "beta_for_loss": beta, "seed": int(cfg["seed"]),
               "max_sequence_length": max_len, "limit": limit, "pairs": {}}

    paths = {"standard": cfg["paths"]["dpo_standard_eval"], "stratified": cfg["paths"]["dpo_length_eval"]}
    for s in sets:
        rows, skipped = filter_fitting_rows(tokenizer, read_jsonl(paths[s]), max_len)
        save_json(results_dir / f"skipped_{paths[s].split('/')[-1].replace('.jsonl', '')}.json", {
            "dataset": paths[s], "max_sequence_length": max_len, "kept": len(rows), "skipped_row_indices": skipped})
        if limit:
            rows = rows[:limit]
        print(f"[{name}] pair evaluation on '{s}': {len(rows)} pairs (skipped {len(skipped)} long prompts)", flush=True)
        recs = score_pairs(model, tokenizer, rows, max_len, beta, device)
        write_jsonl(results_dir / f"eval_{name}_pairs_{s}.jsonl", recs)
        summary["pairs"][s] = summarize_pairs(recs)
        summary["pairs"][s]["skipped_prompt_too_long"] = len(skipped)

    if do_generation:
        rm, rm_tok = load_reward_model(cfg)
        rows, _ = filter_fitting_rows(tokenizer, read_jsonl(paths["standard"]), max_len)
        if limit:
            rows = rows[:limit]
        items = [(r.get("prompt_id"), prompt_messages_from_preference(r)) for r in rows]
        set_seed(int(cfg["seed"]))
        print(f"[{name}] generation on {len(items)} held-out prompts", flush=True)
        gens, kl_token = generate_and_measure(model, tokenizer, rm, rm_tok, items, cfg,
                                              int(cfg["max_generation_tokens"]), gen_batch, max_len)
        write_jsonl(results_dir / f"eval_{name}_generations.jsonl", gens)
        summary["generation"] = {
            "n_prompts": len(gens), "max_new_tokens": int(cfg["max_generation_tokens"]),
            "prompt_ids": [pid for pid, _ in items],
            "kl_token_mean": kl_token,
            "kl_seq_sum": _stats([g["kl_seq_sum"] for g in gens]),
            "rm_score": _stats([g["rm_score"] for g in gens]),
            "response_tokens": _stats([g["response_tokens"] for g in gens]),
            "frac_truncated": float(np.mean([g["truncated"] for g in gens])),
            "frac_eos": float(np.mean([g["terminated_with_eos"] for g in gens])),
        }
        wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
        set_seed(int(cfg["seed"]))
        wl = word_limit_eval(model, tokenizer, wl_rows, cfg, int(cfg["max_generation_tokens"]), wl_samples)
        write_jsonl(results_dir / f"eval_{name}_wordlimit.jsonl", wl)
        comp = [w["compliant"] for w in wl if w["compliant"] is not None]
        summary["word_limit"] = {
            "n_generations": len(wl), "n_prompts": len(wl_rows), "samples_per_prompt": wl_samples,
            "compliance_rate": float(np.mean(comp)) if comp else None,
            "words": _stats([w["words"] for w in wl]),
            "response_tokens": _stats([w["tokens"] for w in wl]),
        }

    summary["wall_clock_s"] = time.perf_counter() - t0
    save_json(results_dir / f"eval_{name}.json", summary)
    print(f"[{name}] done in {summary['wall_clock_s']:.0f}s -> {results_dir / f'eval_{name}.json'}")
    for s, v in summary["pairs"].items():
        print(f"  {s}: n={v['n']} acc={v['preference_accuracy']:.3f} loss={v['dpo_loss']:.4f} margin={v['mean_margin']:.3f}")
        for st, sv in v.get("by_stratum", {}).items():
            print(f"     {st}: n={sv['n']} acc={sv['preference_accuracy']:.3f}")
    if "generation" in summary:
        g = summary["generation"]
        print(f"  KL(token)={g['kl_token_mean']:.4f} RM={g['rm_score']['mean']:.3f} "
              f"len={g['response_tokens']['mean']:.1f}±{g['response_tokens']['std']:.1f} "
              f"wordlimit={summary['word_limit']['compliance_rate']}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the plain base model")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta the model was trained with (used only for the held-out loss)")
    ap.add_argument("--sets", default="standard", help="comma list of: standard, stratified")
    ap.add_argument("--limit", type=int, help="quick test: only the first N pairs/prompts")
    ap.add_argument("--wl-samples", type=int, default=5)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--no-generation", action="store_true")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.beta, tuple(args.sets.split(",")),
             args.limit, args.wl_samples, args.gen_batch, not args.no_generation)


if __name__ == "__main__":
    main()