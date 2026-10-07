from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, get_device, load_policy, load_reward_model, load_tokenizer, reference_mode


def _stats(x):
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "median": float(med),
            "iqr": float(q3 - q1), "min": float(a.min()), "max": float(a.max())}


@torch.no_grad()
def evaluate_rl_policy(config_path, adapter, name, gen_batch=16, lp_chunk=4, limit=None, max_new=None):
    """Common held-out protocol for PPO/GRPO policies: one sampled response per held-out prompt;
    learned reward, sampled-token KL vs the frozen reference (adapter disabled), true token entropy, length."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    results_dir = repo_path(cfg["results_dir"])
    adapter_arg = None if str(adapter).lower() == "none" else adapter
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter_arg, trainable=False)
    rm, rm_tok = load_reward_model(cfg)

    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if limit:
        rows = rows[:limit]
    max_prompt = int(cfg["max_prompt_length"])
    max_new = int(max_new or cfg.get("eval_max_response_length", cfg.get("cache_generation_cap", 768)))
    penalty = float(cfg.get("missing_eos_penalty", 0.0))
    g = cfg["generation"]
    device = get_device()
    t0 = time.perf_counter()
    peak_gb = 0.0
    recs, kl_sum, tok_sum, ent_sum = [], 0.0, 0.0, 0.0
    print(f"[{name}] evaluating on {len(rows)} held-out prompts, max_new_tokens={max_new}", flush=True)

    for i in range(0, len(rows), gen_batch):
        chunk = rows[i:i + gen_batch]
        msgs = [prompt_messages(r) for r in chunk]
        set_seed(int(cfg["seed"]) + i)
        b = batch_generate(policy, tok, msgs, max_prompt, max_new, temperature=float(g["temperature"]),
                           top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
        seq, attn, pw = b["sequences"], b["attention_mask"], b["prompt_width"]
        resp, mask = b["response_ids"], b["response_mask"].float()
        pol_l, ref_l, ent_l = [], [], []
        for s in range(0, seq.shape[0], lp_chunk):
            sl = slice(s, s + lp_chunk)
            lp, logits = response_token_logprobs(policy, seq[sl], attn[sl], pw, resp[sl])
            ent = []
            for j in range(lp.shape[0]):
                lpf = F.log_softmax(logits[j].float(), dim=-1)
                ent.append(-(lpf.exp() * lpf).sum(-1))
                del lpf
            ent_l.append(torch.stack(ent))
            pol_l.append(lp)
            del logits
            with reference_mode(policy):
                rlp, _ = response_token_logprobs(policy, seq[sl], attn[sl], pw, resp[sl])
            ref_l.append(rlp)
        pol_lp, ref_lp, ent = torch.cat(pol_l), torch.cat(ref_l), torch.cat(ent_l)
        diff = (pol_lp - ref_lp) * mask
        raw = score_reward_pairs(rm, rm_tok, msgs, b["responses"], max_length=int(cfg.get("reward_max_length", 1280))).float()
        if device.type == "mps":
            peak_gb = max(peak_gb, torch.mps.driver_allocated_memory() / 2**30)

        kl_sum += diff.sum().item(); tok_sum += mask.sum().item(); ent_sum += (ent * mask).sum().item()
        for j, r in enumerate(chunk):
            n = int(mask[j].sum().item())
            eos = bool(b["terminated_with_eos"][j])
            recs.append({
                "prompt_id": r.get("prompt_id"), "source_index": r.get("source_index"),
                "response": b["responses"][j], "response_tokens": n,
                "terminated_with_eos": eos, "truncated": bool(b["truncated"][j]),
                "reward_raw": raw[j].item(), "reward_effective": raw[j].item() - (0.0 if eos else penalty),
                "kl_seq_sum": diff[j].sum().item(), "kl_token_mean": diff[j].sum().item() / max(n, 1),
                "entropy_mean": (ent[j] * mask[j]).sum().item() / max(n, 1),
            })
        print(f"  {min(i + gen_batch, len(rows))}/{len(rows)}", flush=True)
        del b, seq, attn, resp, mask, pol_lp, ref_lp, ent, diff
        clear_gpu()

    write_jsonl(results_dir / f"eval_{name}_generations.jsonl", recs)
    summary = {
        "name": name, "adapter": adapter, "config": config_path, "n_prompts": len(recs), "max_new_tokens": max_new,
        "seed": int(cfg["seed"]), "prompt_ids": [r["prompt_id"] for r in recs],
        "reward_raw": _stats([r["reward_raw"] for r in recs]),
        "reward_effective": _stats([r["reward_effective"] for r in recs]),
        "kl_token_mean": kl_sum / max(tok_sum, 1.0), "kl_seq_sum": _stats([r["kl_seq_sum"] for r in recs]),
        "entropy_token_mean": ent_sum / max(tok_sum, 1.0),
        "response_tokens": _stats([r["response_tokens"] for r in recs]),
        "frac_truncated": float(np.mean([r["truncated"] for r in recs])),
        "frac_eos": float(np.mean([r["terminated_with_eos"] for r in recs])),
        "wall_clock_s": time.perf_counter() - t0, "peak_mps_driver_gb_sampled": peak_gb,
    }
    save_json(results_dir / f"eval_{name}.json", summary)
    print(f"[{name}] done in {summary['wall_clock_s']:.0f}s | reward {summary['reward_raw']['mean']:.3f} "
          f"(eff {summary['reward_effective']['mean']:.3f}) KL {summary['kl_token_mean']:.5f} "
          f"entropy {summary['entropy_token_mean']:.3f} len {summary['response_tokens']['mean']:.1f}"
          f"+-{summary['response_tokens']['std']:.1f} trunc {summary['frac_truncated']:.3f}")
    return summary