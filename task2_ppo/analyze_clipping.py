from __future__ import annotations

import argparse
import csv
import gc

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import save_json
from common.models import clear_gpu, get_device, load_policy, load_tokenizer, trainable_parameters
from task2_ppo.forks import fork_name, run_fork, summarize
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def build_cached_batch(cfg, tok, rows):
    """Rebuild the fixed batch: token ids, cached old/ref log-probs and values, then rewards -> GAE -> advantages."""
    pool = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    items = []
    for r in rows:
        rendered = tok.apply_chat_template(prompt_messages(pool[r["prompt_id"]]), tokenize=False, add_generation_prompt=True)
        p_ids = tok(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]))["input_ids"]
        rs = tok(r["response"], add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if r["terminated_with_eos"] else [])
        n = min(len(rs), len(r["old_logprobs"]))
        items.append((p_ids, rs[:n], r))
    B, R = len(items), max(len(x[1]) for x in items)
    old, ref, val, mask = torch.zeros(B, R), torch.zeros(B, R), torch.zeros(B, R), torch.zeros(B, R)
    term = torch.zeros(B)
    for i, (_, rs, r) in enumerate(items):
        n = len(rs)
        old[i, :n], ref[i, :n], val[i, :n] = r["old_logprobs"][:n].float(), r["ref_logprobs"][:n].float(), r["values"][:n].float()
        mask[i, :n] = 1.0
        term[i] = float(r["effective_terminal_reward"])
    rewards = shaped_rewards(term, old, ref, mask, float(cfg["kl_beta"]))
    adv, ret = compute_gae(rewards, val, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv = normalize_advantages(adv, mask)
    return items, old, mask, adv


def make_chunks(items, tok, old, mask, adv, chunk, device):
    chunks, pad = [], tok.pad_token_id
    for s in range(0, len(items), chunk):
        sub = items[s:s + chunk]
        P, R = max(len(p) for p, _, _ in sub), max(len(rs) for _, rs, _ in sub)
        seq = torch.full((len(sub), P + R), pad, dtype=torch.long)
        attn = torch.zeros((len(sub), P + R), dtype=torch.long)
        for i, (p, rs, _) in enumerate(sub):
            seq[i, P - len(p):P] = torch.tensor(p)
            attn[i, P - len(p):P] = 1
            seq[i, P:P + len(rs)] = torch.tensor(rs)
            attn[i, P:P + len(rs)] = 1
        chunks.append({"seq": seq.to(device), "attn": attn.to(device), "P": P, "resp": seq[:, P:].to(device),
                       "old": old[s:s + chunk, :R].to(device), "adv": adv[s:s + chunk, :R].to(device),
                       "mask": mask[s:s + chunk, :R].to(device)})
    return chunks


def run_pass(policy, chunks, eps, backward, total_tokens):
    """One full-batch pass. Returns token-level statistics of the ratio rho = pi_theta / pi_old on the cached batch."""
    acc = {"tokens": 0.0, "outside": 0.0, "binding": 0.0, "sur_clip": 0.0, "sur_unclip": 0.0, "absdev": 0.0}
    rmax, rmin = -1e9, 1e9
    for c in chunks:
        with torch.set_grad_enabled(backward):
            new_lp, _ = response_token_logprobs(policy, c["seq"], c["attn"], c["P"], c["resp"])
            loss, ratio, _ = ppo_policy_loss(new_lp, c["old"], c["adv"], c["mask"], eps)
            if backward:
                (loss * (c["mask"].sum() / total_tokens)).backward()
        with torch.no_grad():
            m, r, a = c["mask"], ratio.float(), c["adv"]
            s1, s2 = r * a, r.clamp(1 - eps, 1 + eps) * a
            acc["tokens"] += m.sum().item()
            acc["outside"] += (((r < 1 - eps) | (r > 1 + eps)).float() * m).sum().item()
            acc["binding"] += ((s2 < s1).float() * m).sum().item()
            acc["sur_clip"] += (torch.minimum(s1, s2) * m).sum().item()
            acc["sur_unclip"] += (s1 * m).sum().item()
            acc["absdev"] += ((r - 1).abs() * m).sum().item()
            rmax, rmin = max(rmax, r[m.bool()].max().item()), min(rmin, r[m.bool()].min().item())
        del new_lp, loss
        clear_gpu()
    t = acc["tokens"]
    return {"outside_fraction": acc["outside"] / t, "binding_fraction": acc["binding"] / t,
            "surrogate_clipped": acc["sur_clip"] / t, "surrogate_unclipped": acc["sur_unclip"] / t,
            "mean_abs_ratio_dev": acc["absdev"] / t, "ratio_max": rmax, "ratio_min": rmin, "tokens": int(t)}


def geometry_study(cfg, config_path, steps, lr_scales, chunk, limit):
    tok, device = load_tokenizer(cfg["base_model"]), get_device()
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    if limit:
        rows = rows[:limit]
    items, old, mask, adv = build_cached_batch(cfg, tok, rows)
    chunks = make_chunks(items, tok, old, mask, adv, chunk, device)
    total_tokens = mask.sum().item()
    print(f"cached batch: {len(rows)} rollouts, {int(total_tokens)} response tokens", flush=True)

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    policy.eval()   # no LoRA dropout, so any ratio deviation comes from the update itself (and numerics)
    params = trainable_parameters(policy)
    init = [p.detach().clone() for p in params]
    max_norm = float(cfg["max_grad_norm"])

    # (a) noise floor: locally recomputed log-probs vs the CACHED old log-probs at step 0 (no update yet).
    noise = []
    for eps in cfg["clip_values"]:
        st = run_pass(policy, chunks, float(eps), False, total_tokens)
        noise.append({"kind": "vs_cached_old_logprobs_step0", "eps": float(eps), **st})
        print(f"[noise floor, cached old log-probs] eps {eps:<5}: outside {st['outside_fraction']:.4f} binding {st['binding_fraction']:.4f} "
              f"mean|rho-1| {st['mean_abs_ratio_dev']:.5f}", flush=True)

    # (b) main study: old log-probs recomputed on this hardware from the unchanged midpoint, so rho == 1 at step 0.
    with torch.no_grad():
        for c in chunks:
            lp, _ = response_token_logprobs(policy, c["seq"], c["attn"], c["P"], c["resp"])
            c["old"] = lp.detach().float()
            del lp
            clear_gpu()

    records = []
    for scale in lr_scales:
        for eps in cfg["clip_values"]:
            for p, i0 in zip(params, init):
                p.data.copy_(i0)
            opt = torch.optim.AdamW(params, lr=float(cfg["policy_learning_rate"]) * scale)
            for step in range(steps + 1):
                if step < steps:
                    opt.zero_grad(set_to_none=True)
                    st = run_pass(policy, chunks, float(eps), True, total_tokens)
                    gn = torch.nn.utils.clip_grad_norm_(params, max_norm).item()
                    opt.step()
                else:
                    st = run_pass(policy, chunks, float(eps), False, total_tokens)
                    gn = float("nan")
                rec = {"lr_scale": scale, "lr": float(cfg["policy_learning_rate"]) * scale, "eps": float(eps), "step": step, "grad_norm": gn, **st}
                records.append(rec)
                print(f"lr x{scale:<4} eps {eps:<5} step {step}: outside {st['outside_fraction']:.4f} binding {st['binding_fraction']:.4f} "
                      f"surrogate {st['surrogate_clipped']:.5f} (unclipped {st['surrogate_unclipped']:.5f}) "
                      f"mean|rho-1| {st['mean_abs_ratio_dev']:.5f} rho in [{st['ratio_min']:.3f},{st['ratio_max']:.3f}]", flush=True)
    rdir = repo_path(cfg["results_dir"])
    save_json(rdir / "clipping_cached_batch.json", {"noise_floor": noise, "updates": records})
    with (rdir / "clipping_cached_batch.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    del policy
    gc.collect()
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr-scales", default="1,30")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--skip-geometry", action="store_true")
    ap.add_argument("--skip-forks", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required epsilon values:", cfg["clip_values"])

    if not args.skip_geometry:
        geometry_study(cfg, args.config, 2 if args.smoke else args.steps,
                       [1.0] if args.smoke else [float(x) for x in args.lr_scales.split(",")],
                       args.chunk, 4 if args.smoke else None)

    if not args.skip_forks:
        prefix = "smoke_" if args.smoke else ""
        eps_values = cfg["clip_values"][:1] if args.smoke else cfg["clip_values"]
        for eps in eps_values:
            run_fork(args.config, float(eps), float(cfg["kl_beta"]), updates=1 if args.smoke else None,
                     eval_limit=4 if args.smoke else None, prefix=prefix)
        if not args.smoke:
            summarize(args.config, [("midpoint (start)", "midpoint"), ("SFT base", "sft"), ("standard 20 updates", "standard")]
                      + [(f"fork eps={e}", fork_name(float(e), float(cfg["kl_beta"]))) for e in eps_values], "clipping_summary")


if __name__ == "__main__":
    main()