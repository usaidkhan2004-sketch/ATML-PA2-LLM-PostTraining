from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean
from common.models import (
    clear_gpu,
    get_device,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", diag_length_grads: bool = True):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    tok, policy, opt = bundle["tokenizer"], bundle["policy"], bundle["optimizer"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    params = trainable_parameters(policy)
    K, eps, beta = int(cfg["num_generations"]), float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_prompt, max_comp = int(cfg["max_prompt_length"]), int(cfg["max_completion_length"])
    max_norm, n_updates = float(cfg["max_grad_norm"]), int(cfg["updates"])
    epochs, g = int(cfg.get("policy_epochs", 1)), cfg["generation"]
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))
    tol = 1e-6   # same numerical tolerance as the advantage epsilon in the released helper

    prompts = bundle["prompt_rows"][:n_updates]
    tok_trunc = sum(len(tok.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True)) > max_prompt for r in prompts)
    log_path = repo_path(cfg["results_dir"]) / f"{run_name}_train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    print(f"run={run_name} loss_type={loss_type} updates={n_updates} K={K} eps={eps} beta={beta} "
          f"({tok_trunc} of {len(prompts)} prompts exceed {max_prompt} tokens and are truncated, as in the release)", flush=True)

    policy.eval()    # no LoRA dropout (as in the PPO continuation)
    t_start = time.perf_counter()
    peak_gb, total_gen_tokens, n_uninformative = 0.0, 0, 0

    def mem():
        return torch.mps.driver_allocated_memory() / 2**30 if device.type == "mps" else 0.0

    for u, row in enumerate(prompts, start=1):
        t_u = time.perf_counter()
        set_seed(int(cfg["seed"]) + 1000 * u)
        msgs = prompt_messages(row)

        with torch.no_grad():
            b = batch_generate(policy, tok, [msgs] * K, max_prompt, max_comp, temperature=float(g["temperature"]),
                               top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
            # generation runs under inference_mode: clone so these can be used in autograd-tracked updates
            seq, attn, pw = b["sequences"].clone(), b["attention_mask"].clone(), b["prompt_width"]
            resp, rmask = b["response_ids"].clone(), b["response_mask"].float().clone()
            rewards = score_reward_pairs(rm, rm_tok, [msgs] * K, b["responses"], max_length=int(cfg.get("reward_max_length", 1280))).float()
            old_lp, _ = response_token_logprobs(policy, seq, attn, pw, resp)
            with reference_mode(policy):
                ref_lp, _ = response_token_logprobs(policy, seq, attn, pw, resp)
            adv = group_relative_advantages(rewards, torch.zeros(K, dtype=torch.long, device=rewards.device))
            tmask = mask_truncated_sequences(rmask, b["truncated"]) if mask_trunc else rmask
            kl_tok = masked_mean(old_lp - ref_lp, rmask).item()
        lens = rmask.sum(-1)
        total_gen_tokens += int(lens.sum().item())
        r_std = rewards.std(unbiased=False).item()
        informative = r_std > tol
        n_uninformative += int(not informative)

        entropy, diag = float("nan"), {}
        stats = {"loss": [], "gn": [], "clip": [], "kl_k3": []}
        for ep in range(epochs):
            opt.zero_grad(set_to_none=True)
            new_lp, logits = response_token_logprobs(policy, seq, attn, pw, resp)
            loss, info = grpo_policy_loss(new_lp, old_lp, adv, tmask, ref_lp, eps, beta, loss_type, max_comp)
            if ep == 0:
                with torch.no_grad():
                    ent = []
                    for j in range(K):
                        lpf = F.log_softmax(logits[j].detach().float(), dim=-1)
                        ent.append(-(lpf.exp() * lpf).sum(-1))
                        del lpf
                    entropy = masked_mean(torch.stack(ent), rmask).item()
                if diag_length_grads:
                    # length-conditioned gradient statistic: gradient norm of the shortest vs the longest
                    # (unmasked) completion's own loss term under the chosen sequence normalization
                    ratio = torch.exp(new_lp - old_lp)
                    obj = torch.minimum(ratio * adv[:, None], ratio.clamp(1 - eps, 1 + eps) * adv[:, None])
                    tok_sum = (obj * tmask).sum(-1)
                    norm = tmask.sum(-1).clamp_min(1.0) if loss_type == "grpo" else torch.full_like(tok_sum, float(max_comp))
                    terms = -(tok_sum / norm) / K
                    tl = tmask.sum(-1)
                    valid = [k for k in range(K) if tl[k].item() > 0]
                    if len(valid) >= 2:
                        ks, kg = min(valid, key=lambda k: tl[k].item()), max(valid, key=lambda k: tl[k].item())
                        if tl[ks].item() != tl[kg].item():
                            def gnorm(k):
                                gr = torch.autograd.grad(terms[k], params, retain_graph=True, allow_unused=True)
                                return float(sum((x.float() ** 2).sum() for x in gr if x is not None).sqrt())
                            diag = {"len_short": int(tl[ks].item()), "len_long": int(tl[kg].item()),
                                    "adv_short": adv[ks].item(), "adv_long": adv[kg].item(),
                                    "gnorm_short": gnorm(ks), "gnorm_long": gnorm(kg)}
            loss.backward()
            peak_gb = max(peak_gb, mem())
            gn = torch.nn.utils.clip_grad_norm_(params, max_norm).item()
            opt.step()
            stats["loss"].append(loss.item()); stats["gn"].append(gn)
            stats["clip"].append(info["clip_fraction"].item()); stats["kl_k3"].append(info["sampled_kl"].item())
            del new_lp, logits, loss, info
            clear_gpu()

        rec = {
            "update": u, "prompt_id": row.get("prompt_id"), "source_index": row.get("source_index"),
            "reward_mean": rewards.mean().item(), "reward_std_in_group": r_std, "informative_group": bool(informative),
            "rewards": [round(x, 4) for x in rewards.tolist()], "advantages": [round(x, 4) for x in adv.tolist()],
            "kl_token_mean": kl_tok, "kl_k3": float(np.mean(stats["kl_k3"])), "entropy": entropy,
            "completion_tokens": [int(x) for x in lens.tolist()], "length_mean": lens.mean().item(),
            "frac_truncated": float(np.mean(b["truncated"])), "masked_token_fraction": 1.0 - (tmask.sum() / rmask.sum().clamp_min(1.0)).item(),
            "policy_loss": float(np.mean(stats["loss"])), "grad_norm": float(np.mean(stats["gn"])),
            "clip_fraction": float(np.mean(stats["clip"])), "update_seconds": time.perf_counter() - t_u,
            "elapsed_s": time.perf_counter() - t_start, "loss_type": loss_type, **diag,
        }
        append_jsonl(log_path, rec)
        print(f"update {u}/{n_updates} reward {rec['reward_mean']:.3f} std {r_std:.3f} KL {kl_tok:.4f} "
              f"len {rec['length_mean']:.0f} trunc {rec['frac_truncated']:.2f} ent {entropy:.3f} "
              f"loss {rec['policy_loss']:.4f} gn {rec['grad_norm']:.2f} {rec['update_seconds']:.0f}s"
              + (f" | gnorm short/long {diag['gnorm_short']:.3f}/{diag['gnorm_long']:.3f} (len {diag['len_short']}/{diag['len_long']})" if diag else ""),
              flush=True)
        del b, seq, attn, resp, rmask, tmask, old_lp, ref_lp, adv, rewards
        clear_gpu()

    policy.save_pretrained(str(out))
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_train_summary.json", {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "num_generations": K, "clip_epsilon": eps,
        "kl_beta": beta, "seed": int(cfg["seed"]), "prompt_ids": [r.get("prompt_id") for r in prompts],
        "prompts_truncated_to_max_prompt_length": tok_trunc, "uninformative_group_fraction": n_uninformative / max(n_updates, 1),
        "total_generated_tokens": total_gen_tokens, "wall_clock_s": time.perf_counter() - t_start,
        "peak_mps_driver_gb_sampled": peak_gb, "adapter": str(out),
    })
    print(f"saved adapter to {out} | wall {time.perf_counter() - t_start:.0f}s | peak driver GB (sampled) {peak_gb:.1f} | "
          f"uninformative groups {n_uninformative}/{n_updates}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--no-diag", action="store_true", help="skip the length-conditioned gradient diagnostic")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, not args.no_diag)


if __name__ == "__main__":
    main()