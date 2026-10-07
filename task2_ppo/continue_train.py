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
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def _value_in_fp32(model):
    """AdamW on fp16 weights can underflow; keep every trainable critic parameter and the head in fp32."""
    def cast_in(_mod, args):
        return tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)

    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "score" in name:
            mod.float()
            mod.register_forward_pre_hook(cast_in)


def select_prompts(tokenizer, rows, n, max_prompt_length):
    """First n prompts in file order. As in the release, prompts longer than max_prompt_length are
    truncated by batch_generate (not skipped); we only count how many are affected."""
    chosen = rows[:n]
    n_truncated = sum(
        len(tokenizer.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True)) > max_prompt_length
        for r in chosen
    )
    return chosen, n_truncated


def _explained_variance(returns, values, mask):
    m = mask.bool()
    r, v = returns[m], values[m]
    if r.numel() < 2 or r.var() < 1e-8:
        return float("nan"), float("nan")
    ev = 1.0 - ((r - v).var() / r.var()).item()
    c = torch.corrcoef(torch.stack([r, v]))[0, 1].item() if v.var() > 1e-12 else float("nan")
    return ev, c


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    tok, policy, value_model = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    p_opt, v_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    _value_in_fp32(value_model)

    eps, beta_kl = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    max_norm, v_coef = float(cfg["max_grad_norm"]), float(cfg["value_coef"])
    penalty = float(cfg["missing_eos_penalty"])
    max_new, max_prompt = int(cfg["max_response_length"]), int(cfg["max_prompt_length"])
    n_updates, epochs, g = int(cfg["updates"]), int(cfg["ppo_epochs"]), cfg["generation"]

    prompts, n_trunc = select_prompts(tok, bundle["prompt_rows"], n_updates, max_prompt)
    log_path = repo_path(cfg["results_dir"]) / f"{run_name}_train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    print(f"run={run_name} updates={n_updates} eps={eps} beta_kl={beta_kl} epochs={epochs} "
          f"({n_trunc} of {len(prompts)} prompts exceed {max_prompt} tokens and are truncated, as in the release)", flush=True)

    policy.eval()          # no LoRA dropout, so rho == 1 at the first epoch
    value_model.eval()
    t_start = time.perf_counter()
    peak_gb, total_gen_tokens = 0.0, 0

    def mem():
        return torch.mps.driver_allocated_memory() / 2**30 if device.type == "mps" else 0.0

    for u, row in enumerate(prompts, start=1):
        t_u = time.perf_counter()
        set_seed(int(cfg["seed"]) + 1000 * u)
        msgs = prompt_messages(row)

        # ---- rollout (old policy) ----
        with torch.no_grad():
            b = batch_generate(policy, tok, [msgs], max_prompt, max_new, temperature=float(g["temperature"]),
                               top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]))
            # generation runs under inference_mode: clone so these can be used in autograd-tracked updates
            seq, attn, pw = b["sequences"].clone(), b["attention_mask"].clone(), b["prompt_width"]
            resp, mask = b["response_ids"].clone(), b["response_mask"].float().clone()
            R = resp.shape[1]
            old_lp, _ = response_token_logprobs(policy, seq, attn, pw, resp)
            with reference_mode(policy):
                ref_lp, _ = response_token_logprobs(policy, seq, attn, pw, resp)
            raw = score_reward_pairs(rm, rm_tok, [msgs], b["responses"], max_length=int(cfg["reward_max_length"])).float()
            eos = b["terminated_with_eos"][0]
            eff = raw - (0.0 if eos else penalty)
            vals = token_values(value_model, seq, attn).float()[:, pw - 1: pw - 1 + R]
            rewards = shaped_rewards(eff, old_lp, ref_lp, mask, beta_kl)
            adv, ret = compute_gae(rewards, vals, mask, gamma, lam)
            adv_n = normalize_advantages(adv, mask)
            kl_tok = masked_mean(old_lp - ref_lp, mask).item()
            ev, corr = _explained_variance(ret, vals, mask)
        n_tok = int(mask.sum().item())
        total_gen_tokens += n_tok

        # ---- PPO epochs ----
        stats = {"loss_pi": [], "loss_v": [], "gn_p": [], "gn_v": [], "clip": [], "ratio_dev": []}
        entropy = float("nan")
        for ep in range(epochs):
            p_opt.zero_grad(set_to_none=True)
            new_lp, logits = response_token_logprobs(policy, seq, attn, pw, resp)
            loss_pi, ratio, clip_frac = ppo_policy_loss(new_lp, old_lp, adv_n, mask, eps)
            if ep == 0:
                with torch.no_grad():
                    lp_full = F.log_softmax(logits.detach().float(), dim=-1)
                    entropy = masked_mean(-(lp_full.exp() * lp_full).sum(-1), mask).item()
                    del lp_full
            loss_pi.backward()
            peak_gb = max(peak_gb, mem())
            gn_p = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm).item()
            p_opt.step()

            v_opt.zero_grad(set_to_none=True)
            new_vals = token_values(value_model, seq, attn).float()[:, pw - 1: pw - 1 + R]
            loss_v = value_mse_loss(new_vals, ret.detach(), mask)
            (v_coef * loss_v).backward()
            gn_v = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_norm).item()
            v_opt.step()

            stats["loss_pi"].append(loss_pi.item()); stats["loss_v"].append(loss_v.item())
            stats["gn_p"].append(gn_p); stats["gn_v"].append(gn_v); stats["clip"].append(clip_frac.item())
            stats["ratio_dev"].append(masked_mean((ratio - 1.0).abs(), mask).item())
            del new_lp, logits, loss_pi, new_vals, loss_v
            clear_gpu()

        rec = {
            "update": u, "prompt_id": row.get("prompt_id"), "source_index": row.get("source_index"),
            "raw_reward": raw.item(), "effective_reward": eff.item(), "kl_token_mean": kl_tok,
            "kl_seq_sum": (old_lp - ref_lp).mul(mask).sum().item(),
            "response_tokens": n_tok, "terminated_with_eos": bool(eos), "entropy": entropy,
            "policy_loss": float(np.mean(stats["loss_pi"])), "value_loss": float(np.mean(stats["loss_v"])),
            "grad_norm_policy": float(np.mean(stats["gn_p"])), "grad_norm_value": float(np.mean(stats["gn_v"])),
            "clip_fraction": float(np.mean(stats["clip"])), "clip_fraction_last_epoch": stats["clip"][-1],
            "mean_abs_ratio_dev_last_epoch": stats["ratio_dev"][-1],
            "critic_explained_variance": ev, "critic_corr": corr,
            "mean_value": (vals * mask).sum().item() / max(n_tok, 1), "mean_return": (ret * mask).sum().item() / max(n_tok, 1),
            "update_seconds": time.perf_counter() - t_u, "elapsed_s": time.perf_counter() - t_start,
            "eps": eps, "beta_kl": beta_kl,
        }
        append_jsonl(log_path, rec)
        print(f"update {u}/{n_updates} reward {rec['raw_reward']:.3f} (eff {rec['effective_reward']:.3f}) KL {kl_tok:.4f} "
              f"len {n_tok} ent {entropy:.3f} clip {rec['clip_fraction']:.3f} Lpi {rec['policy_loss']:.4f} "
              f"Lv {rec['value_loss']:.3f} EV {ev:.2f} gn {rec['grad_norm_policy']:.2f}/{rec['grad_norm_value']:.2f} "
              f"{rec['update_seconds']:.0f}s", flush=True)
        del b, seq, attn, resp, mask, old_lp, ref_lp, vals, rewards, adv, ret, adv_n
        clear_gpu()

    policy.save_pretrained(str(out))
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_train_summary.json", {
        "run_name": run_name, "updates": n_updates, "ppo_epochs": epochs, "clip_epsilon": eps, "kl_beta": beta_kl,
        "seed": int(cfg["seed"]), "prompt_ids": [r.get("prompt_id") for r in prompts],
        "prompts_truncated_to_max_prompt_length": n_trunc,
        "total_generated_tokens": total_gen_tokens, "wall_clock_s": time.perf_counter() - t_start,
        "peak_mps_driver_gb_sampled": peak_gb, "adapter": str(out),
    })
    print(f"saved adapter to {out} | wall {time.perf_counter() - t_start:.0f}s | peak driver GB (sampled) {peak_gb:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()