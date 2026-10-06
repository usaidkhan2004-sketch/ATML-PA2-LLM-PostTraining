from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import get_device, load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_fitting_rows(tokenizer, rows, max_length):
    """Drop pairs whose prompt alone does not fit in max_length (the rule enforced by encode_prompt_response)."""
    kept, skipped = [], []
    for i, row in enumerate(rows):
        n = len(tokenizer.apply_chat_template(
            prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
        if n >= max_length:
            skipped.append(i)
        else:
            kept.append(row)
    return kept, skipped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    max_len = int(cfg["max_sequence_length"])

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, skipped = filter_fitting_rows(tokenizer, read_jsonl(path), max_len)
    save_json(repo_path(cfg["results_dir"]) / f"skipped_{Path(path).stem}.json", {
        "dataset": path, "max_sequence_length": max_len,
        "kept": len(rows), "skipped_row_indices": skipped,
    })
    print(f"filtered {path}: kept {len(rows)}, skipped {len(skipped)} (prompt >= {max_len} tokens)")
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, max_len),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _mean(xs):
    return sum(xs) / max(len(xs), 1)


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, loader, optimizer = bundle["cfg"], bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = repo_path(cfg["results_dir"]) / f"{run_name}_train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    device = get_device()
    accum = int(cfg["grad_accum_steps"])
    max_norm = float(cfg["max_grad_norm"])
    n_micro = len(loader)
    total_steps = math.ceil(n_micro / accum)
    print(f"run={run_name} beta={beta} examples={len(bundle['rows'])} micro_batches={n_micro} optimizer_steps={total_steps}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    window = {"loss": [], "acc": [], "margin": []}
    step, peak_gb = 0, 0.0

    for micro, (chosen, rejected) in enumerate(loader, start=1):
        chosen, rejected = _to_device(chosen, device), _to_device(rejected, device)

        with torch.no_grad(), reference_mode(model):
            ref_c, _, _ = response_sequence_logprobs(model, chosen)
            ref_r, _, _ = response_sequence_logprobs(model, rejected)

        pol_c, _, _ = response_sequence_logprobs(model, chosen)
        pol_r, _, _ = response_sequence_logprobs(model, rejected)
        loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)

        group = min(accum, n_micro - step * accum)
        (loss / group).backward()

        margin = ((pol_c - ref_c) - (pol_r - ref_r)).detach().mean().item()
        window["loss"].append(loss.item())
        window["acc"].append(diag["preference_accuracy"].item())
        window["margin"].append(margin)

        if micro % accum == 0 or micro == n_micro:
            gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - t0
            if device.type == "mps":
                peak_gb = max(peak_gb, torch.mps.driver_allocated_memory() / 2**30)
                torch.mps.empty_cache()
            rec = {
                "step": step, "examples_seen": min(step * accum * int(cfg["batch_size"]), len(bundle["rows"])),
                "loss": _mean(window["loss"]), "pref_acc": _mean(window["acc"]),
                "margin": _mean(window["margin"]), "grad_norm": gnorm,
                "elapsed_s": elapsed, "beta": beta,
            }
            append_jsonl(log_path, rec)
            print(f"step {step}/{total_steps} loss {rec['loss']:.4f} acc {rec['pref_acc']:.3f} "
                  f"margin {rec['margin']:.3f} gnorm {gnorm:.2f} {elapsed / step:.1f}s/step", flush=True)
            window = {"loss": [], "acc": [], "margin": []}
            if step % 25 == 0:
                model.save_pretrained(str(output))

    model.save_pretrained(str(output))
    save_json(repo_path(cfg["results_dir"]) / f"{run_name}_train_summary.json", {
        "run_name": run_name, "beta": beta, "seed": int(cfg["seed"]),
        "dataset": dataset_path or cfg["paths"]["dpo_standard_train"],
        "examples": len(bundle["rows"]), "optimizer_steps": step,
        "wall_clock_s": time.perf_counter() - t0, "peak_mps_driver_gb": peak_gb,
        "adapter": str(output),
    })
    print(f"saved adapter to {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()