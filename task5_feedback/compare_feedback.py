from __future__ import annotations

import argparse
import csv

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--outdir", default="task5_feedback")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rdir = repo_path(cfg["results_dir"]) / args.outdir
    gsm = load_json(rdir / "gsm_summary.json")
    tr = load_json(rdir / "transfer_summary.json")
    diag = load_json(rdir / "diagnostic_summary.json")

    rows = []
    for name in ("sft", "rlvr", "rlaif"):
        g, t = gsm["policies"][name], tr["policies"][name]
        row = {"policy": name, "gsm_accuracy": g["accuracy"], "gsm_format": g["format_compliance"], "gsm_len": g["len_tokens_mean"],
               "svamp_accuracy": t["accuracy"], "svamp_format": t["format_compliance"], "svamp_len": t["len_tokens_mean"],
               "accuracy_drop_gsm_to_svamp": g["accuracy"] - t["accuracy"],
               "len_change_gsm_to_svamp": t["len_tokens_mean"] - g["len_tokens_mean"]}
        if name in gsm["pairwise_vs_sft"]:
            pg, pt = gsm["pairwise_vs_sft"][name], tr["pairwise_vs_sft"][name]
            row.update({"gsm_win_vs_sft": pg["win_rate_vs_sft"], "svamp_win_vs_sft": pt["win_rate_vs_sft"],
                        "win_rate_drop": pg["win_rate_vs_sft"] - pt["win_rate_vs_sft"],
                        "gsm_agreement_3way": pg["verifier_judge_agreement_3way"], "gsm_discordant_n": pg["n_verifier_discordant"],
                        "gsm_discordant_judge_picks_verifier": pg.get("discordant_judge_picks_verifier_choice"),
                        "svamp_agreement_3way": pt["verifier_judge_agreement_3way"]})
        row["gsm_failures"], row["svamp_failures"] = g["failure_types"], t["failure_types"]
        rows.append(row)
    save_json(rdir / "feedback_comparison.json", {"policies": rows, "diagnostics": diag})
    flat = [{k: (str(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in rows]
    keys = []
    for r in flat:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (rdir / "feedback_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)

    print("In-domain (GSM8K) vs out-of-domain (SVAMP):")
    for r in rows:
        line = (f"  {r['policy']:<6} GSM acc {r['gsm_accuracy']:.3f} fmt {r['gsm_format']:.3f} len {r['gsm_len']:.0f} | "
                f"SVAMP acc {r['svamp_accuracy']:.3f} fmt {r['svamp_format']:.3f} len {r['svamp_len']:.0f} | drop {r['accuracy_drop_gsm_to_svamp']:+.3f}")
        if "gsm_win_vs_sft" in r:
            line += (f" | win vs SFT GSM {r['gsm_win_vs_sft']:.3f} SVAMP {r['svamp_win_vs_sft']:.3f} | verifier-judge agreement GSM {r['gsm_agreement_3way']:.3f}")
        print(line)
    print("Diagnostics: S_reason verifier/judge =", round(diag["verifier"]["S_reason"], 2), "/", round(diag["judge"]["S_reason"], 2),
          "| S_outcome verifier/judge =", round(diag["verifier"]["S_outcome"]["better_rate"], 2), "/", round(diag["judge"]["S_outcome"]["better_rate"], 2))

    cands = []
    for name in ("rlvr", "rlaif"):
        gen = {r["index"]: r for r in read_jsonl(rdir / f"gsm_generated_{name}.jsonl")}
        sft = {r["index"]: r for r in read_jsonl(rdir / "gsm_generated_sft.jsonl")}
        for rec in read_jsonl(rdir / f"gsm_pairwise_{name}_vs_sft.jsonl"):
            kind = None
            if rec["judge"] == "A" and not rec["policy_correct"] and rec["sft_correct"]:
                kind = "judge_prefers_policy_but_policy_wrong_and_sft_right"
            elif rec["judge"] == "B" and rec["policy_correct"] and not rec["sft_correct"]:
                kind = "judge_prefers_sft_but_policy_right_and_sft_wrong"
            if kind:
                cands.append({"policy": name, "kind": kind, "index": rec["index"], "question": gen[rec["index"]]["question"][:400],
                              "gold": gen[rec["index"]]["gold_final"], "policy_response": gen[rec["index"]]["response"][:900],
                              "sft_response": sft[rec["index"]]["response"][:900]})
    write_jsonl(rdir / "qualitative_candidates.jsonl", cands)
    print(f"{len(cands)} verifier-judge disagreement cases saved to qualitative_candidates.jsonl (plus diagnostic_disagreements.jsonl)")


if __name__ == "__main__":
    main()