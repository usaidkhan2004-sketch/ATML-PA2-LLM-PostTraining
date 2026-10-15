"""List candidate qualitative examples for Task 1 (reads saved files only; no model is run).
Run from the repository root:  python -m task1_dpo.list_examples
Writes results/task1_dpo/example_candidates.txt
"""
from pathlib import Path

from common.data import preference_responses, prompt_messages, read_jsonl

R = Path("results/task1_dpo")
lines = []


def out(s=""):
    print(s)
    lines.append(s)


def flat(s):
    return " ".join(str(s).split())


def head(s, n):
    s = flat(s)
    return s if len(s) <= n else s[:n] + " ..."


def tail(s, n):
    s = flat(s)
    return s if len(s) <= n else "... " + s[-n:]


def ptext(row):
    return prompt_messages(row)[-1]["content"]


def gens(name):
    return {r["prompt_id"]: r for r in read_jsonl(str(R / f"eval_{name}_generations.jsonl"))}


# ---------- A. reward-score gain vs SFT, check whether the answer is actually better ----------
out("=" * 100)
out("A. REWARD vs QUALITY: prompts where standard DPO's RM score rose most over SFT (read both answers)")
std, sft = gens("standard"), gens("sft")
rows = {r["prompt_id"]: r for r in read_jsonl("data/dpo_standard_eval.jsonl")}
ids = [i for i in std if i in sft and i in rows]
for rank, i in enumerate(sorted(ids, key=lambda k: std[k]["rm_score"] - sft[k]["rm_score"], reverse=True)[:8], 1):
    s, b = std[i], sft[i]
    out(f"\n[A{rank}] prompt_id={i[:12]}  RM sft={b['rm_score']:+.2f} -> dpo={s['rm_score']:+.2f}  "
        f"tokens sft={b['response_tokens']} dpo={s['response_tokens']}  truncated sft={b['truncated']} dpo={s['truncated']}")
    out(f"PROMPT: {head(ptext(rows[i]), 300)}")
    out(f"SFT START : {head(b['response'], 350)}")
    out(f"SFT END   : {tail(b['response'], 150)}")
    out(f"DPO START : {head(s['response'], 350)}")
    out(f"DPO END   : {tail(s['response'], 150)}")

# ---------- B. word-limit compliance ----------
out("\n" + "=" * 100)
out("B. INSTRUCTION COMPLIANCE: word-limit prompts (compare standard vs length-balanced vs SFT)")
wlp = {(r.get("prompt_id") or r.get("id")): r for r in read_jsonl("data/word_limit_prompts.jsonl")}
wl = {m: read_jsonl(str(R / f"eval_{m}_wordlimit.jsonl")) for m in ["sft", "standard", "length_balanced"]}


def rate(m, pid):
    xs = [r["compliant"] for r in wl[m] if r["prompt_id"] == pid]
    return sum(xs) / len(xs) if xs else float("nan")


pids = sorted({r["prompt_id"] for r in wl["standard"]})
out("prompt_id  limit  rate(sft)  rate(standard)  rate(balanced)")
for p in pids:
    lim = next(r["limit"] for r in wl["standard"] if r["prompt_id"] == p)
    out(f"{p:9s} {lim:5}  {rate('sft', p):.1f}        {rate('standard', p):.1f}             {rate('length_balanced', p):.1f}")


def sample(m, pid, want):
    for r in wl[m]:
        if r["prompt_id"] == pid and bool(r["compliant"]) == want:
            return r
    return None


for rank, p in enumerate(sorted(pids, key=lambda q: rate("standard", q) - rate("length_balanced", q), reverse=True)[:3], 1):
    out(f"\n[B{rank}] {p}: standard {rate('standard', p):.1f} vs balanced {rate('length_balanced', p):.1f}")
    out(f"PROMPT: {head(ptext(wlp[p]), 300) if p in wlp else '(prompt text not found)'}")
    for m, want in [("standard", True), ("length_balanced", False)]:
        r = sample(m, p, want)
        if r:
            out(f"{m} ({'complies' if want else 'violates'}): {r['words']} words / limit {r['limit']}: {head(r['response'], 400)}")
out("\nStandard-model violations (any prompt):")
for m in ["standard"]:
    bad = sorted([r for r in wl[m] if not r["compliant"]], key=lambda r: r["words"] - r["limit"], reverse=True)[:3]
    for r in bad:
        out(f"[{m}] {r['prompt_id']} {r['words']} words / limit {r['limit']}: {head(r['response'], 300)}")

# ---------- C. length-stratified pairs: model prefers the SHORTER response ----------
out("\n" + "=" * 100)
out("C. LENGTH BEHAVIOUR: preferred-longer pairs where standard DPO's margin is most negative (it favours the shorter rejected answer)")
srows = {r["prompt_id"]: r for r in read_jsonl("data/dpo_length_stratified_eval.jsonl")}
sp = {r["prompt_id"]: r for r in read_jsonl(str(R / "eval_standard_pairs_stratified.jsonl"))}
bp = {r["prompt_id"]: r for r in read_jsonl(str(R / "eval_length_balanced_pairs_stratified.jsonl"))}
cand = [r for r in sp.values() if r["length_stratum"] == "preferred_longer" and r["prompt_id"] in srows]
for rank, r in enumerate(sorted(cand, key=lambda x: x["margin"])[:5], 1):
    row = srows[r["prompt_id"]]
    yc, yr = preference_responses(row)
    b = bp.get(r["prompt_id"], {})
    out(f"\n[C{rank}] prompt_id={r['prompt_id'][:12]}  margin standard={r['margin']:+.2f} balanced={b.get('margin', float('nan')):+.2f}  "
        f"human-score chosen={r['score_chosen']} rejected={r['score_rejected']}  words chosen={len(yc.split())} rejected={len(yr.split())}")
    out(f"PROMPT  : {head(ptext(row), 250)}")
    out(f"CHOSEN  : {head(yc, 250)}")
    out(f"REJECTED: {head(yr, 250)}")

dest = R / "example_candidates.txt"
dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nsaved {dest}")