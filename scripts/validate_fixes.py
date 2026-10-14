"""Numerical validation of the three planted-defect fixes (Tasks 1-3).
Run from the repository root:  python -m scripts.validate_fixes
Writes results/validation_bug_fixes.txt
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import ppo_policy_loss
from task3_grpo.grpo import group_relative_advantages

lines = []


def log(s):
    print(s)
    lines.append(s)


# ---------------- Task 1: DPO ----------------
torch.manual_seed(0)
pc, pr, rc, rr = (torch.randn(8) * 5 for _ in range(4))
beta = 0.1
loss, _ = dpo_loss(pc, pr, rc, rr, beta)
manual = -torch.log(torch.sigmoid(beta * ((pc - rc) - (pr - rr)))).mean()
log("== Task 1: DPO loss  L = -log sigmoid(beta * [(log pi(y+) - log ref(y+)) - (log pi(y-) - log ref(y-))])")
log(f"fixed implementation matches the manual formula: {torch.allclose(loss, manual)}")
l_same, _ = dpo_loss(pc, pr, pc, pr, beta)
buggy_same = -F.logsigmoid(beta * ((pc - pr) + (pc - pr))).mean()   # original defect: policy_margin + ref_margin
log(f"policy == reference: fixed loss {l_same.item():.6f} (ln 2 = {math.log(2):.6f}); original (+) version {buggy_same.item():.6f}")
l_up, _ = dpo_loss(pc + 1, pr, rc, rr, beta)
log(f"raising the chosen log-prob lowers the loss: {l_up.item() < loss.item()}")

# ---------------- Task 2: PPO ----------------
eps = 0.2
mask, old = torch.ones(1, 1), torch.zeros(1, 1)


def ppo(ratio, adv):
    new = torch.log(torch.tensor([[ratio]]))
    l, _, _ = ppo_policy_loss(new, old, torch.tensor([[adv]]), mask, eps)
    return l.item()


log("\n== Task 2: PPO clipped surrogate  L = -E[min(rho*A, clip(rho,1-eps,1+eps)*A)]")
log(f"A=+1, rho=1.5: fixed loss {ppo(1.5, 1.0):+.3f} (expect -1.2); original max-version would give {-max(1.5, 1.2):+.3f}")
log(f"A=-1, rho=0.5: fixed loss {ppo(0.5, -1.0):+.3f} (expect +0.8); original max-version would give {-max(-0.5, -0.8):+.3f}")
x = torch.zeros(1, 1, requires_grad=True)
l, _, _ = ppo_policy_loss(x + torch.log(torch.tensor(1.5)), old, torch.tensor([[1.0]]), mask, eps)
l.backward()
log(f"gradient in the clipped region (A>0, rho=1.5): {x.grad.item():.1f} (expect 0)")

# ---------------- Task 3: GRPO ----------------
log("\n== Task 3: GRPO group-relative advantages  A_k = (r_k - mean_group) / (std_group + eps)")
r = torch.tensor([1., 2., 3., 4., 10., 10., 10., 10.])
g = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
a = group_relative_advantages(r, g)
batch = (r - r.mean()) / r.std(unbiased=False)                        # original defect: ignores group_ids
log(f"group 0 advantages (fixed): {[round(v, 3) for v in a[:4].tolist()]}, mean {a[:4].mean().item():.6f}")
log(f"group 1, all rewards equal (fixed, expect zeros): {[round(v, 3) for v in a[4:].tolist()]}")
log(f"group 1 with the original whole-batch normalization: {[round(v, 3) for v in batch[4:].tolist()]}")

out = Path("results") / "validation_bug_fixes.txt"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nsaved {out}")