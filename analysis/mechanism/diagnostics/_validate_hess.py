"""Validate the Hessian sharpness probe against a full eigendecomposition.

Uses torch.func.functional_call so the loss genuinely depends on the tensors
passed to the Hessian routine. Compares the full spectrum (eigvalsh) with the
power iteration used in gradient_probe.py.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gradient_probe import build_model, load_task_a  # noqa: E402

torch.manual_seed(7)
model = build_model()
X, y = load_task_a()
names = [n for n, _ in model.named_parameters()]
params = [p for p in model.parameters() if p.requires_grad]
print("param count:", sum(p.numel() for p in params))


def f(*ps):
    d = dict(zip(names, ps))
    return F.cross_entropy(functional_call(model, d, (X,)), y)


H = torch.autograd.functional.hessian(f, tuple(params))
n = sum(p.numel() for p in params)
M = torch.zeros(n, n)
oi = 0
for i, pi in enumerate(params):
    oj = 0
    for j, pj in enumerate(params):
        blk = H[i][j].detach().reshape(pi.numel(), pj.numel())
        M[oi : oi + pi.numel(), oj : oj + pj.numel()] = blk
        oj += pj.numel()
    oi += pi.numel()

evals = torch.linalg.eigvalsh(M)
print(f"eig min = {evals.min().item():+.6f}")
print(f"eig max = {evals.max().item():+.6f}")
print(f"largest |eig| = {evals[evals.abs().argmax()].item():+.6f}")

# power iteration, same implementation as gradient_probe.sharpness
v = [torch.randn_like(p) for p in params]
nv = torch.sqrt(sum((vi ** 2).sum() for vi in v))
v = [vi / nv for vi in v]
lam = None
for _ in range(60):
    loss = f(*params)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((g * vi).sum() for g, vi in zip(grads, v))
    h = torch.autograd.grad(dot, params)
    lam = float(sum((vi * hi).sum() for vi, hi in zip(v, h)).detach())
    nv = torch.sqrt(sum((hi ** 2).sum() for hi in h)) + 1e-12
    v = [hi / nv for hi in h]
print(f"power iteration lambda (60 iters) = {lam:+.6f}")
