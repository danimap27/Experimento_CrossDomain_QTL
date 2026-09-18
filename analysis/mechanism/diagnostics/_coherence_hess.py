"""Coherence test: same Hessian operator through both computation paths.

For random unit directions v, compare v^T H v computed from the full Hessian
matrix (torch.autograd.functional.hessian) against the autograd HVPs used by
the power iteration in gradient_probe.py.
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
n = sum(p.numel() for p in params)


def f(*ps):
    d = dict(zip(names, ps))
    return F.cross_entropy(functional_call(model, d, (X,)), y)


H = torch.autograd.functional.hessian(f, tuple(params))
M = torch.zeros(n, n)
oi = 0
for i, pi in enumerate(params):
    oj = 0
    for j, pj in enumerate(params):
        blk = H[i][j].detach().reshape(pi.numel(), pj.numel())
        M[oi : oi + pi.numel(), oj : oj + pj.numel()] = blk
        oj += pj.numel()
    oi += pi.numel()
print("M symmetric?", torch.allclose(M, M.T, atol=1e-5), "| M norm:", M.norm().item())

g = torch.Generator().manual_seed(3)
for k in range(4):
    v = [torch.randn(p.shape, generator=g, dtype=p.dtype) for p in params]
    nv = torch.sqrt(sum((vi ** 2).sum() for vi in v))
    v = [vi / nv for vi in v]
    vflat = torch.cat([vi.flatten() for vi in v])

    A = float(vflat @ (M @ vflat))

    loss = f(*params)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((gr * vi).sum() for gr, vi in zip(grads, v))
    h = torch.autograd.grad(dot, params)
    B = float(sum((vi * hi).sum() for vi, hi in zip(v, h)).detach())

    print(f"dir {k}: M-path vHv = {A:+.6f} | autograd-path vHv = {B:+.6f} | diff = {B - A:+.2e}")
