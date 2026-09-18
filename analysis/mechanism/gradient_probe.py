#!/usr/bin/env python3
"""Mechanism probe — optimization landscape of the hybrid model.

Journal evidence for reviewer #2 ("investigate the loss landscape, parameter
distribution, or barren plateau behavior post-pre-training to obtain insights
into the quantum-specific advantages").

Probes
  bp    Barren-plateau proxy. Variance of cost-function gradients,
        Var[d<Z0>/dtheta], over random parameter draws, swept across qubit
        counts and ansatz topologies (SEL vs TTN).
  grad  Task-A gradient norm and per-coordinate variance at theta_0
        (post-pre-training on the synthetic source) vs random initialization,
        for the full hybrid model (SEL, 4 qubits, 3 layers).
  hess  Top Hessian eigenvalue (sharpness) of the Task-A loss at theta_0 vs
        random initialization, by power iteration on Hessian-vector products.
        WARNING: pending validation. Second-order autodiff through the
        PennyLane TorchLayer is not reliable in this stack (see
        diagnostics/README.md); do not report these numbers until the
        curvature path is recomputed with a validated method (parameter-shift,
        double-precision finite differences, or jax).

Usage
  python gradient_probe.py bp
  python gradient_probe.py grad
  python gradient_probe.py hess
  python gradient_probe.py all

Outputs land in analysis/mechanism/outputs/ as JSON plus a markdown report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "outputs"

import pennylane as qml  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SEED = 1234
N_LAYERS = 3


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def ttn_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Hierarchical (RY+RZ+CNOT) block wiring used by the paper's TTN.

    Adjacent wires are paired level by level and the information accumulates
    on the right wire of each pair (matching the original code: for 4 qubits
    the blocks are (0,1), (2,3), (1,3), i.e. n-1 blocks in total). The
    original implementation crashes for n != 4; this generalization keeps the
    same layout for n = 4 and extends it to any qubit count, so the TTN
    topology can be swept beyond four qubits.
    """
    pairs: list[tuple[int, int]] = []
    active = list(range(n_qubits))
    while len(active) > 1:
        next_active = []
        for k in range(0, len(active) - 1, 2):
            pairs.append((active[k], active[k + 1]))
            next_active.append(active[k + 1])
        if len(active) % 2:
            next_active.append(active[-1])
        active = next_active
    return pairs


def ttn_apply(weights, n_qubits: int):
    """Apply one TTN block per layer. weights: (n_layers, n_blocks, 2)."""
    pairs = ttn_pairs(n_qubits)
    for layer in range(weights.shape[0]):
        for b, (left, right) in enumerate(pairs):
            qml.RY(weights[layer, b, 0], wires=left)
            qml.RZ(weights[layer, b, 1], wires=right)
            qml.CNOT(wires=[left, right])


def param_count(n_qubits: int, ansatz: str, n_layers: int = N_LAYERS) -> int:
    if ansatz == "sel":
        return n_layers * n_qubits * 3
    if ansatz == "ttn":
        return n_layers * (n_qubits - 1) * 2
    raise ValueError(ansatz)


# ---------------------------------------------------------------------------
# Probe 1 — barren plateau sweep
# ---------------------------------------------------------------------------
def build_bp_circuit(n_qubits: int, ansatz: str, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(flat_theta):
        if ansatz == "sel":
            weights = flat_theta.reshape(n_layers, n_qubits, 3)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        elif ansatz == "ttn":
            weights = flat_theta.reshape(n_layers, n_qubits - 1, 2)
            ttn_apply(weights, n_qubits)
        else:
            raise ValueError(ansatz)
        # Global cost: average single-qubit Z expectation over all wires, so
        # every block of every topology contributes to the observable.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def probe_bp(n_qubits_list, ansatze, n_samples: int = 150, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    for ansatz in ansatze:
        for n in n_qubits_list:
            p = param_count(n, ansatz)
            circ = build_bp_circuit(n, ansatz, N_LAYERS)
            grads = []
            t0 = time.time()
            for _ in range(n_samples):
                theta = torch.tensor(rng.uniform(-np.pi, np.pi, p),
                                     dtype=torch.float64, requires_grad=True)
                value = torch.stack(circ(theta)).mean()
                value.backward()
                grads.append(theta.grad.detach().numpy())
            grads = np.stack(grads)
            var_per_coord = grads.var(axis=0)
            key = f"{ansatz}_n{n}"
            results[key] = {
                "ansatz": ansatz,
                "n_qubits": n,
                "n_params": p,
                "n_samples": n_samples,
                "grad_var_mean": float(var_per_coord.mean()),
                "grad_var_median": float(np.median(var_per_coord)),
                "grad_var_max": float(var_per_coord.max()),
                "grad_abs_mean": float(np.abs(grads).mean()),
                "seconds": time.time() - t0,
            }
            print(f"[bp] {key}: var_mean={var_per_coord.mean():.3e} "
                  f"(median {np.median(var_per_coord):.3e}) in {time.time()-t0:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Probe 2/3 — theta_0 vs random, full hybrid model
# ---------------------------------------------------------------------------
def build_model():
    from quantum_net import HybridQuantumNet  # repo module

    return HybridQuantumNet(ansatz="A", n_layers=N_LAYERS, noise=False)


def load_task_a(batch_size: int = 32):
    from data_module import DataModule

    dm = DataModule(data_dir=str(REPO / "data"), batch_size=batch_size)
    train_loader, _, _ = dm.get_fashion_mnist_task(classes=(0, 1))
    X, y = next(iter(train_loader))  # deterministic first batch (shuffle=True per loader)
    return X, y


def pretrain_synthetic(model, epochs: int = 15, lr: float = 0.05, seed: int = SEED):
    """Pre-train on the synthetic Gaussian source (protocol of the paper)."""
    from data_module import DataModule

    torch.manual_seed(seed)
    dm = DataModule(data_dir=str(REPO / "data"), batch_size=32)
    syn_train, _ = dm.get_synthetic_task(n_samples=2500)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for X, y in syn_train:
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            opt.step()
    return model


def flat_params(model):
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def set_params(model, values):
    with torch.no_grad():
        for p, v in zip([p for p in model.parameters() if p.requires_grad], values):
            p.copy_(v)


def probe_grad(n_random: int = 10, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    X, y = load_task_a()

    def grad_stats(model):
        model.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).detach().numpy()
        return {
            "loss": float(loss.detach()),
            "grad_norm": float(np.linalg.norm(g)),
            "grad_var": float(g.var()),
            "grad_abs_mean": float(np.abs(g).mean()),
            "frac_grad_near_zero": float((np.abs(g) < 1e-4).mean()),
        }

    # theta_0 via synthetic pre-training
    model0 = build_model()
    model0 = pretrain_synthetic(model0)
    theta0_stats = grad_stats(model0)
    theta0_params = flat_params(model0)

    # random initializations
    randoms = []
    for r in range(n_random):
        torch.manual_seed(10_000 + r)
        m = build_model()
        randoms.append(grad_stats(m))

    def agg(key):
        vals = np.array([r[key] for r in randoms])
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                "values": vals.tolist()}

    # parameter distribution summary of theta_0 (reviewer's "parameter distribution")
    all_theta0 = np.concatenate([p.flatten().numpy() for p in theta0_params if p.dim() > 1])
    param_dist = {
        "vqc_weights_mean": float(all_theta0.mean()),
        "vqc_weights_std": float(all_theta0.std()),
        "vqc_weights_min": float(all_theta0.min()),
        "vqc_weights_max": float(all_theta0.max()),
        "vqc_weights_hist": np.histogram(all_theta0, bins=12, range=(-np.pi, np.pi))[0].tolist(),
    }

    return {
        "theta0": theta0_stats,
        "random": {k: agg(k) for k in ["loss", "grad_norm", "grad_var", "grad_abs_mean", "frac_grad_near_zero"]},
        "param_distribution": param_dist,
        "n_random": n_random,
    }


def probe_hess(n_random: int = 5, iters: int = 20, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    X, y = load_task_a()

    def sharpness(model):
        params = [p for p in model.parameters() if p.requires_grad]

        def loss_fn():
            return F.cross_entropy(model(X), y)

        v = [torch.randn_like(p) for p in params]
        nv = torch.sqrt(sum((vi ** 2).sum() for vi in v))
        v = [vi / nv for vi in v]
        lam = None
        for _ in range(iters):
            loss = loss_fn()
            grads = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((g * vi).sum() for g, vi in zip(grads, v))
            h = torch.autograd.grad(dot, params)
            lam = float(sum((vi * hi).sum() for vi, hi in zip(v, h)).detach())
            nv = torch.sqrt(sum((hi ** 2).sum() for hi in h)) + 1e-12
            v = [hi / nv for hi in h]
        return lam, float(nv)

    model0 = build_model()
    model0 = pretrain_synthetic(model0)
    lam0, nrm0 = sharpness(model0)

    rands = []
    for r in range(n_random):
        torch.manual_seed(20_000 + r)
        m = build_model()
        lam, nrm = sharpness(m)
        rands.append(lam)

    rands = np.array(rands)
    return {
        "theta0_lambda_max": lam0,
        "random_lambda_max": {"mean": float(rands.mean()), "std": float(rands.std(ddof=1)),
                              "values": rands.tolist()},
        "ratio": float(lam0 / rands.mean()) if rands.mean() else None,
        "n_random": n_random,
    }


def write_report():
    lines = ["# Mechanism probe — preliminary results\n"]
    files = {
        "Barren-plateau sweep": "bp.json",
        "Task-A gradient at theta_0 vs random": "grad.json",
        "Hessian sharpness at theta_0 vs random": "hess.json",
    }
    for title, fname in files.items():
        path = OUT / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        lines.append(f"## {title}\n")

        if fname == "bp.json":
            lines.append("| ansatz | qubits | params | Var[grad] mean | Var[grad] median |")
            lines.append("|--------|--------|--------|----------------|------------------|")
            for key, d in sorted(data.items()):
                lines.append(f"| {d['ansatz']} | {d['n_qubits']} | {d['n_params']} | "
                             f"{d['grad_var_mean']:.3e} | {d['grad_var_median']:.3e} |")
        elif fname == "grad.json":
            t0, rd = data["theta0"], data["random"]
            lines.append(f"- theta_0: loss={t0['loss']:.4f}, |grad|={t0['grad_norm']:.4f}, "
                         f"grad_var={t0['grad_var']:.3e}, frac|g|<1e-4={t0['frac_grad_near_zero']:.3f}")
            for k in ["loss", "grad_norm", "grad_var", "frac_grad_near_zero"]:
                a = rd[k]
                lines.append(f"- random ({data['n_random']}): {k}={a['mean']:.4f} +/- {a['std']:.4f}")
            pd = data["param_distribution"]
            lines.append(f"- theta_0 VQC weights: mean={pd['vqc_weights_mean']:.3f}, "
                         f"std={pd['vqc_weights_std']:.3f}, range=[{pd['vqc_weights_min']:.3f}, "
                         f"{pd['vqc_weights_max']:.3f}]")
        elif fname == "hess.json":
            lines.append(f"- lambda_max(theta_0) = {data['theta0_lambda_max']:.4f}")
            r = data["random_lambda_max"]
            lines.append(f"- lambda_max(random, n={data['n_random']}) = {r['mean']:.4f} +/- {r['std']:.4f}")
            ratio = data.get("ratio")
            lines.append(f"- ratio = {ratio:.3f}" if ratio is not None else "- ratio = n/a")
        lines.append("")

    (OUT / "report_mechanism.md").write_text("\n".join(lines) + "\n")
    print(f"Report written to {OUT / 'report_mechanism.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe", choices=["bp", "grad", "hess", "all"])
    ap.add_argument("--n-qubits", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--ansatze", nargs="+", default=["sel", "ttn"])
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--n-random", type=int, default=10)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.probe in ("bp", "all"):
        res = probe_bp(args.n_qubits, args.ansatze, n_samples=args.samples)
        (OUT / "bp.json").write_text(json.dumps(res, indent=2))
    if args.probe in ("grad", "all"):
        res = probe_grad(n_random=args.n_random)
        (OUT / "grad.json").write_text(json.dumps(res, indent=2))
        print(f"[grad] theta0 |grad|={res['theta0']['grad_norm']:.4f} vs "
              f"random {res['random']['grad_norm']['mean']:.4f}")
    if args.probe in ("hess", "all"):
        print("WARNING: the hess probe is pending validation (TorchLayer "
              "second-order autodiff is unreliable; see diagnostics/README.md). "
              "Results are diagnostic only.", file=sys.stderr)
        res = probe_hess(n_random=max(3, args.n_random // 2))
        (OUT / "hess.json").write_text(json.dumps(res, indent=2))
        print(f"[hess] lambda_max theta0={res['theta0_lambda_max']:.4f} vs "
              f"random {res['random_lambda_max']['mean']:.4f}")

    write_report()


if __name__ == "__main__":
    main()
