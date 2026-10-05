#!/usr/bin/env python3
"""E5 — mechanism study of the cross-domain prior (journal revision).

Checkpoint-based probes for reviewer #2 ("investigate the loss landscape,
parameter distribution, or barren plateau behavior post-pre-training"). This
script complements `gradient_probe.py` (qubit sweep, global Var[grad]) with
the four E5 deliverables of the revision plan:

  1. per-layer gradient variance at theta_0/theta_A/theta_B vs random init;
  2. empirical Fisher spectra (first-order per-sample gradients; eigen-
     spectrum + entropy + effective rank) at the same checkpoints;
  3. loss barrier along the linear interpolation theta_A -> theta_B
     (Task-B and Task-A losses) and the parameter distance ||dB - dA||;
  4. parameter distributions including angle wrapping (period-2pi
     equivalence verified numerically, raw vs wrapped statistics).

All curvature objects are FIRST-ORDER (empirical Fisher) or finite
differences with a self-test. NEVER double-backward through the PennyLane
TorchLayer (caveat T1, analysis/mechanism/diagnostics/README.md).

The training step re-creates the canonical two-task cell of the E2-E4
campaign (pair2, 4-qubit SEL, matched-rate protocol, ideal simulator) for
arms {scratch, synth} x seeds 0..9 and saves theta0/thetaA/thetaB, because
the campaign payloads do not persist parameter snapshots.

Usage
    python e5_study.py train        # (re)build checkpoints (parallel)
    python e5_study.py probes       # run probes from checkpoints -> JSON+MD+fig
    python e5_study.py all

Outputs: analysis/mechanism/outputs/e5/ (ckpts/, probes.json,
report_e5.md) and journal/paper_journal/figures/fig_e5_mechanism.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from continual import snapshot_params, train_task  # noqa: E402
from data_module import DataModule  # noqa: E402
from e234_runner import BASE_SEED, CAMPAIGNS, LR, load_task, set_seed  # noqa: E402
from quantum_net import HybridQuantumNet, get_noise_profile  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "e5"
CKPTS = OUT / "ckpts"
FIGDIR = REPO / "journal" / "paper_journal" / "figures"

N_QUbits = 4
N_LAYERS = 3
N_CLASSES = 4
SEEDS = list(range(10))
ARMS = ["scratch", "synth"]
PRETRAIN_SAMPLES = 2500
PRETRAIN_EPOCHS = 15
EPOCHS = 10
K_GRAD = 120     # samples for gradient-variance probes
K_FISHER = 150   # samples for the empirical Fisher
N_INTERP = 21    # barrier interpolation points


def machine_id() -> str:
    try:
        mid = Path("/etc/machine-id").read_text().strip()[:8]
    except OSError:
        mid = "no-machine-id"
    return f"{socket.gethostname()}:{mid}"


# ---------------------------------------------------------------------------
# Checkpoint training (mirrors e234_runner pair2 std protocol)
# ---------------------------------------------------------------------------

def build_cell_data():
    dm = DataModule(data_dir=str(REPO / "data"), batch_size=32, n_components=N_QUbits)
    tasks = CAMPAIGNS["pair2"]["tasks"]
    loaders, tests = [], []
    for ds, classes in tasks:
        tr, te, _ = load_task(dm, ds, classes, *CAMPAIGNS["pair2"]["limits"],
                              REPO / "cache" / "cifar_feats")
        loaders.append(tr)
        tests.append(te)
    return dm, loaders, tests


def new_model():
    noise, params = get_noise_profile("ideal")
    return HybridQuantumNet(ansatz="A", n_qubits=N_QUbits, n_layers=N_LAYERS,
                            n_classes=N_CLASSES, noise=noise,
                            noise_params=params)


@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct = total = 0
    for X, y in loader:
        preds = model(X).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    model.train()
    return 100.0 * correct / max(total, 1)


def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_one(job):
    arm, seed = job
    out = CKPTS / f"{arm}_s{seed}.pt"
    if out.exists():
        return f"[skip] {out}"
    torch.set_num_threads(1)
    try:
        dm, loaders, tests = build_cell_data()
        criterion = nn.CrossEntropyLoss()

        theta0 = None
        w0 = None
        pt_time = None
        if arm == "synth":
            set_seed(seed * BASE_SEED + 1)
            src = new_model()
            syn_tr, syn_te = dm.get_synthetic_task(n_samples=PRETRAIN_SAMPLES)
            hist = train_task(src, syn_tr, PRETRAIN_EPOCHS, 0.05,
                              criterion=criterion, log_prefix="[pt] ")
            theta0 = snapshot_params(src)
            w0 = state_cpu(src)
            pt_time = hist["train_time"]

        set_seed(seed * BASE_SEED + 0)
        model = new_model()
        if theta0 is not None:
            model.load_state_dict(theta0)
        w_init = state_cpu(model)

        accs = {"b0": [eval_acc(model, te) for te in tests]}
        set_seed(seed * BASE_SEED + 50 + 0)
        train_task(model, loaders[0], EPOCHS, LR, criterion=criterion,
                   log_prefix="[T0] ")
        wA = state_cpu(model)
        accs["a_after_a"] = eval_acc(model, tests[0])
        accs["b_after_a"] = eval_acc(model, tests[1])

        set_seed(seed * BASE_SEED + 50 + 1)
        train_task(model, loaders[1], EPOCHS, LR, criterion=criterion,
                   log_prefix="[T1] ")
        wB = state_cpu(model)
        accs["a_after_b"] = eval_acc(model, tests[0])
        accs["b_after_b"] = eval_acc(model, tests[1])

        CKPTS.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": arm, "seed": seed, "machine_id": machine_id(),
                    "w_init": w_init, "w0": w0 if arm == "synth" else w_init,
                    "wA": wA, "wB": wB, "accs": accs,
                    "pretrain_train_time": pt_time,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, out)
        return f"[done] {out} accs={accs}"
    except Exception as e:  # pragma: no cover
        import traceback
        return f"[fail] {arm} s{seed}: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Probes (per (arm, seed) worker)
# ---------------------------------------------------------------------------

def load_model_from_state(state):
    model = new_model()
    model.load_state_dict({k: v.clone() for k, v in state.items()})
    return model


def flat_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()]).numpy()


def set_flat(model, vec):
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(torch.tensor(vec[off:off + n], dtype=p.dtype).view_as(p))
            off += n


def per_sample_grads(model, criterion, X, y, k):
    """First-order per-sample gradients (caveat T1 compliant)."""
    params = [p for p in model.parameters() if p.requires_grad]
    gs = []
    for i in range(min(k, X.shape[0])):
        model.zero_grad(set_to_none=True)
        loss = criterion(model(X[i:i + 1]), y[i:i + 1])
        g = torch.autograd.grad(loss, params)
        gs.append(np.concatenate([gi.detach().numpy().ravel() for gi in g]))
    return np.array(gs)


def fd_grad_selftest_f64(model, criterion, X, y, coords=4, h=1e-5, seed=0):
    """Central finite differences in float64 against first-order autograd.

    The repo-wide `fd_grad_selftest` (continual.py) runs in float32 and sits
    at the precision margin (observed 2.7-6.2% relative error on the
    pre-trained point); the T1-compliant version for the E5 study operates
    in double precision, where the same check lands at ~1e-5. Returns
    (max_rel_err, errs); it does NOT abort the run so failures are recorded.
    """
    model = load_model_from_state({k: v for k, v in model.state_dict().items()})
    model = model.double()
    X, y = X.to(torch.float64), y
    w = model.vqc.weights
    loss = criterion(model(X), y)
    grads = torch.autograd.grad(loss, w)[0].detach().flatten()
    g = torch.Generator().manual_seed(seed)
    n = w.numel()
    idx = torch.randperm(n, generator=g)[:min(coords, n)]
    errs = []
    with torch.no_grad():
        base = w.detach().clone()
    for i in idx.tolist():
        wp = base.clone().flatten()
        wp[i] += h
        wm = base.clone().flatten()
        wm[i] -= h
        with torch.no_grad():
            w.copy_(wp.reshape_as(w))
            lp = float(criterion(model(X), y))
            w.copy_(wm.reshape_as(w))
            lm = float(criterion(model(X), y))
            w.copy_(base)
        fd = (lp - lm) / (2 * h)
        ag = float(grads[i])
        errs.append(abs(fd - ag) / max(abs(fd), abs(ag), 1e-3))
    return max(errs), errs


def tensor_loss(model, X, y):
    """Deterministic loss on a fixed sample (no DataLoader shuffling)."""
    with torch.no_grad():
        return float(nn.CrossEntropyLoss()(model(X), y))


def batch_loss(model, loader, max_batches=40):
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    n = 0
    with torch.no_grad():
        for bi, (X, y) in enumerate(loader):
            if bi >= max_batches:
                break
            total += float(criterion(model(X), y))
            n += 1
    return total / max(n, 1)


def probe_one(job):
    arm, seed = job
    ck = torch.load(CKPTS / f"{arm}_s{seed}.pt", weights_only=False)

    dm, loaders, tests = build_cell_data()
    criterion = nn.CrossEntropyLoss()
    train_loader = loaders[0]
    B_loader = loaders[1]

    # fixed evaluation tensors (train data of both tasks, first samples)
    def sample_batches(loader, k):
        Xs, ys = [], []
        for X, y in loader:
            Xs.append(X)
            ys.append(y)
            if sum(x.shape[0] for x in Xs) >= k:
                break
        return torch.cat(Xs)[:k], torch.cat(ys)[:k]

    XA, yA = sample_batches(train_loader, max(K_GRAD, K_FISHER, 300))
    XB, yB = sample_batches(B_loader, max(K_GRAD, 300))

    res = {"arm": arm, "seed": seed, "machine_id": ck["machine_id"],
           "accs": ck["accs"]}

    checkpoints = {"theta0": ck["w0"], "thetaA": ck["wA"], "thetaB": ck["wB"]}
    # also random init (theta_init) for both arms
    checkpoints["theta_init"] = ck["w_init"]

    # ---- 1) per-layer gradient variance on Task A data ---------------------
    layers = {}
    for name, state in checkpoints.items():
        m = load_model_from_state(state)
        G = per_sample_grads(m, criterion, XA, yA, K_GRAD)
        v = G.var(axis=0)
        n_vqc = 3 * N_QUbits * N_LAYERS
        per_layer = [float(v[l * 3 * N_QUbits:(l + 1) * 3 * N_QUbits].mean())
                     for l in range(N_LAYERS)]
        layers[name] = {"per_layer": per_layer,
                        "vqc_mean": float(v[:n_vqc].mean()),
                        "fc_mean": float(v[n_vqc:].mean()),
                        "frac_tiny": float((np.abs(G).max(axis=0) < 1e-6).mean())}
    res["layer_var_taskA"] = layers

    # ---- 3) loss barrier thetaA -> thetaB ----------------------------------
    # Evaluated on FIXED 300-sample subsets of each task (deterministic).
    mA = load_model_from_state(ck["wA"])
    vA, vB = flat_vector(mA), flat_vector(load_model_from_state(ck["wB"]))
    dist = float(np.linalg.norm(vB - vA) / np.sqrt(len(vA)))
    curvesA, curvesB = [], []
    for s in np.linspace(0.0, 1.0, N_INTERP):
        m = load_model_from_state(ck["wA"])
        set_flat(m, (1 - s) * vA + s * vB)
        curvesB.append(tensor_loss(m, XB, yB))
        curvesA.append(tensor_loss(m, XA[:300], yA[:300]))
    barrierB = max(curvesB) - max(curvesB[0], curvesB[-1])
    barrierA = max(curvesA) - max(curvesA[0], curvesA[-1])
    res["barrier"] = {"dist_thetaB_thetaA_rms": dist,
                      "curve_taskB": curvesB, "curve_taskA": curvesA,
                      "barrier_taskB": barrierB, "barrier_taskA": barrierA,
                      "subset_n": 300}

    # ---- 2) empirical Fisher spectra on Task A data at theta0/A/B ----------
    fisher = {}
    for name in ("theta0", "thetaA", "thetaB"):
        m = load_model_from_state(checkpoints[name])
        G = per_sample_grads(m, criterion, XA, yA, K_FISHER)
        nan_frac = float(np.isnan(G).mean())
        G = np.nan_to_num(G)
        F = (G.T @ G) / G.shape[0]
        eigs = np.linalg.eigvalsh(F)
        eigs = np.clip(eigs, 0.0, None)
        tot = eigs.sum() + 1e-300
        p = eigs / tot
        nz = p > 0
        entropy = float(-(p[nz] * np.log(p[nz])).sum())
        fisher[name] = {
            "trace": float(eigs.sum()),
            "top1": float(eigs[-1]),
            "top5": [float(x) for x in eigs[-5:][::-1]],
            "top1_frac": float(eigs[-1] / tot),
            "spectral_entropy": entropy,
            "eff_rank": float(np.exp(entropy)),
            "eigs_head": [float(x) for x in eigs[::-1][:24]],
            "nan_frac_grads": nan_frac,
        }
    res["fisher_taskA"] = fisher

    # ---- 4) parameter distributions + angle wrapping -----------------------
    # The circuit is 2pi-periodic in every ROTATION angle of the VQC (global
    # phases are unobservable in the Z expectations). Verified by loss
    # invariance when wrapping the vqc.weights coordinates only: the
    # classical head is NOT periodic and must be left untouched.
    N_VQC = 3 * N_QUbits * N_LAYERS

    def wrap(v):
        return (v + np.pi) % (2 * np.pi) - np.pi

    m0 = load_model_from_state(ck["w0"])
    v0 = flat_vector(m0)
    v0_vqc = v0[:N_VQC]
    v0w_vqc = wrap(v0_vqc)
    m0w = load_model_from_state(ck["w0"])
    set_flat(m0w, np.concatenate([v0w_vqc, v0[N_VQC:]]))
    l_raw = tensor_loss(m0, XA[:K_FISHER], yA[:K_FISHER])
    l_wrapped = tensor_loss(m0w, XA[:K_FISHER], yA[:K_FISHER])
    vA0 = flat_vector(load_model_from_state(ck["wA"]))[:N_VQC]
    vB0 = flat_vector(load_model_from_state(ck["wB"]))[:N_VQC]
    res["wrap"] = {
        "n_vqc": N_VQC,
        "raw_mean": float(v0_vqc.mean()), "raw_std": float(v0_vqc.std()),
        "raw_min": float(v0_vqc.min()), "raw_max": float(v0_vqc.max()),
        "frac_outside_pi": float((np.abs(v0_vqc) > np.pi).mean()),
        "wrapped_mean": float(v0w_vqc.mean()), "wrapped_std": float(v0w_vqc.std()),
        "loss_raw": l_raw, "loss_wrapped": l_wrapped,
        "loss_abs_diff": abs(l_raw - l_wrapped),
        "thetaA_mean": float(vA0.mean()), "thetaB_mean": float(vB0.mean()),
        "raw_frac_at_pi": float((np.abs(np.abs(v0_vqc) - np.pi) < 0.1).mean()),
    }

    # ---- self-tests (caveat T1) -------------------------------------------
    m = load_model_from_state(ck["w0"])
    err, _errs = fd_grad_selftest_f64(m, criterion, XA[:32], yA[:32], coords=4)
    res["selftest_fd_max_rel_err"] = float(err)
    res["selftest_fd_ok"] = bool(err <= 5e-2)
    # Fisher trace consistency: trace(F) == mean_k ||g_k||^2
    G = per_sample_grads(m, criterion, XA, yA, 40)
    trace_direct = float(((G.T @ G) / G.shape[0]).trace())
    norm_sq = float((G ** 2).sum(axis=1).mean())
    res["selftest_fisher_trace_diff"] = abs(trace_direct - norm_sq)
    return res


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def mean_std(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def run(probes):
    report = ["# E5 — mechanism study (cross-domain prior)\n"]
    report.append(f"Checkpoints: {len(probes)} cells; machine ids: "
                  + "; ".join(sorted({p['machine_id'] for p in probes})) + "\n")
    agg = {"per_arm": {}, "probes": probes}

    for arm in ARMS:
        sub = [p for p in probes if p["arm"] == arm]
        if not sub:
            continue
        arm_out = {}
        report.append(f"\n## Arm `{arm}` (n={len(sub)} seeds)\n")

        # accuracies
        acc = {k: mean_std([p["accs"][k] for p in sub])
               for k in ("a_after_a", "a_after_b", "b_after_b")}
        arm_out["accs"] = acc
        report.append(f"- acc(A|A)={acc['a_after_a'][0]:.2f}, "
                      f"acc(A|B)={acc['a_after_b'][0]:.2f}, "
                      f"acc(B|B)={acc['b_after_b'][0]:.2f}")

        # per-layer grad var at each checkpoint
        layer_tab = {}
        for ckpt in ("theta_init", "theta0", "thetaA", "thetaB"):
            per_layer = [p["layer_var_taskA"][ckpt]["per_layer"] for p in sub]
            arr = np.array(per_layer)
            layer_tab[ckpt] = {"per_layer_mean": arr.mean(axis=0).tolist(),
                               "per_layer_std": arr.std(axis=0, ddof=1).tolist(),
                               "vqc_mean": mean_std([p["layer_var_taskA"][ckpt]["vqc_mean"] for p in sub])}
            report.append(f"- Var[grad|A] {ckpt}: per-layer "
                          + ", ".join(f"L{i}={m:.3e}" for i, m in enumerate(arr.mean(axis=0)))
                          + f"; vqc mean {layer_tab[ckpt]['vqc_mean'][0]:.3e}")
        arm_out["layer_var"] = layer_tab

        # Fisher summary
        fisher_tab = {}
        for ckpt in ("theta0", "thetaA", "thetaB"):
            rows = [p["fisher_taskA"][ckpt] for p in sub]
            fisher_tab[ckpt] = {
                "trace": mean_std([r["trace"] for r in rows]),
                "top1_frac": mean_std([r["top1_frac"] for r in rows]),
                "entropy": mean_std([r["spectral_entropy"] for r in rows]),
                "eff_rank": mean_std([r["eff_rank"] for r in rows]),
                "eigs_head_mean": np.mean([r["eigs_head"] for r in rows], axis=0).tolist(),
            }
            ft = fisher_tab[ckpt]
            report.append(f"- Fisher {ckpt}: trace={ft['trace'][0]:.2e}, "
                          f"top1frac={ft['top1_frac'][0]:.3f}, "
                          f"H={ft['entropy'][0]:.2f}, effrank={ft['eff_rank'][0]:.1f}")
        arm_out["fisher"] = fisher_tab

        # barrier
        barr = {
            "dist": mean_std([p["barrier"]["dist_thetaB_thetaA_rms"] for p in sub]),
            "barrierB": mean_std([p["barrier"]["barrier_taskB"] for p in sub]),
            "barrierA": mean_std([p["barrier"]["barrier_taskA"] for p in sub]),
            "curve_taskB_mean": np.mean([p["barrier"]["curve_taskB"] for p in sub], axis=0).tolist(),
            "curve_taskA_mean": np.mean([p["barrier"]["curve_taskA"] for p in sub], axis=0).tolist(),
        }
        arm_out["barrier"] = barr
        report.append(f"- barrier: ||dB-dA||_rms={barr['dist'][0]:.3f}, "
                      f"barrier(B)={barr['barrierB'][0]:+.3f}±{barr['barrierB'][1]:.3f}, "
                      f"barrier(A)={barr['barrierA'][0]:+.3f}±{barr['barrierA'][1]:.3f}")

        # wrap
        wr = {
            "frac_outside_pi": mean_std([p["wrap"]["frac_outside_pi"] for p in sub]),
            "raw_mean": mean_std([p["wrap"]["raw_mean"] for p in sub]),
            "raw_std": mean_std([p["wrap"]["raw_std"] for p in sub]),
            "loss_abs_diff_max": max(p["wrap"]["loss_abs_diff"] for p in sub),
        }
        arm_out["wrap"] = wr
        report.append(f"- angles: mean={wr['raw_mean'][0]:.2f}±{wr['raw_std'][0]:.2f}, "
                      f"frac|θ|>π={wr['frac_outside_pi'][0]:.3f}, "
                      f"max|ΔL| wrapping={wr['loss_abs_diff_max']:.2e}")

        agg["per_arm"][arm] = arm_out

    # selftest summary
    report.append("\n## Self-tests (caveat T1)\n")
    report.append(f"- FD grad self-test max rel err: "
                  f"{max(p['selftest_fd_max_rel_err'] for p in probes):.2e} (rtol 5e-2)")
    report.append(f"- Fisher trace consistency max diff: "
                  f"{max(p['selftest_fisher_trace_diff'] for p in probes):.2e}")
    report.append(f"- wrap invariance max |ΔL|: "
                  f"{max(p['wrap']['loss_abs_diff'] for p in probes):.2e}")

    (OUT / "probes.json").write_text(json.dumps(agg, indent=2, default=str))
    (OUT / "report_e5.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return agg


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def figure(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    only = {}
    for arm in ARMS:
        a = agg["per_arm"].get(arm)
        if a:
            only[arm] = a
    if not only:
        return
    colors = {"scratch": "#e63946", "synth": "#457b9d"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # (a) per-layer gradient variance at theta0
    ax = axes[0]
    x = np.arange(N_LAYERS)
    for arm, a in only.items():
        m = a["layer_var"]["theta0"]["per_layer_mean"]
        s = a["layer_var"]["theta0"]["per_layer_std"]
        ax.errorbar(x, m, yerr=s, marker="o", capsize=3, color=colors[arm],
                    label=arm)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"L{i+1}" for i in x])
    ax.set_xlabel("ansatz layer")
    ax.set_ylabel(r"Var$[\partial\mathcal{L}_A/\partial\theta]$ (per layer)")
    ax.set_title("(a) gradient variance per layer, $\\theta_0$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", ls="--", alpha=0.5)

    # (b) Fisher spectra
    ax = axes[1]
    for arm, a in only.items():
        e = np.asarray(a["fisher"]["theta0"]["eigs_head_mean"])
        ax.plot(np.arange(1, len(e) + 1), e, marker=".", color=colors[arm],
                label=f"{arm} @ $\\theta_0$")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("empirical Fisher eigenvalue")
    ax.set_title("(b) Fisher spectrum, Task A at $\\theta_0$")
    ax.grid(axis="y", ls="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)

    # (c) interpolation curves thetaA -> thetaB (no barrier on either task)
    ax = axes[2]
    t = np.linspace(0, 1, N_INTERP)
    for arm, a in only.items():
        ax.plot(t, a["barrier"]["curve_taskB_mean"], marker="o", ms=3,
                color=colors[arm], label=f"{arm}: Task B loss")
        ax.plot(t, a["barrier"]["curve_taskA_mean"], marker="s", ms=3,
                color=colors[arm], alpha=0.45, ls="--",
                label=f"{arm}: Task A loss")
    ax.set_xlabel("$\\theta_A + t(\\theta_B-\\theta_A)$")
    ax.set_ylabel("training loss (fixed subset)")
    ax.set_title("(c) interpolation $\\theta_A\\to\\theta_B$")
    ax.grid(axis="y", ls="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center")
    fig.tight_layout()
    out = FIGDIR / "fig_e5_mechanism.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure written: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_train(workers):
    jobs = [(a, s) for a in ARMS for s in SEEDS]
    CKPTS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for msg in ex.map(train_one, jobs):
            print(msg, flush=True)


def cmd_probes(workers):
    jobs = [(a, s) for a in ARMS for s in SEEDS
            if (CKPTS / f"{a}_s{s}.pt").exists()]
    if not jobs:
        print("no checkpoints; run `train` first")
        sys.exit(1)
    results = []
    if workers == 1:
        for j in jobs:
            results.append(probe_one(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(probe_one, jobs):
                results.append(r)
    agg = run(results)
    figure(agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "probes", "all"])
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    if args.cmd in ("train", "all"):
        cmd_train(args.workers)
    if args.cmd in ("probes", "all"):
        cmd_probes(args.workers)


if __name__ == "__main__":
    main()
