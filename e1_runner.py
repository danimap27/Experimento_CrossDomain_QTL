"""E1 — controlled 2x2 + CL baselines campaign (checklist 3.1 + 2.3 + 2.4).

One *cell* = (noise profile, seed). A cell runs every arm of the fixed design
with checkpoint sharing where the arms are identical up to a phase:

  Group  scratch05  (Task A at lr 0.05 from random init):
      B1        plain replay-free baseline (0.05 / 0.05)
      B5_l1e2   EWC lam=1e2   (0.05 / 0.05)
      B5_l1e3   EWC lam=1e3   (0.05 / 0.05)
      B5_l1e4   EWC lam=1e4   (0.05 / 0.05)
      B6        Replay 25%    (0.05 / 0.05)
      B7        DER++ 25%     (0.05 / 0.05)  [optional arm of the design]
  Group  scratch01  (Task A at lr 0.01 from random init):
      B2        plain (0.01 / 0.005)   <- LR-confound control
  Group  synth05 / synth01 (share ONE synthetic pre-training checkpoint):
      B3        synth prior (0.05 / 0.05)     <- pre-training without LR drop
      B4        synth prior (0.01 / 0.005)     <- original QTL arm

Persisted per arm and seed (item 2.3): Acc_A_init, Acc_A_final, Acc_B, delta_A,
retention r_A, full loss/accuracy histories, phase timings, theta snapshots
(theta_source, theta_A, theta_B), Fisher summary where computed, the complete
config, library versions and the REAL machine_id of the executing node
(requirement 1.11 -- never the literal 'local').

Design decisions (fixed for this campaign, see card t_687c7d83):
  * Pre-training: 15 epochs at lr 0.05 on the synthetic source, ONE shared
    checkpoint per cell (isolates the fine-tuning LR as the only difference
    between B3 and B4).
  * No layer freezing in any arm (the protocol dropped the freeze claim --
    checklist 1.4). The corrected functional freeze ships in quantum_net.py
    for future ablations.
  * EWC Fisher: empirical diagonal (mean of squared per-minibatch first-order
    gradients over the Task A training set at theta_A). Never double-backward
    through the TorchLayer (caveat T1).
  * Replay/DER++ buffer: 25% of the Task A training set, sampled once per cell
    (identical buffer for B6 and B7). DER++ beta = 0.5.

Usage:
    python e1_runner.py --profile heron_r2 --seed 3        # one cell
    python e1_runner.py --smoke                            # minimal smoke test
    python e1_runner.py --print-cells                      # job list
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import random
import socket
import sys
import time

import numpy as np
import torch

from data_module import DataModule
from quantum_net import HybridQuantumNet, NOISE_PROFILES, get_noise_profile
from continual import (
    ReplayBuffer,
    empirical_fisher,
    evaluate_accuracy,
    ewc_penalty,
    snapshot_params,
    stored_logits,
    train_task,
)

REPO = pathlib.Path(__file__).resolve().parent

PRETRAIN_LR = 0.05
PRETRAIN_EPOCHS = 15
DEFAULT_EPOCHS = 10
DEFAULT_SEEDS = list(range(20))
PROFILES = ["ideal", "heron_r2"]
BUFFER_FRACTION = 0.25
DERPP_BETA = 0.5

ARMS = {
    "B1":      dict(init="scratch", lr_a=0.05,   lr_b=0.05,   method="plain"),
    "B2":      dict(init="scratch", lr_a=0.01,   lr_b=0.005,  method="plain"),
    "B3":      dict(init="synth",   lr_a=0.05,   lr_b=0.05,   method="plain"),
    "B4":      dict(init="synth",   lr_a=0.01,   lr_b=0.005,  method="plain"),
    "B5_l1e2": dict(init="scratch", lr_a=0.05,   lr_b=0.05,   method="ewc", lam=1e2),
    "B5_l1e3": dict(init="scratch", lr_a=0.05,   lr_b=0.05,   method="ewc", lam=1e3),
    "B5_l1e4": dict(init="scratch", lr_a=0.05,   lr_b=0.05,   method="ewc", lam=1e4),
    "B6":      dict(init="scratch", lr_a=0.05,   lr_b=0.05,   method="replay"),
    "B7":      dict(init="scratch", lr_a=0.05,   lr_b=0.05,   method="derpp"),
}

GROUPS = [
    dict(id="scratch05", init="scratch", lr_a=0.05, arms=["B1", "B5_l1e2", "B5_l1e3", "B5_l1e4", "B6", "B7"]),
    dict(id="scratch01", init="scratch", lr_a=0.01, arms=["B2"]),
    dict(id="synth05",   init="synth",   lr_a=0.05, arms=["B3"]),
    dict(id="synth01",   init="synth",   lr_a=0.01, arms=["B4"]),
]


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_machine_id() -> str:
    """REAL node identity (hostname + OS machine-id) -- requirement 1.11."""
    mid = ""
    p = pathlib.Path("/etc/machine-id")
    try:
        mid = p.read_text().strip()[:8]
    except OSError:
        mid = "no-machine-id"
    return f"{socket.gethostname()}:{mid}"


def library_versions() -> dict:
    import pennylane
    import sklearn
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pennylane": pennylane.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }


def sd_to_lists(model) -> dict:
    return {k: np.asarray(v.detach().cpu(), dtype=float).tolist()
            for k, v in model.state_dict().items()}


def theta_to_lists(theta: dict) -> dict:
    return {k: np.asarray(v.detach().cpu(), dtype=float).tolist()
            for k, v in theta.items()}


def fisher_summary(fisher) -> dict:
    out = {}
    for n, f in fisher.items():
        a = f.detach().cpu().numpy()
        out[n] = {"mean": float(a.mean()), "max": float(a.max()),
                  "min": float(a.min()), "frac_lt_1e-8": float((a < 1e-8).mean())}
    return out


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------

def run_cell(profile: str, seed: int, arms: list[str], epochs: int, data_dir: str,
             out_root: pathlib.Path, smoke: bool = False) -> list[pathlib.Path]:
    noise, noise_params = get_noise_profile(profile)
    machine_id = get_machine_id()
    t_cell = time.time()

    n_syn = 250 if smoke else 2500
    lim_tr = 128 if smoke else 1000
    lim_te = 64 if smoke else 200

    set_seed(seed * 10 + 0)
    dm = DataModule(data_dir=data_dir, batch_size=32, n_components=4)
    syn_train, syn_test = dm.get_synthetic_task(n_samples=n_syn)
    fa_train, fa_test, _ = dm.get_fashion_mnist_task(classes=(0, 1))
    mn_train, mn_test, _ = dm.get_mnist_task(classes=(2, 3))
    if smoke:
        # shrink the loaders for the smoke run only
        def shrink(loader, n):
            X = torch.cat([b[0] for b in loader])[:n]
            y = torch.cat([b[1] for b in loader])[:n]
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, y), batch_size=32, shuffle=True)
        fa_train, fa_test = shrink(fa_train, lim_tr), shrink(fa_test, lim_te)
        mn_train, mn_test = shrink(mn_train, lim_tr), shrink(mn_test, lim_te)

    model_kwargs = dict(ansatz="A", n_qubits=4, n_layers=3, n_classes=2,
                        noise=noise, noise_params=noise_params)

    def new_model():
        return HybridQuantumNet(**model_kwargs)

    criterion = torch.nn.CrossEntropyLoss()
    phases = {}
    arm_payloads = {}

    # ---------------- shared synthetic pre-training (B3/B4) ----------------
    theta0 = None
    if any(ARMS[a]["init"] == "synth" for a in arms):
        set_seed(seed * 10 + 1)
        src = new_model()
        hist = train_task(src, syn_train, PRETRAIN_EPOCHS, PRETRAIN_LR,
                          criterion=criterion, log_prefix="[pretrain] ")
        acc_syn = evaluate_accuracy(src, syn_test)
        theta0 = snapshot_params(src)
        phases["pretrain"] = {
            "epochs": PRETRAIN_EPOCHS, "lr": PRETRAIN_LR, "loader": "synthetic",
            "acc_syn_test": acc_syn, **hist,
        }

    # ---------------- shared Task A per group ----------------
    for gi, grp in enumerate(GROUPS):
        grp_arms = [a for a in arms if a in grp["arms"]]
        if not grp_arms:
            continue

        set_seed(seed * 10 + 2 + gi)
        model = new_model()
        if grp["init"] == "synth":
            model.load_state_dict(theta0)

        hist_a = train_task(model, fa_train, epochs, grp["lr_a"],
                            criterion=criterion, log_prefix=f"[{grp['id']}/A] ")
        acc_a_init = evaluate_accuracy(model, fa_test)
        theta_A = snapshot_params(model)
        phases[grp["id"] + "/task_a"] = {
            "epochs": epochs, "lr": grp["lr_a"], "init": grp["init"],
            "acc_a_init": acc_a_init, **hist_a,
        }

        # arm-shared extras (computed once at theta_A)
        ewc_shared = None
        if any(ARMS[a]["method"] == "ewc" for a in grp_arms):
            t0 = time.time()
            fisher = empirical_fisher(model, criterion, fa_train)
            ewc_shared = dict(fisher=fisher, theta_star=theta_A,
                              fisher_summary=fisher_summary(fisher),
                              fisher_time=time.time() - t0)
        buffer = None
        derpp_logits = None
        if any(ARMS[a]["method"] in ("replay", "derpp") for a in grp_arms):
            gen = torch.Generator().manual_seed(seed * 100 + 7)
            buffer = ReplayBuffer(fa_train.dataset, fraction=BUFFER_FRACTION, rng=gen)

        # ---------------- Task B per arm (from the same theta_A) ------------
        for arm in grp_arms:
            spec = ARMS[arm]
            set_seed(seed * 10 + 5)  # identical batch order for every arm
            m = new_model()
            m.load_state_dict(theta_A)

            ewc = None
            ewc_extra = {}
            if spec["method"] == "ewc":
                ewc = dict(fisher=ewc_shared["fisher"], theta_star=ewc_shared["theta_star"],
                           lam=spec["lam"])
                ewc_extra = {"fisher_summary": ewc_shared["fisher_summary"],
                             "fisher_time_s": ewc_shared["fisher_time"]}

            replay_dataset, replay_mode, replay_logits = None, None, None
            if spec["method"] == "replay":
                replay_dataset, replay_mode = buffer.dataset, "er"
            elif spec["method"] == "derpp":
                derpp_logits = stored_logits(m, buffer.loader(batch_size=32, shuffle=False))
                replay_dataset, replay_mode, replay_logits = buffer.dataset, "derpp", derpp_logits

            hist_b = train_task(
                m, mn_train, epochs, spec["lr_b"], criterion=criterion,
                eval_old_loader=fa_test,
                ewc=ewc, replay_dataset=replay_dataset, replay_mode=replay_mode,
                replay_logits=replay_logits, beta=DERPP_BETA,
                log_prefix=f"[{arm}/B] ")

            acc_a_final = evaluate_accuracy(m, fa_test)
            acc_b_final = evaluate_accuracy(m, mn_test)
            delta_a = acc_a_init - acc_a_final
            r_a = acc_a_final / acc_a_init if acc_a_init else float("nan")

            arm_payloads[arm] = {
                "experiment": "e1_controlled_v1",
                "arm": arm,
                "arm_spec": spec,
                "group": grp["id"],
                "seed": seed,
                "noise_profile": profile,
                "noise_params": noise_params,
                "machine_id": machine_id,
                "hostname": socket.gethostname(),
                "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_cell)),
                "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cell_wall_time_s": time.time() - t_cell,
                # --- headline metrics (item 2.3) ---
                "acc_a_init": acc_a_init,
                "acc_a_final": acc_a_final,
                "acc_b_final": acc_b_final,
                "delta_a": delta_a,
                "retention_r_a": r_a,
                # --- full payload ---
                "phases": {"pretrain": phases.get("pretrain"),
                           "task_a": phases[grp["id"] + "/task_a"],
                           "task_b": hist_b},
                "ewc": ewc_extra,
                "replay": ({"buffer_size": len(buffer), "buffer_fraction": BUFFER_FRACTION,
                            "mode": replay_mode, "beta": DERPP_BETA if replay_mode == "derpp" else None}
                           if buffer is not None else None),
                "weights": {"theta_A": theta_to_lists(theta_A), "theta_B": sd_to_lists(m)},
                "config": {
                    "epochs_per_task": epochs, "pretrain_epochs": PRETRAIN_EPOCHS,
                    "pretrain_lr": PRETRAIN_LR, "lr_a": spec["lr_a"], "lr_b": spec["lr_b"],
                    "batch_size": 32, "n_qubits": 4, "n_layers": 3, "ansatz": "A",
                    "n_classes": 2, "freeze": None,
                    "data": {"synthetic": n_syn, "fashion_train": lim_tr, "fashion_test": lim_te,
                             "mnist_train": lim_tr, "mnist_test": lim_te,
                             "classes_A": [0, 1], "classes_B": [2, 3]},
                },
                "library_versions": library_versions(),
            }

    # ---------------- persist one JSON per arm ----------------
    tag = "e1_smoke" if smoke else "e1"
    written = []
    for arm, payload in arm_payloads.items():
        run_dir = out_root / f"{tag}_{arm}__{profile}__s{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"results_seed_{seed}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path)

    manifest = {
        "profile": profile, "seed": seed, "machine_id": machine_id,
        "arms": list(arm_payloads), "cells_wall_time_s": time.time() - t_cell,
        "files": [str(p) for p in written],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    man_path = out_root / f"{tag}_manifest__{profile}__s{seed}.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    written.append(man_path)

    print(f"[cell done] profile={profile} seed={seed} machine_id={machine_id} "
          f"arms={len(arm_payloads)} wall={time.time() - t_cell:.1f}s")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="ideal", choices=list(NOISE_PROFILES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out-root", default=str(REPO / "results"))
    ap.add_argument("--smoke", action="store_true",
                    help="Minimal configuration: 1 seed, arms B1+B4, 2 epochs, tiny data.")
    ap.add_argument("--print-cells", action="store_true",
                    help="Print the campaign job list (profile seed) and exit.")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.print_cells:
        for p in PROFILES:
            for s in DEFAULT_SEEDS:
                print(p, s)
        return

    arms = args.arms or (["B1", "B4"] if args.smoke else list(ARMS))
    epochs = 2 if args.smoke else args.epochs

    print(f"=== E1 cell | profile={args.profile} seed={args.seed} arms={arms} "
          f"epochs={epochs} machine_id={get_machine_id()} ===")
    run_cell(args.profile, args.seed, arms, epochs, args.data_dir,
             pathlib.Path(args.out_root), smoke=args.smoke)


if __name__ == "__main__":
    main()
