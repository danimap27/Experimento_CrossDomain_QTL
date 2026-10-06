"""E2/E3/E4 — scenario, data-scale and qubit/depth scaling campaigns.

Generalises the E1 two-task cell to *arbitrary task sequences* so that one
runner covers the three journal experiments:

  E2  scenario definition (checklist 3.2):
        chain4  = FMNIST{0,1} -> MNIST{2,3} -> KMNIST{4,5} -> FMNIST{8,9}
                  (yields TIL-2 / CIL-2 after 2 tasks, TIL-3 / CIL-3 after 3,
                   CIL-4 after 4 from a single training run)
        smnist5 / sfmnist5 = the community-standard class-IL benchmarks:
                   split-MNIST / split-Fashion-MNIST, 5 tasks x 2 classes,
                   shared head, evaluated with and without task oracle.
        scifar5 = split-CIFAR-10 on cached MobileNetV2+PCA features.
  E3  data scale (checklist 3.3):
        pair2 with --limit-train {500, 2000, 12000} (full binary splits).
  E4  scaling (checklist 3.4):
        pair2 with --n-qubits {4,6,8} (PCA dimension = # qubits) and
        --n-layers {2,3,4}; task-count scaling comes from E2's chain.

Protocol (fixed for this campaign, consistent with the E1 controlled design):
  * Matched learning rates: every arm trains at lr 0.05 in every task, so the
    scratch-vs-synth contrast is *not* contaminated by the E1 LR confound
    (checklist 1.5 / 3.1).
  * `synth` arms share one synthetic pre-training checkpoint (2500 samples,
    15 epochs, lr 0.05) per cell, identical to E1 (B3 protocol).
  * `er`   = rehearsal with a 25% accumulated reservoir of every past task.
  * `ewc`  = sequential Fisher penalties from every completed task (lam 1e4).
  * `si`   = synaptic intelligence (Zenke et al. 2017): online importance
    accumulation with lam 5e3, usable on top of scratch or synth.
  * `l2`   = uniform L2 drift penalty towards every past-task optimum (lam 5e3).
  * `derpp` = dark experience replay++: 25% reservoir like `er` plus the
    stored logits of every buffered chunk (beta 0.5, same recipe as E1/B7).
  * `synth_si` / `synth_l2` / `synth_derpp` = the same mechanisms initialised
    from the synthetic prior (init x method grid of the E6 study).
  * `--label-mode global` remaps every task's labels to their global class
    offset (2*i) BEFORE training/evaluation, so the shared multi-class head
    allocates a distinct pair of output units per task: this is the true
    task-IL/class-IL protocol for chain4/smnist5/sfmnist5. The default
    `local` mode keeps the two local labels (a shared binary readout trained
    sequentially on both tasks, the semantics of the E1 two-task protocol);
    it is kept for the pair2 cells so the scale/scaling axes remain
    comparable with Experiment 1.
  * Metrics after every task: full accuracy matrices for class-IL (argmax over
    classes seen so far), class-IL over the full head, and task-IL (argmax
    restricted to the task's classes), plus AA/AF/BWT/FWT (van de Ven 2022).
  * Payload: complete config, histories, timings, REAL machine_id, library
    versions (same persistence contract as E1).

Usage:
    python e234_runner.py --campaign chain4 --arm synth --seed 0
    python e234_runner.py --campaign pair2 --arm scratch --seed 3 \
        --limit-train 12000 --limit-test 2000 --tag sz12k
    python e234_runner.py --campaign pair2 --arm synth --seed 0 --n-qubits 8 \
        --n-components 8 --tag q8
    python e234_runner.py --extract-cifar --classes 0 1   # cache MobileNet feats
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import random
import socket
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_module import DataModule
from quantum_net import HybridQuantumNet, NOISE_PROFILES, get_noise_profile
from continual import (
    ReplayBuffer,
    SITracker,
    empirical_fisher,
    evaluate_accuracy,
    snapshot_params,
    train_task,
)

REPO = pathlib.Path(__file__).resolve().parent

DEFAULT_EPOCHS = 10
PRETRAIN_EPOCHS = 15
PRETRAIN_LR = 0.05
PRETRAIN_SAMPLES = 2500
LR = 0.05                      # matched-rate protocol for every arm
BUFFER_FRACTION = 0.25         # ER reservoir fraction per task
EWC_LAM = 1e4
EWC_MAX_BATCHES = 60           # fisher batches per completed task (cost control)
BASE_SEED = 10                 # E1 seed conventions

CAMPAIGNS = {
    "chain4": dict(
        tasks=[("fashion_mnist", (0, 1)), ("mnist", (2, 3)),
               ("kmnist", (4, 5)), ("fashion_mnist", (8, 9))],
        limits=(1000, 200)),
    "pair2": dict(
        tasks=[("fashion_mnist", (0, 1)), ("mnist", (2, 3))],
        limits=(1000, 200)),
    "smnist5": dict(
        tasks=[("mnist", (0, 1)), ("mnist", (2, 3)), ("mnist", (4, 5)),
               ("mnist", (6, 7)), ("mnist", (8, 9))],
        limits=(12000, 2000)),
    "sfmnist5": dict(
        tasks=[("fashion_mnist", (0, 1)), ("fashion_mnist", (2, 3)),
               ("fashion_mnist", (4, 5)), ("fashion_mnist", (6, 7)),
               ("fashion_mnist", (8, 9))],
        limits=(12000, 2000)),
    "scifar5": dict(
        tasks=[("cifar10", (0, 1)), ("cifar10", (2, 3)), ("cifar10", (4, 5)),
               ("cifar10", (6, 7)), ("cifar10", (8, 9))],
        limits=(2000, 400)),
}

ARMS = ["scratch", "synth", "er", "ewc", "si", "l2", "derpp",
        "synth_si", "synth_l2", "synth_derpp"]
PROFILES = ["ideal", "heron_r2"]
SI_LAM = 5.0                    # synaptic-intelligence penalty (calibrated:
                                # the QTCL-scale 5e3 freezes this model)
L2_LAM = 0.2                    # uniform L2 drift penalty (same calibration)
DERPP_BETA = 0.5                # DER++ logit-matching weight (same as E1)


def arm_parts(arm):
    """Split an arm name into (init, method).

    init   in {scratch, synth}: whether the model starts from the synthetic
           prior or from a random initialisation.
    method in {None, 'er', 'ewc', 'si', 'l2'}: the continual-learning
           mechanism applied on top of the initialisation, keeping every
           scratch/synth pair comparable under the matched-rate protocol.
    """
    if arm.startswith("synth"):
        method = arm[len("synth_"):] if arm != "synth" else None
        return "synth", method
    return "scratch", (None if arm == "scratch" else arm)


# ---------------------------------------------------------------------------
# Bookkeeping helpers (same contract as e1_runner.py)
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_machine_id() -> str:
    mid = ""
    try:
        mid = pathlib.Path("/etc/machine-id").read_text().strip()[:8]
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_task(dm: DataModule, dataset: str, classes, limit_train: int,
              limit_test: int, cifar_cache: pathlib.Path):
    """Task loaders for one (dataset, classes) pair with explicit limits."""
    if dataset == "mnist":
        tr, te, pca = dm.get_mnist_task(classes=classes, limit_train=limit_train,
                                        limit_test=limit_test)
    elif dataset == "fashion_mnist":
        tr, te, pca = dm.get_fashion_mnist_task(classes=classes,
                                                limit_train=limit_train,
                                                limit_test=limit_test)
    elif dataset == "kmnist":
        tr, te, pca = dm.get_kmnist_task(classes=classes, limit_train=limit_train,
                                         limit_test=limit_test)
    elif dataset == "cifar10":
        tr, te, pca = load_cifar_task(dm, classes, limit_train, limit_test,
                                      cifar_cache)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return tr, te, pca


def _cifar_cache_path(cache_dir: pathlib.Path, classes, limit_train, limit_test):
    return cache_dir / f"cifar10_{classes[0]}_{classes[1]}_tr{limit_train}_te{limit_test}.npz"


def extract_cifar_features(dm: DataModule, classes, limit_train, limit_test,
                           cache_dir: pathlib.Path):
    """Extract + cache MobileNetV2 features for one CIFAR-10 class pair.

    The transform follows the original pipeline (Resize 224 + ImageNet
    normalisation); features are the 1280-dim penultimate activations. Cached
    as .npz so parallel campaign cells never re-run the CNN.
    """
    import torchvision
    import torchvision.transforms as transforms

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cifar_cache_path(cache_dir, classes, limit_train, limit_test)
    if path.exists():
        return path

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dm._ensure_mobilenet()
    train_set = torchvision.datasets.CIFAR10(root=dm.data_dir, train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=dm.data_dir, train=False,
                                            download=True, transform=transform)

    def extract(dataset, limit):
        feats, labels = [], []
        with torch.no_grad():
            for img, label in dataset:
                if label in classes:
                    out = dm.mobilenet(img.unsqueeze(0)).squeeze().numpy()
                    feats.append(out)
                    labels.append(classes.index(label))
                    if len(feats) >= limit:
                        break
        return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    print(f"[cifar] extracting pair {classes} train<={limit_train}", flush=True)
    Xtr, ytr = extract(train_set, limit_train)
    print(f"[cifar] extracting pair {classes} test<={limit_test}", flush=True)
    Xte, yte = extract(test_set, limit_test)
    np.savez_compressed(path, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    print(f"[cifar] cached {path}", flush=True)
    return path


def load_cifar_task(dm: DataModule, classes, limit_train, limit_test, cache_dir):
    from sklearn.decomposition import PCA
    path = _cifar_cache_path(cache_dir, classes, limit_train, limit_test)
    if not path.exists():
        extract_cifar_features(dm, classes, limit_train, limit_test, cache_dir)
    blob = np.load(path)
    Xtr, ytr = blob["X_train"], blob["y_train"]
    Xte, yte = blob["X_test"], blob["y_test"]
    pca = PCA(n_components=min(dm.n_components, Xtr.shape[0], Xtr.shape[1]))
    Xtr_p = pca.fit_transform(Xtr)
    Xte_p = pca.transform(Xte)
    # same [0, pi] normalisation as the rest of the pipeline (per split)
    Xtr_p = (Xtr_p - Xtr_p.min(axis=0)) / (Xtr_p.max(axis=0) - Xtr_p.min(axis=0) + 1e-8) * np.pi
    Xte_p = (Xte_p - Xte_p.min(axis=0)) / (Xte_p.max(axis=0) - Xte_p.min(axis=0) + 1e-8) * np.pi
    tr = DataLoader(TensorDataset(torch.tensor(Xtr_p, dtype=torch.float32),
                                  torch.tensor(ytr, dtype=torch.long)),
                    batch_size=dm.batch_size, shuffle=True)
    te = DataLoader(TensorDataset(torch.tensor(Xte_p, dtype=torch.float32),
                                  torch.tensor(yte, dtype=torch.long)),
                    batch_size=dm.batch_size, shuffle=False)
    return tr, te, pca


def dataset_size(loader) -> int:
    return len(loader.dataset)


# ---------------------------------------------------------------------------
# Evaluation (class-IL seen-so-far / full-head / task-IL oracle)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_on_task(model, loader, offset, mode, seen_max):
    """Accuracy (%) of the model on one task's test set.

    mode='til'  -> argmax restricted to the task's two classes (task oracle)
    mode='cil'  -> argmax over classes [0, seen_max) (class-IL, seen so far)
    labels `y` are global (offset + 0/1).
    """
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        logits = model(X)
        if mode == "til":
            local = logits[:, offset:offset + 2]
            preds = local.argmax(dim=1) + offset
        else:
            masked = logits.clone()
            masked[:, seen_max:] = float("-inf")
            preds = masked.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    model.train()
    return 100.0 * correct / max(total, 1)


def eval_matrix_row(model, loaders, t, offsets, mode, seen_max):
    return [eval_on_task(model, loaders[j], offsets[j], mode, seen_max)
            for j in range(t + 1)]


def cl_metrics(matrix, t_final, b0, before):
    """AA / AF / BWT / FWT from the accuracy matrix A[i][j] (i>=j) for T tasks.

    matrix[i][j] = acc on task j after training task i (i=0..T-1), rows trimmed
                   to the defined entries (j <= i).
    b0[j]        = acc of the *initial* model on task j (before any training).
    before[j]    = acc on task j of the model as it enters task j (= zero-shot
                   transfer after tasks 0..j-1; equals b0[0] at j=0).
    """
    T = t_final + 1
    A = matrix
    if T == 1:
        return {"aa": A[0][0], "af": 0.0, "bwt": 0.0, "fwt": 0.0}
    aa = float(np.mean([A[T - 1][j] for j in range(T)]))
    af = float(np.mean([A[j][j] - A[T - 1][j] for j in range(T)]))
    bwt = float(np.mean([A[T - 1][j] - A[j][j] for j in range(T - 1)]))
    fwt = float(np.mean([before[j] - b0[j] for j in range(1, T)]))
    return {"aa": aa, "af": af, "bwt": bwt, "fwt": fwt}


# ---------------------------------------------------------------------------
# Run one cell
# ---------------------------------------------------------------------------

def run_cell(args) -> pathlib.Path | None:
    camp = CAMPAIGNS[args.campaign]
    noise, noise_params = get_noise_profile(args.profile)

    limit_train = args.limit_train or camp["limits"][0]
    limit_test = args.limit_test or camp["limits"][1]
    nq = args.n_qubits
    ncomp = args.n_components or nq
    nl = args.n_layers
    tag = args.tag or "std"
    cell_init, cell_method = arm_parts(args.arm)

    out_dir = (pathlib.Path(args.out_root) /
               f"e234_{args.campaign}__{tag}__{args.arm}__{args.profile}__s{args.seed}")
    out_path = out_dir / f"results_seed_{args.seed}.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path}")
        return None

    t_cell = time.time()
    machine_id = get_machine_id()
    seed = args.seed
    tasks = camp["tasks"]
    T = len(tasks)
    n_classes = 2 * T
    offsets = [2 * i for i in range(T)]

    # ---------------- data ----------------
    dm = DataModule(data_dir=args.data_dir, batch_size=32, n_components=ncomp)
    loaders, tests, seq_info = [], [], []
    for i, (ds, classes) in enumerate(tasks):
        tr, te, _ = load_task(dm, ds, classes, limit_train, limit_test,
                              pathlib.Path(args.cifar_cache))
        loaders.append(tr)
        tests.append(te)
        seq_info.append({"dataset": ds, "classes": list(classes), "offset": offsets[i],
                         "n_train": dataset_size(tr), "n_test": dataset_size(te)})

    if args.label_mode == "global":
        # DataModule emits LOCAL labels (classes.index -> 0/1); remap them to
        # the global class offsets so the shared multi-class head allocates a
        # distinct output pair per task (true task-IL / class-IL).
        for i in range(T):
            for ld in (loaders[i], tests[i]):
                ds_ = ld.dataset
                X_, y_ = ds_.tensors
                ds_.tensors = (X_, y_ + 2 * i)

    model_kwargs = dict(ansatz="A", n_qubits=nq, n_layers=nl, n_classes=n_classes,
                        noise=noise, noise_params=noise_params)
    criterion = nn.CrossEntropyLoss()

    def new_model():
        return HybridQuantumNet(**model_kwargs)

    payload = {
        "experiment": "e234_v1",
        "campaign": args.campaign,
        "tag": tag,
        "arm": args.arm,
        "seed": seed,
        "noise_profile": args.profile,
        "noise_params": noise_params,
        "machine_id": machine_id,
        "hostname": socket.gethostname(),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_cell)),
        "sequence": seq_info,
        "config": {
            "epochs_per_task": args.epochs, "lr": LR, "batch_size": 32,
            "n_qubits": nq, "n_components": ncomp, "n_layers": nl, "ansatz": "A",
            "n_classes": n_classes, "limit_train": limit_train,
            "limit_test": limit_test, "arm": args.arm,
            "label_mode": args.label_mode,
            "method": {"er": "rehearsal 25% accumulated",
                       "derpp": "dark experience replay++ 25% (beta 0.5)",
                       "si": "synaptic intelligence",
                       "l2": "uniform L2 drift penalty"}.get(cell_method, cell_method or args.arm),
            "ewc_lam": EWC_LAM if cell_method == "ewc" else None,
            "reg_lam": (args.si_lam if cell_method == "si" else
                        (args.l2_lam if cell_method == "l2" else None)),
            "pretrain": {"samples": PRETRAIN_SAMPLES, "epochs": PRETRAIN_EPOCHS,
                         "lr": PRETRAIN_LR} if cell_init == "synth" else None,
        },
    }

    # ---------------- initialisation ----------------
    theta0 = None
    if cell_init == "synth":
        set_seed(seed * BASE_SEED + 1)
        src = new_model()
        syn_tr, syn_te = dm.get_synthetic_task(n_samples=args.pretrain_samples)
        hist = train_task(src, syn_tr, args.pretrain_epochs, PRETRAIN_LR,
                          criterion=criterion, log_prefix="[pretrain] ")
        payload["pretrain"] = {"acc_syn_test": evaluate_accuracy(src, syn_te), **hist}
        theta0 = snapshot_params(src)

    set_seed(seed * BASE_SEED + 0)
    model = new_model()
    if theta0 is not None:
        model.load_state_dict(theta0)

    # initial-model accuracy on every task (FWT b0)
    b0_cil = [eval_on_task(model, tests[j], offsets[j], "cil", 2 * T) for j in range(T)]
    b0_til = [eval_on_task(model, tests[j], offsets[j], "til", 2 * T) for j in range(T)]

    # ---------------- sequential training ----------------
    matrices = {m: [[None] * T for _ in range(T)] for m in ("cil", "cil_full", "til")}
    before = {m: [None] * T for m in ("cil", "til")}   # acc just before training task j
    phases = {}
    ewc_chain = []
    buffers = []
    replay_all = None
    si_tracker = SITracker(lam=args.si_lam) if cell_method == "si" else None
    l2_anchors = [] if cell_method == "l2" else None
    derpp_z = [] if cell_method == "derpp" else None

    for t in range(T):
        # accuracy on task t of the model as it enters task t (b_j)
        before["cil"][t] = eval_on_task(model, tests[t], offsets[t], "cil", 2 * T)
        before["til"][t] = eval_on_task(model, tests[t], offsets[t], "til", 2 * T)

        set_seed(seed * BASE_SEED + 50 + t)  # identical batch order across arms
        replay_dataset = None
        replay_logits = None
        if cell_method in ("er", "derpp") and buffers:
            replay_dataset = TensorDataset(
                torch.cat([b.dataset.tensors[0] for b in buffers]),
                torch.cat([b.dataset.tensors[1] for b in buffers]))
            if cell_method == "derpp":
                replay_logits = TensorDataset(
                    replay_dataset.tensors[0], replay_dataset.tensors[1],
                    torch.cat(derpp_z))
        ewc_arg = ewc_chain if (cell_method == "ewc" and ewc_chain) else None
        l2_arg = {"anchors": l2_anchors, "lam": args.l2_lam} if cell_method == "l2" else None

        hist = train_task(model, loaders[t], args.epochs, LR, criterion=criterion,
                          ewc=ewc_arg, replay_dataset=replay_dataset,
                          replay_mode=("derpp" if cell_method == "derpp" else "er"),
                          replay_logits=replay_logits, beta=DERPP_BETA,
                          si=si_tracker, l2=l2_arg,
                          log_prefix=f"[{args.arm}/T{t}] ")

        row_cil = eval_matrix_row(model, tests, t, offsets, "cil", 2 * (t + 1))
        row_full = eval_matrix_row(model, tests, t, offsets, "cil", 2 * T)
        row_til = eval_matrix_row(model, tests, t, offsets, "til", 2 * T)
        for j in range(t + 1):
            matrices["cil"][t][j] = row_cil[j]
            matrices["cil_full"][t][j] = row_full[j]
            matrices["til"][t][j] = row_til[j]

        phases[f"task_{t}"] = {
            "dataset": seq_info[t]["dataset"], "classes": seq_info[t]["classes"],
            "epochs": args.epochs, "lr": LR, "loss": hist["loss"],
            "epoch_time": hist["epoch_time"], "train_time": hist["train_time"],
            "acc_after": {"cil": row_cil, "til": row_til},
        }

        if cell_method == "ewc":
            fisher = empirical_fisher(model, criterion, loaders[t],
                                      max_batches=EWC_MAX_BATCHES)
            ewc_chain.append({"fisher": fisher, "theta_star": snapshot_params(model),
                              "lam": EWC_LAM})
        if cell_method in ("er", "derpp"):
            gen = torch.Generator().manual_seed(seed * 100 + 7 + t)
            buf = ReplayBuffer(loaders[t].dataset, fraction=BUFFER_FRACTION, rng=gen)
            buffers.append(buf)
            if cell_method == "derpp":
                with torch.no_grad():
                    derpp_z.append(model(buf.dataset.tensors[0]).detach())
        if cell_method == "si":
            si_tracker.consolidate(model)
        if cell_method == "l2":
            l2_anchors.append(snapshot_params(model))

    # ---------------- metrics ----------------
    metrics = {}
    for m in ("cil", "cil_full", "til"):
        mm = [[v for v in row if v is not None] for row in matrices[m]]
        cl = cl_metrics(mm, T - 1, b0_til if m == "til" else b0_cil,
                        before[m] if m in before else before["cil"])
        cl["final_accs"] = mm[T - 1]
        cl["matrix"] = mm
        cl["before"] = before[m] if m in before else None
        metrics[m] = cl
    metrics["b0"] = {"cil": b0_cil, "til": b0_til}

    payload["metrics"] = metrics
    payload["phases"] = phases
    payload["cell_wall_time_s"] = time.time() - t_cell
    payload["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["library_versions"] = library_versions()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[done] {out_path} wall={payload['cell_wall_time_s']:.1f}s "
          f"machine_id={machine_id}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="chain4", choices=list(CAMPAIGNS))
    ap.add_argument("--arm", default="scratch", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="ideal", choices=list(NOISE_PROFILES))
    ap.add_argument("--label-mode", choices=["local", "global"], default="local",
                    help="global: remap per-task labels to their class offsets "
                         "(true task-IL/class-IL with the shared head); "
                         "local: keep the two local labels (shared binary "
                         "readout, the E1 two-task protocol semantics).")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--n-components", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--pretrain-samples", type=int, default=PRETRAIN_SAMPLES)
    ap.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    ap.add_argument("--si-lam", type=float, default=SI_LAM,
                    help="synaptic-intelligence penalty strength (arm 'si')")
    ap.add_argument("--l2-lam", type=float, default=L2_LAM,
                    help="uniform L2 drift penalty strength (arm 'l2')")
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--cifar-cache", default=str(REPO / "cache" / "cifar_feats"))
    ap.add_argument("--out-root", default=str(REPO / "results"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extract-cifar", action="store_true",
                    help="Only extract + cache CIFAR features and exit.")
    ap.add_argument("--classes", nargs=2, type=int, default=[0, 1])
    ap.add_argument("--print-jobs", action="store_true",
                    help="Print the default campaign job list and exit.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.extract_cifar:
        dm = DataModule(data_dir=args.data_dir, batch_size=32,
                        n_components=args.n_components or args.n_qubits)
        extract_cifar_features(dm, tuple(args.classes), args.limit_train or 2000,
                               args.limit_test or 400,
                               pathlib.Path(args.cifar_cache))
        return
    if args.print_jobs:
        for camp in ("chain4", "smnist5", "sfmnist5", "scifar5"):
            for arm in ("scratch", "synth"):
                for s in range(10):
                    print(f"--campaign {camp} --arm {arm} --seed {s}")
        return
    run_cell(args)


if __name__ == "__main__":
    main()
