"""Continual-learning utilities for the hybrid VQC pipeline (checklist 3.7).

Implements the baselines requested by the reviewers (EWC, rehearsal) and the
optional DER++ arm for Experiment E1:

  * `empirical_fisher`  — Fisher diagonal by *empirical accumulation*: the
    mean of the squared per-minibatch gradients of the task loss, computed
    with FIRST-ORDER autograd only.
  * `ewc_penalty`       — lambda/2 * sum_i F_i (theta_i - theta*_i)^2.
  * `ReplayBuffer`      — fixed reservoir of past-task samples (25% default).
  * `train_task`        — task training loop supporting the EWC penalty and
    rehearsal (ER) / DER++ batches.

Caveat T1 compliance: curvature/gradient-diagnostics for this model must
never use double-backward through the PennyLane TorchLayer (validated as
unreliable in `analysis/mechanism/diagnostics/`). Everything here is either
first-order gradients (squared for the Fisher diagonal) or finite differences
(`fd_grad_selftest` below cross-checks the first-order autograd gradients of
the VQC weights against central finite differences in float64).
"""

from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Fisher diagonal (empirical accumulation, first-order only)
# ---------------------------------------------------------------------------

def empirical_fisher(model, criterion, loader, max_batches=None):
    """Diagonal of the empirical Fisher: F_i = mean_b (dL_b/dtheta_i)^2.

    One backward pass per minibatch of `loader` (the task data at the end of
    the task). Returns a dict keyed by parameter name with a detached tensor
    of the same shape as the parameter.
    """
    fisher = {n: torch.zeros_like(p, memory_format=torch.preserve_format)
              for n, p in model.named_parameters() if p.requires_grad}
    n_batches = 0
    for bi, (X, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        model.zero_grad(set_to_none=False)
        loss = criterion(model(X), y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        n_batches += 1
    if n_batches == 0:
        raise ValueError("empirical_fisher: empty loader")
    for n in fisher:
        fisher[n] /= n_batches
    return fisher


def snapshot_params(model):
    """Detached copy of the trainable parameters (anchor theta* for EWC)."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def ewc_penalty(model, fisher, theta_star, lam):
    """EWC quadratic penalty: lam/2 * sum_i F_i (theta_i - theta*_i)^2."""
    total = None
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        term = (fisher[n] * (p - theta_star[n]) ** 2).sum()
        total = term if total is None else total + term
    return 0.5 * lam * total


def l2_penalty(model, anchors, lam):
    """Uniform L2 drift penalty: lam/2 * sum_anchors sum_i (theta_i - theta*_i)^2.

    `anchors` is a list of `snapshot_params` dicts (every completed task
    optimum). Returns a zero tensor when the list is empty (first task).
    """
    if not anchors:
        return torch.zeros((), dtype=next(model.parameters()).dtype)
    total = None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for anchor in anchors:
            term = ((p - anchor[name].to(p.device)) ** 2).sum()
            total = term if total is None else total + term
    return 0.5 * lam * total


class SITracker:
    """Synaptic Intelligence (Zenke et al. 2017) for `train_task`.

    Per-parameter importance is accumulated online during training as the
    path integral of each weight's contribution to the loss decrease
    (`pre_step`/`post_step` around every optimisation step), consolidated at
    the end of every task (`consolidate`), and applied as a quadratic penalty
    from the second task onwards (`penalty`). First-order gradients only
    (caveat T1 compliance).
    """

    def __init__(self, lam: float = 5e3, epsilon: float = 1e-3):
        self.lam = lam
        self.epsilon = epsilon
        self._omega = {}
        self._w = {}
        self._prev = {}
        self._anchor = {}
        self._n_tasks = 0

    def _ensure_init(self, model):
        if self._omega:
            return
        for n, p in model.named_parameters():
            if p.requires_grad:
                self._omega[n] = torch.zeros_like(p)
                self._w[n] = torch.zeros_like(p)
                self._prev[n] = p.detach().clone()
                self._anchor[n] = p.detach().clone()

    def pre_step(self, model):
        self._ensure_init(model)

    def post_step(self, model):
        """Accumulate the online contribution w += -grad * delta after a step."""
        self._ensure_init(model)
        for n, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            delta = p.detach() - self._prev[n]
            self._w[n] += -p.grad.detach() * delta
            self._prev[n] = p.detach().clone()

    def consolidate(self, model):
        """Fold the task's online contribution into omega and re-anchor."""
        self._ensure_init(model)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            theta_curr = p.detach().clone()
            change = (theta_curr - self._anchor[n]).pow(2) + self.epsilon
            self._omega[n] += torch.clamp(self._w[n] / change, min=0)
            self._w[n].zero_()
            self._anchor[n] = theta_curr
            self._prev[n] = theta_curr.clone()
        self._n_tasks += 1

    def penalty(self, model):
        if self._n_tasks == 0 or not self._omega:
            return torch.zeros((), dtype=next(model.parameters()).dtype)
        total = None
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._omega:
                term = (self._omega[n].to(p.device) *
                        (p - self._anchor[n].to(p.device)) ** 2).sum()
                total = term if total is None else total + term
        if total is None:
            return torch.zeros((), dtype=next(model.parameters()).dtype)
        return self.lam * total

    def n_tasks_seen(self) -> int:
        return self._n_tasks


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed random subset of a task's training data (simple rehearsal)."""

    def __init__(self, dataset, fraction=0.25, rng=None):
        n = len(dataset)
        k = max(1, int(round(fraction * n)))
        rng = rng if rng is not None else torch.Generator().manual_seed(0)
        idx = torch.randperm(n, generator=rng)[:k]
        X = torch.stack([dataset[i][0] for i in idx])
        y = torch.tensor([int(dataset[i][1]) for i in idx], dtype=torch.long)
        self.dataset = TensorDataset(X, y)
        self.fraction = fraction

    def __len__(self):
        return len(self.dataset)

    def loader(self, batch_size=32, shuffle=True, generator=None):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def stored_logits(model, loader):
    """Logits of the buffer samples at the end of the previous task (DER++)."""
    out_x, out_y, out_z = [], [], []
    for X, y in loader:
        out_x.append(X)
        out_y.append(y)
        out_z.append(model(X))
    return TensorDataset(torch.cat(out_x), torch.cat(out_y), torch.cat(out_z))


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(model, loader):
    """Accuracy (%) of `model` on `loader` (mirrors ExperimentRunner.evaluate)."""
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        preds = torch.argmax(model(X), dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    model.train()
    return (correct / total) * 100.0


# ---------------------------------------------------------------------------
# Training loop with optional EWC penalty and rehearsal
# ---------------------------------------------------------------------------

def train_task(model, train_loader, epochs, lr, criterion=None, eval_old_loader=None,
               ewc=None, replay_dataset=None, replay_mode=None, replay_logits=None,
               beta=0.5, si=None, l2=None, log_prefix=""):
    """Train `model` on one task.

    ewc            -- dict(fisher=..., theta_star=..., lam=...) to add the EWC
                      penalty to every minibatch loss (Task B regularisation).
    replay_dataset -- TensorDataset with past-task samples; when given, each
                      step also trains on a rehearsal batch:
                        replay_mode='er'    -> CE on the rehearsal batch (ER)
                        replay_mode='derpp' -> CE on the rehearsal batch plus
                                               beta * MSE(logits, stored logits)
    si             -- SITracker instance: runs the per-step synaptic-intensity
                      hooks around every optimisation step and adds its penalty
                      from the second task onwards (zero in the first task).
    l2             -- dict(anchors=[...], lam=...) adding a uniform L2 drift
                      penalty towards every past-task optimum.
    Returns a history dict with loss curves, timings and per-epoch old-task
    accuracy when `eval_old_loader` is given.
    """
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    hist = {"loss": [], "loss_task": [], "loss_replay": [], "loss_ewc": [],
            "loss_si": [], "loss_l2": [], "epoch_time": [], "old_acc": []}

    replay_loader = None
    if replay_dataset is not None:
        replay_loader = DataLoader(replay_dataset, batch_size=32, shuffle=True)
    rlogit_loader = None
    if replay_logits is not None:
        rlogit_loader = DataLoader(replay_logits, batch_size=32, shuffle=True)
    replay_iter, rlogit_iter = None, None

    start_train = time.time()
    for ep in range(epochs):
        ep_start = time.time()
        ep_loss = ep_task = ep_rep = ep_ewc = ep_si = ep_l2 = 0.0
        n_steps = 0

        if replay_loader is not None:
            replay_iter = iter(replay_loader)
        if rlogit_loader is not None:
            rlogit_iter = iter(rlogit_loader)

        for X, y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss_task = criterion(out, y)
            loss = loss_task
            l_rep = torch.zeros((), dtype=loss_task.dtype)
            l_ewc = torch.zeros((), dtype=loss_task.dtype)

            if replay_loader is not None or rlogit_loader is not None:
                if rlogit_loader is not None:
                    try:
                        Xr, yr, zr = next(rlogit_iter)
                    except StopIteration:
                        rlogit_iter = iter(rlogit_loader)
                        Xr, yr, zr = next(rlogit_iter)
                    out_r = model(Xr)
                    l_rep = criterion(out_r, yr)
                    if replay_mode == 'derpp':
                        l_rep = l_rep + beta * nn.functional.mse_loss(out_r, zr)
                else:
                    try:
                        Xr, yr = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(replay_loader)
                        Xr, yr = next(replay_iter)
                    l_rep = criterion(model(Xr), yr)
                loss = loss + l_rep

            if ewc is not None:
                ewc_specs = ewc if isinstance(ewc, (list, tuple)) else [ewc]
                l_ewc = None
                for spec in ewc_specs:
                    term = ewc_penalty(model, spec["fisher"], spec["theta_star"], spec["lam"])
                    l_ewc = term if l_ewc is None else l_ewc + term
                loss = loss + l_ewc

            l_si = torch.zeros((), dtype=loss_task.dtype)
            l_l2 = torch.zeros((), dtype=loss_task.dtype)
            if l2 is not None:
                l_l2 = l2_penalty(model, l2.get("anchors") or [], l2["lam"])
                loss = loss + l_l2
            if si is not None:
                l_si = si.penalty(model)
                loss = loss + l_si
                si.pre_step(model)

            loss.backward()
            optimizer.step()
            if si is not None:
                si.post_step(model)

            ep_loss += float(loss.detach())
            ep_task += float(loss_task.detach())
            ep_rep += float(l_rep.detach())
            ep_ewc += float(l_ewc.detach())
            ep_si += float(l_si.detach())
            ep_l2 += float(l_l2.detach())
            n_steps += 1

        hist["loss"].append(ep_loss / max(n_steps, 1))
        hist["loss_task"].append(ep_task / max(n_steps, 1))
        hist["loss_replay"].append(ep_rep / max(n_steps, 1))
        hist["loss_ewc"].append(ep_ewc / max(n_steps, 1))
        hist["loss_si"].append(ep_si / max(n_steps, 1))
        hist["loss_l2"].append(ep_l2 / max(n_steps, 1))
        hist["epoch_time"].append(time.time() - ep_start)

        if eval_old_loader is not None:
            acc = evaluate_accuracy(model, eval_old_loader)
            hist["old_acc"].append(acc)
            print(f"{log_prefix}Epoch {ep+1}/{epochs} | Loss: {hist['loss'][-1]:.4f} | "
                  f"Old Task Acc: {acc:.2f}% | Time: {hist['epoch_time'][-1]:.2f}s")
        else:
            print(f"{log_prefix}Epoch {ep+1}/{epochs} | Loss: {hist['loss'][-1]:.4f} | "
                  f"Time: {hist['epoch_time'][-1]:.2f}s")

    hist["train_time"] = time.time() - start_train
    return hist


# ---------------------------------------------------------------------------
# Self-test helpers (T1 caveat: first-order gradients only)
# ---------------------------------------------------------------------------

def fd_grad_selftest(model, criterion, X, y, coords=6, h=1e-3, rtol=5e-2, seed=0):
    """Cross-check first-order autograd gradients of `vqc.weights` against
    central finite differences of the same scalar loss.

    Returns (max_rel_error, ok). Raises AssertionError when a coordinate
    disagrees beyond `rtol` (a real failure of the gradient path).
    """
    g = torch.Generator().manual_seed(seed)
    w = model.vqc.weights
    loss = criterion(model(X), y)
    grads = torch.autograd.grad(loss, w)[0].detach().flatten()

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
        denom = max(abs(fd), abs(ag), 1e-3)
        errs.append(abs(fd - ag) / denom)

    max_err = max(errs) if errs else 0.0
    ok = bool(max_err <= rtol)
    assert ok, f"fd_grad_selftest failed: max rel err {max_err:.4f} > {rtol}"
    return max_err, ok
