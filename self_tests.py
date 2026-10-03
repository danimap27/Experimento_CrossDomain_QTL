"""Self-tests for checklist-3.7 repairs (run once, evidence for the campaign).

1. Functional freeze: the historical `if '0' in name` filter matched NOTHING
   (vqc.weights / fc.weight / fc.bias); the repaired freeze_ansatz_layers()
   freezes real rows of vqc.weights and the test asserts (a) names match,
   (b) frozen rows do not move, (c) the rest does.
2. General TTN: builds and runs for n_qubits in {2,4,6,8}; for n=4 the merge
   schedule equals the historical [(0,1),(2,3),(1,3)]; demonstrates the H9
   inert-parameter effect of the readout-limited variant vs 'root'.
3. Multi-class head: n_classes=5 returns (batch, 5).
4. PCA to n components: DataModule(n_components=6) yields 6-feature batches
   that run through a 6-qubit model.
5. Gradient path (T1 caveat): first-order autograd gradients of the VQC
   weights agree with central finite differences (NO double-backward used).

Prints a PASS/FAIL line per test; exit code 0 only if all pass.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from data_module import DataModule
from quantum_net import HybridQuantumNet, ttn_pairs
from continual import empirical_fisher, fd_grad_selftest, snapshot_params

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} :: {detail}")
    if not cond:
        FAILURES.append(name)


def test_freeze():
    torch.manual_seed(0)
    m = HybridQuantumNet(ansatz="A", n_qubits=4, n_layers=3)
    names = sorted(n for n, _ in m.named_parameters())
    check("freeze/param-names", names == ["fc.bias", "fc.weight", "vqc.weights"], str(names))

    # the historical filter matched nothing:
    legacy_matched = [n for n, _ in m.named_parameters() if "0" in n]
    check("freeze/legacy-bug-reproduced", legacy_matched == [], f"legacy matched {legacy_matched}")

    frozen = m.freeze_ansatz_layers([0])
    check("freeze/real-name-match", frozen == ["vqc.weights[0]"], str(frozen))

    X = torch.rand(8, 4) * np.pi
    y = torch.randint(0, 2, (8,))
    crit = torch.nn.CrossEntropyLoss()
    before = m.vqc.weights.detach().clone()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        loss = crit(m(X), y)
        loss.backward()
        opt.step()
    after = m.vqc.weights.detach().clone()
    check("freeze/lower-block-static", torch.allclose(before[0], after[0]), "layer 0 unchanged")
    check("freeze/rest-updated", not torch.allclose(before[1:], after[1:]), "layers 1-2 moved")


def test_ttn_general():
    legacy_pairs_4 = [(0, 1), (2, 3), (1, 3)]
    check("ttn/pairs-n4-legacy", ttn_pairs(4) == legacy_pairs_4, str(ttn_pairs(4)))
    check("ttn/pairs-count", all(len(ttn_pairs(n)) == n - 1 for n in (2, 4, 8)),
          str({n: len(ttn_pairs(n)) for n in (2, 4, 6, 8)}))

    for n in (2, 4, 6, 8):
        torch.manual_seed(0)
        m = HybridQuantumNet(ansatz="C", n_qubits=n, n_layers=3)
        out = m(torch.rand(4, n))
        check(f"ttn/forward-n{n}", out.shape == (4, 2), str(tuple(out.shape)))

    # H9 / audit A7: in the readout-limited variant whole coordinates are
    # structurally inert (commutation argument, verified against gradients at
    # generic weights): the {2,3} block of the LAST layer and its root RZ never
    # reach the Z0/Z1 readout. The 'root' readout makes the subtree effective.
    torch.manual_seed(0)
    ml = HybridQuantumNet(ansatz="C", n_qubits=4, n_layers=3, ttn_readout="limited")
    torch.manual_seed(0)
    mr = HybridQuantumNet(ansatz="C", n_qubits=4, n_layers=3, ttn_readout="root")
    X = torch.rand(8, 4)
    y = torch.randint(0, 2, (8,))
    crit = torch.nn.CrossEntropyLoss()
    grads = {}
    for tag, m in (("limited", ml), ("root", mr)):
        loss = crit(m(X), y)
        grads[tag] = torch.autograd.grad(loss, m.vqc.weights)[0].detach().abs()

    inert_limited = [grads["limited"][2, 1, 0], grads["limited"][2, 1, 1],
                     grads["limited"][2, 2, 1]]
    check("ttn/h9-inert-params-limited",
          all(float(g) < 1e-6 for g in inert_limited),
          "last-layer {2,3} block + root RZ inert under Z0/Z1 readout: "
          + ", ".join(f"{float(g):.2e}" for g in inert_limited))
    check("ttn/root-variant-activates-subtree",
          float(grads["root"][2, 1, 0]) > 1e-6,
          f"subtree RY grad under root readout = {float(grads['root'][2, 1, 0]):.2e}")


def test_multiclass_head():
    torch.manual_seed(0)
    m = HybridQuantumNet(ansatz="A", n_qubits=4, n_layers=2, n_classes=5)
    out = m(torch.rand(6, 4))
    check("head/multiclass-shape", out.shape == (6, 5), str(tuple(out.shape)))


def test_pca_n():
    dm = DataModule(data_dir="/tmp/e1_selftest_data", batch_size=8, n_components=6)
    tr, te, _ = dm.get_mnist_task(classes=(0, 1))
    X, _ = next(iter(tr))
    check("pca/n-components", X.shape[1] == 6, f"batch shape {tuple(X.shape)}")
    torch.manual_seed(0)
    m = HybridQuantumNet(ansatz="A", n_qubits=6, n_layers=2)
    out = m(X)
    check("pca/6q-forward", out.shape == (X.shape[0], 2), str(tuple(out.shape)))


def test_fisher_and_grads():
    torch.manual_seed(0)
    m = HybridQuantumNet(ansatz="A", n_qubits=4, n_layers=2)
    X = torch.rand(8, 4)
    y = torch.randint(0, 2, (8,))
    crit = torch.nn.CrossEntropyLoss()
    err, ok = fd_grad_selftest(m, crit, X, y, coords=6, h=1e-3, rtol=5e-2)
    check("grad/first-order-vs-fd", ok, f"max rel err {err:.4f} (no double-backward used)")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=4)
    fisher = empirical_fisher(m, crit, loader)
    f = fisher["vqc.weights"]
    check("fisher/empirical-diag", f.shape == m.vqc.weights.shape and float(f.min()) >= 0.0,
          f"mean={float(f.mean()):.4f}")
    # snapshot + EWC penalty sanity
    theta = snapshot_params(m)
    from continual import ewc_penalty
    with torch.no_grad():
        pen = float(ewc_penalty(m, fisher, theta, 1e3))
    check("ewc/penalty-zero-at-anchor", abs(pen) < 1e-9, f"penalty={pen:.2e}")


def main():
    test_freeze()
    test_ttn_general()
    test_multiclass_head()
    test_pca_n()
    test_fisher_and_grads()
    print("\n=== SELF-TEST SUMMARY ===")
    if FAILURES:
        print("FAILED:", FAILURES)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
