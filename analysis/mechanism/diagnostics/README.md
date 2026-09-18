# Diagnostic scripts — second-order autodiff through the PennyLane TorchLayer

These scripts document why the Hessian-sharpness probe (`hess` in
`../gradient_probe.py`) is marked **pending validation** instead of reported.

## What was tested

1. `_validate_hess.py` — full Hessian of the Task-A loss via
   `torch.autograd.functional.hessian` + `functional_call`, compared with the
   power iteration used by the probe.
2. `_coherence_hess.py` — for random unit directions `v`, compares `v^T H v`
   computed from the explicit Hessian matrix against the autograd
   double-backward HVP path used by the power iteration.

## Result (4-qubit SEL model, Task-A batch, 2026-09-17)

- On a **purely classical** control model the two paths agree to ~1e-9.
- Through the **PennyLane TorchLayer** both paths disagree at O(1) relative
  error, and their eigenvalues do not match (0.48 vs 0.55), so neither can be
  reported as the Hessian sharpness of the hybrid model.

## Implication

Second-order autodiff through `qml.qnn.TorchLayer` (torch interface, backprop
diff method) is not trustworthy for curvature analyses in this stack. The
journal campaign must recompute curvature with a validated method:
parameter-shift on a second-order graph, finite differences in double
precision, or a jax-based path, each with a finite-difference self-test before
use. The first-order gradient probe (`grad`) is unaffected and validated by
construction (single backward pass).
