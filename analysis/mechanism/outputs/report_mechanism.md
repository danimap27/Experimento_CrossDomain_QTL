# Mechanism probe — preliminary results

## Barren-plateau sweep

| ansatz | qubits | params | Var[grad] mean | Var[grad] median |
|--------|--------|--------|----------------|------------------|
| sel | 10 | 90 | 3.410e-05 | 3.120e-05 |
| sel | 12 | 108 | 1.174e-05 | 1.060e-05 |
| sel | 4 | 36 | 5.204e-03 | 5.668e-03 |
| sel | 6 | 54 | 6.514e-04 | 4.803e-04 |
| sel | 8 | 72 | 1.435e-04 | 1.349e-04 |
| ttn | 10 | 54 | 5.415e-04 | 5.041e-05 |
| ttn | 12 | 66 | 3.638e-04 | 1.022e-04 |
| ttn | 4 | 18 | 6.824e-03 | 5.535e-03 |
| ttn | 6 | 30 | 1.796e-03 | 7.601e-04 |
| ttn | 8 | 42 | 1.006e-03 | 3.864e-04 |

## Task-A gradient at theta_0 vs random

- theta_0: loss=1.8341, |grad|=3.0154, grad_var=2.087e-01, frac|g|<1e-4=0.143
- random (10): loss=0.7811 +/- 0.1167
- random (10): grad_norm=0.3548 +/- 0.1646
- random (10): grad_var=0.0036 +/- 0.0028
- random (10): frac_grad_near_zero=0.1524 +/- 0.0123
- theta_0 VQC weights: mean=3.200, std=2.732, range=[-7.967, 7.462]

