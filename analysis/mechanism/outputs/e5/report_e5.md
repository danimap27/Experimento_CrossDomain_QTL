# E5 — mechanism study (cross-domain prior)

Checkpoints: 20 cells; machine ids: quantum-nas:4d306c63


## Arm `scratch` (n=10 seeds)

- acc(A|A)=96.00, acc(A|B)=27.70, acc(B|B)=89.60
- Var[grad|A] theta_init: per-layer L0=2.725e-03, L1=3.026e-03, L2=1.360e-03; vqc mean 2.370e-03
- Var[grad|A] theta0: per-layer L0=2.725e-03, L1=3.026e-03, L2=1.360e-03; vqc mean 2.370e-03
- Var[grad|A] thetaA: per-layer L0=8.817e-02, L1=6.555e-02, L2=3.154e-02; vqc mean 6.175e-02
- Var[grad|A] thetaB: per-layer L0=1.553e+00, L1=1.195e+00, L2=6.930e-01; vqc mean 1.147e+00
- Fisher theta0: trace=1.13e+00, top1frac=0.558, H=1.05, effrank=2.9
- Fisher thetaA: trace=2.23e+00, top1frac=0.631, H=1.04, effrank=2.9
- Fisher thetaB: trace=5.56e+01, top1frac=0.526, H=1.42, effrank=4.2
- barrier: ||dB-dA||_rms=1.040, barrier(B)=+0.000±0.000, barrier(A)=+0.002±0.007
- angles: mean=2.92±1.74, frac|θ|>π=0.422, max|ΔL| wrapping=2.38e-07

## Arm `synth` (n=10 seeds)

- acc(A|A)=96.05, acc(A|B)=24.50, acc(B|B)=90.00
- Var[grad|A] theta_init: per-layer L0=2.798e+00, L1=3.668e+00, L2=2.091e+00; vqc mean 2.852e+00
- Var[grad|A] theta0: per-layer L0=2.798e+00, L1=3.668e+00, L2=2.091e+00; vqc mean 2.852e+00
- Var[grad|A] thetaA: per-layer L0=2.200e-01, L1=2.086e-01, L2=8.527e-02; vqc mean 1.713e-01
- Var[grad|A] thetaB: per-layer L0=2.201e+00, L1=3.082e+00, L2=1.073e+00; vqc mean 2.119e+00
- Fisher theta0: trace=1.27e+02, top1frac=0.510, H=1.43, effrank=4.2
- Fisher thetaA: trace=6.30e+00, top1frac=0.578, H=1.12, effrank=3.1
- Fisher thetaB: trace=1.09e+02, top1frac=0.542, H=1.41, effrank=4.1
- barrier: ||dB-dA||_rms=1.063, barrier(B)=+0.000±0.000, barrier(A)=+0.050±0.096
- angles: mean=3.29±2.01, frac|θ|>π=0.519, max|ΔL| wrapping=3.34e-06

## Self-tests (caveat T1)

- FD grad self-test max rel err: 6.09e-08 (rtol 5e-2)
- Fisher trace consistency max diff: 1.53e-05
- wrap invariance max |ΔL|: 3.34e-06
