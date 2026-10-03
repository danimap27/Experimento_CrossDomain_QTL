# E1 controlled campaign -- statistical report

Seeds available per profile: heron_r2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]; ideal: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

## Descriptives per arm (mean +/- sd over seeds)

- **heron_r2 / B1** (n=20): AccA_init 95.95+/-0.81, AccA_final 26.27+/-3.68, AccB 89.92+/-2.23, dA 69.67+/-3.97, r_A 0.274+/-0.039
- **heron_r2 / B2** (n=20): AccA_init 95.78+/-0.60, AccA_final 29.88+/-7.76, AccB 89.05+/-2.32, dA 65.90+/-7.79, r_A 0.312+/-0.081
- **heron_r2 / B3** (n=20): AccA_init 96.00+/-0.92, AccA_final 27.27+/-4.51, AccB 89.88+/-1.74, dA 68.72+/-4.54, r_A 0.284+/-0.047
- **heron_r2 / B4** (n=20): AccA_init 95.83+/-0.98, AccA_final 32.25+/-4.21, AccB 87.80+/-2.00, dA 63.58+/-4.45, r_A 0.337+/-0.045
- **heron_r2 / B5_l1e2** (n=20): AccA_init 95.95+/-0.81, AccA_final 34.52+/-9.38, AccB 86.62+/-4.29, dA 61.42+/-9.70, r_A 0.360+/-0.099
- **heron_r2 / B5_l1e3** (n=20): AccA_init 95.95+/-0.81, AccA_final 54.85+/-16.35, AccB 70.55+/-9.85, dA 41.10+/-16.72, r_A 0.572+/-0.173
- **heron_r2 / B5_l1e4** (n=20): AccA_init 95.95+/-0.81, AccA_final 91.08+/-9.24, AccB 34.12+/-10.09, dA 4.88+/-9.50, r_A 0.949+/-0.098
- **heron_r2 / B6** (n=20): AccA_init 95.95+/-0.81, AccA_final 84.20+/-5.41, AccB 73.03+/-3.66, dA 11.75+/-5.57, r_A 0.878+/-0.058
- **heron_r2 / B7** (n=20): AccA_init 95.95+/-0.81, AccA_final 91.92+/-1.38, AccB 60.85+/-3.45, dA 4.03+/-1.67, r_A 0.958+/-0.017
- **ideal / B1** (n=20): AccA_init 96.28+/-0.72, AccA_final 24.85+/-2.78, AccB 90.40+/-1.70, dA 71.42+/-2.79, r_A 0.258+/-0.029
- **ideal / B2** (n=20): AccA_init 95.78+/-0.70, AccA_final 31.00+/-5.67, AccB 89.12+/-1.72, dA 64.78+/-5.72, r_A 0.324+/-0.059
- **ideal / B3** (n=20): AccA_init 96.25+/-0.68, AccA_final 26.25+/-3.52, AccB 89.67+/-1.78, dA 70.00+/-3.41, r_A 0.273+/-0.036
- **ideal / B4** (n=20): AccA_init 95.35+/-1.27, AccA_final 30.30+/-4.67, AccB 88.35+/-2.03, dA 65.05+/-4.78, r_A 0.318+/-0.049
- **ideal / B5_l1e2** (n=20): AccA_init 96.28+/-0.72, AccA_final 35.55+/-8.14, AccB 86.83+/-4.27, dA 60.73+/-8.34, r_A 0.369+/-0.085
- **ideal / B5_l1e3** (n=20): AccA_init 96.28+/-0.72, AccA_final 57.02+/-16.97, AccB 66.85+/-11.87, dA 39.25+/-17.20, r_A 0.593+/-0.177
- **ideal / B5_l1e4** (n=20): AccA_init 96.28+/-0.72, AccA_final 93.30+/-5.15, AccB 31.55+/-6.08, dA 2.98+/-5.37, r_A 0.969+/-0.055
- **ideal / B6** (n=20): AccA_init 96.28+/-0.72, AccA_final 83.30+/-5.52, AccB 73.15+/-4.03, dA 12.97+/-5.62, r_A 0.865+/-0.058
- **ideal / B7** (n=20): AccA_init 96.28+/-0.72, AccA_final 92.17+/-1.89, AccB 61.12+/-4.26, dA 4.10+/-1.99, r_A 0.957+/-0.021

Machine ids (requirement 1.11): `cluster-MS-7885:9b2b046e` -> 9 cells; `quantum-nas:4d306c63` -> 18 cells

## Paired contrasts on delta_A (Holm within family)


### Family F1_design


**ideal**:

  - LR effect (scratch): B2-B1: d=-6.65 pp [-9.60, -3.88], t=-4.41, p=$<$0.001, p_holm=0.001, wilcoxon=$<$0.001, dz=-0.99, n=20
  - Init effect (high LR): B3-B1: d=-1.43 pp [-3.42, +0.53], t=-1.39, p=0.179, p_holm=0.359, wilcoxon=0.212, dz=-0.31, n=20
  - LR effect (synth): B4-B3: d=-4.95 pp [-7.50, -2.52], t=-3.77, p=0.001, p_holm=0.005, wilcoxon=0.002, dz=-0.84, n=20
  - Init effect (low LR): B4-B2: d=+0.28 pp [-2.12, +2.62], t=+0.22, p=0.827, p_holm=0.827, wilcoxon=0.831, dz=+0.05, n=20
  - Headline (confounded) contrast: B4-B1: d=-6.38 pp [-8.60, -4.08], t=-5.33, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-1.19, n=20
  - Cross contrast: B3-B2: d=+5.22 pp [+2.15, +8.18], t=+3.33, p=0.004, p_holm=0.011, wilcoxon=0.006, dz=+0.74, n=20

**heron_r2**:

  - LR effect (scratch): B2-B1: d=-3.77 pp [-7.33, -0.35], t=-2.07, p=0.053, p_holm=0.210, wilcoxon=0.089, dz=-0.46, n=20
  - Init effect (high LR): B3-B1: d=-0.95 pp [-3.45, +1.45], t=-0.73, p=0.474, p_holm=0.474, wilcoxon=0.570, dz=-0.16, n=20
  - LR effect (synth): B4-B3: d=-5.15 pp [-7.85, -2.60], t=-3.73, p=0.001, p_holm=0.007, wilcoxon=0.002, dz=-0.83, n=20
  - Init effect (low LR): B4-B2: d=-2.33 pp [-5.75, +0.88], t=-1.34, p=0.195, p_holm=0.391, wilcoxon=0.191, dz=-0.30, n=20
  - Headline (confounded) contrast: B4-B1: d=-6.10 pp [-8.68, -3.42], t=-4.43, p=$<$0.001, p_holm=0.002, wilcoxon=0.001, dz=-0.99, n=20
  - Cross contrast: B3-B2: d=+2.83 pp [-0.40, +6.17], t=+1.65, p=0.115, p_holm=0.346, wilcoxon=0.161, dz=+0.37, n=20

**pooled**:

  - LR effect (scratch): B2-B1: d=-5.21 pp [-7.51, -2.95], t=-4.38, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-0.69, n=40
  - Init effect (high LR): B3-B1: d=-1.19 pp [-2.76, +0.38], t=-1.45, p=0.154, p_holm=0.309, wilcoxon=0.229, dz=-0.23, n=40
  - LR effect (synth): B4-B3: d=-5.05 pp [-6.91, -3.26], t=-5.37, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-0.85, n=40
  - Init effect (low LR): B4-B2: d=-1.02 pp [-3.15, +1.00], t=-0.96, p=0.345, p_holm=0.345, wilcoxon=0.394, dz=-0.15, n=40
  - Headline (confounded) contrast: B4-B1: d=-6.24 pp [-7.95, -4.47], t=-6.93, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-1.10, n=40
  - Cross contrast: B3-B2: d=+4.03 pp [+1.76, +6.26], t=+3.46, p=0.001, p_holm=0.004, wilcoxon=0.002, dz=+0.55, n=40

### Family F2_baselines


**ideal**:

  - EWC 1e2 vs scratch: B5-B1: d=-10.70 pp [-14.65, -7.10], t=-5.41, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-1.21, n=20
  - EWC 1e3 vs scratch: B5-B1: d=-32.17 pp [-39.95, -24.70], t=-8.12, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-1.82, n=20
  - EWC 1e4 vs scratch: B5-B1: d=-68.45 pp [-70.72, -65.85], t=-54.01, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-12.08, n=20
  - Replay vs scratch: B6-B1: d=-58.45 pp [-60.92, -55.77], t=-42.93, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-9.60, n=20
  - DER++ vs scratch: B7-B1: d=-67.33 pp [-68.97, -65.67], t=-78.00, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-17.44, n=20
  - QTL vs EWC 1e2: B4-B5: d=+4.33 pp [+1.15, +7.67], t=+2.54, p=0.020, p_holm=0.020, wilcoxon=0.030, dz=+0.57, n=20
  - QTL vs EWC 1e3: B4-B5: d=+25.80 pp [+19.52, +32.25], t=+7.78, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+1.74, n=20
  - QTL vs EWC 1e4: B4-B5: d=+62.08 pp [+59.23, +64.78], t=+42.62, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+9.53, n=20
  - QTL vs Replay: B4-B6: d=+52.08 pp [+49.17, +54.85], t=+35.15, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+7.86, n=20
  - QTL vs DER++: B4-B7: d=+60.95 pp [+58.62, +63.27], t=+49.75, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+11.12, n=20

**heron_r2**:

  - EWC 1e2 vs scratch: B5-B1: d=-8.25 pp [-12.38, -4.22], t=-3.86, p=0.001, p_holm=0.002, wilcoxon=0.002, dz=-0.86, n=20
  - EWC 1e3 vs scratch: B5-B1: d=-28.57 pp [-35.62, -21.93], t=-7.97, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-1.78, n=20
  - EWC 1e4 vs scratch: B5-B1: d=-64.80 pp [-68.20, -60.60], t=-32.38, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-7.24, n=20
  - Replay vs scratch: B6-B1: d=-57.92 pp [-60.60, -55.33], t=-41.37, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-9.25, n=20
  - DER++ vs scratch: B7-B1: d=-65.65 pp [-67.30, -63.92], t=-73.82, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-16.51, n=20
  - QTL vs EWC 1e2: B4-B5: d=+2.15 pp [-2.00, +6.35], t=+0.99, p=0.333, p_holm=0.333, wilcoxon=0.444, dz=+0.22, n=20
  - QTL vs EWC 1e3: B4-B5: d=+22.48 pp [+15.47, +29.88], t=+5.99, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+1.34, n=20
  - QTL vs EWC 1e4: B4-B5: d=+58.70 pp [+54.65, +62.20], t=+29.40, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+6.57, n=20
  - QTL vs Replay: B4-B6: d=+51.83 pp [+49.00, +54.62], t=+35.33, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+7.90, n=20
  - QTL vs DER++: B4-B7: d=+59.55 pp [+57.58, +61.42], t=+59.09, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+13.21, n=20

**pooled**:

  - EWC 1e2 vs scratch: B5-B1: d=-9.47 pp [-12.36, -6.72], t=-6.53, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-1.03, n=40
  - EWC 1e3 vs scratch: B5-B1: d=-30.38 pp [-35.51, -25.34], t=-11.45, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-1.81, n=40
  - EWC 1e4 vs scratch: B5-B1: d=-66.62 pp [-68.78, -64.12], t=-55.28, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-8.74, n=40
  - Replay vs scratch: B6-B1: d=-58.19 pp [-60.05, -56.30], t=-60.31, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-9.54, n=40
  - DER++ vs scratch: B7-B1: d=-66.49 pp [-67.69, -65.26], t=-106.17, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=-16.79, n=40
  - QTL vs EWC 1e2: B4-B5: d=+3.24 pp [+0.61, +5.88], t=+2.36, p=0.023, p_holm=0.023, wilcoxon=0.037, dz=+0.37, n=40
  - QTL vs EWC 1e3: B4-B5: d=+24.14 pp [+19.47, +28.99], t=+9.71, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+1.54, n=40
  - QTL vs EWC 1e4: B4-B5: d=+60.39 pp [+57.85, +62.70], t=+48.34, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+7.64, n=40
  - QTL vs Replay: B4-B6: d=+51.95 pp [+49.96, +53.95], t=+50.48, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+7.98, n=40
  - QTL vs DER++: B4-B7: d=+60.25 pp [+58.70, +61.76], t=+76.17, p=$<$0.001, p_holm=$<$0.001, wilcoxon=$<$0.001, dz=+12.04, n=40

## Robustness (exploratory): paired contrasts on Acc_A_final

- ideal: Init effect (high LR): d=+1.40 pp, p=0.205, p_holm=0.410, dz=+0.29
- ideal: Init effect (low LR): d=-0.70 pp, p=0.610, p_holm=0.610, dz=-0.12
- heron_r2: Init effect (high LR): d=+1.00 pp, p=0.429, p_holm=0.429, dz=+0.18
- heron_r2: Init effect (low LR): d=+2.38 pp, p=0.193, p_holm=0.387, dz=+0.30
- pooled: Init effect (high LR): d=+1.20 pp, p=0.145, p_holm=0.290, dz=+0.24
- pooled: Init effect (low LR): d=+0.84 pp, p=0.460, p_holm=0.460, dz=+0.12
