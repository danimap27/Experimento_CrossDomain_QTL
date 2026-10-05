# E2/E3/E4 campaigns -- statistical report

Cells loaded: 290/290 (missing 0, incomplete 0, stale local-label 0).

Executing machines (real machine_id per run): `nodo3051:b73ce5ab` -> 10 cells; `nodo4011:39da159c` -> 11 cells; `nodo4012:f761356e` -> 8 cells; `nodo4071:3f12fae7` -> 10 cells; `nodo4072:388733b7` -> 5 cells; `nodo4074:57608ec1` -> 12 cells; `nodo4082:7d0e004e` -> 13 cells; `nodo4091:6403e3ac` -> 6 cells


## H1 chain4: synth vs scratch on scenario metrics (Holm, 5-metric family)

- ideal/TIL-2: d=+1.07 pp [-7.45,+9.15], p=0.817, wilcoxon=0.922, dz=+0.08, n=10
- ideal/CIL-2: d=-0.40 pp [-2.38,+1.23], p=0.692, wilcoxon=1.000, dz=-0.13, n=10
- ideal/TIL-3: d=-2.67 pp [-12.40,+6.67], p=0.616, wilcoxon=0.695, dz=-0.16, n=10
- ideal/CIL-3: d=+0.43 pp [-0.22,+1.15], p=0.274, wilcoxon=0.320, dz=+0.37, n=10
- ideal/CIL-4: d=+0.03 pp [-0.35,+0.38], p=0.901, wilcoxon=0.848, dz=+0.04, n=10
- heron_r2/TIL-2: d=-1.90 pp [-9.50,+5.42], p=0.651, wilcoxon=0.756, dz=-0.15, n=10
- heron_r2/CIL-2: d=+1.12 pp [-0.05,+2.23], p=0.098, wilcoxon=0.133, dz=+0.58, n=10
- heron_r2/TIL-3: d=-2.70 pp [-10.95,+5.92], p=0.569, wilcoxon=0.492, dz=-0.19, n=10
- heron_r2/CIL-3: d=-0.53 pp [-1.22,+0.17], p=0.192, wilcoxon=0.201, dz=-0.45, n=10
- heron_r2/CIL-4: d=-0.17 pp [-0.45,+0.07], p=0.242, wilcoxon=0.312, dz=-0.40, n=10
- pooled/TIL-2: d=-0.41 pp [-6.10,+5.19], p=0.891, wilcoxon=0.867, dz=-0.03, n=20
- pooled/CIL-2: d=+0.36 pp [-0.84,+1.41], p=0.544, wilcoxon=0.231, dz=+0.14, n=20
- pooled/TIL-3: d=-2.68 pp [-9.17,+3.78], p=0.432, wilcoxon=0.409, dz=-0.18, n=20
- pooled/CIL-3: d=-0.05 pp [-0.58,+0.49], p=0.861, wilcoxon=0.808, dz=-0.04, n=20
- pooled/CIL-4: d=-0.07 pp [-0.31,+0.15], p=0.537, wilcoxon=0.686, dz=-0.14, n=20

## H2 five-task benchmarks (Holm within benchmark scope)

- smnist5/ideal/scratch (n=10): AA 18.97, AF 77.42, BWT -96.77, FWT -13.78
- smnist5/ideal/synth (n=10): AA 19.02, AF 77.44, BWT -96.80, FWT 0.00
- smnist5/ideal/er (n=10): AA 34.45, AF 50.40, BWT -63.00, FWT -13.78
- smnist5/ideal/ewc (n=10): AA 17.26, AF 74.60, BWT -93.25, FWT -13.78
  - synth-scratch: d=+0.05 pp [-0.02,+0.13], p=0.240, p_holm=0.240, wilcoxon=0.375, dz=+0.40, n=10
  - er-scratch: d=+15.49 pp [+14.73,+16.10], p=$<$0.001, p_holm=$<$0.001, wilcoxon=0.002, dz=+13.20, n=10
  - ewc-scratch: d=-1.71 pp [-2.13,-1.28], p=$<$0.001, p_holm=$<$0.001, wilcoxon=0.002, dz=-2.37, n=10
- smnist5/heron_r2/scratch (n=5): AA 19.06, AF 77.34, BWT -96.67, FWT -13.39
- smnist5/heron_r2/synth (n=5): AA 19.01, AF 77.43, BWT -96.79, FWT 0.00
  - synth-scratch: d=-0.06 pp [-0.20,+0.05], p=0.480, p_holm=0.480, wilcoxon=0.625, dz=-0.35, n=5
- sfmnist5/ideal/scratch (n=10): AA 19.97, AF 77.53, BWT -96.91, FWT -10.34
- sfmnist5/ideal/synth (n=10): AA 20.57, AF 76.94, BWT -96.18, FWT 0.00
  - synth-scratch: d=+0.59 pp [-0.01,+1.51], p=0.204, p_holm=0.204, wilcoxon=0.164, dz=+0.43, n=10

### Friedman + Nemenyi (smnist5 ideal, 4 arms)

- Friedman chi2=27.12, p=$<$0.001 (N=10 seeds, k=4)
- mean ranks: scratch=2.40, synth=2.60, er=4.00, ewc=1.00
- Nemenyi critical difference (alpha=0.05): 1.48

## H3 data scale (Holm within the three-size family)

- sz500/ideal: d=+0.02 pp [-1.10,+0.96], p=0.968, dz=+0.01, n=10
- sz2k/ideal: d=+0.61 pp [-0.64,+1.69], p=0.356, dz=+0.31, n=10
- sz12k/ideal: d=-1.71 pp [-3.72,+0.13], p=0.136, dz=-0.52, n=10
- sz12k/heron_r2: d=-0.21 pp [-1.93,+1.49], p=0.825, dz=-0.07, n=10

## H4 qubit/depth grid (Holm within the five-config family)

- std: d=+3.05 pp [+0.00,+6.25], p=0.108, dz=+0.57, n=10
- q6: d=-0.85 pp [-3.80,+1.70], p=0.585, dz=-0.18, n=10
- q8: d=-0.55 pp [-6.50,+5.25], p=0.868, dz=-0.05, n=10
- L2: d=-2.45 pp [-7.25,+1.60], p=0.334, dz=-0.32, n=10
- L4: d=+0.00 pp [-2.60,+2.90], p=1.000, dz=+0.00, n=10
