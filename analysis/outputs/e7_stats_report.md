# E7 -- long sequences (10 tasks) and tuning

## smnist10 (10 tasks x 1 class)

- scratch: aa 10.01+/-0.01 | af 89.99+/-0.01 | bwt -99.99+/-0.01 | fwt -11.15+/-1.12
- synth: aa 10.04+/-0.06 | af 89.96+/-0.06 | bwt -99.95+/-0.06 | fwt -5.44+/-0.40
- si_0.5: aa 10.66+/-1.54 | af 25.38+/-6.25 | bwt -28.21+/-6.94 | fwt -11.07+/-1.14
- si_1.5: aa 10.76+/-1.14 | af 11.47+/-4.27 | bwt -12.75+/-4.74 | fwt -11.08+/-1.12
- si_5: aa 11.09+/-1.21 | af 8.35+/-1.74 | bwt -9.28+/-1.93 | fwt -11.15+/-1.12
- er: aa 15.69+/-0.72 | af 69.68+/-1.86 | bwt -77.43+/-2.07 | fwt -11.15+/-1.12
- derpp: aa 18.46+/-0.57 | af 49.39+/-1.41 | bwt -54.87+/-1.56 | fwt -11.15+/-1.12
- ewc_1e4: aa 9.95+/-0.07 | af 89.76+/-0.17 | bwt -99.73+/-0.18 | fwt -11.15+/-1.12
- ewc_1e2: aa 10.00+/-0.01 | af 90.00+/-0.01 | bwt -100.00+/-0.01 | fwt -11.15+/-1.12

## fmnist10 (10 tasks x 1 class)

- scratch: aa 10.02+/-0.05 | af 89.97+/-0.05 | bwt -99.97+/-0.05 | fwt -10.91+/-2.60
- synth: aa 10.03+/-0.05 | af 89.94+/-0.06 | bwt -99.93+/-0.07 | fwt -3.41+/-0.58
- si_5: aa 10.54+/-1.12 | af 9.03+/-1.27 | bwt -10.04+/-1.41 | fwt -10.91+/-2.61
- er: aa 17.30+/-1.29 | af 70.51+/-3.06 | bwt -78.34+/-3.40 | fwt -10.91+/-2.60
- derpp: aa 18.79+/-0.77 | af 54.27+/-1.79 | bwt -60.30+/-1.99 | fwt -10.91+/-2.60

## smnist5 (tuning)

- scratch: aa 18.97+/-0.14 | af 77.42+/-0.37 | bwt -96.77+/-0.47 | fwt -13.78+/-4.38
- er_25: aa 34.45+/-1.12 | af 50.40+/-3.26 | bwt -63.00+/-4.07 | fwt -13.78+/-4.38
- er_50: aa 34.67+/-1.35 | af 50.57+/-1.77 | bwt -63.22+/-2.21 | fwt -13.78+/-4.38
- ewc_1e4: aa 17.26+/-0.62 | af 74.60+/-1.07 | bwt -93.25+/-1.33 | fwt -13.78+/-4.38
- ewc_1e2: aa 17.76+/-1.44 | af 76.45+/-0.58 | bwt -95.56+/-0.72 | fwt -13.78+/-4.38

## F1 smnist10 vs scratch

- smnist10 synth - scratch: d=+0.03 pp [-0.00,+0.07], t=+1.70, p=0.123, p_holm=0.368, wilcoxon=0.156, dz=+0.54, n=10
- smnist10 si_0.5 - scratch: d=+0.66 pp [-0.19,+1.60], t=+1.35, p=0.210, p_holm=0.420, wilcoxon=0.322, dz=+0.43, n=10
- smnist10 si_1.5 - scratch: d=+0.76 pp [+0.11,+1.45], t=+2.10, p=0.065, p_holm=0.262, wilcoxon=0.131, dz=+0.66, n=10
- smnist10 si_5 - scratch: d=+1.08 pp [+0.38,+1.78], t=+2.84, p=0.019, p_holm=0.116, wilcoxon=0.037, dz=+0.90, n=10
- smnist10 er - scratch: d=+5.68 pp [+5.26,+6.11], t=+24.88, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+7.87, n=10
- smnist10 derpp - scratch: d=+8.45 pp [+8.12,+8.78], t=+47.18, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+14.92, n=10
- smnist10 ewc_1e4 - scratch: d=-0.06 pp [-0.11,-0.01], t=-2.24, p=0.052, p_holm=0.260, wilcoxon=0.064, dz=-0.71, n=10
- smnist10 ewc_1e2 - scratch: d=-0.00 pp [-0.01,+0.00], t=-0.94, p=0.370, p_holm=0.420, wilcoxon=1.000, dz=-0.30, n=10

## F2 fmnist10 vs scratch

- fmnist10 synth - scratch: d=+0.01 pp [-0.03,+0.06], t=+0.45, p=0.665, p_holm=0.665, wilcoxon=0.625, dz=+0.14, n=10
- fmnist10 si_5 - scratch: d=+0.52 pp [-0.09,+1.20], t=+1.50, p=0.168, p_holm=0.335, wilcoxon=0.275, dz=+0.47, n=10
- fmnist10 er - scratch: d=+7.28 pp [+6.58,+8.10], t=+17.72, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+5.60, n=10
- fmnist10 derpp - scratch: d=+8.77 pp [+8.30,+9.24], t=+34.96, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+11.05, n=10

## F3 smnist5 tuning

- smnist5 er50 - er25: d=+0.22 pp [-0.62,+1.05], t=+0.47, p=0.646, p_holm=0.746, wilcoxon=0.770, dz=+0.15, n=10
- smnist5 er50 - scratch: d=+15.70 pp [+14.99,+16.44], t=+39.73, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+12.56, n=10
- smnist5 ewc1e2 - ewc1e4: d=+0.50 pp [-0.53,+1.42], t=+0.94, p=0.373, p_holm=0.746, wilcoxon=0.322, dz=+0.30, n=10
- smnist5 ewc1e2 - scratch: d=-1.20 pp [-2.08,-0.48], t=-2.72, p=0.024, p_holm=0.071, wilcoxon=0.004, dz=-0.86, n=10

## F4 SI lambda on 10 tasks

- smnist10 si0.5 - si5: d=-0.42 pp [-1.58,+0.63], t=-0.70, p=0.500, p_holm=1.000, wilcoxon=0.695, dz=-0.22, n=10
- smnist10 si1.5 - si5: d=-0.33 pp [-1.28,+0.64], t=-0.62, p=0.548, p_holm=1.000, wilcoxon=0.695, dz=-0.20, n=10
