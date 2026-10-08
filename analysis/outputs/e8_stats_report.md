# E8 -- input capacity (qubits x components) on class-IL

## smnist10

- scratch q4: aa 10.01+/-0.01 | af 89.99+/-0.01 | bwt -99.99+/-0.01
- scratch q6: aa 10.01+/-0.03 | af 89.88+/-0.05 | bwt -99.87+/-0.06
- scratch q8: aa 10.00+/-0.02 | af 89.99+/-0.01 | bwt -99.99+/-0.01
- er q4: aa 15.69+/-0.72 | af 69.68+/-1.86 | bwt -77.43+/-2.07
- er q6: aa 18.17+/-1.59 | af 69.56+/-1.99 | bwt -77.29+/-2.21
- er q8: aa 20.46+/-1.64 | af 68.26+/-2.02 | bwt -75.84+/-2.25
- derpp q4: aa 18.46+/-0.57 | af 49.39+/-1.41 | bwt -54.87+/-1.56
- derpp q6: aa 20.69+/-0.88 | af 49.32+/-2.95 | bwt -54.80+/-3.28
- derpp q8: aa 21.35+/-0.58 | af 48.35+/-2.38 | bwt -53.72+/-2.64

## smnist5

- scratch q4: aa 18.97+/-0.14 | af 77.42+/-0.37 | bwt -96.77+/-0.47
- scratch q8: aa 19.32+/-0.20 | af 78.50+/-0.24 | bwt -98.13+/-0.29
- er q4: aa 34.45+/-1.12 | af 50.40+/-3.26 | bwt -63.00+/-4.07
- er q8: aa 40.92+/-1.81 | af 46.08+/-3.10 | bwt -57.60+/-3.87
- derpp q4: aa 30.74+/-1.63 | af 27.41+/-2.93 | bwt -34.27+/-3.66
- derpp q8: aa 34.82+/-1.49 | af 25.60+/-2.58 | bwt -32.00+/-3.23

## F1 smnist10 capacity vs q4

- smnist10 scratch q6 - q4: d=+0.00 pp [-0.01,+0.02], t=+0.37, p=0.720, p_holm=0.720, wilcoxon=0.977, dz=+0.12, n=10
- smnist10 scratch q8 - q4: d=-0.01 pp [-0.02,+0.00], t=-1.10, p=0.299, p_holm=0.598, wilcoxon=0.438, dz=-0.35, n=10
- smnist10 er q6 - q4: d=+2.49 pp [+1.34,+3.43], t=+4.37, p=0.002, p_holm=0.005, wilcoxon=0.006, dz=+1.38, n=10
- smnist10 er q8 - q4: d=+4.77 pp [+3.72,+5.74], t=+8.73, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+2.76, n=10
- smnist10 derpp q6 - q4: d=+2.23 pp [+1.78,+2.71], t=+8.91, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+2.82, n=10
- smnist10 derpp q8 - q4: d=+2.89 pp [+2.37,+3.37], t=+10.70, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+3.38, n=10

## F2 smnist5 capacity vs q4

- smnist5 scratch q8 - q4: d=+0.36 pp [+0.21,+0.50], t=+4.61, p=0.001, p_holm=0.001, wilcoxon=0.004, dz=+1.46, n=10
- smnist5 er q8 - q4: d=+6.46 pp [+5.17,+7.54], t=+10.01, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+3.17, n=10
- smnist5 derpp q8 - q4: d=+4.08 pp [+2.86,+5.51], t=+5.62, p=0.000, p_holm=0.001, wilcoxon=0.002, dz=+1.78, n=10
