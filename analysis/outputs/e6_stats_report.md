# E6 -- regularization family (split-MNIST class-IL, ideal)

Cells loaded: scratch/std=10, synth/std=10, er/std=10, ewc/std=10, si/si5=10, si/si50=10, l2/l2a=10, l2/l2b=10, derpp/std=10, synth_si/si5=10, synth_si/si50=10, synth_l2/l2a=10, synth_l2/l2b=10, synth_derpp/std=10

## Descriptives on AA / AF / BWT / FWT (mean +/- sd, n=10)

- scratch/std: aa 18.97+/-0.14 | af 77.42+/-0.37 | bwt -96.77+/-0.47 | fwt -13.78+/-4.38
- synth/std: aa 19.02+/-0.14 | af 77.44+/-0.29 | bwt -96.80+/-0.36 | fwt 0.00+/-0.00
- er/std: aa 34.45+/-1.12 | af 50.40+/-3.26 | bwt -63.00+/-4.07 | fwt -13.78+/-4.38
- ewc/std: aa 17.26+/-0.62 | af 74.60+/-1.07 | bwt -93.25+/-1.33 | fwt -13.78+/-4.38
- si/si5: aa 23.03+/-3.30 | af 6.81+/-4.38 | bwt -8.51+/-5.47 | fwt -13.78+/-4.38
- si/si50: aa 19.95+/-0.04 | af 0.01+/-0.03 | bwt -0.02+/-0.03 | fwt -13.78+/-4.38
- l2/l2a: aa 12.23+/-1.94 | af 31.70+/-9.32 | bwt -39.62+/-11.65 | fwt -13.78+/-4.38
- l2/l2b: aa 14.29+/-4.36 | af 5.96+/-4.19 | bwt -7.44+/-5.23 | fwt -13.78+/-4.38
- derpp/std: aa 30.74+/-1.63 | af 27.41+/-2.93 | bwt -34.27+/-3.66 | fwt -13.78+/-4.38
- synth_si/si5: aa 22.16+/-4.40 | af 25.91+/-10.66 | bwt -32.39+/-13.32 | fwt 0.00+/-0.00
- synth_si/si50: aa 18.43+/-1.85 | af 18.82+/-1.69 | bwt -23.53+/-2.11 | fwt 0.00+/-0.00
- synth_l2/l2a: aa 11.45+/-5.30 | af 8.57+/-5.27 | bwt -10.71+/-6.59 | fwt 0.00+/-0.00
- synth_l2/l2b: aa 16.14+/-3.86 | af 3.81+/-3.87 | bwt -4.76+/-4.84 | fwt 0.00+/-0.00
- synth_derpp/std: aa 29.39+/-1.22 | af 18.53+/-1.78 | bwt -23.17+/-2.22 | fwt 0.00+/-0.00

## F1 vs scratch

- er/std - scratch: d=+15.49 pp [+14.73,+16.10], t=+41.75, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+13.20, n=10
- ewc/std - scratch: d=-1.71 pp [-2.12,-1.27], t=-7.49, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=-2.37, n=10
- si/si5 - scratch: d=+4.07 pp [+2.15,+5.97], t=+3.94, p=0.003, p_holm=0.007, wilcoxon=0.006, dz=+1.25, n=10
- si/si50 - scratch: d=+0.98 pp [+0.89,+1.09], t=+18.16, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+5.74, n=10
- l2/l2a - scratch: d=-6.73 pp [-7.84,-5.63], t=-11.21, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=-3.55, n=10
- l2/l2b - scratch: d=-4.67 pp [-7.27,-2.19], t=-3.37, p=0.008, p_holm=0.008, wilcoxon=0.006, dz=-1.07, n=10
- derpp/std - scratch: d=+11.78 pp [+10.75,+12.67], t=+22.66, p=0.000, p_holm=0.000, wilcoxon=0.002, dz=+7.17, n=10

## F2 prior effect

- synth - scratch: d=+0.05 pp [-0.02,+0.13], t=+1.26, p=0.240, p_holm=0.962, wilcoxon=0.375, dz=+0.40, n=10
- synth_si5 - si5: d=-0.87 pp [-5.03,+2.95], t=-0.41, p=0.694, p_holm=1.000, wilcoxon=0.922, dz=-0.13, n=10
- synth_si50 - si50: d=-1.51 pp [-2.49,-0.33], t=-2.57, p=0.030, p_holm=0.181, wilcoxon=0.037, dz=-0.81, n=10
- synth_l2a - l2a: d=-0.79 pp [-4.58,+3.08], t=-0.38, p=0.713, p_holm=1.000, wilcoxon=0.695, dz=-0.12, n=10
- synth_l2b - l2b: d=+1.85 pp [-2.52,+5.91], t=+0.81, p=0.439, p_holm=1.000, wilcoxon=0.432, dz=+0.26, n=10
- synth_derpp - derpp: d=-1.35 pp [-2.33,-0.33], t=-2.51, p=0.033, p_holm=0.181, wilcoxon=0.049, dz=-0.79, n=10

## F3 lambda

- si50 - si5 (lambda): d=-3.09 pp [-5.01,-1.14], t=-2.97, p=0.016, p_holm=0.032, wilcoxon=0.020, dz=-0.94, n=10
- l2b - l2a (lambda): d=+2.06 pp [-0.54,+4.60], t=+1.47, p=0.176, p_holm=0.176, wilcoxon=0.232, dz=+0.46, n=10
