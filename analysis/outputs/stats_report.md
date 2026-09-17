# Statistical re-analysis — journal evidence

## Experiment 1 — forgetting drop (paired over seeds)

- **ideal**: baseline 72.10 +/- 2.61, QTL 65.10 +/- 5.09, delta +7.00 pp (9.7%), t=+2.14 p=0.099, wilcoxon p=0.125, dz=+0.96, CI95 [+0.70, +12.40]
- **heron_r2**: baseline 72.90 +/- 2.01, QTL 65.10 +/- 3.73, delta +7.80 pp (10.7%), t=+3.37 p=0.028, wilcoxon p=0.062, dz=+1.51, CI95 [+4.50, +12.30]
- **legacy_nisq**: baseline 70.50 +/- 7.19, QTL 61.20 +/- 4.04, delta +9.30 pp (13.2%), t=+2.00 p=0.116, wilcoxon p=0.188, dz=+0.90, CI95 [+0.80, +17.10]
- **pooled (15 pairs)**: delta +8.03 pp, t=+4.21 p=$<$0.001, wilcoxon p=0.003, dz=+1.09, CI95 [+4.47, +11.63]

Field audit: Experiment 1 stores ['drop_base', 'drop_qtl']. Absolute Task~A accuracies after Task~A training are NOT stored in this campaign (reviewer #2 finding confirmed). The journal campaign must store acc_a_init/acc_a_final per run and report them in the manuscript tables.

## Experiment 2 — retained Task A accuracy per ansatz

- **ideal**: Strongly Entangling AccA 96.30+/-0.91; Basic Entangler AccA 72.90+/-1.52; TTN AccA 95.40+/-1.56
  - Strongly Entangling vs Basic Entangler: delta +23.40 pp, p=$<$0.001, p_holm=$<$0.001, dz=+9.72
  - Strongly Entangling vs TTN: delta +0.90 pp, p=0.286, p_holm=0.286, dz=+0.55
  - Basic Entangler vs TTN: delta -22.50 pp, p=$<$0.001, p_holm=$<$0.001, dz=-9.28
- **heron_r2**: Strongly Entangling AccA 96.50+/-0.61; Basic Entangler AccA 73.10+/-1.34; TTN AccA 95.40+/-1.56
  - Strongly Entangling vs Basic Entangler: delta +23.40 pp, p=$<$0.001, p_holm=$<$0.001, dz=+15.03
  - Strongly Entangling vs TTN: delta +1.10 pp, p=0.282, p_holm=0.282, dz=+0.56
  - Basic Entangler vs TTN: delta -22.30 pp, p=$<$0.001, p_holm=$<$0.001, dz=-8.54
- **legacy_nisq**: Strongly Entangling AccA 96.70+/-0.57; Basic Entangler AccA 72.30+/-0.91; TTN AccA 95.30+/-1.20
  - Strongly Entangling vs Basic Entangler: delta +24.40 pp, p=$<$0.001, p_holm=$<$0.001, dz=+20.44
  - Strongly Entangling vs TTN: delta +1.40 pp, p=0.141, p_holm=0.141, dz=+0.82
  - Basic Entangler vs TTN: delta -23.00 pp, p=$<$0.001, p_holm=$<$0.001, dz=-23.00

## Experiment 3 — cross-domain initialization (target accuracy)

- **ideal**: scr_acc 90.10+/-2.13; qtl_acc 90.90+/-1.60; mob_qtl_acc 89.70+/-1.15
  - qtl_acc vs scr_acc: delta +0.80 pp, p=0.614, p_holm=1.000, dz=+0.24
  - mob_qtl_acc vs scr_acc: delta -0.40 pp, p=0.732, p_holm=1.000, dz=-0.16
  - qtl_acc vs mob_qtl_acc: delta +1.20 pp, p=0.319, p_holm=0.958, dz=+0.51
- **heron_r2**: scr_acc 90.80+/-1.72; qtl_acc 89.90+/-0.65; mob_qtl_acc 88.60+/-1.52
  - qtl_acc vs scr_acc: delta -0.90 pp, p=0.374, p_holm=0.430, dz=-0.45
  - mob_qtl_acc vs scr_acc: delta -2.20 pp, p=0.143, p_holm=0.430, dz=-0.81
  - qtl_acc vs mob_qtl_acc: delta +1.30 pp, p=0.144, p_holm=0.430, dz=+0.81
- **legacy_nisq**: scr_acc 89.60+/-2.61; qtl_acc 89.90+/-1.39; mob_qtl_acc 89.70+/-1.72
  - qtl_acc vs scr_acc: delta +0.30 pp, p=0.864, p_holm=1.000, dz=+0.08
  - mob_qtl_acc vs scr_acc: delta +0.10 pp, p=0.925, p_holm=1.000, dz=+0.05
  - qtl_acc vs mob_qtl_acc: delta +0.20 pp, p=0.870, p_holm=1.000, dz=+0.08

Consistency note: the local `full` directory duplicates `heron_r2` (1 small difference(s) across 5 seeds in Experiment 3); it is excluded from the reported statistics.

