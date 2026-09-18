# Journal Conversion Plan — Cross-Domain QCL

> Base: IEEE QAI 2026 submission (paper ID 191, rejected Sep 2026, 2x weak
> reject). Goal: a journal version that closes every reviewer objection with
> new evidence, not just rewrites.
> Working repo: `danimap27/Experimento_CrossDomain_QTL` (this repository).
> Status of this document: v1, 2026-09-17. Owner: D. Martín-Pérez.

---

## 1. What the journal version must deliver

Three things separate a journal paper from the rejected conference version:

1. **A mechanism, not just a result.** Why does synthetic pre-training help
   quantum circuits? Answers via loss-landscape, gradient-variance (barren
   plateau) and parameter-distribution evidence. (R2.1, R4.8, R4.9)
2. **A fair, rigorous evaluation.** Established CL baselines (EWC, rehearsal),
   standard scenarios (task-IL and class-IL), larger task sequences, multiple
   seeds with paired statistics, absolute accuracies reported. (R2.2, R4.2,
   R4.3, R4.4, R4.13, R4.14, R4.15)
3. **Correct, complete, honest reporting.** Fixed circuit definitions, a
   separated methodology, quantitative SOTA table, moderated hardware claims.
   (R4.11, R4.12, R4.16, R4.17)

Additionally, an internal audit (Section 3) found defects in the submitted
version that must be fixed regardless of reviewer comments.

---

## 2. Reviewer map (comment → action → evidence)

### Reviewer #2

| # | Comment | Action | Where it lands | Status |
|---|---------|--------|----------------|--------|
| R2.1 | Novelty overclaimed; no justification of why synthetic data helps; asks for loss landscape / parameter distribution / barren plateau analysis | (a) Reposition novelty as *first systematic mechanism study* of cross-domain initialization for QCL; (b) new mechanism workstream M (gradient variance, curvature, parameter distributions, task-gradient conflict) | Sec. Intro, Sec. Methodology, new Sec. Mechanism | designed (W2) |
| R2.2a | Forgetting drop metric not explained theoretically | Formal definition (AA/AF/BWT/FWT per CL literature) + explanation of ΔA as parameter interference; task-gradient alignment analysis | Sec. Metrics | designed (W1) |
| R2.2b | No statistical tests; std ~6.43 | Paired t / Wilcoxon + Holm + Cohen dz + bootstrap CI; 20 seeds; CIs in all main tables | Sec. Results | **done for existing 5-seed data** (`analysis/statistical_analysis.py`); expand to 20 seeds (W1) |
| R2.2c | Missing QCL baselines: EWC [R1], simple rehearsal | Implement quantum EWC (Fisher penalty, λ sweep), simple rehearsal (buffer), QTL+EWC combo; include classical head reference | Sec. Results | designed (W1) |
| R2.2d | Exp. 1 never reports absolute Task A accuracy | Store `acc_a_init` / `acc_a_final` per run; report both + ΔA | Sec. Results | audit confirmed missing; campaign fix (W1) |
| R2.3 | Scalability: 4 qubits only; ablate qubit count, depth, task count; classical pre-training cost dominates; fault-tolerant resource utilization | Scaling workstream: qubits {4,6,8,10,12} × depth {1,2,4} × tasks {2,5}; resource accounting (gates, depth, params, classical cost breakdown); FT-resource estimate appendix | New Sec. Scaling + Appendix | designed (W2-W3) |

### Reviewer #4

| # | Comment | Action | Where it lands | Status |
|---|---------|--------|----------------|--------|
| R4.1 | Chain of arguments not established | Rewrite Introduction as an explicit argument chain (context → problem → prior solutions → gap → hypothesis → roadmap) | Intro | rewrite (W3) |
| R4.2, R4.10 | Coverage of classical CL methods limited; work looks preliminary | Expand Related Work (regularization: EWC/SI/MAS; replay: ER/DER++; architectural; taxonomy) **and** run the corresponding baselines | Related Work + Results | designed (W1-W2) |
| R4.3, R4.13 | CL scenario not well defined; crafted settings | Formal scenario definition (task-IL / domain-IL / class-IL after van de Ven), protocol stated precisely, both task-IL and class-IL evaluated | Methodology + Results | designed (W2) |
| R4.4, R4.15 | Datasets too small | Headline protocol = 5-task sequences on split-MNIST / split-Fashion-MNIST / split-CIFAR-10; CIFAR-100 as extension if time allows | Results | designed (W2-W3) |
| R4.5, R4.6 | Not mature; "no empirical evidence" of forgetting | Full per-task retention matrices (who forgets what, quantified); corrected Experiment 2 (see audit finding A1); comparison with published hardware CL (Zhang et al.) | Results | designed (W1) |
| R4.7 | Motivation for CL study unconvincing | Intro rewrite: NISQ constraints, sequential data streams, no-memory requirement; tie to demonstrated QCL on hardware | Intro | rewrite (W3) |
| R4.8, R4.9 | Rationale of cross-domain synthetic pre-training unclear | Mechanism workstream (M) + "what makes a good source" ablation (Gaussian vs uniform vs rings vs XOR vs Fourier; source size) | Methodology + Results | designed (W2) |
| R4.11 | Circuit descriptions incomplete / inconsistent | Rewrite circuit section with exact PennyLane-consistent definitions; fix two factual mismatches (audit A3-A4); add parameter/gate count table | Methodology | **audit done**; rewrite (W2) |
| R4.12 | Mixes methodology with experimental pipeline | Split into Methodology (formal) and Experimental Setup (implementation) sections | Paper structure | rewrite (W3) |
| R4.14 | Task-incremental too simple | class-IL protocol (single head, no task oracle) as the headline demanding scenario | Methodology + Results | designed (W2) |
| R4.16 | SOTA comparison descriptive | Quantitative comparison table (tasks, qubits, hardware, method, reported metrics) incl. Zhang et al. 2026 (92% AA, EWC, 3 tasks, superconducting) | Related Work | designed (W3) |
| R4.17 | Hardware-deployment claims from wall-clock not justified | Remove deployment claims; keep wall-clock only as reproduction cost; add optional real-hardware validation (IBM) or an explicit simulation-only limitation | Intro, Results, Limitations | rewrite (W3); HW decision open |

---

## 3. Internal audit findings (defects to fix in any resubmission)

**A1 — Experiment 2 reports the wrong quantity as "retained accuracy"
(CRITICAL).** The submitted tables/text describe accuracies measured after
Task A training (SE 96.5, BE 73.1, TTN 95.4) as "Task A retained accuracy after
exposure to Task B". The actual retention in the stored `acc_history` is
dramatically different:

| Ansatz | Acc_A after Task A | Acc_A after Task B (true retention) |
|--------|--------------------|-------------------------------------|
| SE | 96.3–96.7 | **23.4–24.8** |
| BE | 72.3–73.1 | **62.5–63.4** |
| TTN | 95.3–95.4 | **22.0–23.8** |

The submitted claim "SE and TTN retain above 95%" is wrong; the honest finding
is a capacity-retention trade-off (BE forgets least but is weak everywhere;
SE/TTN learn more but forget almost everything). Fix: report both columns,
correct every affected sentence, figure and table, and build the topology
discussion on the trade-off. This finding also directly explains why R4 wrote
"no empirical evidence of forgetting".

**A2 — Layer freezing described but never applied (Exp. 1).** The code
`if '0' in name: param.requires_grad = False` never matches any parameter
(`vqc.weights`, `fc.weight`, `fc.bias`); the submitted text says the lowest
layer was frozen. Fix: implement the freeze correctly, or drop it from the
protocol, and add a freeze on/off ablation to isolate its effect (this was also
flagged at QCE26).

**A3 — Embedding described as Ry, implemented as Rx.** `qml.AngleEmbedding`
defaults to `rotation='X'` and the code does not override it. Fix: describe Rx
correctly, or make the rotation an ablation axis.

**A4 — TTN definition mismatch.** The submitted equation uses Ry on both wires
of a block; the code applies Ry then Rz. Fix the write-up to match the code
(RY + RZ + CNOT).

**A5 — Missing stored magnitudes.** Experiment 1 does not persist absolute
accuracies (only drops) — see R2.2d. The journal campaign must store the full
per-task accuracy matrix per run.

**A6 — Table generator bug.** `paper/tables/tab_forgetting.tex` contains a
stray unexpanded `full` row (a duplicated noise profile). The `full` results
directory duplicates `heron_r2` (verified: identical except 1 value across 5
seeds). Exclude `full` from all reporting.

**A7 — TTN wiring and readout are inconsistent (found during mechanism-probe
construction).** The manuscript states that the TTN's active wires "combine
towards qubits 0 and 1, where the readout occurs". In the implementation the
CNOT chain propagates towards the **last** wire (for 4 qubits the final block
is `CNOT(1 -> 3)`, and the readout is `Z_0, Z_1`). Consequently the parameters
of the `{2,3}` subtree and the `RZ` on wire 3 never influence the measured
expectations: their gradients are exactly zero (confirmed empirically:
`Var[grad]` median ~1e-33 for those coordinates in the barren-plateau probe).
The evaluated "TTN" is therefore effectively a much smaller circuit than the
18-parameter ansatz the paper describes (about half the parameters are
functionally dead). Fix for the journal version: align the tree with the
standard definition (combine towards the readout wires, as in Grant et al.) or
move the readout to the root wires, document it exactly, and re-run.
This also affects how the topology comparison of Experiment 2 is interpreted.

**A8 — Experiment 1 does not isolate pre-training from the learning-rate
schedule.** The QTL arm runs with lr 0.01/0.005 while the scratch baseline
runs at 0.05, so the reported 7-9 pp gain mixes the initialization effect with
the optimization regime. Workstream A must include the full 2x2 design
(scratch/synthetic x lr high/low) plus the EWC and rehearsal arms.

**A9 — Figure scripts contain hardcoded values.** Several plotting scripts in
the repository carry literal numbers instead of reading the JSON payloads,
which risks silently regenerating figures that do not match the data.
Regenerate every figure from stored results only.

**A10 — The three noise profiles are statistically indistinguishable in the
current results** (<0.8 pp between profiles, p>0.1). The noise-robustness
narrative needs either the stronger support of the 20-seed rerun or a softer
claim, and the total-error budget of the 4-qubit circuits should be reported so
the reader can see why the profiles barely separate.

---

## 4. Journal experiment campaign

**Fixed design decisions (apply to all workstreams unless stated):**

- Framework: PennyLane (simulation only, `default.qubit` / `default.mixed`),
  consistent with the submitted stack; PyTorch 2.x, scikit-learn PCA.
- Metrics: AA, AF, BWT, FWT (standard CL definitions) plus per-task retention
  matrices stored per run (JSON) and rendered as heatmaps.
- Statistics: ≥20 seeds for headline tables; paired tests (t + Wilcoxon),
  Holm correction within families, Cohen dz, bootstrap 95% CIs; all reported in
  tables. (Existing 5-seed re-analysis: `analysis/outputs/stats_report.md`.)
- Storage: every run serializes the full protocol payload (all accuracies,
  losses, histories, timings, config, seed) so no follow-up analysis requires
  re-execution.
- Compute: Hercules SLURM (standard partition), array jobs, one seed per task;
  `uv`-managed env mirroring `requirements.txt`.

### Workstream A — Main protocol rerun (fixes + baselines)  [W1]

- Conditions: {scratch, QTL-Syn} × {EWC, rehearsal, none} × {SE, TTN} plus
  classical head reference; 4 qubits, 3 layers, 2-task binary protocol as in
  the submission (comparability), 20 seeds, 3 noise profiles.
- Deliverable: Table 1 (absolute accuracies + ΔA + stats) replacing the
  defective Exp. 1; answers R2.2b/c/d, R4.2.

### Workstream B — Standard benchmarks, task-IL + class-IL  [W2]

- Sequential datasets: split-MNIST (5×2 classes), split-Fashion-MNIST,
  split-CIFAR-10 (5×2 classes; MobileNetV2+PCA features). Multi-class head
  maps quantum readout (2–4 expectation values) through the classical layer.
- Scenarios: task-IL (multi-head) and class-IL (single head, no task oracle).
- Conditions: {scratch, QTL-Syn} × {none, EWC, rehearsal, QTL-Syn+EWC} ×
  {SE, TTN} × 10 seeds × {ideal, heron_r2}.
- Deliverable: AA/AF/BWT/FWT tables + retention matrices; answers R4.3, R4.4,
  R4.13, R4.14, R4.15.

### Workstream C — Scaling  [W2-W3]

- Qubits {4, 6, 8, 10, 12} × depth {1, 2, 4} × {scratch, QTL-Syn}, 2-task
  protocol, 10 seeds (subset of the grid for 12 qubits if runtime demands).
- Task count {2, 3, 5} at 6 qubits.
- Deliverable: ΔA and AA vs qubits/depth/tasks; runtime table; answers R2.3.

### Workstream D — Mechanism study  [W2]

- D1 Gradient-variance probe (barren-plateau proxy) at θ0 (post-pre-training)
  vs random init, vs qubits {4..12}: variance of cost gradients over random
  parameter samplings.
- D2 Curvature: top Hessian / Fisher eigenvalues (Lanczos / power iteration on
  Hessian-vector products) and sharpness at θ0 vs random.
- D3 Parameter distributions: angular histograms pre/post pre-training.
- D4 Task-gradient conflict: cos(∇L_A, ∇L_B) at checkpoints along the
  sequential protocol.
- D5 Loss-barrier interpolation between θ_A and θ_B endpoints.
- Deliverable: mechanism section + figures; answers R2.1, R4.8, R4.9.

### Workstream E — Source ablation  [W3]

- Synthetic families: Gaussian clusters (baseline), uniform, rings/two-moons,
  XOR grid, Fourier patterns; source sizes {500, 2000, 8000}; margin sweep at
  fixed family.
- Deliverable: which source properties drive retention; answers R4.8, R4.9 and
  strengthens R2.1.

### Workstream F — Resource accounting & fault-tolerant estimate  [W3]

- Per configuration: #params, 1q/2q gate counts, depth; classical pipeline cost
  (MobileNetV2 feature extraction + PCA, measured); total wall-clock breakdown.
- Appendix: estimate of Clifford+T compilation cost of the ansatz rotations
  and what it implies for FT-era feasibility; framed as an estimate with
  explicit assumptions.
- Deliverable: resource tables; answers R2.3 (last part).

### Workstream G — Real-hardware validation (optional, budget-dependent)  [W4]

- Minimum viable: validate the QTL-Syn vs scratch inference on one IBM QPU;
  project cost from a probing run first (see qpu-budget discipline).
- Decision needed: which account/budget, and whether to include at all.
- Deliverable: simulator-vs-hardware gap paragraph; answers R4.17/R2.3
  partially; makes npj QI a realistic stretch venue.

---

## 5. Paper structure (journal)

1. Introduction (argument chain; moderated claims)
2. Background: VQCs, continual learning taxonomy (task-/domain-/class-IL),
   QCL state of the art (qualitative)
3. Related Work: axes + **quantitative** comparison table (R4.16)
4. Methodology: formal CL protocol; cross-domain pre-training; ansätze with
   exact definitions and parameter counts; metrics (AA/AF/BWT/FWT)
5. Experimental Setup: datasets, feature extraction, noise profiles, hardware
   and software stack, statistical protocol
6. Results: main benchmark; scenarios; scaling; mechanism; ablations; each with
   CIs and tests
7. Discussion: why (does) it work; capacity-retention trade-off; limitations and
   threats to validity; implications for NISQ practice
8. Conclusions
- Appendix: FT resource estimate; extra grids; code/data availability

**Baseline framing note:** the contribution is repositioned as *the mechanism
study + rigorous benchmark of cross-domain initialization*, not a new
state-of-the-art method. This framing is supported by the corrected results and
is robust to the "well-established technique" objection (R2.1).

---

## 6. Venue options

| Venue | Fit | Notes |
|-------|-----|-------|
| Quantum Machine Intelligence (Springer) | Best thematic fit; method + analysis papers welcome | Rolling submissions; precedent in references ([30] Skolik) |
| Quantum (quantum-journal.org) | Good fit; values honest, well-scoped studies | Rolling; APC |
| npj Quantum Information | Stretch; strongest signal | Wants hardware-level impact — only realistic with workstream G |
| IEEE Trans. on Quantum Engineering | Good fit; IEEE ecosystem | Rolling |
| Neurocomputing (special issue) | Q1, general ML audience; invitation was discussed in the group | Check SI scope/deadline before committing |
| EPJ Quantum Technology / QST | Alternatives | Lower priority |

Recommendation: **QMI as primary target**, Neurocomputing SI as schedule-driven
alternative, npj QI only if hardware validation lands. Final call: team
(F. Martínez-Álvarez).

---

## 7. Timeline (relative weeks from 2026-09-17)

| Week | Deliverable |
|------|-------------|
| W1 | Audit fixes in code (A1-A6); statistical module finalized; baseline implementations (EWC, rehearsal); workstream A launched on Hercules |
| W2 | Workstreams B, C, D launched; mechanism analysis drafted; paper skeleton |
| W3 | Workstreams E, F; results tables auto-generated; full draft v1 |
| W4 | Internal review (D. Gutiérrez, F. Rodríguez-Díaz); revisions; venue decision; submission package |
| W5+ | Submission; workstream G if hardware budget is granted |

Reality check: campaigns A-C on Hercules dominate the schedule; launch early
and keep the paper writing in parallel.

---

## 8. Open decisions

1. **Venue** (team) — see Section 6.
2. **Hardware budget** for workstream G (IBM account; ~30–60 min QPU estimate,
   to be revised after a probing run).
3. **Scope of workstream B** — include split-CIFAR-10 as headline or keep as
   extension (depends on MobileNet feature cost).
4. **Hercules allocation** — confirm project account and queue limits for the
   combined arrays (~2-3k jobs).
5. **Author list / corresponding author** — unchanged from submission unless
   the team decides otherwise.

## 9. Evidence produced so far

- `analysis/statistical_analysis.py` + `analysis/outputs/` — paired tests,
  Holm correction, effect sizes, bootstrap CIs over the 20 stored runs;
  field audit of the missing Exp. 1 magnitudes; `full`-vs-`heron_r2`
  consistency check.
- Key numbers: pooled Exp. 1 effect +8.03 pp [4.47, 11.63], dz=+1.09,
  p<0.001 (15 pairs); per-profile significance reached only under Heron r2
  at n=5 (p=0.028) — the 20-seed rerun is required for the journal tables.
- Corrected Experiment 2 retention values (finding A1).
- `analysis/mechanism/gradient_probe.py` — first mechanism evidence (D1):
  - **Barren-plateau sweep** (Var[grad] of a global cost over random parameter
    draws, 150 samples per point): SEL decays ~exponentially with qubit count
    (5.2e-3 at 4q, 6.5e-4 at 6q, 1.4e-4 at 8q, 3.4e-5 at 10q, 1.2e-5 at 12q)
    while TTN decays far more gently (6.8e-3, 1.8e-3, 1.0e-3, 5.4e-4,
    3.6e-4). At 12 qubits the hierarchical topology keeps ~31x more gradient
    variance than SEL. Direct evidence for the scalability argument (R2.3).
  - **Task-A gradient at theta_0 vs random** (4-qubit SEL, 10 random inits):
    |grad| = 3.02 at theta_0 vs 0.35 +/- 0.16 at random (about 8.5x larger),
    per-coordinate variance 0.209 vs 0.0036 +/- 0.0028, fraction of
    near-zero coordinates ~0.14 in both cases. Interpretation caveat: theta_0
    is not a minimum of Task A (its Task-A loss, 1.83, is above the random
    mean of 0.78), so the result is best read as "the synthetic prior lands
    in an active region of the landscape rather than the weak-gradient region
    random initialization starts from", which is precisely the mechanism the
    journal version must characterize fully in W2.
  - Parameter-distribution note: post-pre-training VQC angles occupy
    [-7.97, 7.46], i.e. far outside the canonical [-pi, pi] interval, with
    mean 3.20 and std 2.73. The role of angle wrapping deserves an explicit
    treatment in the parameter-distribution analysis.
  - **Technical caveat (T1)**: second-order autodiff through the PennyLane
    TorchLayer is not trustworthy (classical control agrees to ~1e-9 across
    computation paths, the hybrid model disagrees at O(1), eigenvalues
    0.48 vs 0.55). Curvature/sharpness analyses in W2 must use parameter-shift
    or double-precision finite differences with an FD self-test; the
    `hess` probe ships disabled-by-warning with diagnostics under
    `analysis/mechanism/diagnostics/`.
  Outputs land in `analysis/mechanism/outputs/`.
- Corrected figures and LaTeX tables built from the raw JSON payloads, a
  draft response-to-reviewers and a coauthor message draft live in the team
  review packet (see Section 10). Both audits agree on the essential findings.

## 10. Related working material (team review packet)

Companion packet prepared in parallel (17 Sep 2026), in Spanish, for the
team: `~/papers/reviews/QAI2026-191/`. It contains the full review report, the
data-integrity findings H1-H9 (coincident with audit findings A1-A10 here), the
experiment specification E1-E8, a draft response-to-reviewers, corrected
figures/tables generated only from the raw JSON, and a message draft for the
coauthors (F. Martínez-Álvarez, D. Gutiérrez-Avilés).

Housekeeping notes from that packet that affect this plan:

- The LaTeX source of the *submitted* PDF ("Mitigating" version) is not on the
  homelab; the local `.tex` files correspond to earlier drafts. The journal
  manuscript will therefore be written from the corrected skeleton rather than
  patched onto the submitted source.
- The two audits should be treated as one workstream: this document is the
  working plan (repo, English), the packet is the review dossier and the
  artifact set (Spanish) for the team.
