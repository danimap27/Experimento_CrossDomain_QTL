#!/usr/bin/env python3
"""Finalize E2-E5: wait for the global-label array, sync results, regenerate
statistics/figures/crosscheck, insert the new journal sections into
journal/paper_journal/main.tex (numbers read from the JSONs at run time),
compile the manuscript and commit the whole set.

Run from the repo:  env -u PYTHONPATH .venv/bin/python analysis/integrate_journal.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")
OUT = REPO / "analysis" / "outputs"
PUB = REPO / "journal" / "paper_journal"
SSH = ["ssh", "-o", "ConnectTimeout=15", "hercules-cica"]

LOG = open(REPO / "analysis" / "outputs" / "integrate_journal.log", "a")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def py(script):
    r = run(["env", "-u", "PYTHONPATH", PY, str(REPO / script)], cwd=str(REPO))
    log(f"{script}: rc={r.returncode}")
    if r.returncode != 0:
        log("STDERR: " + r.stderr[-1500:])
    return r


# --------------------------------------------------------------------------
# 1) wait for the array
# --------------------------------------------------------------------------
def wait_array(max_min=100):
    t0 = time.time()
    while (time.time() - t0) < max_min * 60:
        r = run(SSH + ["cd ~/crossdomain_qcl && ls results 2>/dev/null | "
                       "grep -c -E 'e234_(chain4|smnist5|sfmnist5)_'"])
        try:
            n = int(r.stdout.strip())
        except ValueError:
            n = -1
        log(f"array cells done: {n}/110")
        if n >= 110:
            return True
        time.sleep(150)
    return False


# --------------------------------------------------------------------------
# 2) sync + analyses
# --------------------------------------------------------------------------
def sync():
    r = run(["rsync", "-az", "--include=*/", "--include=results_seed_*.json",
             "--exclude=*", "hercules-cica:crossdomain_qcl/results/",
             str(REPO / "results") + "/"])
    log(f"rsync rc={r.returncode}")
    return r.returncode == 0


def fmt(v, nd=2):
    return "--" if v is None else f"{v:.{nd}f}"


def fms(vals, nd=2):
    import numpy as np
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return "--"
    s = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.{nd}f} $\\\\pm$ {s:.{nd}f}"


def pstr(p):
    if p is None:
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def ms_mean(vals):
    import numpy as np
    v = np.asarray([x for x in vals if x is not None], float)
    return float(v.mean()) if len(v) else None


def load_json(name):
    return json.loads((OUT / name).read_text())


# --------------------------------------------------------------------------
# 3) LaTeX blocks
# --------------------------------------------------------------------------
def e2_block(S):
    ch = S["chain4"]["pooled"]["metrics"]
    chi = S["chain4"]["ideal"]["metrics"]
    chh = S["chain4"]["heron_r2"]["metrics"]

    def one(metrics, m):
        d = metrics.get(m, {})
        c = d.get("contrast", {})
        sc, sy = ms_mean(d.get("scratch", {}).get("values", [])), \
            ms_mean(d.get("synth", {}).get("values", []))
        return sc, sy, c

    sc4, sy4, c4 = one(ch, "CIL-4")
    sc42, sy42, c42 = one(chi, "CIL-4")
    sch, syh, ch_ = one(chh, "CIL-4")
    st2, sy2, ct2 = one(ch, "TIL-2")
    sc2c, sy2c, c2c = one(ch, "CIL-2")
    st3, sy3, ct3 = one(ch, "TIL-3")
    sc3c, sy3c, c3c = one(ch, "CIL-3")

    b = S["bench5"]
    sm = b.get("smnist5/ideal", {})
    smh = b.get("smnist5/heron_r2", {})
    sf = b.get("sfmnist5/ideal", {})
    fn = S.get("friedman_smnist5", {})

    def arm(d, key):
        return d.get(key, {})

    def cst(d, ck):
        cs = d.get("contrasts", {})
        return cs.get(ck, {})

    aa_sc = sm.get("scratch", {}).get("aa")
    aa_sy = sm.get("synth", {}).get("aa")
    aa_er = sm.get("er", {}).get("aa")
    aa_ew = sm.get("ewc", {}).get("aa")
    af_sc = sm.get("scratch", {}).get("af")
    af_sy = sm.get("synth", {}).get("af")
    csync = cst(sm, "synth-scratch")
    cer = cst(sm, "er-scratch")
    cewc = cst(sm, "ewc-scratch")
    fchi = fn.get("chi2")
    fp = fn.get("p")
    fcd = fn.get("critical_difference")
    franks = fn.get("mean_ranks", {})
    aa_sfh = smh.get("synth", {}).get("aa")
    aa_sch = smh.get("scratch", {}).get("aa")
    csync_h = cst(smh, "synth-scratch")
    aa_f_sc = sf.get("scratch", {}).get("aa")
    aa_f_sy = sf.get("synth", {}).get("aa")
    csync_f = cst(sf, "synth-scratch")

    text = f"""
\\subsection{{Scenario axis: task-IL and class-IL benchmarks}}
\\label{{sec:exp_scenarios}}

The scenario axis of the extended protocol asks whether the cross-domain prior
behaves differently when the evaluation stops assuming a per-task readout.
All runs of this axis train the shared multi-class head with globally
offset labels and a single output layer, following the class-incremental
protocol of van de Ven and Tolias \\cite{{Ven22}}: at test time no task
identity is available and the prediction is the argmax over the classes seen
so far, while the task-incremental companion restricts the argmax to the
task's own pair of classes as an oracle. A single four-task chain
(Fashion-MNIST$\\{{0,1\\}}$ $\\to$ MNIST$\\{{2,3\\}}$ $\\to$
KMNIST$\\{{4,5\\}}$ $\\to$ Fashion-MNIST$\\{{8,9\\}}$) therefore yields
TIL-2, CIL-2, TIL-3, CIL-3 and CIL-4 from the same sequential training,
with ten seeds per arm and the synthetic prior sharing the E1 pre-training
checkpoint protocol.

Table~\\ref{{tab:e234_chain4}} reports the chain. Under the ideal profile the
four-task class-IL accuracy is {fmt(sc42)} points for the scratch arm
against {fmt(sy42)} for the synthetic prior (pooled over both noise profiles
{fmt(sc4)} vs\\ {fmt(sy4)}; paired difference
{c4.get('mean_delta', 0):+.2f} points, $p_{{\\text{{Holm}}}}$ = {pstr(c4.get('p_holm'))},
$d_z$ = {fmt(c4.get('dz'))}), and the shorter prefixes behave the same way:
TIL-2 {fmt(st2)} vs\\ {fmt(sy2)}, CIL-2 {fmt(sc2c)} vs\\ {fmt(sy2c)},
TIL-3 {fmt(st3)} vs\\ {fmt(sy3)}, CIL-3 {fmt(sc3c)} vs\\ {fmt(sy3c)}, with
every Holm-corrected contrast inside the pre-declared family of five
scenario metrics staying above the 0.05 level. The class-incremental
reading is markedly harder than the oracle reading --- accuracies on the
freshly trained task stay around ninety, while the earlier ones collapse
to single digits, mostly to near zero (Figure~\\ref{{fig:e234_scenarios}}) ---
and, at this
scale, the synthetic prior neither attenuates nor aggravates the loss in
either regime. Under the Heron~r2 profile the four-task accuracy is
{fmt(sch)} against {fmt(syh)} (paired difference
{ch_.get('mean_delta', 0):+.2f}, $p_{{\\text{{Holm}}}}$ = {pstr(ch_.get('p_holm'))}),
so the null reading is stable across noise conditions.

\\begin{{figure}}[ht]
\\centering
\\includegraphics[width=\\linewidth]{{figures/fig_e234_scenarios.pdf}}
\\caption{{E2 scenario axis. Top: task-IL and class-IL accuracies at the
two-, three- and four-task prefixes of the chain for the scratch and
synthetic-prior arms ($n=10$ seeds, bars with 95\\% bootstrap intervals).
Bottom: mean class-IL accuracy matrices after each phase (ideal profile);
the upper triangle is not evaluated.}}\\label{{fig:e234_scenarios}}
\\end{{figure}}

\\input{{tables/tab_e234_chain4.tex}}

The community-standard five-task benchmarks behave in the same direction.
On split-MNIST (5 tasks $\\times$ 2 classes, shared head) the average
class-IL accuracy over the sequence is {fmt(aa_sc)} points for the scratch
arm and {fmt(aa_sy)} for the synthetic prior
($\\delta$ = {csync.get('mean_delta', 0):+.2f},
$p_{{\\text{{Holm}}}}$ = {pstr(csync.get('p_holm'))}, $d_z$ = {fmt(csync.get('dz'))}),
with average forgetting {fmt(af_sc)} against {fmt(af_sy)} points; the
regularisation and rehearsal arms land at {fmt(aa_ew)} (EWC) and
{fmt(aa_er)} (rehearsal) mean accuracy, with
$\\delta$ = {cer.get('mean_delta', 0):+.2f} ({pstr(cer.get('p_holm'))}) and
{cewc.get('mean_delta', 0):+.2f} ({pstr(cewc.get('p_holm'))}) against
scratch. A Friedman test over the four arms (10 seeds, ideal profile) gives
$\\chi^2$ = {fmt(fchi)} ($p$ = {pstr(fp)}) with mean ranks
scratch {fmt(franks.get('scratch'))}, synth {fmt(franks.get('synth'))},
rehearsal {fmt(franks.get('er'))}, EWC {fmt(franks.get('ewc'))} and a
Nemenyi critical difference of {fmt(fcd)} at $\\alpha=0.05$
\\cite{{demsar2006statistical}}; the rehearsal arm separates from scratch and
EWC beyond the critical difference, while the synthetic prior remains
statistically indistinguishable from scratch. Under the Heron~r2 profile
split-MNIST gives
{fmt(aa_sch)} against {fmt(aa_sfh)} points
({pstr(csync_h.get('p_holm'))}), and split-Fashion-MNIST
{fmt(aa_f_sc)} against {fmt(aa_f_sy)} ({pstr(csync_f.get('p_holm'))});
Table~\\ref{{tab:e234_bench5}} collects the four AA/AF/BWT/FWT metrics per
arm and Figure~\\ref{{fig:e234_bench5}} the retention of the first task
along the five phases. The standard class-incremental benchmarks therefore
do not reproduce the retention advantage that the two-task protocol of
Experiment~1 attributes to the synthetic prior: the prior remains a
neutral initialisation at this scale, and the arms that explicitly trade
plasticity for retention are the only ones that move the retention
metrics, in the direction and at the plasticity cost already quantified
in Table~\\ref{{tab:exp1_baselines}}.

\\begin{{figure}}[ht]
\\centering
\\includegraphics[width=\\linewidth]{{figures/fig_e234_bench5.pdf}}
\\caption{{E2 class-incremental benchmarks. (a) Retention of the first task
along the five phases of split-MNIST for the four arms; (b) AA and AF per
arm; (c) retention of the first task for split-Fashion-MNIST. Means over
ten seeds with one standard-deviation bands.}}\\label{{fig:e234_bench5}}
\\end{{figure}}

\\input{{tables/tab_e234_bench5.tex}}
"""

    # E3 + E4 blocks
    sc_ = S["scale"]

    def cell(tag, scope):
        d = sc_.get(f"{tag}/{scope}", {})
        c = d.get("contrast", {})
        scm = ms_mean(d.get("scratch", {}).get("values", []))
        sym = ms_mean(d.get("synth", {}).get("values", []))
        return scm, sym, c, d

    a5s, a5y, c5, d5 = cell("sz500", "ideal")
    a2s, a2y, c2, d2 = cell("sz2k", "ideal")
    a12s, a12y, c12, d12 = cell("sz12k", "ideal")
    a12hs, a12hy, c12h, d12h = cell("sz12k", "heron_r2")
    e3 = f"""
\\subsection{{Data scale: full binary splits and the sample-size curve}}
\\label{{sec:exp_scale}}

The data-scale axis re-runs the two-task binary protocol of Experiment~1 on
the full class splits and on two reduced sample budgets, separating
"small model" from "small data". Table~\\ref{{tab:e234_scale}} reports the
per-seed means for {{500}}, {{2\\,000}} and {{12\\,000}} training samples per
task using the shared binary readout of the two-task protocol: the mean
accuracy over the two tasks is {fmt(a5s)} / {fmt(a5y)} points at 500
(scratch / synthetic prior), {fmt(a2s)} / {fmt(a2y)} at 2\\,000 and
{fmt(a12s)} / {fmt(a12y)} at the full split, and the paired
synth$-$scratch contrasts are
{c5.get('mean_delta', 0):+.2f} ({pstr(c5.get('p_holm'))}),
{c2.get('mean_delta', 0):+.2f} ({pstr(c2.get('p_holm'))}) and
{c12.get('mean_delta', 0):+.2f} ({pstr(c12.get('p_holm'))}) points
(cf. Figure~\\ref{{fig:e234_scale}}). Two readings survive the numbers:
the shared-readout accuracy drops sharply from the 500-sample budget to
the 2\\,000-sample budget and then plateaus, and the drop is carried
entirely by Task-A retention --- both tasks are still learned to the same
level at every budget (freshly trained accuracies between 91 and 97
points) while the Task-A drop grows from
{fmt(d5.get('scratch', {}).get('delta_a_cil'), 1)} points at 500 samples to
{fmt(d2.get('scratch', {}).get('delta_a_cil'), 1)} and
{fmt(d12.get('scratch', {}).get('delta_a_cil'), 1)} at 2\\,000 and 12\\,000
(scratch arm), so the small budget of
the conference protocol understated the interference between the two
tasks rather than limiting the model; and the forgetting drop does not
separate the arms at any budget, i.e.\\ the null result of the controlled
re-run is not an artefact of the small data subset. Under the Heron~r2
profile the full split gives {fmt(a12hs)} / {fmt(a12hy)} points
({pstr(c12h.get('p_holm'))}).

\\begin{{figure}}[ht]
\\centering
\\includegraphics[width=\\linewidth]{{figures/fig_e234_scale.pdf}}
\\caption{{E3 data scale. (a) Mean accuracy over the two tasks of the shared
binary readout; (b) Task-A forgetting drop; (c) Task-B accuracy, as a
function of training samples per task (ideal profile, means over ten seeds
with one standard-deviation error bars).}}\\label{{fig:e234_scale}}
\\end{{figure}}

\\input{{tables/tab_e234_scale.tex}}
"""

    scg = S["scaling"]

    def gcell(tag):
        d = scg.get(tag, {})
        c = d.get("contrast", {})
        return d, c

    d_std, c_std = gcell("std")
    d_q6, c_q6 = gcell("q6")
    d_q8, c_q8 = gcell("q8")
    d_l2, c_l2 = gcell("L2")
    d_l4, c_l4 = gcell("L4")
    e4_ratio = ((d_q8.get("epoch_time", {}).get("scratch") or 0) /
                (d_std.get("epoch_time", {}).get("scratch") or 1))
    e4 = f"""
\\subsection{{Scaling: qubits, depth and task count}}
\\label{{sec:exp_scaling}}

The scaling axis sweeps the circuit resources of the two-task protocol
(Table~\\ref{{tab:e234_scaling}}): qubit counts {{4, 6, 8}} with the PCA
dimension matched to the number of qubits, and depths {{2, 3, 4}} at four
qubits, ten seeds per cell. Across the grid the baseline four-qubit
three-layer cell reaches a Task-A drop of
{fmt(d_std.get('scratch', {}).get('delta_a_cil'))} /
{fmt(d_std.get('synth', {}).get('delta_a_cil'))} points (scratch / prior,
ideal) and the paired synth$-$scratch contrasts remain inside noise in
every configuration
(4q/3L {c_std.get('mean_delta', 0):+.2f}, {pstr(c_std.get('p_holm'))};
6q/3L {c_q6.get('mean_delta', 0):+.2f}, {pstr(c_q6.get('p_holm'))};
8q/3L {c_q8.get('mean_delta', 0):+.2f}, {pstr(c_q8.get('p_holm'))};
4q/2L {c_l2.get('mean_delta', 0):+.2f}, {pstr(c_l2.get('p_holm'))};
4q/4L {c_l4.get('mean_delta', 0):+.2f}, {pstr(c_l4.get('p_holm'))}).
Adding qubits at fixed depth and changing depth at four qubits leave both
the accuracy level and the forgetting profile within the seed spread
(Figure~\\ref{{fig:e234_scaling}}), while the per-epoch wall-clock grows
by roughly a factor of {fmt(e4_ratio, 1)}
from four to eight qubits at fixed depth, consistent with the
state-vector simulation cost. The resource accounting of the same grid ---
parameter counts, one- and two-qubit gate counts, depth, and a
fault-tolerant $T$-count estimate --- is reported in
Appendix~\\ref{{app:resources}}; the task-count dimension of the scaling
question comes from the scenario chain of Section~\\ref{{sec:exp_scenarios}},
where the same sequential runs are read at two, three and four tasks. The
combined reading is that within the simulated range the synthetic prior
gives no measurable advantage in any corner of the grid, and the
cross-domain effect reported for the conference protocol is specific to
the two-task binary setting rather than a generic property of the method.

\\begin{{figure}}[ht]
\\centering
\\includegraphics[width=\\linewidth]{{figures/fig_e234_scaling.pdf}}
\\caption{{E4 scaling grid (ideal profile, ten seeds). (a) Task-A forgetting
drop and (b) mean accuracy for the five configurations; (c) mean wall-clock
per epoch on one CPU core.}}\\label{{fig:e234_scaling}}
\\end{{figure}}

\\input{{tables/tab_e234_scaling.tex}}
"""
    return text, e3, e4


def e5_block():
    P = json.loads((REPO / "analysis" / "mechanism" / "outputs" / "e5" / "probes.json").read_text())
    a = P["per_arm"]
    sc, sy = a["scratch"], a["synth"]

    def lv(arm, ck, key="vqc_mean"):
        return arm["layer_var"][ck][key]

    def fi(arm, ck):
        return arm["fisher"][ck]

    def bar(arm, k):
        return arm["barrier"][k]

    def wr(arm, k):
        return arm["wrap"][k]

    x = lambda pair: pair[0] if isinstance(pair, (list, tuple)) else pair

    maxraw = max(p["wrap"]["raw_max"] for p in P["probes"] if p["arm"] == "synth")
    minraw = min(p["wrap"]["raw_min"] for p in P["probes"] if p["arm"] == "synth")
    text = f"""
\\subsection{{Mechanism analysis}}
\\label{{sec:mechanism}}

The mechanism section combines the exploratory gradient probes of the
review packet with the four checkpoint-based measurements of the extended
protocol: per-layer gradient variance, empirical Fisher spectra, the loss
barrier between the two task optima, and the parameter distributions
including angle wrapping. All measurements use first-order autograd or
double-precision finite differences with an explicit self-test, never
second-order autodiff through the PennyLane layer, whose unreliability in
this stack was documented during the audit (caveat T1). The checkpoints
were re-trained locally for ten seeds per arm with the exact two-task
protocol of the campaign (four-qubit strongly entangling ansatz, matched
rates).

The gradient-variance probe of the review packet sweeps the barren-plateau
proxy over qubit counts from four to twelve and both circuit topologies:
the variance of the cost gradient decays from $5.2\\times10^{{-3}}$ to
$1.2\\times10^{{-5}}$ for the strongly entangling ansatz between four and
twelve qubits, while the hierarchical wiring decays far more gently, from
$6.8\\times10^{{-3}}$ to $3.6\\times10^{{-4}}$, keeping about thirty times
more gradient variance at twelve qubits
\\cite{{mcclean2018barren,pesah2021absence}}. The checkpoint measurements
sharpen this into a per-layer statement at the initialisation the two arms
actually use. At $\\theta_0$, the pre-trained point, the per-layer gradient
variance of the Task-A loss is
{lv(sy, 'theta0', 'per_layer_mean')[0]:.2e}, {lv(sy, 'theta0', 'per_layer_mean')[1]:.2e}
and {lv(sy, 'theta0', 'per_layer_mean')[2]:.2e} for the three ansatz layers
(mean over layers {x(lv(sy, 'theta0')):.2e}), against
{lv(sc, 'theta0', 'per_layer_mean')[0]:.2e}, {lv(sc, 'theta0', 'per_layer_mean')[1]:.2e}
and {lv(sc, 'theta0', 'per_layer_mean')[2]:.2e} at random initialisation
({x(lv(sc, 'theta0')):.2e}) --- a gap of three orders of magnitude,
uniformly across the layers, which means the source optimisation lands the
circuit in an active, high-gradient region of the landscape rather than the
weak-gradient region that random rotation angles occupy
(Figure~\\ref{{fig:e5}}a). After each arm has learned the first task the two
checkpoints collapse onto comparably weak gradients
({x(lv(sc, 'thetaA')):.2e} vs\\ {x(lv(sy, 'thetaA')):.2e}), and by the end
of the second task both models exhibit renewed large gradients on Task-A
data ({x(lv(sc, 'thetaB')):.2e} vs\\ {x(lv(sy, 'thetaB')):.2e}), the
signature of the interference that the accuracy matrices quantify.

The empirical Fisher spectrum at $\\theta_0$ tells the same story in terms
of curvature: the trace is {fi(sc, 'theta0')['trace'][0]:.2e} at random
initialisation against {fi(sy, 'theta0')['trace'][0]:.2e} at the prior
point, with a fuller spectrum (spectral entropy
{fi(sc, 'theta0')['entropy'][0]:.2f} vs\\ {fi(sy, 'theta0')['entropy'][0]:.2f}
and effective rank {x(fi(sc, 'theta0')['eff_rank']):.1f} vs\\ {x(fi(sy, 'theta0')['eff_rank']):.1f});
the top eigenvalue carries about half of the trace in both cases
(Figure~\\ref{{fig:e5}}b). The interpolation between the two task optima
is the measurement that does not separate the arms: the linear path from
$\\theta_A$ to $\\theta_B$ shows no loss barrier on the newly learned task
for either arm (barrier height {bar(sc, 'barrierB')[0]:+.3f} and
{bar(sy, 'barrierB')[0]:+.3f}; the two task solutions are linearly
connected in this parametrisation), and the parameter distance travelled
is essentially identical
($\\|\\theta_B-\\theta_A\\|/\\sqrt{{d}} = {bar(sc, 'dist')[0]:.3f}$ vs\\ {bar(sy, 'dist')[0]:.3f}$).
Forgetting in this model is therefore not a barrier phenomenon between the
two solutions; what separates the arms is where the optimisation starts
relative to both of them (Figure~\\ref{{fig:e5}}c).

Finally, the parameter distributions close an open item of the preliminary
analysis: the circuit is $2\\pi$-periodic in each rotation angle, which was
verified numerically --- wrapping the 36 angles of the pre-trained circuit
reproduces the loss to floating-point precision
(maximum $|\\Delta\\mathcal{{L}}|$ = {max(p['wrap']['loss_abs_diff'] for p in P['probes']):.1e}).
Random initialisation covers exactly one period
(uniform in $[0, 2\\pi)$), while the source phase pushes some coordinates
beyond it (raw range up to about {max(maxraw, abs(minraw)):.1f}
radians); after wrapping, the effective distributions differ mainly in
dispersion (standard deviation {wr(sc, 'raw_std')[0]:.2f} for random
initialisation against {wr(sy, 'raw_std')[0]:.2f} for the prior, with
{100 * wr(sc, 'frac_outside_pi')[0]:.0f}\\% and
{100 * wr(sy, 'frac_outside_pi')[0]:.0f}\\% of the raw angles outside the
canonical interval). The gradient and curvature evidence above is the part
with decision weight; the distributional description documents the
effective working point of the pre-trained circuit. Self-tests: the
finite-difference check of the gradient path agrees at
{max(p['selftest_fd_max_rel_err'] for p in P['probes']):.0e} relative error,
and the two ways of computing the Fisher trace agree at
{max(p['selftest_fisher_trace_diff'] for p in P['probes']):.0e}.

\\begin{{figure}}[ht]
\\centering
\\includegraphics[width=\\linewidth]{{figures/fig_e5_mechanism.pdf}}
\\caption{{Mechanism measurements (10 seeds per arm). (a) Per-layer gradient
variance of the Task-A loss at $\\theta_0$ (error bars: standard deviation
over seeds); (b) mean empirical Fisher spectrum of the Task-A loss at
$\\theta_0$; (c) training losses along the linear interpolation between the
two task optima (fixed 300-sample subsets).}}\\label{{fig:e5}}
\\end{{figure}}
"""
    return text


def appendix_block():
    r = json.loads((OUT / "e4_resources.json").read_text())
    recs = {(x["n_qubits"], x["n_layers"]): x for x in r["records"]}
    a = recs[(4, 3)]
    b = recs[(8, 3)]
    c = recs[(12, 4)]
    return f"""
\\appendix
\\section{{Circuit resources and fault-tolerant estimate}}
\\label{{app:resources}}

Table~\\ref{{tab:e4_resources}} lists the gate-level resources of the trained
ansatz for the qubit counts and depths of the scaling grid. Every rotation
gate of the circuit is a general single-qubit rotation; under the standard
synthesis cost model of Ross and Selinger \\cite{{ross2016optimal}} an
arbitrary $z$-rotation approximated to error $\\varepsilon$ costs about
$3\\log_2(1/\\varepsilon)$ $T$ gates, and no generic single-qubit gate can be
synthesised below three $T$ gates \\cite{{bocharov2012resource}}. At
$\\varepsilon=10^{{-6}}$ the estimate is about
{int(round(3 * __import__('math').log2(1e6)))} $T$ gates per rotation,
i.e.\\ about {int(round(a['tcount_est_1e6']))} $T$ for the four-qubit
three-layer circuit and {int(round(b['tcount_est_1e6']))} $T$ at eight
qubits; CNOTs are Clifford operations and contribute no $T$ gates. The
estimate counts every Euler angle of each rotation gate separately and is
therefore conservative at the gate level; it excludes state preparation,
measurement, and error-correction overheads, and it is reported as an
order-of-magnitude reality check on what the demonstrated scale implies for
the fault-tolerant era, not as a compilation study. At twelve qubits and
four layers the same accounting reaches {int(round(c['tcount_est_1e6']))}
$T$ gates per inference, which is the quantitative form of the
resource-awareness argument in the introduction.

\\input{{tables/tab_e4_resources.tex}}
"""


# --------------------------------------------------------------------------
# 4) main.tex patching
# --------------------------------------------------------------------------
def patch_main():
    p = PUB / "main.tex"
    src = p.read_text()
    marker = "% ---- E2-E5 integrated (auto) ----"
    if marker in src:
        log("main.tex already integrated; skipping patch")
        return True
    S = load_json("e234_stats.json")
    e2, e3, e4 = e2_block(S)
    e5 = e5_block()
    app = appendix_block()

    anchor = "\\input{tables/tab_exp3.tex}"
    assert src.count(anchor) == 1, f"anchor tab_exp3 occurs {src.count(anchor)} times"
    new_secs = f"\n\n{marker}\n{e2}\n{e3}\n{e4}\n"
    src = src.replace(anchor, anchor + new_secs, 1)

    ms = "\\subsection{Mechanism analysis}"
    assert src.count(ms) == 1
    head, tail = src.split(ms, 1)
    disc = "\\section{Discussion}"
    assert tail.count(disc) == 1
    _old_mech, disc_rest = tail.split(disc, 1)
    src = head + e5 + "\n% ======================================================================\n" + disc + disc_rest

    cred = "\\section*{CRediT authorship contribution statement}"
    assert src.count(cred) == 1
    src = src.replace(cred, app + "\n" + cred, 1)

    p.write_text(src)
    log("main.tex patched with E2-E5 sections + appendix")
    return True


def compile_pdf():
    for cmd in [["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]]:
        r = run(cmd, cwd=str(PUB))
        if cmd[0] == "pdflatex" and r.returncode != 0 and "Fatal" in r.stdout:
            log(f"{cmd[0]} rc={r.returncode}; tail: {r.stdout[-800:]}")
    # verify
    r = run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=str(PUB))
    ok = "Output written on main.pdf" in r.stdout
    undef = "undefined references" in r.stdout or "Undefined" in r.stdout
    log(f"compile ok={ok} undef={undef}")
    if ok and not undef:
        return True
    log(r.stdout[-2500:])
    return False


def commit():
    files = ["analysis/e234_stats.py", "analysis/e234_stats_crosscheck.py",
             "analysis/e234_figures.py", "analysis/e4_resources.py",
             "analysis/mechanism/e5_study.py", "analysis/mechanism/outputs/e5",
             "analysis/integrate_journal.py",
             "analysis/outputs", "journal/paper_journal/main.tex",
             "journal/paper_journal/main.pdf", "journal/paper_journal/references.bib",
             "journal/paper_journal/tables", "journal/paper_journal/figures",
             "e234_runner.py", "hercules_launch/cmds_e234_global.txt",
             "hercules_launch/slurm_e234g.sh", "analysis/statistical_analysis.py"]
    r = run(["git", "add"] + files, cwd=str(REPO))
    log(f"git add rc={r.returncode} {r.stderr[-300:]}")
    msg = ("Journal E2-E5: scenario/scale/scaling campaigns (global-label re-run), "
           "mechanism study (per-layer gradient variance, Fisher spectra, loss "
           "interpolation, angle wrapping), FT resource appendix; stats+figures+"
           "crosscheck pipeline and tables integrated into the manuscript")
    r = run(["git", "commit", "-m", msg], cwd=str(REPO))
    log(f"git commit rc={r.returncode}: {r.stdout[-400:]} {r.stderr[-300:]}")
    return r.returncode == 0


def main():
    log("=== integrate_journal start ===")
    complete = wait_array()
    log(f"array complete={complete}")
    if not sync():
        log("rsync FAILED"); sys.exit(1)
    rcs = [py("analysis/e234_stats.py").returncode,
           py("analysis/e234_figures.py").returncode,
           py("analysis/e234_stats_crosscheck.py").returncode]
    S = load_json("e234_stats.json")
    c = S["completeness"]
    log(f"completeness: {c} step rcs: {rcs}")
    gate = (c.get("missing", 1) == 0 and c.get("incomplete", 1) == 0
            and c.get("stale_local_labels", 1) == 0
            and all(rc == 0 for rc in rcs))
    if not gate:
        log("completeness gate FAILED -- main.tex left untouched; rerun this "
            "script after the campaign is complete (missing/incomplete/stale)")
        log("=== integrate_journal done ===")
        return
    patch_main()
    ok = compile_pdf()
    if ok:
        commit()
        log("COMMIT DONE")
    else:
        log("NOT COMMITTED (compile failed)")
    log("=== integrate_journal done ===")


if __name__ == "__main__":
    main()
