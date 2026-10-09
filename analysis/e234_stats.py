#!/usr/bin/env python3
"""Statistical analysis of the E2/E3/E4 campaigns (e234_runner.py).

Reads every `results/e234_<campaign>__<tag>__<arm>__<profile>__s<seed>/`
payload that corresponds to one of the 290 pre-declared cells in
`hercules_launch/cmds_e234.txt` and produces:

  * analysis/outputs/e234_stats.json         machine-readable summary
  * analysis/outputs/e234_stats_report.md    human-readable report
  * journal/paper_journal/tables/tab_e234_chain4.tex    (E2 scenarios)
  * journal/paper_journal/tables/tab_e234_bench5.tex    (E2 5-task benchmarks)
  * journal/paper_journal/tables/tab_e234_scale.tex     (E3 data scale)
  * journal/paper_journal/tables/tab_e234_scaling.tex   (E4 qubits/layers)

Protocol (matches the journal draft, Section "Statistical protocol"):
  * comparisons are PAIRED BY SEED; the paired test of record is the
    two-sided paired t-test, with the Wilcoxon signed-rank reported as the
    non-parametric companion; Cohen dz and a 95% bootstrap CI (20k
    resamples) accompany every contrast;
  * Holm correction within each PRE-DECLARED family of contrasts:
      H1 chain4   : synth-scratch on the five scenario metrics
                    [TIL-2, CIL-2, TIL-3, CIL-3, CIL-4] per scope;
      H2 bench5   : synth/er/ewc - scratch on AA (class-IL) per benchmark
                    and scope (smnist5 four-arm, sfmnist5 two-arm);
      H3 scale    : synth-scratch on AA (class-IL) at train sizes
                    {500, 2000, 12000} per scope;
      H4 grid     : synth-scratch on the forgetting drop delta_A (class-IL)
                    across the five scaling configs {4q/3L, 6q, 8q, 2L, 4L};
    scopes are 'ideal', 'heron_r2' and their 'pooled' combination (pooled
    is confirmatory; the per-profile tests are the primary reading);
  * >2 arms on one benchmark (smnist5 ideal, four arms) additionally get a
    Friedman test + Nemenyi post-hoc with the critical-difference value.

Primary metric definitions (fixed here, before reading any result):
  * chain4   : class-IL accuracy with argmax over classes seen so far
               ("cil"); TIL versions use the task oracle ("til").
  * bench5   : AA/AF/BWT/FWT over the class-IL accuracy matrix + final-row
               accuracies; AA is the contrast metric.
  * pair2    : delta_A = A[0][0] - A[1][0] on the class-IL matrix (drop of
               task-A accuracy after the task-B phase); Acc_B = A[1][1];
               AA = mean of the final row.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from statistical_analysis import holm, paired_stats, fmt_p  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent / "outputs"
TABLES = REPO / "journal" / "paper_journal" / "tables"
CMDS = REPO / "hercules_launch" / "cmds_e234.txt"

CAMPAIGNS = ["chain4", "pair2", "smnist5", "sfmnist5"]
TAGS = ["std", "sz500", "sz2k", "sz12k", "q6", "q8", "L2", "L4"]
ARMS = ["scratch", "synth", "er", "ewc"]
PROFILES = ["ideal", "heron_r2"]
PNAME = {"ideal": "Ideal (noiseless)", "heron_r2": "IBM Heron r2",
         "pooled": "Pooled"}
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}  # infinite-df table


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def expected_dirs() -> set[str]:
    """The 290 pre-declared cell directories, parsed from the cmds file."""
    dirs = set()
    for line in CMDS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()

        def val(flag, default):
            return parts[parts.index(flag) + 1] if flag in parts else default

        camp = val("--campaign", None)
        tag = val("--tag", "std")
        arm = val("--arm", None)
        seed = val("--seed", None)
        prof = val("--profile", "ideal")
        dirs.add(f"e234_{camp}__{tag}__{arm}__{prof}__s{seed}")
    return dirs


def load_runs():
    """-> {dir_name: payload} for every completed pre-declared cell.

    Label-mode guard: chain4/smnist5/sfmnist5 cells must carry
    `label_mode == "global"` (true task-IL/class-IL with the shared head);
    a stale local-label payload for those campaigns is EXCLUDED and reported,
    so a failed re-run can never leak its predecessor's numbers into the
    tables. pair2 cells keep the local-label shared-binary semantics
    (Experiment-1 protocol) and are exempt from the check.
    """
    want = expected_dirs()
    runs, missing, incomplete, stale = {}, [], [], []
    for d in sorted(want):
        path = RESULTS / d
        json_files = list(path.glob("results_seed_*.json"))
        if not path.is_dir():
            missing.append(d)
            continue
        if not json_files:
            incomplete.append(d)
            continue
        payload = json.loads(json_files[0].read_text())
        if payload.get("experiment") != "e234_v1":
            raise ValueError(f"unexpected payload at {json_files[0]}")
        camp = payload["campaign"]
        mode = payload.get("config", {}).get("label_mode", "local")
        if camp in ("chain4", "smnist5", "sfmnist5") and mode != "global":
            stale.append(d)
            continue
        runs[d] = payload
    return runs, sorted(missing), sorted(incomplete), sorted(stale)


def key(payload):
    return (payload["campaign"], payload["tag"], payload["noise_profile"],
            payload["arm"], payload["seed"])


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def scenario_metrics(payload):
    """Sub-sequence metrics for chain4: means of trimmed matrix rows."""
    out = {}
    m = payload["metrics"]
    for mat, name in (("til", "TIL"), ("cil", "CIL"), ("cil_full", "CILfull")):
        mm = m[mat]["matrix"]
        for T in range(2, len(mm) + 1):
            row = mm[T - 1][:T]
            out[f"{name}-{T}"] = mean(row)
    return out


def pair_metrics(payload):
    """Two-task metrics for pair2 (class-IL primary, task-IL companion)."""
    m = payload["metrics"]
    cil = m["cil"]
    til = m["til"]
    a0_cil = cil["matrix"][0][0]
    a1_cil = cil["matrix"][1][0]
    b_cil = cil["matrix"][1][1]
    a0_til = til["matrix"][0][0]
    a1_til = til["matrix"][1][0]
    b_til = til["matrix"][1][1]
    return {
        "aa_cil": cil["aa"], "af_cil": cil["af"], "bwt_cil": cil["bwt"],
        "fwt_cil": cil["fwt"],
        "acc_a_init_cil": a0_cil, "acc_a_final_cil": a1_cil, "acc_b_cil": b_cil,
        "delta_a_cil": a0_cil - a1_cil,
        "retention_cil": a1_cil / a0_cil if a0_cil else None,
        "aa_til": til["aa"], "delta_a_til": a0_til - a1_til, "acc_b_til": b_til,
    }


def bench_metrics(payload):
    """Five-task benchmark metrics (class-IL primary)."""
    m = payload["metrics"]
    cil = m["cil"]
    return {
        "aa": cil["aa"], "af": cil["af"], "bwt": cil["bwt"], "fwt": cil["fwt"],
        "final_accs": cil["final_accs"],
        "acc_t0_final": cil["matrix"][-1][0],
        "acc_t4_final": cil["matrix"][-1][-1],
        "aa_til": m["til"]["aa"],
    }


def epoch_time(payload):
    e = [p["epoch_time"] for p in payload["phases"].values()]
    flat = [t for lst in e for t in lst if t is not None]
    return float(np.mean(flat)) if flat else None


def abs_acc(payload):
    """Absolute task accuracies after each task (list over tasks)."""
    return [payload["metrics"]["cil"]["matrix"][t][t] for t in range(len(payload["metrics"]["cil"]["matrix"]))]


# ---------------------------------------------------------------------------
# Pivot helpers
# ---------------------------------------------------------------------------

def series(runs, scope, campaign, tag, arm, extract):
    """{seed: value} for one cell, pooled over profiles when scope='pooled'."""
    profiles = PROFILES if scope == "pooled" else [scope]
    out = {}
    for payload in runs:
        if (payload["campaign"] == campaign and payload["tag"] == tag
                and payload["noise_profile"] in profiles and payload["arm"] == arm):
            out[(payload["noise_profile"], payload["seed"])] = extract(payload)
    return out


class Runs:
    """Indexed access to every loaded payload."""

    def __init__(self, payloads):
        self.by_cell = {}
        for payload in payloads:
            self.by_cell.setdefault(
                (payload["campaign"], payload["tag"], payload["noise_profile"],
                 payload["arm"]), {})[payload["seed"]] = payload

    def seeds(self, campaign, tag, profile, arm):
        return self.by_cell.get((campaign, tag, profile, arm), {})

    def values(self, scope, campaign, tag, arm, extract):
        profiles = PROFILES if scope == "pooled" else [scope]
        out = {}
        for prof in profiles:
            for s, payload in self.seeds(campaign, tag, prof, arm).items():
                out[(prof, s)] = extract(payload)
        return out

    def contrast(self, scope, campaign, tag, arm_a, arm_b, extract):
        sa = self.values(scope, campaign, tag, arm_a, extract)
        sb = self.values(scope, campaign, tag, arm_b, extract)
        keys = sorted(set(sa) & set(sb))
        if len(keys) < 2:
            return {"n": len(keys), "mean_delta": None, "error": "insufficient paired seeds"}
        st = paired_stats(np.array([sa[k] for k in keys]),
                          np.array([sb[k] for k in keys]))
        st["seeds"] = [k[1] for k in keys]
        st["n_profiles"] = len(set(k[0] for k in keys))
        return st


def stats_block(runs: Runs, scope, campaign, tag, arms, extract):
    """Descriptives + paired contrasts vs the first arm, Holm within block."""
    fam = []
    for arm in arms:
        vals = runs.values(scope, campaign, tag, arm, extract)
        fam.append({"arm": arm, "n": len(vals),
                    "values": [v for v in vals.values() if v is not None]})
    contrasts = []
    base = arms[0]
    for arm in arms[1:]:
        c = runs.contrast(scope, campaign, tag, arm, base, extract)
        c["contrast"] = f"{arm}-{base}"
        contrasts.append(c)
    ph = holm([c.get("p_t") for c in contrasts])
    for c, p in zip(contrasts, ph):
        c["p_holm"] = p
    return {"descriptives": fam, "contrasts": contrasts}


def fmt(v, nd=2):
    return "--" if v is None else f"{v:.{nd}f}"


def ms(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return "--"
    return f"{v.mean():.2f} $\\pm$ {v.std(ddof=1) if len(v) > 1 else 0.0:.2f}"


def arm_means(runs: Runs, scope, campaign, tag, arm, extract_list):
    """Per-arm means of several metrics, combined over profiles per scope."""
    profiles = PROFILES if scope == "pooled" else [scope]
    out = {}
    for key, ex in extract_list:
        vals = [ex(p) for prof in profiles
                for p in runs.seeds(campaign, tag, prof, arm).values()]
        out[key] = mean(vals)
    return out


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def tex_header(path, caption, label, colspec, head):
    lines = [
        "% Generated by analysis/e234_stats.py -- recomputed from results/e234_* payloads.",
        "\\begin{table}[ht]", "\\centering", "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", head, "\\midrule",
    ]
    return lines


def tex_write(path, lines):
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path(path).write_text("\n".join(lines))
    print(f"table written: {path}")


def pcell(p):
    s = fmt_p(p)
    return s if s.startswith("$") else f"${s}$"


def dcell(c):
    if c.get("mean_delta") is None:
        return "--", "--", "--", ""
    return (f"${c['mean_delta']:+.2f}$ [${c['ci95_low']:+.2f}$, ${c['ci95_high']:+.2f}$]",
            pcell(c.get("p_holm")), f"${c['dz']:+.2f}$", f"${c['n']}$")


def write_chain4_table(runs: Runs, out):
    lines = tex_header(
        TABLES / "tab_e234_chain4.tex",
        "E2 scenarios on the four-task chain (Fashion-MNIST$\\{0,1\\}$ $\\to$ "
        "MNIST$\\{2,3\\}$ $\\to$ KMNIST$\\{4,5\\}$ $\\to$ Fashion-MNIST$\\{8,9\\}$), "
        "strongly entangling ansatz, four qubits. Task-IL (task oracle) and "
        "class-IL (argmax over classes seen so far) accuracies at the "
        "two-, three- and four-task prefixes of the same training run. "
        "$\\delta$ (p$_{\\text{Holm}}$): paired $t$-test of synth$-$scratch, "
        "Holm-corrected within the family of five scenario metrics per "
        "scope; $d_z$ paired effect size. Mean $\\pm$ SD over seeds.",
        "tab:e234_chain4", "lcccccccc",
        "Scenario & Scratch & Synth & $\\delta$ [95\\% CI] & $p_{\\text{Holm}}$ & $d_z$ & $n$ \\\\")
    summary = {}
    for scope in PROFILES + ["pooled"]:
        sub = {"scope": scope, "metrics": {}}
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{PNAME[scope]}}}}} \\\\")
        for metric in ["TIL-2", "CIL-2", "TIL-3", "CIL-3", "CIL-4"]:
            ex = (lambda m: (lambda p: scenario_metrics(p)[m]))(metric)
            block = stats_block(runs, scope, "chain4", "std", ["scratch", "synth"], ex)
            d_sc, d_sy = block["descriptives"]
            c = block["contrasts"][0]
            sc4, sy4, dc4, dc5, dc6, dc7 = (ms(d_sc["values"]), ms(d_sy["values"]), *dcell(c))
            sub["metrics"][metric] = {"scratch": d_sc, "synth": d_sy, "contrast": c}
            label = ("Task-IL " if metric.startswith("TIL") else "Class-IL ") + metric[-1]
            lines.append(f"{label} "
                         f"& {sc4} & {sy4} & {dc4} & {dc5} & {dc6} & {dc7} \\\\")
        lines.append("\\midrule")
        summary[scope] = sub
    del lines[-1]
    tex_write(TABLES / "tab_e234_chain4.tex", lines)
    out["chain4"] = summary


def write_bench5_table(runs: Runs, out):
    lines = tex_header(
        TABLES / "tab_e234_bench5.tex",
        "E2 standard class-incremental benchmarks: split-MNIST and "
        "split-Fashion-MNIST (5 tasks $\\times$ 2 classes, shared head, "
        "class-IL evaluation). AA/AF/BWT/FWT follow van de Ven et al. "
        "(2022); $\\text{Acc}_{t_0}$ is the accuracy on the first task after "
        "the whole sequence. $\\delta$ (p$_{\\text{Holm}}$): paired "
        "$t$-test vs.\\ scratch on AA, Holm-corrected within each block; "
        "$d_z$ paired effect size. Mean $\\pm$ SD over seeds.",
        "tab:e234_bench5", "llccccc@{\\hskip 4pt}ccc",
        "Bench. & Arm & AA & AF & BWT & FWT & $\\text{Acc}_{t_0}$ & "
        "$\\delta$AA [95\\% CI] & $p_{\\text{Holm}}$ & $d_z$ \\\\")
    summary = {}
    bench_defs = [
        ("smnist5", "std", "ideal", ["scratch", "synth", "er", "ewc"], "split-MNIST (ideal)"),
        ("smnist5", "std", "heron_r2", ["scratch", "synth"], "split-MNIST (Heron r2)"),
        ("sfmnist5", "std", "ideal", ["scratch", "synth"], "split-Fashion-MNIST (ideal)"),
    ]
    for camp, tag, scope, arms, title in bench_defs:
        sub = {}
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{title}}}}} \\\\")
        block = stats_block(runs, scope, camp, tag, arms, lambda p: bench_metrics(p)["aa"])
        contrasts = {c["contrast"]: c for c in block["contrasts"]}
        sub["contrasts"] = contrasts
        for d in block["descriptives"]:
            arm = d["arm"]
            means = arm_means(runs, scope, camp, tag, arm, [
                ("aa", lambda p: bench_metrics(p)["aa"]),
                ("af", lambda p: bench_metrics(p)["af"]),
                ("bwt", lambda p: bench_metrics(p)["bwt"]),
                ("fwt", lambda p: bench_metrics(p)["fwt"]),
                ("acc_t0_final", lambda p: bench_metrics(p)["acc_t0_final"]),
            ])
            vals = d["values"]
            aa, af, bwt, fwt = means["aa"], means["af"], means["bwt"], means["fwt"]
            t0 = means["acc_t0_final"]
            if arm == "scratch":
                dc = ("--", "--", "--", "")
            else:
                c = contrasts.get(f"{arm}-scratch")
                dc = dcell(c) if c else ("--", "--", "--", "")
            lines.append(f"& {arm} & {fmt(aa)} & {fmt(af)} & {fmt(bwt)} & {fmt(fwt)} "
                         f"& {fmt(t0)} & {dc[0]} & {dc[1]} & {dc[2]} \\\\")
            sub[arm] = {"n": d["n"], "aa": aa, "af": af, "bwt": bwt, "fwt": fwt,
                        "acc_t0_final": t0, "values": vals}
        lines.append("\\midrule")
        summary[f"{camp}/{scope}"] = sub
    del lines[-1]
    tex_write(TABLES / "tab_e234_bench5.tex", lines)
    out["bench5"] = summary


def write_scale_table(runs: Runs, out):
    lines = tex_header(
        TABLES / "tab_e234_scale.tex",
        "E3 data scale: binary Fashion-MNIST$\\{0,1\\}$ $\\to$ "
        "MNIST$\\{2,3\\}$ protocol re-run on {500, 2\\,000, 12\\,000} "
        "training samples per task (full binary splits at 12k). The two-task "
        "protocol uses the shared binary readout of Experiment~1: AA is the "
        "mean accuracy over the two tasks of that readout and $\\Delta_A$ is "
        "the Task-A drop. $\\delta$AA (p$_{\\text{Holm}}$): paired "
        "synth$-$scratch $t$-test, Holm within the three-size family; "
        "$d_z$ paired effect size. Mean $\\pm$ SD over seeds.",
        "tab:e234_scale", "lccccccc",
        "Samples & Scratch AA & Synth AA & $\\delta$AA [95\\% CI] & "
        "$p_{\\text{Holm}}$ & $d_z$ & Scratch $\\Delta_A$ & Synth $\\Delta_A$ \\\\")
    summary = {}
    dA = lambda p: pair_metrics(p)["delta_a_cil"]
    aa = lambda p: pair_metrics(p)["aa_cil"]
    for size, tag, scope in [("500", "sz500", "ideal"), ("2k", "sz2k", "ideal"),
                             ("12k", "sz12k", "ideal"), ("12k (Heron)", "sz12k", "heron_r2")]:
        block = stats_block(runs, scope, "pair2", tag, ["scratch", "synth"], aa)
        d_sc, d_sy = block["descriptives"]
        c = block["contrasts"][0]
        dc4, dc5, dc6, _ = dcell(c)
        dA_sc = mean([dA(p) for p in runs.seeds("pair2", tag, scope, "scratch").values()])
        dA_sy = mean([dA(p) for p in runs.seeds("pair2", tag, scope, "synth").values()])
        lines.append(f"{size} & {ms(d_sc['values'])} & {ms(d_sy['values'])} & {dc4} "
                     f"& {dc5} & {dc6} & {fmt(dA_sc)} & {fmt(dA_sy)} \\\\")
        summary[(tag, scope)] = {"scratch": d_sc, "synth": d_sy, "contrast": c,
                                 "delta_a": {"scratch": dA_sc, "synth": dA_sy}}
    tex_write(TABLES / "tab_e234_scale.tex", lines)
    out["scale"] = {f"{k[0]}/{k[1]}": v for k, v in summary.items()}


def write_scaling_table(runs: Runs, out):
    lines = tex_header(
        TABLES / "tab_e234_scaling.tex",
        "E4 scaling of the two-task protocol: qubits {4, 6, 8} and depth "
        "{2, 3, 4} (PCA dimension matched to the qubit count). "
        "$\\Delta_A$ (Task-A drop of the shared binary readout), AA and "
        "$\\text{Acc}_B$; $t_{\\text{ep}}$ "
        "is the mean elapsed time per epoch (one CPU core, PennyLane "
        "state-vector). $\\delta\\Delta_A$ (p$_{\\text{Holm}}$): paired "
        "synth$-$scratch $t$-test, Holm within the five-config family; "
        "$d_z$ paired effect size. Mean over seeds.",
        "tab:e234_scaling", "lccccccccc",
        "Config & $\\Delta_A$ scr. & $\\Delta_A$ syn. & $\\delta\\Delta_A$ "
        "[95\\% CI] & $p_{\\text{Holm}}$ & $d_z$ & AA scr. & AA syn. & "
        "$\\text{Acc}_B$ scr. & $\\text{Acc}_B$ syn. \\\\")
    configs = [("4 qubits, 3 layers", "std"), ("6 qubits, 3 layers", "q6"),
               ("8 qubits, 3 layers", "q8"), ("4 qubits, 2 layers", "L2"),
               ("4 qubits, 4 layers", "L4")]
    summary = {}
    for label, tag in configs:
        block = stats_block(runs, "ideal", "pair2", tag, ["scratch", "synth"],
                            lambda p: pair_metrics(p)["delta_a_cil"])
        d_sc, d_sy = block["descriptives"]

        def avg(arm):
            return arm_means(runs, "ideal", "pair2", tag, arm, [
                ("delta_a_cil", lambda p: pair_metrics(p)["delta_a_cil"]),
                ("aa_cil", lambda p: pair_metrics(p)["aa_cil"]),
                ("acc_b_cil", lambda p: pair_metrics(p)["acc_b_cil"]),
            ])

        m_sc, m_sy = avg("scratch"), avg("synth")
        c = block["contrasts"][0]
        dc4, dc5, dc6, _ = dcell(c)
        lines.append(f"{label} & {fmt(m_sc['delta_a_cil'])} & {fmt(m_sy['delta_a_cil'])} "
                     f"& {dc4} & {dc5} & {dc6} "
                     f"& {fmt(m_sc['aa_cil'])} & {fmt(m_sy['aa_cil'])} "
                     f"& {fmt(m_sc['acc_b_cil'])} & {fmt(m_sy['acc_b_cil'])} \\\\")
        summary[tag] = {"scratch": {"n": d_sc["n"], **m_sc},
                        "synth": {"n": d_sy["n"], **m_sy}}
        summary[tag]["contrast"] = c
        summary[tag]["epoch_time"] = {
            arm: mean([epoch_time(p) for p in runs.seeds("pair2", tag, "ideal", arm).values()])
            for arm in ("scratch", "synth")}
    tex_write(TABLES / "tab_e234_scaling.tex", lines)
    out["scaling"] = summary


# ---------------------------------------------------------------------------
# Friedman + Nemenyi (smnist5, four arms, ideal)
# ---------------------------------------------------------------------------

def friedman_nemenyi(runs: Runs, campaign, tag, scope, arms, extract):
    from scipy import stats
    per_seed = {a: runs.values(scope, campaign, tag, a, extract) for a in arms}
    seeds = sorted(set.intersection(*[set(v) for v in per_seed.values()]))
    if len(seeds) < 2 or len(arms) < 3:
        return {"note": "insufficient arms/seeds"}
    mat = np.array([[per_seed[a][s] for a in arms] for s in seeds])
    chi2, p = stats.friedmanchisquare(*[mat[:, i] for i in range(len(arms))])
    ranks = np.array([stats.rankdata(row) for row in mat]).mean(axis=0)
    k, N = len(arms), len(seeds)
    q = NEMENYI_Q05.get(k)
    cd = None if q is None else q * np.sqrt(k * (k + 1) / (6 * N))
    return {"arms": arms, "seeds": seeds, "chi2": float(chi2), "p": float(p),
            "mean_ranks": {a: float(r) for a, r in zip(arms, ranks)},
            "critical_difference": cd, "k": k, "N": N}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    payloads, missing, incomplete, stale = load_runs()
    runs = Runs(payloads.values())
    total_expected = len(expected_dirs())
    summary = {"completeness": {"expected": total_expected, "loaded": len(payloads),
                                "missing": len(missing), "incomplete": len(incomplete),
                                "stale_local_labels": len(stale)},
               "missing_dirs": missing, "incomplete_dirs": incomplete,
               "stale_dirs": stale}
    report = ["# E2/E3/E4 campaigns -- statistical report\n",
              f"Cells loaded: {len(payloads)}/{total_expected} "
              f"(missing {len(missing)}, incomplete {len(incomplete)}, "
              f"stale local-label {len(stale)})."]
    if stale:
        report.append(f"\n_Stale (local-label) cells excluded: {len(stale)}_")
    report.append("")

    # machines / provenance
    machines = {}
    for p in payloads.values():
        machines.setdefault(p.get("machine_id", "?"), set()).add(
            f"{p['campaign']}/{p['tag']}/{p['noise_profile']}")
    summary["machines"] = {m: sorted(c) for m, c in machines.items()}
    report.append("Executing machines (real machine_id per run): "
                  + "; ".join(f"`{m}` -> {len(c)} cells" for m, c in sorted(machines.items())) + "\n")

    write_chain4_table(runs, summary)
    write_bench5_table(runs, summary)
    write_scale_table(runs, summary)
    write_scaling_table(runs, summary)

    # ---- H1 report detail
    report.append("\n## H1 chain4: synth vs scratch on scenario metrics (Holm, 5-metric family)\n")
    for scope in PROFILES + ["pooled"]:
        for metric in ["TIL-2", "CIL-2", "TIL-3", "CIL-3", "CIL-4"]:
            ex = (lambda m: (lambda p: scenario_metrics(p)[m]))(metric)
            c = runs.contrast(scope, "chain4", "std", "synth", "scratch", ex)
            if c.get("error"):
                report.append(f"- {scope}/{metric}: {c['error']}")
                continue
            report.append(f"- {scope}/{metric}: d={c['mean_delta']:+.2f} pp "
                          f"[{c['ci95_low']:+.2f},{c['ci95_high']:+.2f}], "
                          f"p={fmt_p(c['p_t'])}, wilcoxon={fmt_p(c.get('p_wilcoxon'))}, "
                          f"dz={c['dz']:+.2f}, n={c['n']}")

    # ---- H2 detail + Friedman/Nemenyi
    report.append("\n## H2 five-task benchmarks (Holm within benchmark scope)\n")
    for camp, scope, arms in [("smnist5", "ideal", ["scratch", "synth", "er", "ewc"]),
                              ("smnist5", "heron_r2", ["scratch", "synth"]),
                              ("sfmnist5", "ideal", ["scratch", "synth"])]:
        block = stats_block(runs, scope, camp, "std", arms, lambda p: bench_metrics(p)["aa"])
        for d in block["descriptives"]:
            m = arm_means(runs, scope, camp, "std", d["arm"], [
                ("aa", lambda p: bench_metrics(p)["aa"]),
                ("af", lambda p: bench_metrics(p)["af"]),
                ("bwt", lambda p: bench_metrics(p)["bwt"]),
                ("fwt", lambda p: bench_metrics(p)["fwt"]),
            ])
            report.append(f"- {camp}/{scope}/{d['arm']} (n={d['n']}): "
                          f"AA {fmt(m['aa'])}, AF {fmt(m['af'])}, "
                          f"BWT {fmt(m['bwt'])}, FWT {fmt(m['fwt'])}")
        for c in block["contrasts"]:
            if c.get("error"):
                report.append(f"  - {c['contrast']}: {c['error']}")
                continue
            report.append(f"  - {c['contrast']}: d={c['mean_delta']:+.2f} pp "
                          f"[{c['ci95_low']:+.2f},{c['ci95_high']:+.2f}], "
                          f"p={fmt_p(c['p_t'])}, p_holm={fmt_p(c.get('p_holm'))}, "
                          f"wilcoxon={fmt_p(c.get('p_wilcoxon'))}, dz={c['dz']:+.2f}, n={c['n']}")
    fn = friedman_nemenyi(runs, "smnist5", "std", "ideal",
                          ["scratch", "synth", "er", "ewc"],
                          lambda p: bench_metrics(p)["aa"])
    summary["friedman_smnist5"] = fn
    report.append(f"\n### Friedman + Nemenyi (smnist5 ideal, 4 arms)\n")
    if "note" in fn:
        report.append(f"- {fn['note']}")
    else:
        report.append(f"- Friedman chi2={fn['chi2']:.2f}, p={fmt_p(fn['p'])} "
                      f"(N={fn['N']} seeds, k={fn['k']})")
        report.append(f"- mean ranks: " + ", ".join(f"{a}={r:.2f}" for a, r in fn["mean_ranks"].items()))
        report.append(f"- Nemenyi critical difference (alpha=0.05): {fn['critical_difference']:.2f}")

    # ---- H3 detail
    report.append("\n## H3 data scale (Holm within the three-size family)\n")
    aa = lambda p: pair_metrics(p)["aa_cil"]
    for tag, scope in [("sz500", "ideal"), ("sz2k", "ideal"), ("sz12k", "ideal"),
                       ("sz12k", "heron_r2")]:
        c = runs.contrast(scope, "pair2", tag, "synth", "scratch", aa)
        if c.get("error"):
            report.append(f"- {tag}/{scope}: {c['error']}")
            continue
        report.append(f"- {tag}/{scope}: d={c['mean_delta']:+.2f} pp "
                      f"[{c['ci95_low']:+.2f},{c['ci95_high']:+.2f}], "
                      f"p={fmt_p(c['p_t'])}, dz={c['dz']:+.2f}, n={c['n']}")

    # ---- H4 detail
    report.append("\n## H4 qubit/depth grid (Holm within the five-config family)\n")
    dA = lambda p: pair_metrics(p)["delta_a_cil"]
    for tag in ["std", "q6", "q8", "L2", "L4"]:
        c = runs.contrast("ideal", "pair2", tag, "synth", "scratch", dA)
        if c.get("error"):
            report.append(f"- {tag}: {c['error']}")
            continue
        report.append(f"- {tag}: d={c['mean_delta']:+.2f} pp "
                      f"[{c['ci95_low']:+.2f},{c['ci95_high']:+.2f}], "
                      f"p={fmt_p(c['p_t'])}, dz={c['dz']:+.2f}, n={c['n']}")

    (OUT / "e234_stats.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "e234_stats_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nOutputs written to {OUT} and {TABLES}")


if __name__ == "__main__":
    main()
