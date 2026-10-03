#!/usr/bin/env python3
"""Statistical analysis of the E1 controlled campaign (2x2 + CL baselines).

Reads every `results/e1_<arm>__<profile>__s<seed>/results_seed_<seed>.json`
produced by e1_runner.py and produces:

  * analysis/outputs/e1_stats.json          machine-readable summary
  * analysis/outputs/e1_stats_report.md     human-readable report
  * journal/paper_journal/tables/tab_exp1_controlled.tex   (2x2 absolute accs)
  * journal/paper_journal/tables/tab_exp1_baselines.tex    (EWC/Replay/DER++)

Protocol (matches the draft, Section "Statistical protocol"):
  * comparisons are PAIRED BY SEED on the primary metric delta_A
    (forgetting drop on Task A, percentage points; lower is better);
  * every contrast reports mean difference, two-sided paired t-test,
    Wilcoxon signed-rank, Cohen dz and a 95% bootstrap CI (20k resamples);
  * Holm correction within each pre-declared family:
      F1 "design"     = the six contrasts that decompose the 2x2
                        (B2-B1, B3-B1, B4-B3, B4-B2, B4-B1, B3-B2);
      F2 "baselines"  = the five EWC/Replay/DER++ vs plain-scratch contrasts
                        plus the five QTL(B4) vs baseline contrasts.
    Each family is corrected per noise profile AND pooled across profiles.
  * Acc_A_init / Acc_A_final / Acc_B and retention r_A are reported as
    descriptives per arm (item 2.3); Acc_A_final and Acc_B carry the
    robustness contrasts labelled as such in the report.
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

PROFILES = ["ideal", "heron_r2"]
ARM_ORDER = ["B1", "B2", "B3", "B4", "B5_l1e2", "B5_l1e3", "B5_l1e4", "B6", "B7"]
ARM_LABEL = {
    "B1": "B1 scratch, $0.05/0.05$",
    "B2": "B2 scratch, $0.01/0.005$",
    "B3": "B3 synth, $0.05/0.05$",
    "B4": "B4 synth, $0.01/0.005$",
    "B5_l1e2": "B5 EWC $\\lambda{=}10^2$",
    "B5_l1e3": "B5 EWC $\\lambda{=}10^3$",
    "B5_l1e4": "B5 EWC $\\lambda{=}10^4$",
    "B6": "B6 Replay 25\\%",
    "B7": "B7 DER++ 25\\%",
}

F1_DESIGNS = [
    ("B2", "B1", "LR effect (scratch): B2-B1"),
    ("B3", "B1", "Init effect (high LR): B3-B1"),
    ("B4", "B3", "LR effect (synth): B4-B3"),
    ("B4", "B2", "Init effect (low LR): B4-B2"),
    ("B4", "B1", "Headline (confounded) contrast: B4-B1"),
    ("B3", "B2", "Cross contrast: B3-B2"),
]
F2_BASELINES = [
    ("B5_l1e2", "B1", "EWC 1e2 vs scratch: B5-B1"),
    ("B5_l1e3", "B1", "EWC 1e3 vs scratch: B5-B1"),
    ("B5_l1e4", "B1", "EWC 1e4 vs scratch: B5-B1"),
    ("B6", "B1", "Replay vs scratch: B6-B1"),
    ("B7", "B1", "DER++ vs scratch: B7-B1"),
    ("B4", "B5_l1e2", "QTL vs EWC 1e2: B4-B5"),
    ("B4", "B5_l1e3", "QTL vs EWC 1e3: B4-B5"),
    ("B4", "B5_l1e4", "QTL vs EWC 1e4: B4-B5"),
    ("B4", "B6", "QTL vs Replay: B4-B6"),
    ("B4", "B7", "QTL vs DER++: B4-B7"),
]


def load_runs():
    """-> {(profile, arm): {seed: payload}}"""
    runs: dict[tuple[str, str], dict[int, dict]] = {}
    pat = re.compile(r"results/e1_(?!smoke)([A-Za-z0-9_]+?)__([a-z0-9_]+)__s(\d+)/results_seed_\d+\.json$")
    for path in sorted(glob.glob(str(RESULTS / "e1_*" / "results_seed_*.json"))):
        m = pat.search(path)
        if not m:
            continue
        arm, profile, seed = m.group(1), m.group(2), int(m.group(3))
        payload = json.loads(Path(path).read_text())
        if payload.get("experiment") != "e1_controlled_v1":
            continue
        runs.setdefault((profile, arm), {})[seed] = payload
    return runs


def mean_std(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def contrast_family(runs, profile, contrasts, metric="delta_a"):
    """Paired contrasts within one family for one profile ('pooled' allowed)."""
    if profile == "pooled":
        profiles = PROFILES
    else:
        profiles = [profile]

    def series(arm):
        vals = {}
        for prof in profiles:
            for seed, payload in runs.get((prof, arm), {}).items():
                vals[(prof, seed)] = payload[metric]
        return vals

    out = []
    for a, b, label in contrasts:
        sa, sb = series(a), series(b)
        keys = sorted(set(sa) & set(sb))
        if len(keys) < 2:
            out.append({"contrast": f"{a}-{b}", "label": label, "n": len(keys),
                        "error": "insufficient paired seeds"})
            continue
        st = paired_stats(np.array([sa[k] for k in keys]),
                          np.array([sb[k] for k in keys]))
        st.update({"contrast": f"{a}-{b}", "label": label,
                   "seeds": [k[1] for k in keys]})
        out.append(st)
    ph = holm([c.get("p_t") for c in out])
    for c, p in zip(out, ph):
        c["p_holm"] = p
    return out


def fmt(v, nd=2):
    return "--" if v is None else f"{v:.{nd}f}"


def tex_escape_row(cells):
    return " & ".join(cells) + " \\\\"


def write_tables(runs, desc):
    TABLES.mkdir(parents=True, exist_ok=True)

    # ---- Table 1: the 2x2 design with absolute accuracies ------------------
    lines = [
        "% Generated by analysis/e1_stats.py -- E1 controlled campaign (2x2).",
        "% Values recomputed from results/e1_*/results_seed_*.json (paired by seed).",
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Experiment 1 (controlled re-run): absolute accuracies and "
        "forgetting for the $2{\\times}2$ design "
        "(initialisation $\\times$ learning-rate regime), strongly entangling "
        "ansatz, four qubits. "
        "$\\text{Acc}_A^{\\text{init}}$: Task~A accuracy after the Task~A phase; "
        "$\\text{Acc}_A^{\\text{final}}$: Task~A accuracy after the Task~B phase; "
        "$\\Delta_A = \\text{Acc}_A^{\\text{init}} - \\text{Acc}_A^{\\text{final}}$; "
        "$r_A = \\text{Acc}_A^{\\text{final}} / \\text{Acc}_A^{\\text{init}}$. "
        "Mean $\\pm$ standard deviation over seeds; every run stores the "
        "executing node identity and the full per-seed payload.}",
        "\\label{tab:exp1_controlled}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Arm & Pre-training / $\\eta_A/\\eta_B$ & $\\text{Acc}_A^{\\text{init}}$ & "
        "$\\text{Acc}_A^{\\text{final}}$ & $\\text{Acc}_B$ & $\\Delta_A$ & $r_A$ \\\\",
        "\\midrule",
    ]
    for prof in PROFILES:
        pname = {"ideal": "Ideal (noiseless)", "heron_r2": "IBM Heron r2"}[prof]
        n = len(desc[(prof, ARM_ORDER[0])]["seeds"]) if (prof, ARM_ORDER[0]) in desc else 0
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{pname} ($n={n}$ seeds)}}}} \\\\")
        for arm in ["B1", "B2", "B3", "B4"]:
            d = desc.get((prof, arm))
            if d is None:
                continue
            lr = {"B1": "none / $0.05/0.05$", "B2": "none / $0.01/0.005$",
                  "B3": "synthetic / $0.05/0.05$", "B4": "synthetic / $0.01/0.005$"}[arm]
            lines.append(tex_escape_row([
                arm.split()[0], lr,
                f"${fmt(d['acc_a_init'][0])} \\pm {fmt(d['acc_a_init'][1])}$",
                f"${fmt(d['acc_a_final'][0])} \\pm {fmt(d['acc_a_final'][1])}$",
                f"${fmt(d['acc_b_final'][0])} \\pm {fmt(d['acc_b_final'][1])}$",
                f"${fmt(d['delta_a'][0])} \\pm {fmt(d['delta_a'][1])}$",
                f"${fmt(d['retention_r_a'][0], 3)} \\pm {fmt(d['retention_r_a'][1], 3)}$",
            ]))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}", ""]
    (TABLES / "tab_exp1_controlled.tex").write_text("\n".join(lines))

    # ---- Table 2: CL baselines --------------------------------------------
    lines = [
        "% Generated by analysis/e1_stats.py -- E1 controlled campaign (baselines).",
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Experiment 1 (controlled re-run): continual-learning "
        "baselines against the plain scratch arm (B1) and the synthetic-prior "
        "arm (B4). $\\Delta_A$ is the Task~A forgetting drop in percentage "
        "points (lower is better). $p_{\\text{Holm}}$ is the two-sided paired "
        "$t$-test over seeds after Holm correction within the family of ten "
        "baseline contrasts; $d_z$ is the paired Cohen effect size of the "
        "contrast against B1 (negative values mean less forgetting than B1). "
        "EWC uses the empirical Fisher diagonal computed at the Task~A "
        "optimum; Replay and DER++ replay a buffer with 25\\% of the Task~A "
        "training set.}",
        "\\label{tab:exp1_baselines}",
        "\\begin{tabular}{llccc@{\\hskip 6pt}ccc}",
        "\\toprule",
        "& & & & & \\multicolumn{2}{c}{vs. B1} \\\\",
        "\\cmidrule(lr){6-7}",
        "Arm & Method & $\\Delta_A$ & $\\text{Acc}_A^{\\text{final}}$ & $\\text{Acc}_B$ & "
        "$\\delta$ ($p_{\\text{Holm}}$) & $d_z$ \\\\",
        "\\midrule",
    ]
    fam2 = {}
    for prof in PROFILES:
        fam2[prof] = {c["contrast"]: c for c in f2_by_profile[prof]}
    for prof in PROFILES:
        pname = {"ideal": "Ideal (noiseless)", "heron_r2": "IBM Heron r2"}[prof]
        n = len(desc[(prof, ARM_ORDER[0])]["seeds"]) if (prof, ARM_ORDER[0]) in desc else 0
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{pname} ($n={n}$ seeds)}}}} \\\\")
        for arm in ["B1", "B5_l1e2", "B5_l1e3", "B5_l1e4", "B6", "B7", "B4"]:
            d = desc.get((prof, arm))
            if d is None:
                continue
            method = {"B1": "scratch", "B5_l1e2": "EWC $10^2$", "B5_l1e3": "EWC $10^3$",
                      "B5_l1e4": "EWC $10^4$", "B6": "Replay", "B7": "DER++",
                      "B4": "synth prior"}[arm]
            if arm == "B1":
                vs = ("--", "--")
            elif arm == "B4":
                # B4 has no contrast against B1 inside family F2; show '--'
                # and keep its comparison in the design family (text).
                vs = ("--", "--")
            else:
                c = fam2[prof].get(f"{arm}-B1")
                if c and "p_t" in c:
                    vs = (f"${c['mean_delta']:+.2f}$ ({fmt_p(c['p_holm'])})",
                          f"${c['dz']:+.2f}$")
                else:
                    vs = ("--", "--")
            arm_label = {"B1": "B1", "B5_l1e2": "B5-$10^2$",
                         "B5_l1e3": "B5-$10^3$", "B5_l1e4": "B5-$10^4$",
                         "B6": "B6", "B7": "B7", "B4": "B4"}[arm]
            lines.append(tex_escape_row([
                arm_label,
                method,
                f"${fmt(d['delta_a'][0])} \\pm {fmt(d['delta_a'][1])}$",
                f"${fmt(d['acc_a_final'][0])} \\pm {fmt(d['acc_a_final'][1])}$",
                f"${fmt(d['acc_b_final'][0])} \\pm {fmt(d['acc_b_final'][1])}$",
                vs[0], vs[1],
            ]))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}", ""]
    (TABLES / "tab_exp1_baselines.tex").write_text("\n".join(lines))

    # ---- Table 3: paired contrasts of the 2x2 decomposition ---------------
    # Same reporting format as tab_exp1.tex v0.2 (delta [95% CI], p, p_Holm, dz).
    lines = [
        "% Generated by analysis/e1_stats.py -- E1 controlled campaign (2x2 contrasts).",
        "% Paired by seed on delta_A; Holm within the F1 design family of six contrasts.",
        "\\begin{table}[ht]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Experiment 1 (controlled re-run): paired contrasts of the "
        "$2{\\times}2$ decomposition (initialisation $\\times$ learning-rate "
        "regime) on the Task~A forgetting drop $\\Delta_A$, in percentage "
        "points (positive values mean more forgetting in the first arm of the "
        "contrast). Significance from a two-sided paired $t$-test over seeds; "
        "$p_{\\text{Holm}}$ applies the Holm procedure over the six "
        "pre-registered design contrasts treated as one family; $d_z$ is the "
        "paired Cohen effect size. The pooled rows combine both noise "
        "profiles ($n=40$ seed pairs).}",
        "\\label{tab:exp1_contrasts}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Scope & Contrast & $\\delta$ [95\\% CI] & $p$ & $p_{\\text{Holm}}$ & $d_z$ & $n$ \\\\",
        "\\midrule",
    ]
    short = {
        "B2-B1": "LR effect (scratch): B2$-$B1",
        "B3-B1": "Init effect (high LR): B3$-$B1",
        "B4-B3": "LR effect (synth): B4$-$B3",
        "B4-B2": "Init effect (low LR): B4$-$B2",
        "B4-B1": "Headline (confounded): B4$-$B1",
        "B3-B2": "Cross contrast: B3$-$B2",
    }
    for scope in PROFILES + ["pooled"]:
        pname = {"ideal": "Ideal", "heron_r2": "Heron r2", "pooled": "Pooled"}[scope]
        fam = contrast_family(runs, scope, F1_DESIGNS, metric="delta_a")
        first = True

        def p_cell(p):
            s = fmt_p(p)
            return s if s.startswith("$") else f"${s}$"

        for c in fam:
            if "error" in c:
                continue
            lines.append(tex_escape_row([
                pname if first else "",
                short.get(c["contrast"], c["label"]),
                f"${c['mean_delta']:+.2f}$ "
                f"[${c['ci95_low']:+.2f}$, ${c['ci95_high']:+.2f}$]",
                p_cell(c["p_t"]),
                p_cell(c["p_holm"]),
                f"${c['dz']:+.2f}$",
                f"${c['n']}$",
            ]))
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}", ""]
    (TABLES / "tab_exp1_contrasts.tex").write_text("\n".join(lines))


def main():
    global f2_by_profile
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No E1 results found under results/e1_*")
        sys.exit(1)

    report = ["# E1 controlled campaign -- statistical report\n"]
    summary = {"arms": {}, "families": {}, "machines": {}}

    seeds_by_profile = {}
    for (prof, arm), per_seed in sorted(runs.items()):
        seeds_by_profile.setdefault(prof, set()).update(per_seed.keys())
    report.append("Seeds available per profile: "
                  + "; ".join(f"{p}: {sorted(s)}" for p, s in sorted(seeds_by_profile.items())) + "\n")

    # ---- descriptives ------------------------------------------------------
    desc = {}
    machines = {}
    report.append("## Descriptives per arm (mean +/- sd over seeds)\n")
    for (prof, arm), per_seed in sorted(runs.items()):
        vals = {k: [p[k] for p in per_seed.values()]
                for k in ["acc_a_init", "acc_a_final", "acc_b_final", "delta_a", "retention_r_a"]}
        desc[(prof, arm)] = {k: mean_std(v) for k, v in vals.items()}
        desc[(prof, arm)]["seeds"] = sorted(per_seed)
        for p in per_seed.values():
            machines.setdefault(p.get("machine_id", "?"), set()).add(f"{prof}:{arm}")
        d = desc[(prof, arm)]
        report.append(
            f"- **{prof} / {arm}** (n={len(per_seed)}): "
            f"AccA_init {d['acc_a_init'][0]:.2f}+/-{d['acc_a_init'][1]:.2f}, "
            f"AccA_final {d['acc_a_final'][0]:.2f}+/-{d['acc_a_final'][1]:.2f}, "
            f"AccB {d['acc_b_final'][0]:.2f}+/-{d['acc_b_final'][1]:.2f}, "
            f"dA {d['delta_a'][0]:.2f}+/-{d['delta_a'][1]:.2f}, "
            f"r_A {d['retention_r_a'][0]:.3f}+/-{d['retention_r_a'][1]:.3f}")
    summary["arms"] = {f"{p}/{a}": d for (p, a), d in desc.items()}
    summary["machines"] = {m: sorted(cells) for m, cells in machines.items()}
    report.append("\nMachine ids (requirement 1.11): "
                  + "; ".join(f"`{m}` -> {len(c)} cells" for m, c in machines.items()) + "\n")

    # ---- families ---------------------------------------------------------
    f2_by_profile = {}
    report.append("## Paired contrasts on delta_A (Holm within family)\n")
    for fam_name, contrasts in (("F1_design", F1_DESIGNS), ("F2_baselines", F2_BASELINES)):
        report.append(f"\n### Family {fam_name}\n")
        summary["families"][fam_name] = {}
        for scope in PROFILES + ["pooled"]:
            fam = contrast_family(runs, scope, contrasts, metric="delta_a")
            if fam_name == "F2_baselines" and scope in PROFILES:
                f2_by_profile[scope] = fam
            summary["families"][fam_name][scope] = fam
            report.append(f"\n**{scope}**:\n")
            for c in fam:
                if "error" in c:
                    report.append(f"  - {c['label']}: {c['error']}")
                    continue
                report.append(
                    f"  - {c['label']}: d={c['mean_delta']:+.2f} pp "
                    f"[{c['ci95_low']:+.2f}, {c['ci95_high']:+.2f}], "
                    f"t={c['t']:+.2f}, p={fmt_p(c['p_t'])}, p_holm={fmt_p(c['p_holm'])}, "
                    f"wilcoxon={fmt_p(c['p_wilcoxon'])}, dz={c['dz']:+.2f}, n={c['n']}")

    # ---- robustness on Acc_A_final (labelled exploratory) ------------------
    report.append("\n## Robustness (exploratory): paired contrasts on Acc_A_final\n")
    summary["robustness"] = {}
    robust = [("B3", "B1", "Init effect (high LR)"), ("B4", "B2", "Init effect (low LR)")]
    for scope in PROFILES + ["pooled"]:
        fam = contrast_family(runs, scope, robust, metric="acc_a_final")
        summary["robustness"][scope] = fam
        for c in fam:
            if "error" in c:
                continue
            report.append(
                f"- {scope}: {c['label']}: d={c['mean_delta']:+.2f} pp, "
                f"p={fmt_p(c['p_t'])}, p_holm={fmt_p(c['p_holm'])}, dz={c['dz']:+.2f}")

    write_tables(runs, desc)

    (OUT / "e1_stats.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "e1_stats_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nOutputs written to {OUT} and {TABLES}")


if __name__ == "__main__":
    main()
