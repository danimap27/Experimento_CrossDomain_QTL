#!/usr/bin/env python3
"""Statistical re-analysis of the IEEE QAI submission results (journal version).

Addresses reviewer requests for the journal conversion:
  - paired significance tests over the per-seed runs instead of means alone
  - effect sizes and bootstrap confidence intervals
  - absolute accuracies are reported wherever the JSON stores them, with an
    explicit audit of the fields missing from Experiment 1

Input : results/<profile>__s<seed>/results_seed_<seed>.json
        profiles used in the paper: ideal, heron_r2, legacy_nisq
        (the local `full` directory is a near-duplicate of `heron_r2` and is
        reported separately as a consistency check)
Output: analysis/outputs/stats.json          machine-readable summary
        analysis/outputs/stats_report.md     human-readable report
        analysis/outputs/tables/*.tex        booktabs snippets for the paper
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent / "outputs"
TABLES = OUT / "tables"

PROFILES = ["ideal", "heron_r2", "legacy_nisq"]
SEEDS = [0, 1, 2, 3, 4]
ANSATZE = {"A": "Strongly Entangling", "B": "Basic Entangler", "C": "TTN"}
N_BOOT = 20000
BOOT_SEED = 12345


def load_runs(profile: str) -> list[dict]:
    runs = []
    for s in SEEDS:
        path = RESULTS / f"{profile}__s{s}" / f"results_seed_{s}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        runs.append(json.loads(path.read_text()))
    return runs


def mean_std(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def paired_stats(a, b) -> dict:
    """Paired comparison of a vs b (a - b). t-test, Wilcoxon, Cohen's dz, bootstrap CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    out = {"n": int(len(d)), "mean_delta": float(d.mean())}
    if len(d) > 1 and d.std(ddof=1) > 0:
        t, p_t = stats.ttest_rel(a, b)
        out["t"] = float(t)
        out["p_t"] = float(p_t)
        out["dz"] = float(d.mean() / d.std(ddof=1))
    else:
        out["t"] = out["p_t"] = out["dz"] = None
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        out["p_wilcoxon"] = None
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    out["ci95_low"], out["ci95_high"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    if len(d) >= 3:
        out["p_shapiro"] = float(stats.shapiro(d).pvalue)
    return out


def holm(pvals: list[float | None]) -> list[float | None]:
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    m = len(indexed)
    out: list[float | None] = [None] * len(pvals)
    running = 0.0
    for rank, (i, p) in enumerate(sorted(indexed, key=lambda x: x[1])):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[i] = running
    return out


def fmt_p(p) -> str:
    if p is None:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    report: list[str] = ["# Statistical re-analysis — journal evidence\n"]
    summary: dict = {}

    # ---------------------------------------------------------------- Exp 1
    report.append("## Experiment 1 — forgetting drop (paired over seeds)\n")
    exp1 = {}
    rows = []
    for prof in PROFILES:
        runs = load_runs(prof)
        drop_base = [r["exp1"]["drop_base"] for r in runs]
        drop_qtl = [r["exp1"]["drop_qtl"] for r in runs]
        bm, bs = mean_std(drop_base)
        qm, qs = mean_std(drop_qtl)
        st = paired_stats(np.array(drop_base), np.array(drop_qtl))
        rel = 100.0 * st["mean_delta"] / bm
        exp1[prof] = {"base": (bm, bs), "qtl": (qm, qs), "rel_pct": rel, **st}
        rows.append((prof, bm, bs, qm, qs, st, rel))
        report.append(
            f"- **{prof}**: baseline {bm:.2f} +/- {bs:.2f}, QTL {qm:.2f} +/- {qs:.2f}, "
            f"delta {st['mean_delta']:+.2f} pp ({rel:.1f}%), "
            f"t={st['t']:+.2f} p={fmt_p(st['p_t'])}, wilcoxon p={fmt_p(st['p_wilcoxon'])}, "
            f"dz={st['dz']:+.2f}, CI95 [{st['ci95_low']:+.2f}, {st['ci95_high']:+.2f}]"
        )
    # pooled across profiles (15 pairs)
    pooled_base, pooled_qtl = [], []
    for prof in PROFILES:
        runs = load_runs(prof)
        pooled_base += [r["exp1"]["drop_base"] for r in runs]
        pooled_qtl += [r["exp1"]["drop_qtl"] for r in runs]
    pooled = paired_stats(pooled_base, pooled_qtl)
    exp1["pooled"] = pooled
    report.append(
        f"- **pooled (15 pairs)**: delta {pooled['mean_delta']:+.2f} pp, "
        f"t={pooled['t']:+.2f} p={fmt_p(pooled['p_t'])}, wilcoxon p={fmt_p(pooled['p_wilcoxon'])}, "
        f"dz={pooled['dz']:+.2f}, CI95 [{pooled['ci95_low']:+.2f}, {pooled['ci95_high']:+.2f}]"
    )
    summary["exp1"] = exp1

    # missing-field audit
    runs = load_runs("ideal")
    exp1_fields = sorted(runs[0]["exp1"].keys())
    report.append(
        f"\nField audit: Experiment 1 stores {exp1_fields}. "
        "Absolute Task~A accuracies after Task~A training are NOT stored in this campaign "
        "(reviewer #2 finding confirmed). The journal campaign must store acc_a_init/acc_a_final "
        "per run and report them in the manuscript tables.\n"
    )

    # ---------------------------------------------------------------- Exp 2
    report.append("## Experiment 2 — retained Task A accuracy per ansatz\n")
    exp2 = {}
    for prof in PROFILES:
        runs = load_runs(prof)
        accA = {k: [r["exp2"][k]["acc_A"] for r in runs] for k in ANSATZE}
        accB = {k: [r["exp2"][k]["acc_B"] for r in runs] for k in ANSATZE}
        pairs = [("A", "B"), ("A", "C"), ("B", "C")]
        tests = [paired_stats(accA[a], accA[b]) for a, b in pairs]
        adj = holm([t["p_t"] for t in tests])
        for t, p_adj in zip(tests, adj):
            t["p_holm"] = p_adj
        exp2[prof] = {"accA": {k: mean_std(v) for k, v in accA.items()},
                      "accB": {k: mean_std(v) for k, v in accB.items()},
                      "tests": {f"{a}_vs_{b}": t for (a, b), t in zip(pairs, tests)}}
        report.append(f"- **{prof}**: " + "; ".join(
            f"{ANSATZE[k]} AccA {mean_std(v)[0]:.2f}+/-{mean_std(v)[1]:.2f}" for k, v in accA.items()))
        for (a, b), t in zip(pairs, tests):
            report.append(
                f"  - {ANSATZE[a]} vs {ANSATZE[b]}: delta {t['mean_delta']:+.2f} pp, "
                f"p={fmt_p(t['p_t'])}, p_holm={fmt_p(t['p_holm'])}, dz={t['dz']:+.2f}"
            )
    summary["exp2"] = exp2

    # ---------------------------------------------------------------- Exp 3
    report.append("\n## Experiment 3 — cross-domain initialization (target accuracy)\n")
    exp3 = {}
    for prof in PROFILES:
        runs = load_runs(prof)
        vals = {k: [r["exp3"][k] for r in runs] for k in ["scr_acc", "qtl_acc", "mob_qtl_acc"]}
        pairs = [("qtl_acc", "scr_acc"), ("mob_qtl_acc", "scr_acc"), ("qtl_acc", "mob_qtl_acc")]
        tests = [paired_stats(vals[a], vals[b]) for a, b in pairs]
        adj = holm([t["p_t"] for t in tests])
        for t, p_adj in zip(tests, adj):
            t["p_holm"] = p_adj
        exp3[prof] = {"accs": {k: mean_std(v) for k, v in vals.items()},
                      "tests": {f"{a}_vs_{b}": t for (a, b), t in zip(pairs, tests)}}
        report.append(f"- **{prof}**: " + "; ".join(f"{k} {mean_std(v)[0]:.2f}+/-{mean_std(v)[1]:.2f}"
                                                    for k, v in vals.items()))
        for (a, b), t in zip(pairs, tests):
            report.append(
                f"  - {a} vs {b}: delta {t['mean_delta']:+.2f} pp, p={fmt_p(t['p_t'])}, "
                f"p_holm={fmt_p(t['p_holm'])}, dz={t['dz']:+.2f}")
    summary["exp3"] = exp3

    # -------------------------------------------------- consistency: full vs heron
    diffs = 0
    for s in SEEDS:
        a = json.loads((RESULTS / f"full__s{s}" / f"results_seed_{s}.json").read_text())
        b = json.loads((RESULTS / f"heron_r2__s{s}" / f"results_seed_{s}.json").read_text())
        if a["exp3"]["mob_qtl_acc"] != b["exp3"]["mob_qtl_acc"]:
            diffs += 1
    report.append(
        f"\nConsistency note: the local `full` directory duplicates `heron_r2` "
        f"({diffs} small difference(s) across 5 seeds in Experiment 3); it is excluded "
        "from the reported statistics.\n"
    )
    summary["full_vs_heron_diff_seeds"] = diffs

    # ---------------------------------------------------------------- LaTeX
    def tex_table(prof_rows: list[str], header: str, caption: str, label: str) -> str:
        body = "\n".join(prof_rows)
        return (
            "\\begin{table}[ht]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\begin{tabular}{lccccc}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            "\\end{tabular}\n\\end{table}\n"
        )

    tex1_rows = []
    for prof, bm, bs, qm, qs, st, rel in rows:
        name = {"ideal": "Ideal", "heron_r2": "IBM Heron~r2", "legacy_nisq": "Legacy NISQ"}[prof]
        tex1_rows.append(
            f"{name} & ${bm:.2f} \\pm {bs:.2f}$ & ${qm:.2f} \\pm {qs:.2f}$ & "
            f"${st['mean_delta']:+.2f}$ & ${rel:.1f}$ & ${fmt_p(st['p_t'])}$ & ${st['dz']:+.2f}$ \\\\"
        )
    (TABLES / "exp1_stats.tex").write_text(tex_table(
        tex1_rows,
        "\\textbf{Noise profile} & \\textbf{Baseline $\\Delta_A$} & \\textbf{QTL $\\Delta_A$} & "
        "\\textbf{$\\delta$ (pp)} & \\textbf{Rel. (\\%)} & \\textbf{$p$-value} & \\textbf{Cohen $d_z$}",
        "Experiment 1 paired statistics over five seeds per profile ($\\Delta_A$ in percentage points).",
        "tab:exp1_stats",
    ))

    tex2_rows = []
    for prof in PROFILES:
        name = {"ideal": "Ideal", "heron_r2": "IBM Heron~r2", "legacy_nisq": "Legacy NISQ"}[prof]
        aA, bB, cC = exp2[prof]["accA"], exp2[prof]["accB"], exp2[prof]["accA"]
        tAB = exp2[prof]["tests"]["A_vs_B"]
        tAC = exp2[prof]["tests"]["A_vs_C"]
        tex2_rows.append(
            f"{name} & ${aA['A'][0]:.2f} \\pm {aA['A'][1]:.2f}$ & ${aA['B'][0]:.2f} \\pm {aA['B'][1]:.2f}$ & "
            f"${aA['C'][0]:.2f} \\pm {aA['C'][1]:.2f}$ & ${fmt_p(tAB['p_holm'])}$ & ${fmt_p(tAC['p_holm'])}$ \\\\"
        )
    (TABLES / "exp2_stats.tex").write_text(tex_table(
        tex2_rows,
        "\\textbf{Noise profile} & \\textbf{SEL} & \\textbf{Basic Ent.} & \\textbf{TTN} & "
        "\\textbf{$p_\\text{Holm}$ SEL--BE} & \\textbf{$p_\\text{Holm}$ SEL--TTN}",
        "Experiment 2 retained Task~A accuracy (\\%) per ansatz with Holm-adjusted paired $p$-values.",
        "tab:exp2_stats",
    ))

    tex3_rows = []
    for prof in PROFILES:
        name = {"ideal": "Ideal", "heron_r2": "IBM Heron~r2", "legacy_nisq": "Legacy NISQ"}[prof]
        s = exp3[prof]["accs"]
        tq = exp3[prof]["tests"]["qtl_acc_vs_scr_acc"]
        tex3_rows.append(
            f"{name} & ${s['scr_acc'][0]:.2f} \\pm {s['scr_acc'][1]:.2f}$ & "
            f"${s['qtl_acc'][0]:.2f} \\pm {s['qtl_acc'][1]:.2f}$ & "
            f"${s['mob_qtl_acc'][0]:.2f} \\pm {s['mob_qtl_acc'][1]:.2f}$ & "
            f"${tq['mean_delta']:+.2f}$ & ${fmt_p(tq['p_holm'])}$ \\\\"
        )
    (TABLES / "exp3_stats.tex").write_text(tex_table(
        tex3_rows,
        "\\textbf{Noise profile} & \\textbf{Scratch} & \\textbf{QTL-Syn} & \\textbf{QTL-Mob} & "
        "\\textbf{$\\delta$ Syn--Scr} & \\textbf{$p_\\text{Holm}$}",
        "Experiment 3 target accuracy (\\%) per initialization with paired statistics.",
        "tab:exp3_stats",
    ))

    # ---------------------------------------------------------------- write
    (OUT / "stats.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "stats_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nOuputs written to {OUT}")


if __name__ == "__main__":
    main()
