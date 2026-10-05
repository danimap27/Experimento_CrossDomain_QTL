#!/usr/bin/env python3
"""Figures for the E2/E3/E4 campaigns (journal revision).

Reads the same payloads as analysis/e234_stats.py (no hardcoded values) and
writes into journal/paper_journal/figures/:

  fig_e234_scenarios.pdf  E2 chain: scenario metrics (ideal/heron) + mean
                          class-IL matrices for scratch and synth.
  fig_e234_bench5.pdf     E2 standard benchmarks: split-MNIST retention
                          curves + AA/AF + split-FMNIST curves (+Nemenyi CD).
  fig_e234_scale.pdf      E3 data scale: AA / dA / AccB vs train size.
  fig_e234_scaling.pdf    E4 grid: dA / AA / epoch time vs config.

Usage: python analysis/e234_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e234_stats import (PROFILES, Runs, bench_metrics, load_runs,  # noqa: E402
                        pair_metrics, scenario_metrics, friedman_nemenyi)

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "journal" / "paper_journal" / "figures"
COL = {"scratch": "#e63946", "synth": "#457b9d", "er": "#2a9d8f", "ewc": "#e9c46a"}
PNAME = {"ideal": "Ideal (noiseless)", "heron_r2": "IBM Heron r2"}


def boot_ci(v, n=20000, seed=12345):
    v = np.asarray(v, float)
    if len(v) < 2:
        return v.mean(), v.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def grouped_bars(ax, groups, series, colors, labels, ylabel, title=None,
                 group_labels=None, ci=True, legend_loc="upper right"):
    """groups: list of group keys; series: {key: [values per group]}."""
    x = np.arange(len(groups))
    keys = list(series.keys())
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals = series[k]
        m = np.array([np.mean(np.asarray(v, float)) if len(v) else np.nan for v in vals])
        lo = np.array([boot_ci(np.asarray(v, float))[0] if len(v) else np.nan for v in vals])
        hi = np.array([boot_ci(np.asarray(v, float))[1] if len(v) else np.nan for v in vals])
        err = np.vstack([m - lo, hi - m])
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, m,
               yerr=(err if ci else None), width=w * 0.92,
               color=colors.get(k, None), label=labels.get(k, k),
               edgecolor="black", linewidth=0.5, capsize=2,
               error_kw={"elinewidth": 1.0})
    ax.set_xticks(x, group_labels if group_labels else groups)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", ls="--", alpha=0.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.legend(frameon=True, fontsize=8, loc=legend_loc, framealpha=0.9)


def fig_scenarios(runs: Runs):
    metrics = ["TIL-2", "CIL-2", "TIL-3", "CIL-3", "CIL-4"]
    labels = ["TIL-2", "CIL-2", "TIL-3", "CIL-3", "CIL-4"]
    have = {p: any(runs.seeds("chain4", "std", p, "scratch") for _ in [0])
            for p in PROFILES}
    if not have["ideal"]:
        return
    panels = [p for p in PROFILES if have[p]]

    ncols = min(len(panels), 2)
    fig = plt.figure(figsize=(13.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, max(ncols, 2), height_ratios=[1.15, 1.0])

    # top row: scenario bars per profile
    for j, prof in enumerate(panels[:2]):
        ax = fig.add_subplot(gs[0, j] if ncols > 1 else gs[0, :])
        series = {"scratch": [], "synth": []}
        for m in metrics:
            for arm in series:
                vals = [scenario_metrics(p)[m]
                        for p in runs.seeds("chain4", "std", prof, arm).values()]
                series[arm].append(vals)
        grouped_bars(ax, metrics, series, COL, {"scratch": "scratch", "synth": "synth"},
                     "accuracy (%)", title=f"E2 chain scenarios — {PNAME[prof]}",
                     group_labels=labels, ci=True)

    # bottom row: mean class-IL matrices (ideal) for scratch and synth
    for j, arm in enumerate(("scratch", "synth")):
        ax = fig.add_subplot(gs[1, j])
        per = [p["metrics"]["cil"]["matrix"]
               for p in runs.seeds("chain4", "std", "ideal", arm).values()]
        if not per:
            ax.axis("off")
            continue
        T = 4
        M = np.full((T, T), np.nan)
        for mat in per:
            for i, row in enumerate(mat):
                for jj, v in enumerate(row):
                    M[i, jj] = np.nanmean([M[i, jj], v])
        im = ax.imshow(M, vmin=0, vmax=100, cmap="viridis")
        ax.set_xticks(range(T), [f"T{i+1}" for i in range(T)])
        ax.set_yticks(range(T), [f"after T{i+1}" for i in range(T)])
        ax.set_xlabel("evaluated task")
        for i in range(T):
            for jj in range(i + 1):
                v = M[i, jj]
                ax.text(jj, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < 60 else "black", fontsize=8)
        ax.set_title(f"class-IL accuracy matrix — {arm} (ideal, mean over seeds)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("accuracy (%)", fontsize=8)
    out = FIGDIR / "fig_e234_scenarios.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure written: {out}")


def fig_bench5(runs: Runs):
    if not runs.seeds("smnist5", "std", "ideal", "scratch"):
        return
    arms = [a for a in ("scratch", "synth", "er", "ewc")
            if runs.seeds("smnist5", "std", "ideal", a)]
    fig = plt.figure(figsize=(13.5, 3.9))

    # (a) split-MNIST retention curve on task 0 (class-IL acc per phase)
    ax = fig.add_subplot(1, 3, 1)
    for arm in arms:
        curves = []
        for p in runs.seeds("smnist5", "std", "ideal", arm).values():
            mat = p["metrics"]["cil"]["matrix"]
            curves.append([mat[t][0] for t in range(5)])
        c = np.mean(curves, axis=0)
        s = np.std(curves, axis=0, ddof=1)
        ax.errorbar(range(1, 6), c, yerr=s, marker="o", capsize=3,
                    color=COL[arm], label=arm)
    ax.set_xticks(range(1, 6), [f"after T{i}" for i in range(1, 6)])
    ax.set_ylabel("class-IL accuracy on Task 1 (%)")
    ax.set_title("(a) split-MNIST: retention of Task 1")
    ax.grid(axis="y", ls="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)

    # (b) AA / AF per arm
    ax = fig.add_subplot(1, 3, 2)
    series = {"aa": [], "af": []}
    xs = np.arange(len(arms))
    for arm in arms:
        aa = [bench_metrics(p)["aa"] for p in runs.seeds("smnist5", "std", "ideal", arm).values()]
        af = [bench_metrics(p)["af"] for p in runs.seeds("smnist5", "std", "ideal", arm).values()]
        series["aa"].append(aa)
        series["af"].append(af)
    w = 0.38
    for k, (color, lab) in enumerate([("aa", "AA"), ("af", "AF")]):
        vals = series[k]
        m = np.array([np.mean(v) for v in vals])
        s = np.array([np.std(v, ddof=1) for v in vals])
        ax.bar(xs + (k - 0.5) * w, m, yerr=s, width=w * 0.9, capsize=2,
               color=["#6c757d", "#f4a261"][k], edgecolor="black", linewidth=0.5,
               label=lab)
    ax.set_xticks(xs, arms)
    ax.set_ylabel("percentage points")
    ax.set_title("(b) split-MNIST: AA and AF")
    ax.grid(axis="y", ls="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)

    # (c) split-FMNIST retention curves
    ax = fig.add_subplot(1, 3, 3)
    for arm in ("scratch", "synth"):
        per = runs.seeds("sfmnist5", "std", "ideal", arm)
        if not per:
            continue
        curves = []
        for p in per.values():
            mat = p["metrics"]["cil"]["matrix"]
            curves.append([mat[t][0] for t in range(5)])
        c = np.mean(curves, axis=0)
        s = np.std(curves, axis=0, ddof=1)
        ax.errorbar(range(1, 6), c, yerr=s, marker="o", capsize=3,
                    color=COL[arm], label=arm)
    ax.set_xticks(range(1, 6), [f"after T{i}" for i in range(1, 6)])
    ax.set_ylabel("class-IL accuracy on Task 1 (%)")
    ax.set_title("(c) split-Fashion-MNIST: retention of Task 1")
    ax.grid(axis="y", ls="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = FIGDIR / "fig_e234_bench5.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure written: {out}")


def fig_scale(runs: Runs):
    sizes = [("sz500", "500"), ("sz2k", "2000"), ("sz12k", "12000")]
    if not runs.seeds("pair2", "sz2k", "ideal", "scratch"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # (a) AA vs size
    ax = axes[0]
    for arm in ("scratch", "synth"):
        xs, ms, ss = [], [], []
        for tag, lab in sizes:
            per = runs.seeds("pair2", tag, "ideal", arm)
            if not per:
                continue
            v = [pair_metrics(p)["aa_cil"] for p in per.values()]
            xs.append(int(lab))
            ms.append(np.mean(v))
            ss.append(np.std(v, ddof=1))
        ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, color=COL[arm], label=arm)
    ax.set_xscale("log")
    ax.set_xticks([500, 2000, 12000], ["500", "2k", "12k"])
    ax.set_xlabel("training samples per task")
    ax.set_ylabel("AA, binary readout (%)")
    ax.set_title("(a) accuracy vs data scale")
    ax.grid(ls="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)

    # (b) Delta_A vs size
    ax = axes[1]
    series = {"scratch": [], "synth": []}
    for tag, lab in sizes:
        for arm in series:
            v = [pair_metrics(p)["delta_a_cil"]
                 for p in runs.seeds("pair2", tag, "ideal", arm).values()]
            series[arm].append(v)
    grouped_bars(ax, [("500" if lab == "500" else ("2k" if lab == "2000" else "12k"))
                      for _, lab in sizes], series, COL,
                 {"scratch": "scratch", "synth": "synth"},
                 r"$\Delta_A$ (pp)", title=r"(b) forgetting drop $\Delta_A$ vs data scale")

    # (c) Acc_B vs size
    ax = axes[2]
    for arm in ("scratch", "synth"):
        xs, ms, ss = [], [], []
        for tag, lab in sizes:
            per = runs.seeds("pair2", tag, "ideal", arm)
            if not per:
                continue
            v = [pair_metrics(p)["acc_b_cil"] for p in per.values()]
            xs.append(int(lab))
            ms.append(np.mean(v))
            ss.append(np.std(v, ddof=1))
        ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, color=COL[arm], label=arm)
    ax.set_xscale("log")
    ax.set_xticks([500, 2000, 12000], ["500", "2k", "12k"])
    ax.set_xlabel("training samples per task")
    ax.set_ylabel("Acc$_B$ (%)")
    ax.set_title("(c) Task-B plasticity vs data scale")
    ax.grid(ls="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = FIGDIR / "fig_e234_scale.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure written: {out}")


def fig_scaling(runs: Runs):
    configs = [("4q/3L", "std"), ("6q/3L", "q6"), ("8q/3L", "q8"),
               ("4q/2L", "L2"), ("4q/4L", "L4")]
    if not runs.seeds("pair2", "std", "ideal", "scratch"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # (a) Delta_A bars
    ax = axes[0]
    series = {"scratch": [], "synth": []}
    for lab, tag in configs:
        for arm in series:
            v = [pair_metrics(p)["delta_a_cil"]
                 for p in runs.seeds("pair2", tag, "ideal", arm).values()]
            series[arm].append(v)
    grouped_bars(ax, [c for c, _ in configs], series, COL,
                 {"scratch": "scratch", "synth": "synth"},
                 r"$\Delta_A$ (pp)", title=r"(a) forgetting drop $\Delta_A$ across configs")

    # (b) AA bars
    ax = axes[1]
    series = {"scratch": [], "synth": []}
    for lab, tag in configs:
        for arm in series:
            v = [pair_metrics(p)["aa_cil"]
                 for p in runs.seeds("pair2", tag, "ideal", arm).values()]
            series[arm].append(v)
    grouped_bars(ax, [c for c, _ in configs], series, COL,
                 {"scratch": "scratch", "synth": "synth"},
                 "AA, binary readout (%)", title="(b) accuracy across configs")

    # (c) epoch time
    ax = axes[2]
    from e234_stats import epoch_time
    xs = np.arange(len(configs))
    for arm in ("scratch", "synth"):
        m, s = [], []
        for lab, tag in configs:
            v = [epoch_time(p) for p in runs.seeds("pair2", tag, "ideal", arm).values()]
            v = [x for x in v if x]
            m.append(np.mean(v))
            s.append(np.std(v, ddof=1))
        ax.bar(xs + (0.2 if arm == "synth" else -0.2), m, yerr=s, width=0.38,
               capsize=2, color=COL[arm], edgecolor="black", linewidth=0.5, label=arm)
    ax.set_xticks(xs, [c for c, _ in configs])
    ax.set_ylabel("seconds per epoch")
    ax.set_title("(c) per-epoch wall-clock")
    ax.grid(axis="y", ls="--", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = FIGDIR / "fig_e234_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure written: {out}")


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    payloads, missing, incomplete, _stale = load_runs()
    runs = Runs(payloads.values())
    print(f"loaded {len(payloads)} cells ({len(missing)} missing)")
    fig_scenarios(runs)
    fig_bench5(runs)
    fig_scale(runs)
    fig_scaling(runs)


if __name__ == "__main__":
    main()
