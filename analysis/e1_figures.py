#!/usr/bin/env python3
"""Figure for the E1 controlled campaign: forgetting drop per arm per profile.

Output: journal/paper_journal/figures/fig_exp1_controlled.pdf
Reads the same JSON payloads as analysis/e1_stats.py (no hardcoded values).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_stats import ARM_ORDER, PROFILES, load_runs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "journal" / "paper_journal" / "figures"


def boot_ci(v, n=20000, seed=12345):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    runs = load_runs()
    if not runs:
        print("No E1 results found")
        sys.exit(1)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    arms = [a for a in ARM_ORDER if any((p, a) in runs for p in PROFILES)]

    fig, axes = plt.subplots(1, len(PROFILES), figsize=(11, 4.2), sharey=True)
    if len(PROFILES) == 1:
        axes = [axes]
    for ax, prof in zip(axes, PROFILES):
        xs, means, errs_lo, errs_hi, seeds_all = [], [], [], [], []
        for i, arm in enumerate(arms):
            per_seed = runs.get((prof, arm))
            if not per_seed:
                continue
            vals = [p["delta_a"] for p in per_seed.values()]
            m = float(np.mean(vals))
            lo, hi = boot_ci(vals)
            xs.append(i)
            means.append(m)
            errs_lo.append(m - lo)
            errs_hi.append(hi - m)
            seeds_all.append((i, vals))
        colors = ["#e63946" if a in ("B1", "B2") else
                  "#457b9d" if a in ("B3", "B4") else "#2a9d8f" for a in arms]
        ax.bar(xs, means, yerr=[errs_lo, errs_hi], capsize=4,
               color=[colors[i] for i in xs], edgecolor="black", linewidth=0.7,
               error_kw={"elinewidth": 1.2})
        for i, vals in seeds_all:
            jitter = np.linspace(-0.15, 0.15, len(vals))
            ax.scatter([i + j for j in jitter], vals, s=9, zorder=3,
                       facecolor="black", alpha=0.45, linewidths=0)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, fontsize=8)
        ax.set_title({"ideal": "Ideal (noiseless)", "heron_r2": "IBM Heron r2"}[prof])
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].set_ylabel(r"Forgetting drop $\Delta_A$ (pp) $\downarrow$", fontsize=11)
    fig.suptitle("E1 controlled campaign: 2x2 design + CL baselines", fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "fig_exp1_controlled.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
