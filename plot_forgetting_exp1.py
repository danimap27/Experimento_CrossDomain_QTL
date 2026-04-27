"""Standalone plot for Experiment 1 (catastrophic forgetting mitigation).

Reads aggregated results from `results_noisy_5seeds.json` produced by main.py
and renders a bar chart with 1-sigma error bars over seeds. Mirrors the style
of plot_topology_decay.py (Exp 2) and plot_convergence_exp3.py (Exp 3).
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_JSON = "results_noisy_5seeds.json"


def _load_exp1():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            data = json.load(f)
        exp1 = data["exp1"]
        cfg = data.get("config", {})
        noise_label = "IBM Heron r2 noise" if cfg.get("noise") else "noiseless"
        n_seeds = len(cfg.get("seeds", []))
        return exp1, noise_label, n_seeds

    # Fallback: hardcoded values if no JSON yet
    print(f"[warn] {RESULTS_JSON} not found, using fallback values.")
    exp1 = {
        "drop_base": (45.0, 5.0),
        "drop_qtl":  (12.0, 3.0),
        "raw": {"drop_base": [], "drop_qtl": []},
    }
    return exp1, "fallback", 0


def plot_exp1_forgetting():
    exp1, noise_label, n_seeds = _load_exp1()

    drop_base_mean, drop_base_std = exp1["drop_base"]
    drop_qtl_mean,  drop_qtl_std  = exp1["drop_qtl"]
    raw = exp1.get("raw", {})

    labels = ['Baseline\n(Random Init)', 'QTL\n(Synthetic Prior)']
    means  = [drop_base_mean, drop_qtl_mean]
    stds   = [drop_base_std,  drop_qtl_std]
    colors = ['#e63946', '#457b9d']

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(labels, means, yerr=stds, capsize=10,
                  color=colors, edgecolor='black', linewidth=1.0,
                  width=0.55, error_kw={'elinewidth': 1.8, 'ecolor': 'black'})

    # Scatter individual seed points if available
    for i, key in enumerate(['drop_base', 'drop_qtl']):
        seeds_vals = raw.get(key, [])
        if seeds_vals:
            jitter = np.random.normal(0, 0.04, size=len(seeds_vals))
            ax.scatter([i + j for j in jitter], seeds_vals,
                       color='black', zorder=5, s=35, alpha=0.7,
                       edgecolor='white', linewidth=0.6,
                       label='Per-seed' if i == 0 else None)

    # Annotation: improvement
    delta = drop_base_mean - drop_qtl_mean
    ax.annotate(
        f'Forgetting reduced by\n{delta:.1f} pp',
        xy=(1, drop_qtl_mean + drop_qtl_std),
        xytext=(0.5, max(means) * 0.85),
        textcoords='data',
        ha='center', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#06d6a0", lw=1.8),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.3, headwidth=7),
    )

    ax.set_ylabel(r'Forgetting Drop (\% Accuracy Loss) $\downarrow$', fontsize=12)
    title = f'Experiment 1: Catastrophic Forgetting Mitigation ({noise_label}'
    if n_seeds:
        title += f', {n_seeds} seeds)'
    else:
        title += ')'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(means) + max(stds) + 12)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 1.2,
                f'{mean:.1f} ± {std:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    if any(raw.get(k) for k in ('drop_base', 'drop_qtl')):
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('figure1_forgetting.png', dpi=300)
    print("Standalone Figure saved as figure1_forgetting.png")
    plt.show(block=False)


if __name__ == "__main__":
    plot_exp1_forgetting()
