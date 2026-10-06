#!/usr/bin/env python3
"""E6 stats -- regularization family on split-MNIST class-IL (global labels).

Cells: e234_smnist5__{tag}__{arm}__ideal__s{0..9} for tag std | si5 | si50 |
l2a | l2b. New arms (si/l2/derpp and synth variants) plus the existing
scratch/synth/er/ewc baselines of the same campaign.

Families for Holm:
  F1  vs-scratch (AA): er, ewc, si5, si50, l2a, l2b, derpp
  F2  prior effect (synth - scratch, same method/lambda): 6 contrasts
  F3  lambda sensitivity: si50-si5, l2b-l2a

Writes analysis/outputs/e6_stats_report.md and e6_stats.json.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent / "hercules_results" / "e234"
OUT = pathlib.Path(__file__).resolve().parent / "outputs"
SEEDS = range(10)
METRICS = ("aa", "af", "bwt", "fwt")
CONFIGS = [
    ("scratch", "std"), ("synth", "std"), ("er", "std"), ("ewc", "std"),
    ("si", "si5"), ("si", "si50"), ("l2", "l2a"), ("l2", "l2b"),
    ("derpp", "std"), ("synth_si", "si5"), ("synth_si", "si50"),
    ("synth_l2", "l2a"), ("synth_l2", "l2b"), ("synth_derpp", "std"),
]


def load(arm: str, tag: str) -> dict:
    out = {}
    for s in SEEDS:
        p = ROOT / f"e234_smnist5__{tag}__{arm}__ideal__s{s}" / f"results_seed_{s}.json"
        if p.exists():
            j = json.load(open(p))
            m = j["metrics"]["cil"]
            out[s] = {k: float(m[k]) for k in METRICS}
    return out


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, idx in enumerate(order):
        val = (len(ps) - rank) * ps[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def paired(a: dict, b: dict, label: str, family: str) -> dict:
    ks = sorted(set(a) & set(b))
    da = np.array([a[s]["aa"] for s in ks])
    db = np.array([b[s]["aa"] for s in ks])
    diff = da - db
    t, p = stats.ttest_rel(da, db)
    try:
        _, pw = stats.wilcoxon(da, db)
    except ValueError:
        pw = float("nan")
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd > 0 else 0.0
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(20000)])
    return dict(family=family, label=label, n=len(ks), d=float(diff.mean()),
                ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                t=float(t), p=float(p), wilcoxon=float(pw), dz=dz)


def main() -> None:
    data = {c: load(*c) for c in CONFIGS}
    missing = [c for c, d in data.items() if len(d) != 10]
    if missing:
        print("WARN celdas incompletas:", missing)

    lines = ["# E6 -- regularization family (split-MNIST class-IL, ideal)", ""]
    lines.append("Cells loaded: " + ", ".join(f"{a}/{t}={len(d)}" for (a, t), d in data.items()))
    lines.append("")
    lines.append("## Descriptives on AA / AF / BWT / FWT (mean +/- sd, n=10)")
    lines.append("")
    for (arm, tag), d in data.items():
        if not d:
            continue
        arrs = {k: np.array([d[s][k] for s in sorted(d)]) for k in METRICS}
        lines.append(f"- {arm}/{tag}: " + " | ".join(
            f"{k} {arrs[k].mean():.2f}+/-{arrs[k].std(ddof=1):.2f}" for k in METRICS))
    lines.append("")

    # family F1: vs scratch on AA
    f1 = []
    base = data[("scratch", "std")]
    for arm, tag in [("er", "std"), ("ewc", "std"), ("si", "si5"), ("si", "si50"),
                     ("l2", "l2a"), ("l2", "l2b"), ("derpp", "std")]:
        f1.append(paired(data[(arm, tag)], base, f"{arm}/{tag} - scratch", "F1_vs_scratch"))
    # family F2: prior effect (synth - scratch_init), same method and lambda
    f2 = [
        paired(data[("synth", "std")], base, "synth - scratch", "F2_prior"),
        paired(data[("synth_si", "si5")], data[("si", "si5")], "synth_si5 - si5", "F2_prior"),
        paired(data[("synth_si", "si50")], data[("si", "si50")], "synth_si50 - si50", "F2_prior"),
        paired(data[("synth_l2", "l2a")], data[("l2", "l2a")], "synth_l2a - l2a", "F2_prior"),
        paired(data[("synth_l2", "l2b")], data[("l2", "l2b")], "synth_l2b - l2b", "F2_prior"),
        paired(data[("synth_derpp", "std")], data[("derpp", "std")], "synth_derpp - derpp", "F2_prior"),
    ]
    # family F3: lambda sensitivity
    f3 = [
        paired(data[("si", "si50")], data[("si", "si5")], "si50 - si5 (lambda)", "F3_lambda"),
        paired(data[("l2", "l2b")], data[("l2", "l2a")], "l2b - l2a (lambda)", "F3_lambda"),
    ]

    for fam, rows in (("F1 vs scratch", f1), ("F2 prior effect", f2), ("F3 lambda", f3)):
        adj = holm([r["p"] for r in rows])
        lines.append(f"## {fam}")
        lines.append("")
        for r, ph in zip(rows, adj):
            lines.append(
                f"- {r['label']}: d={r['d']:+.2f} pp [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}], "
                f"t={r['t']:+.2f}, p={r['p']:.3f}, p_holm={ph:.3f}, "
                f"wilcoxon={r['wilcoxon']:.3f}, dz={r['dz']:+.2f}, n={r['n']}")
        lines.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    (OUT / "e6_stats_report.md").write_text(report)
    (OUT / "e6_stats.json").write_text(json.dumps(
        dict(descriptives={f"{a}/{t}": {k: float(np.mean([d[s][k] for s in d]))
                                        for k in METRICS} for (a, t), d in data.items() if d},
             contrasts=f1 + f2 + f3), indent=2))
    print(report)


if __name__ == "__main__":
    main()
