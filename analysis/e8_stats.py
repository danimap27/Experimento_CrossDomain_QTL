#!/usr/bin/env python3
"""E8 stats -- input capacity (qubits x PCA components) on the hard benchmarks.

Compares q6/q8 against the q4 baseline for scratch/er/derpp on smnist10 and
smnist5 (class-IL, global labels). Writes outputs/e8_stats_report.md + json.
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


def load(camp: str, arm: str, tag: str) -> dict:
    out = {}
    for s in SEEDS:
        p = ROOT / f"e234_{camp}__{tag}__{arm}__ideal__s{s}" / f"results_seed_{s}.json"
        if p.exists():
            j = json.load(open(p))
            m = j["metrics"]["cil"]
            out[s] = {k: float(m[k]) for k in METRICS}
    return out


def holm(ps):
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
    g = {}
    for arm in ("scratch", "er", "derpp"):
        g[("smnist10", arm, "q4")] = load("smnist10", arm, "std")
        g[("smnist10", arm, "q6")] = load("smnist10", arm, "q6")
        g[("smnist10", arm, "q8")] = load("smnist10", arm, "q8")
    for arm in ("scratch", "er", "derpp"):
        g[("smnist5", arm, "q4")] = load("smnist5", arm, "std")
        g[("smnist5", arm, "q8")] = load("smnist5", arm, "q8")

    lines = ["# E8 -- input capacity (qubits x components) on class-IL", ""]
    for camp in ("smnist10", "smnist5"):
        lines.append(f"## {camp}")
        lines.append("")
        for arm in ("scratch", "er", "derpp"):
            for q in ("q4", "q6", "q8"):
                d = g.get((camp, arm, q))
                if not d:
                    continue
                arrs = {k: np.array([d[s][k] for s in sorted(d)]) for k in METRICS}
                lines.append(f"- {arm} {q}: aa {arrs['aa'].mean():.2f}+/-{arrs['aa'].std(ddof=1):.2f}"
                             f" | af {arrs['af'].mean():.2f}+/-{arrs['af'].std(ddof=1):.2f}"
                             f" | bwt {arrs['bwt'].mean():.2f}+/-{arrs['bwt'].std(ddof=1):.2f}")
        lines.append("")

    f1 = []
    for arm in ("scratch", "er", "derpp"):
        f1.append(paired(g[("smnist10", arm, "q6")], g[("smnist10", arm, "q4")], f"smnist10 {arm} q6 - q4", "F1_sm10_capacity"))
        f1.append(paired(g[("smnist10", arm, "q8")], g[("smnist10", arm, "q4")], f"smnist10 {arm} q8 - q4", "F1_sm10_capacity"))
    f2 = []
    for arm in ("scratch", "er", "derpp"):
        f2.append(paired(g[("smnist5", arm, "q8")], g[("smnist5", arm, "q4")], f"smnist5 {arm} q8 - q4", "F2_sm5_capacity"))

    for fam, rows in (("F1 smnist10 capacity vs q4", f1), ("F2 smnist5 capacity vs q4", f2)):
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
    (OUT / "e8_stats_report.md").write_text(report)
    (OUT / "e8_stats.json").write_text(json.dumps(dict(contrasts=f1 + f2), indent=2))
    print(report)


if __name__ == "__main__":
    main()
