#!/usr/bin/env python3
"""E7 stats -- long sequences (10 tasks) and hyperparameter tuning.

smnist10 / fmnist10: ten single-class tasks (0..9) per dataset.
smnist5 tuning: EWC lambda 1e2 and ER buffer 50% (vs the E6 settings).
Writes analysis/outputs/e7_stats_report.md and e7_stats.json.
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
    sm10 = {
        "scratch": load("smnist10", "scratch", "std"),
        "synth": load("smnist10", "synth", "std"),
        "si_0.5": load("smnist10", "si", "si05"),
        "si_1.5": load("smnist10", "si", "si15"),
        "si_5": load("smnist10", "si", "std"),
        "er": load("smnist10", "er", "std"),
        "derpp": load("smnist10", "derpp", "std"),
        "ewc_1e4": load("smnist10", "ewc", "std"),
        "ewc_1e2": load("smnist10", "ewc", "ewc1e2"),
    }
    fm10 = {
        "scratch": load("fmnist10", "scratch", "std"),
        "synth": load("fmnist10", "synth", "std"),
        "si_5": load("fmnist10", "si", "std"),
        "er": load("fmnist10", "er", "std"),
        "derpp": load("fmnist10", "derpp", "std"),
    }
    sm5 = {
        "scratch": load("smnist5", "scratch", "std"),
        "er_25": load("smnist5", "er", "std"),
        "er_50": load("smnist5", "er", "er50"),
        "ewc_1e4": load("smnist5", "ewc", "std"),
        "ewc_1e2": load("smnist5", "ewc", "ewc1e2"),
    }

    lines = ["# E7 -- long sequences (10 tasks) and tuning", ""]
    for title, groups in (("smnist10 (10 tasks x 1 class)", sm10),
                          ("fmnist10 (10 tasks x 1 class)", fm10),
                          ("smnist5 (tuning)", sm5)):
        lines.append(f"## {title}")
        lines.append("")
        for g, d in groups.items():
            if not d:
                lines.append(f"- {g}: MISSING")
                continue
            arrs = {k: np.array([d[s][k] for s in sorted(d)]) for k in METRICS}
            lines.append(f"- {g}: " + " | ".join(
                f"{k} {arrs[k].mean():.2f}+/-{arrs[k].std(ddof=1):.2f}" for k in METRICS))
        lines.append("")

    f1 = [paired(sm10[k], sm10["scratch"], f"smnist10 {k} - scratch", "F1_sm10")
          for k in ("synth", "si_0.5", "si_1.5", "si_5", "er", "derpp", "ewc_1e4", "ewc_1e2")]
    f2 = [paired(fm10[k], fm10["scratch"], f"fmnist10 {k} - scratch", "F2_fm10")
          for k in ("synth", "si_5", "er", "derpp")]
    f3 = [
        paired(sm5["er_50"], sm5["er_25"], "smnist5 er50 - er25", "F3_tune"),
        paired(sm5["er_50"], sm5["scratch"], "smnist5 er50 - scratch", "F3_tune"),
        paired(sm5["ewc_1e2"], sm5["ewc_1e4"], "smnist5 ewc1e2 - ewc1e4", "F3_tune"),
        paired(sm5["ewc_1e2"], sm5["scratch"], "smnist5 ewc1e2 - scratch", "F3_tune"),
    ]
    f4 = [
        paired(sm10["si_0.5"], sm10["si_5"], "smnist10 si0.5 - si5", "F4_lambda"),
        paired(sm10["si_1.5"], sm10["si_5"], "smnist10 si1.5 - si5", "F4_lambda"),
    ]
    for fam, rows in (("F1 smnist10 vs scratch", f1), ("F2 fmnist10 vs scratch", f2),
                      ("F3 smnist5 tuning", f3), ("F4 SI lambda on 10 tasks", f4)):
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
    (OUT / "e7_stats_report.md").write_text(report)
    (OUT / "e7_stats.json").write_text(json.dumps(dict(contrasts=f1 + f2 + f3 + f4), indent=2))
    print(report)


if __name__ == "__main__":
    main()
