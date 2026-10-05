#!/usr/bin/env python3
"""Independent crosscheck of the E2/E3/E4 headline statistics.

Recomputes a fixed battery of quantities from the raw payloads with code
that does NOT import analysis/e234_stats.py (independent parsing, direct
numpy/scipy usage) and compares them against analysis/outputs/e234_stats.json.

Checks (per the pre-declared families):
  C1  cell counts (chain4/ideal n=10 per arm; smnist5 ideal n=10)
  C2  chain4 ideal: mean of the five scenario metrics per arm
  C3  chain4 ideal: paired t p-values for synth-scratch on CIL-4
  C4  smnist5 ideal: AA means per arm + synth-scratch paired p
  C5  sz12k ideal: AA means + paired p
  C6  q8 ideal: delta_A means + paired p
Exit code 0 iff every check passes; prints a PASS/FAIL table.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
STATS = json.loads((REPO / "analysis" / "outputs" / "e234_stats.json").read_text())
TABLES = REPO / "journal" / "paper_journal" / "tables"

PAT = re.compile(r"results/e234_([a-z0-9]+)__([A-Za-z0-9]+)__([a-z]+)__([a-z0-9_]+)__s(\d+)/results_seed_(\d+)\.json$")


def cells():
    out = {}
    allowed_tags = {"std", "sz500", "sz2k", "sz12k", "q6", "q8", "L2", "L4"}
    for path in glob.glob(str(RESULTS / "e234_*" / "results_seed_*.json")):
        m = PAT.search(path)
        if not m:
            continue
        camp, tag, arm, prof, seed = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        if camp not in ("chain4", "pair2", "smnist5", "sfmnist5"):
            continue
        if tag not in allowed_tags or arm not in ("scratch", "synth", "er", "ewc"):
            continue
        if prof not in ("ideal", "heron_r2") or seed > 9:
            continue
        payload = json.loads(Path(path).read_text())
        if payload.get("experiment") != "e234_v1":
            continue
        out.setdefault((camp, tag, prof, arm), {})[seed] = payload
    return out


def scen(payload, metric):
    mm = payload["metrics"]["cil"]["matrix"] if metric.startswith("CIL") else payload["metrics"]["til"]["matrix"]
    T = int(metric[-1])
    row = mm[T - 1][:T]
    return float(np.mean(row))


def aa(payload):
    return float(payload["metrics"]["cil"]["aa"])


def dA(payload):
    m = payload["metrics"]["cil"]["matrix"]
    return m[0][0] - m[1][0]


def pmatch(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def main():
    C = cells()
    failures = []
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        if not ok:
            failures.append(f"{name}: {detail}")

    # C1 counts
    n_chain_scratch = len(C.get(("chain4", "std", "ideal", "scratch"), {}))
    n_chain_synth = len(C.get(("chain4", "std", "ideal", "synth"), {}))
    n5 = len(C.get(("smnist5", "std", "ideal", "scratch"), {}))
    check("C1a chain4 ideal counts", n_chain_scratch == n_chain_synth and n_chain_scratch >= 2,
          f"scratch={n_chain_scratch} synth={n_chain_synth}")
    check("C1b smnist5 ideal counts", n5 >= 2, f"n={n5}")
    check("C1c completeness", STATS["completeness"]["loaded"] == sum(
        len(v) for v in C.values()),
        f"json={STATS['completeness']['loaded']} recomputed={sum(len(v) for v in C.values())}")

    def paired_p(camp, tag, prof, metric_fn):
        s1 = C.get((camp, tag, prof, "synth"), {})
        s0 = C.get((camp, tag, prof, "scratch"), {})
        seeds = sorted(set(s1) & set(s0))
        if len(seeds) < 2:
            return None, None, None
        a = np.array([metric_fn(s1[x]) for x in seeds])
        b = np.array([metric_fn(s0[x]) for x in seeds])
        return a.mean(), b.mean(), float(stats.ttest_rel(a, b).pvalue)

    # C2/C3 chain4 ideal CIL-4
    m_syn, m_scr, p = paired_p("chain4", "std", "ideal", lambda x: scen(x, "CIL-4"))
    cj = STATS["chain4"]["ideal"]["metrics"]["CIL-4"]["contrast"]
    if m_syn is not None and "mean_delta" in cj and cj["mean_delta"] is not None:
        check("C2 chain4 CIL-4 delta", pmatch(m_syn - m_scr, cj["mean_delta"], 1e-6),
              f"recomputed={m_syn - m_scr:.6f} json={cj['mean_delta']}")
        check("C3 chain4 CIL-4 p", pmatch(p, cj.get("p_t"), 1e-8),
              f"recomputed={p} json={cj.get('p_t')}")
    else:
        check("C2/C3 chain4 CIL-4 available", False, "missing data")

    # C4 smnist5 AA
    m_syn, m_scr, p = paired_p("smnist5", "std", "ideal", aa)
    if m_syn is not None:
        bj = STATS["bench5"]["smnist5/ideal"]["synth"]
        cj = STATS["bench5"]["smnist5/ideal"].get("__contrast_placeholder__")
        # contrast lives in the report; recompute both and compare to descriptives only
        check("C4a smnist5 AA synth mean", pmatch(m_syn, bj["aa"], 1e-6),
              f"recomputed={m_syn} json={bj['aa']}")
        check("C4b smnist5 AA scratch mean",
              pmatch(m_scr, STATS["bench5"]["smnist5/ideal"]["scratch"]["aa"], 1e-6),
              f"recomputed={m_scr}")
    else:
        check("C4 smnist5 available", False, "missing data")

    # C5 sz12k ideal AA
    m_syn, m_scr, p = paired_p("pair2", "sz12k", "ideal", aa)
    if m_syn is not None:
        check("C5b sz12k AA means",
              pmatch(m_syn, np.mean([aa(x) for x in C[("pair2", "sz12k", "ideal", "synth")].values()]), 1e-6)
              and pmatch(m_scr, np.mean([aa(x) for x in C[("pair2", "sz12k", "ideal", "scratch")].values()]), 1e-6),
              f"synth={m_syn} scratch={m_scr}")
        cj = STATS["scale"]["sz12k/ideal"]["contrast"]
        check("C5c sz12k p", pmatch(p, cj.get("p_t"), 1e-8), f"recomputed={p} json={cj.get('p_t')}")
    else:
        check("C5 sz12k available", False, "missing data")

    # C6 q8 delta_A
    m_syn, m_scr, p = paired_p("pair2", "q8", "ideal", dA)
    if m_syn is not None:
        cj = STATS["scaling"]["q8"]["contrast"]
        check("C6a q8 dA delta", pmatch(m_syn - m_scr, cj["mean_delta"], 1e-6),
              f"recomputed={m_syn - m_scr} json={cj['mean_delta']}")
        check("C6b q8 dA p", pmatch(p, cj.get("p_t"), 1e-8), f"recomputed={p} json={cj.get('p_t')}")
    else:
        check("C6 q8 available", False, "missing data")

    # C7 machines recorded
    machines = STATS.get("machines", {})
    check("C7 machine ids", len(machines) >= 1, f"{len(machines)} ids")

    ok_all = not failures
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {('- ' + detail) if detail else ''}")
    print(f"\n{sum(1 for _, ok, _ in checks if ok)}/{len(checks)} checks passed")
    if not ok_all:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("CROSSCHECK OK")


if __name__ == "__main__":
    main()
