#!/usr/bin/env python3
"""Independent cross-check of the E1 statistics (task requirement: replicate
with `~/papers/reviews/QAI2026-191/stats_full.py` as a contrast).

This script re-implements the tstats()/ci_boot() functions of stats_full.py
(copied verbatim below, provenance: QAI2026-191 review packet, 2026-10-03)
and recomputes the headline E1 contrasts from the raw JSON payloads, then
compares them numerically against analysis/outputs/e1_stats.json.

Exit code 0 only if every checked figure agrees within tolerance.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent / "outputs"

# ---- copied verbatim from ~/papers/reviews/QAI2026-191/stats_full.py --------
def ci_boot(x, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def tstats(a, b):
    """paired test a vs b (same seeds)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n) if n > 1 else float('nan')
    t = d.mean() / se if se else float('inf')
    try:
        from scipy import stats as st
        p = float(2 * st.t.sf(abs(t), n - 1))
        pw = float(st.wilcoxon(a, b).pvalue)
        tb, pb = st.ttest_rel(a, b)
    except Exception:
        p = pw = tb = pb = None
    return dict(n=n, mean=float(d.mean()), sd=float(sd), se=se if isinstance(se, float) else float(se),
                t=float(t), p=p, wilcoxon_p=pw,
                cohen_d=float(d.mean() / sd) if sd else None,
                ci=ci_boot(d), per_seed=[float(v) for v in d])
# ---------------------------------------------------------------------------


def load_metric(metric, arm_a, arm_b, profile=None):
    pat = re.compile(r"results/e1_(?!smoke)([A-Za-z0-9_]+?)__([a-z0-9_]+)__s(\d+)/results_seed_\d+\.json$")
    vals = {}
    for path in glob.glob(str(RESULTS / "e1_*" / "results_seed_*.json")):
        m = pat.search(path)
        if not m:
            continue
        arm, prof, seed = m.group(1), m.group(2), int(m.group(3))
        if profile is not None and prof != profile:
            continue
        if arm not in (arm_a, arm_b):
            continue
        payload = json.loads(Path(path).read_text())
        vals.setdefault(arm, {})[(prof, seed)] = payload[metric]
    keys = sorted(set(vals.get(arm_a, {})) & set(vals.get(arm_b, {})))
    return ([vals[arm_a][k] for k in keys], [vals[arm_b][k] for k in keys], [k[1] for k in keys])


def main():
    stats = json.loads((OUT / "e1_stats.json").read_text())
    checks = [
        ("B2", "B1"), ("B3", "B1"), ("B4", "B2"), ("B4", "B1"),
        ("B5_l1e2", "B1"), ("B6", "B1"), ("B7", "B1"),
    ]
    ref = {}
    for fam, entries in stats["families"].items():
        for scope, contrasts in entries.items():
            for c in contrasts:
                if "mean_delta" in c:
                    ref[(fam, scope, c["contrast"])] = c

    failures = 0
    checked = 0
    for scope in ["ideal", "heron_r2", "pooled"]:
        for a, b in checks:
            xs, ys, seeds = load_metric("delta_a", a, b, None if scope == "pooled" else scope)
            if len(xs) < 2:
                continue
            st = tstats(xs, ys)
            fam = "F1_design" if {a, b} <= {"B1", "B2", "B3", "B4"} else "F2_baselines"
            key = (fam, scope, f"{a}-{b}")
            r = ref.get(key)
            if r is None:
                print(f"[MISS] {scope} {a}-{b}: not in e1_stats.json")
                failures += 1
                continue
            ok = (abs(st["mean"] - r["mean_delta"]) < 1e-9
                  and abs(st["cohen_d"] - r["dz"]) < 1e-6
                  and abs(st["p"] - r["p_t"]) < 1e-9
                  # bootstrap CIs use different RNG streams (stats_full.py uses
                  # seed=0 + per-resample choice; e1_stats.py BOOT_SEED=12345 +
                  # vectorized resampling), so agreement is checked within the
                  # resampling noise of 20k draws, not exactly.
                  and abs(st["ci"][0] - r["ci95_low"]) < 0.5
                  and abs(st["ci"][1] - r["ci95_high"]) < 0.5
                  and st["n"] == r["n"])
            checked += 1
            status = "OK" if ok else "MISMATCH"
            failures += 0 if ok else 1
            print(f"[{status}] {scope:9s} {a}-{b}: mean {st['mean']:+.4f} vs {r['mean_delta']:+.4f} | "
                  f"p {st['p']:.6f} vs {r['p_t']:.6f} | dz {st['cohen_d']:+.4f} vs {r['dz']:+.4f} | "
                  f"CI [{st['ci'][0]:+.2f},{st['ci'][1]:+.2f}] vs [{r['ci95_low']:+.2f},{r['ci95_high']:+.2f}] | n={st['n']}")

    print(f"\nchecked={checked} failures={failures}")
    if failures or not checked:
        sys.exit(1)
    print("CROSSCHECK PASSED (stats_full.py-style recompute agrees with e1_stats.py)")


if __name__ == "__main__":
    main()
