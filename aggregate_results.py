"""Combine per-seed JSON files (results_seed_*.json) into one aggregated JSON
+ regenerate the 3 figures with mean ± std error bars / bands.

Usage:
    python3 aggregate_results.py --pattern 'results_seed_*.json' --out results_noisy_5seeds.json
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

from main import aggregate_seeds, plot_aggregated  # reuse
from quantum_net import HERON_R2_NOISE


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results_seed_*.json")
    ap.add_argument("--out", default="results_noisy_5seeds.json")
    ap.add_argument("--out-dir", default=".")
    return ap.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"[error] no files match {args.pattern}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} per-seed files:")
    for f in files: print("  -", f)

    per_seed = []
    for f in files:
        with open(f) as fh:
            per_seed.append(json.load(fh))

    seeds = [r.get("seed") for r in per_seed]
    cfg0 = per_seed[0].get("config", {})
    noise = cfg0.get("noise", True)

    exp1_agg, exp2_agg, exp3_agg = aggregate_seeds(per_seed)

    payload = {
        "config": {
            "seeds": seeds,
            "noise": noise,
            "noise_params": HERON_R2_NOISE if noise else None,
            "epochs": cfg0.get("epochs"),
            "lr": cfg0.get("lr"),
        },
        "exp1": exp1_agg, "exp2": exp2_agg, "exp3": exp3_agg,
        "per_seed": per_seed,
    }
    out_path = os.path.join(args.out_dir, args.out)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nAggregated -> {out_path}")

    print("\n=== AGGREGATED METRICS (mean ± std over %d seeds) ===" % len(seeds))
    print(f"Exp 1 | Forgetting Drop (Baseline): {exp1_agg['drop_base'][0]:.2f} ± {exp1_agg['drop_base'][1]:.2f} %")
    print(f"Exp 1 | Forgetting Drop (QTL):      {exp1_agg['drop_qtl'][0]:.2f} ± {exp1_agg['drop_qtl'][1]:.2f} %")
    print("\nExp 2 (Topologies):")
    for ans in ['A', 'B', 'C']:
        m, s = exp2_agg[ans]['acc_A']
        mb, sb = exp2_agg[ans]['acc_B']
        print(f"  Ansatz {ans} | Acc.A: {m:.2f} ± {s:.2f}% | Acc.B: {mb:.2f} ± {sb:.2f}%")
    print("\nExp 3 (Cross-Domain):")
    print(f"  Pre-Train Syn Acc:    {exp3_agg['pretrain_acc'][0]:.2f} ± {exp3_agg['pretrain_acc'][1]:.2f}%")
    print(f"  Pre-Train Mob Acc:    {exp3_agg['mob_pretrain_acc'][0]:.2f} ± {exp3_agg['mob_pretrain_acc'][1]:.2f}%")
    print(f"  Scratch Acc:          {exp3_agg['scr_acc'][0]:.2f} ± {exp3_agg['scr_acc'][1]:.2f}%")
    print(f"  QTL-Syn Acc:          {exp3_agg['qtl_acc'][0]:.2f} ± {exp3_agg['qtl_acc'][1]:.2f}%")
    print(f"  QTL-Mob Acc:          {exp3_agg['mob_qtl_acc'][0]:.2f} ± {exp3_agg['mob_qtl_acc'][1]:.2f}%")

    noise_label = "IBM Heron r2 noise" if noise else "noiseless"
    plot_aggregated(exp1_agg, exp2_agg, exp3_agg, noise_label, len(seeds), out_dir=args.out_dir)


if __name__ == "__main__":
    main()
