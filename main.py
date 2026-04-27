"""Run all 3 experiments for one seed (or several) and dump per-seed results.

Default: runs SEEDS list and aggregates locally (legacy behaviour).
SLURM-friendly: pass --seed N --out results_seed_N.json to run a single seed
and persist its raw payload. Aggregation across seeds is done by
`aggregate_results.py`.
"""

import argparse
import json
import os
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless safe (cluster nodes have no display)
import matplotlib.pyplot as plt
import torch

from data_module import DataModule
from quantum_net import HybridQuantumNet, HERON_R2_NOISE
from experiment_runner import ExperimentRunner

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_NOISE = True
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.05


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_curves(curves):
    arr = np.asarray(curves, dtype=float)
    return arr.mean(axis=0).tolist(), arr.std(axis=0, ddof=0).tolist()


def run_single_seed(seed, runner_kwargs, model_kwargs, loaders):
    """Execute the 3 experiments for a single seed and return raw results."""
    source_loader, task_a_loader, target_loader, mob_source_loader = loaders
    set_seed(seed)
    runner = ExperimentRunner(**runner_kwargs, model_kwargs=model_kwargs)

    drop_base, drop_qtl = runner.run_experiment_1(
        HybridQuantumNet, source_loader, task_a_loader, target_loader)

    exp2 = runner.run_experiment_2(HybridQuantumNet, task_a_loader, target_loader)

    scr_res, qtl_res, pre_res, mob_qtl_res, mob_pre_res = runner.run_experiment_3(
        HybridQuantumNet, source_loader, mob_source_loader, target_loader)

    return {
        "seed": seed,
        "exp1": {"drop_base": drop_base, "drop_qtl": drop_qtl},
        "exp2": {ans: {k: met[k] for k in
                       ['acc_A', 'acc_B', 'train_time_A', 'test_time_A',
                        'train_time_B', 'test_time_B', 'acc_history']}
                 for ans, met in exp2.items()},
        "exp3": {
            "scratch_loss": scr_res[0],
            "scr_train_time": scr_res[1], "scr_acc": scr_res[2], "scr_test_time": scr_res[3],
            "qtl_loss": qtl_res[0],
            "qtl_train_time": qtl_res[1], "qtl_acc": qtl_res[2], "qtl_test_time": qtl_res[3],
            "mob_qtl_loss": mob_qtl_res[0],
            "mob_qtl_train_time": mob_qtl_res[1], "mob_qtl_acc": mob_qtl_res[2], "mob_qtl_test_time": mob_qtl_res[3],
            "pretrain_t": pre_res[0], "pretrain_acc": pre_res[1], "pretest_t": pre_res[2],
            "mob_pretrain_t": mob_pre_res[0], "mob_pretrain_acc": mob_pre_res[1], "mob_pretest_t": mob_pre_res[2],
        },
    }


def plot_aggregated(exp1_agg, exp2_agg, exp3_agg, noise_label, n_seeds, out_dir="."):
    print("Generating Matplotlib visualizations...")

    # ---------------- Figure 1 ----------------
    drop_base_mean, drop_base_std = exp1_agg["drop_base"]
    drop_qtl_mean,  drop_qtl_std  = exp1_agg["drop_qtl"]

    plt.figure(figsize=(7, 5))
    labels = ['Baseline (No Transfer)', 'QTL (Synthetic Prior)']
    means = [drop_base_mean, drop_qtl_mean]
    stds  = [drop_base_std,  drop_qtl_std]
    bars = plt.bar(labels, means, yerr=stds, capsize=8, color=['#e63946', '#457b9d'],
                   edgecolor='black', linewidth=0.8, error_kw={'elinewidth': 1.5})

    plt.ylabel(r'Forgetting Drop (\% Accuracy Loss) $\downarrow$', fontsize=12)
    plt.title(f'Experiment 1: Mitigation of Catastrophic Forgetting ({noise_label}, {n_seeds} seeds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + std + 1.0,
                 f"{mean:.1f}±{std:.1f}%", ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure1_forgetting.png'), dpi=300)
    plt.close()

    # ---------------- Figure 2 ----------------
    plt.figure(figsize=(8, 5))
    ansatze = ['A (Strongly Entangling)', 'B (Basic Entangler)', 'C (Tree Tensor Network)']
    means2 = [exp2_agg['A']['acc_A'][0], exp2_agg['B']['acc_A'][0], exp2_agg['C']['acc_A'][0]]
    stds2  = [exp2_agg['A']['acc_A'][1], exp2_agg['B']['acc_A'][1], exp2_agg['C']['acc_A'][1]]
    bars2 = plt.bar(ansatze, means2, yerr=stds2, capsize=8,
                    color=['#8338ec', '#ffbe0b', '#3a86ff'],
                    edgecolor='black', linewidth=0.8, error_kw={'elinewidth': 1.5})
    plt.ylabel(r'Retained Task A Accuracy (\%) $\uparrow$', fontsize=12)
    plt.title(f'Experiment 2: Topology Resilience ({noise_label}, {n_seeds} seeds)', fontsize=12)
    plt.ylim(0, 115)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, mean, std in zip(bars2, means2, stds2):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + std + 1.5,
                 f"{mean:.1f}±{std:.1f}%", ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure2_ansatz_decay.png'), dpi=300)
    plt.close()

    # ---------------- Figure 3 ----------------
    scratch_mean, scratch_std = exp3_agg['scratch_loss']
    qtl_mean,     qtl_std     = exp3_agg['qtl_loss']
    mob_mean,     mob_std     = exp3_agg['mob_qtl_loss']
    epochs = range(1, len(scratch_mean) + 1)

    plt.figure(figsize=(7.5, 5))
    for mean, std, marker, color, label in [
        (scratch_mean, scratch_std, 's', '#d62828', 'Target from Scratch'),
        (qtl_mean,     qtl_std,     'o', '#003049', 'QTL (Synthetic Source)'),
        (mob_mean,     mob_std,     '^', '#2a9d8f', 'QTL (MobileNetV2 Source)'),
    ]:
        mean = np.asarray(mean); std = np.asarray(std)
        plt.plot(epochs, mean, marker=marker, label=label, color=color, linewidth=2.2)
        plt.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18)
    plt.xlabel('Target Training Epochs', fontsize=12)
    plt.ylabel(r'Cross-Entropy Loss $\downarrow$', fontsize=12)
    plt.title(f'Experiment 3: Convergence via Cross-Domain Prior ({noise_label}, {n_seeds} seeds)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'figure3_convergence.png'), dpi=300)
    plt.close()

    print("Figures saved.")


def aggregate_seeds(per_seed_results):
    """Take list of single-seed payloads and return aggregated (mean, std)."""
    drops_base = [r["exp1"]["drop_base"] for r in per_seed_results]
    drops_qtl  = [r["exp1"]["drop_qtl"]  for r in per_seed_results]
    exp1_agg = {
        "drop_base": aggregate(drops_base),
        "drop_qtl":  aggregate(drops_qtl),
        "raw": {"drop_base": drops_base, "drop_qtl": drops_qtl},
    }

    exp2_agg = {}
    for ans in ['A', 'B', 'C']:
        d = {k: [r["exp2"][ans][k] for r in per_seed_results]
             for k in ['acc_A', 'acc_B', 'train_time_A', 'test_time_A',
                       'train_time_B', 'test_time_B']}
        exp2_agg[ans] = {k: aggregate(d[k]) for k in d}
        exp2_agg[ans]['acc_history'] = aggregate_curves(
            [r["exp2"][ans]['acc_history'] for r in per_seed_results])
        exp2_agg[ans]['raw'] = d

    exp3_curves = {key: [r["exp3"][key] for r in per_seed_results]
                   for key in ['scratch_loss', 'qtl_loss', 'mob_qtl_loss']}
    exp3_scalars = {key: [r["exp3"][key] for r in per_seed_results]
                    for key in ['scr_acc', 'qtl_acc', 'mob_qtl_acc',
                                'pretrain_acc', 'mob_pretrain_acc']}
    exp3_agg = {
        "scratch_loss": aggregate_curves(exp3_curves['scratch_loss']),
        "qtl_loss":     aggregate_curves(exp3_curves['qtl_loss']),
        "mob_qtl_loss": aggregate_curves(exp3_curves['mob_qtl_loss']),
        "scr_acc":      aggregate(exp3_scalars['scr_acc']),
        "qtl_acc":      aggregate(exp3_scalars['qtl_acc']),
        "mob_qtl_acc":  aggregate(exp3_scalars['mob_qtl_acc']),
        "pretrain_acc": aggregate(exp3_scalars['pretrain_acc']),
        "mob_pretrain_acc": aggregate(exp3_scalars['mob_pretrain_acc']),
        "raw_curves": exp3_curves,
        "raw_scalars": exp3_scalars,
    }
    return exp1_agg, exp2_agg, exp3_agg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="Run a single seed and dump its payload to --out (SLURM array mode).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Run multiple seeds locally and aggregate (default: %s)" % DEFAULT_SEEDS)
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSON path. Default: results_noisy_5seeds.json (multi) or results_seed_<N>.json (single).")
    ap.add_argument("--noise", action="store_true", default=DEFAULT_NOISE)
    ap.add_argument("--no-noise", dest="noise", action="store_false")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--no-mobilenet", action="store_true",
                    help="Skip MobileNetV2 source (offline-friendly when CIFAR/weights aren't pre-cached).")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"\n=== NOISE={args.noise} (Heron r2: {HERON_R2_NOISE if args.noise else None}) ===")
    print(f"=== epochs={args.epochs} lr={args.lr} ===")

    print("Initializing Quantum Modules and Loading Data...")
    data_mod = DataModule(data_dir=args.data_dir, batch_size=32)

    syn_train, syn_test = data_mod.get_synthetic_task(n_samples=2500)
    source_loader = (syn_train, syn_test)

    fa_train, fa_test, _ = data_mod.get_fashion_mnist_task(classes=(0, 1))
    task_a_loader = (fa_train, fa_test)

    mn_train, mn_test, _ = data_mod.get_mnist_task(classes=(2, 3))
    target_loader = (mn_train, mn_test)

    if args.no_mobilenet:
        print("WARNING: --no-mobilenet, reusing synthetic source as MobileNet placeholder.")
        mob_source_loader = source_loader
    else:
        mob_train, mob_test = data_mod.get_mobilenet_features_task(classes=(2, 3), n_samples=1000)
        mob_source_loader = (mob_train, mob_test)

    loaders = (source_loader, task_a_loader, target_loader, mob_source_loader)
    model_kwargs = {"noise": args.noise, "noise_params": HERON_R2_NOISE if args.noise else None}
    runner_kwargs = {"data_module": data_mod, "epochs": args.epochs, "lr": args.lr}

    # ---------- Single-seed mode (SLURM array) ----------
    if args.seed is not None:
        out_path = args.out or f"results_seed_{args.seed}.json"
        print(f"\n############ SEED {args.seed} ############")
        payload = run_single_seed(args.seed, runner_kwargs, model_kwargs, loaders)
        payload["config"] = {
            "noise": args.noise,
            "noise_params": HERON_R2_NOISE if args.noise else None,
            "epochs": args.epochs, "lr": args.lr,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nDumped seed {args.seed} -> {out_path}")
        return

    # ---------- Multi-seed local mode ----------
    seeds = args.seeds or DEFAULT_SEEDS
    out_path = args.out or "results_noisy_5seeds.json"

    per_seed = []
    for s in seeds:
        print(f"\n############ SEED {s} ############")
        per_seed.append(run_single_seed(s, runner_kwargs, model_kwargs, loaders))

    exp1_agg, exp2_agg, exp3_agg = aggregate_seeds(per_seed)

    payload = {
        "config": {"seeds": seeds, "noise": args.noise,
                   "noise_params": HERON_R2_NOISE if args.noise else None,
                   "epochs": args.epochs, "lr": args.lr},
        "exp1": exp1_agg, "exp2": exp2_agg, "exp3": exp3_agg,
        "per_seed": per_seed,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults dumped to {out_path}")

    print("\n=== AGGREGATED METRICS (mean ± std over %d seeds) ===" % len(seeds))
    print(f"Exp 1 | Forgetting Drop (Baseline): {exp1_agg['drop_base'][0]:.2f} ± {exp1_agg['drop_base'][1]:.2f} %")
    print(f"Exp 1 | Forgetting Drop (QTL):      {exp1_agg['drop_qtl'][0]:.2f} ± {exp1_agg['drop_qtl'][1]:.2f} %")
    print("\nExp 2 (Topologies):")
    for ans in ['A', 'B', 'C']:
        m, s = exp2_agg[ans]['acc_A']
        mb, sb = exp2_agg[ans]['acc_B']
        print(f"  Ansatz {ans} | Acc.A: {m:.2f} ± {s:.2f}% | Acc.B: {mb:.2f} ± {sb:.2f}%")
    print("\nExp 3 (Cross-Domain):")
    print(f"  Pre-Train Synthetic Acc:  {exp3_agg['pretrain_acc'][0]:.2f} ± {exp3_agg['pretrain_acc'][1]:.2f}%")
    print(f"  Pre-Train MobileNet Acc:  {exp3_agg['mob_pretrain_acc'][0]:.2f} ± {exp3_agg['mob_pretrain_acc'][1]:.2f}%")
    print(f"  Scratch Target Acc:       {exp3_agg['scr_acc'][0]:.2f} ± {exp3_agg['scr_acc'][1]:.2f}%")
    print(f"  QTL-Syn Target Acc:       {exp3_agg['qtl_acc'][0]:.2f} ± {exp3_agg['qtl_acc'][1]:.2f}%")
    print(f"  QTL-Mob Target Acc:       {exp3_agg['mob_qtl_acc'][0]:.2f} ± {exp3_agg['mob_qtl_acc'][1]:.2f}%")

    noise_label = "IBM Heron r2 noise" if args.noise else "noiseless"
    plot_aggregated(exp1_agg, exp2_agg, exp3_agg, noise_label, len(seeds))


if __name__ == "__main__":
    main()
