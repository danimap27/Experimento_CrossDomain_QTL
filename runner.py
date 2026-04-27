"""runner.py — Cross-Domain QTL adapter for hercules-framework manager.py.

For each (seed, model) combination, calls the existing main.py pipeline with
--seed N --out results/<run_id>/results_seed_<seed>.json, then writes a flat
results.csv next to it with the headline metrics that manager.py [M] and
generate_tables.py [T] consume.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Sweep enumeration
# =============================================================================

def make_run_id(*args) -> str:
    return "__".join(str(a) for a in args)


def iter_runs(cfg: dict):
    models = [m["name"] for m in cfg.get("models", [])]
    seeds  = cfg.get("seeds", [0])

    for model, seed in product(models, seeds):
        run_id = make_run_id(model, f"s{seed}")
        yield {
            "run_id": run_id,
            "model": model,
            "seed": seed,
        }


def apply_filter(runs, phase):
    filters = phase.get("filters", {})
    return [r for r in runs if all(r.get(k) == v for k, v in filters.items())]


# =============================================================================
# Single run execution
# =============================================================================

def execute_run(run_spec: dict, cfg: dict, machine_id: str = "local"):
    run_id  = run_spec["run_id"]
    seed    = int(run_spec["seed"])
    model   = run_spec["model"]
    epochs  = int(cfg.get("epochs", 10))
    lr      = float(cfg.get("lr", 0.05))
    # `model` doubles as the noise-profile selector
    from quantum_net import get_noise_profile
    noise, noise_params = get_noise_profile(model)

    results_dir = cfg.get("output_dir", "./results")
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / f"results_seed_{seed}.json"
    csv_path  = run_dir / "results.csv"

    if csv_path.exists():
        logger.info(f"[SKIP] {run_id} already completed.")
        return

    logger.info(f"[START] {run_id} seed={seed} model={model} noise={noise}")
    t0 = time.time()

    from main import run_single_seed
    from data_module import DataModule
    from experiment_runner import ExperimentRunner  # noqa: F401 (used by run_single_seed)

    data_mod = DataModule(data_dir=cfg.get("data_dir", "./data"), batch_size=32)

    syn_train, syn_test = data_mod.get_synthetic_task(n_samples=2500)
    fa_train, fa_test, _ = data_mod.get_fashion_mnist_task(classes=(0, 1))
    mn_train, mn_test, _ = data_mod.get_mnist_task(classes=(2, 3))
    mob_train, mob_test = data_mod.get_mobilenet_features_task(classes=(2, 3), n_samples=1000)

    loaders = (
        (syn_train, syn_test),
        (fa_train, fa_test),
        (mn_train, mn_test),
        (mob_train, mob_test),
    )
    model_kwargs = {"noise": noise, "noise_params": noise_params}
    runner_kwargs = {"data_module": data_mod, "epochs": epochs, "lr": lr}

    payload = run_single_seed(seed, runner_kwargs, model_kwargs, loaders)
    payload["config"] = {"noise": noise, "noise_params": noise_params,
                         "noise_profile": model, "epochs": epochs, "lr": lr}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    elapsed = time.time() - t0

    # Flatten headline metrics into a CSV row for manager.py [M] and [T]
    e1 = payload["exp1"]
    e3 = payload["exp3"]

    # Per-ansatz Exp 2 retained accuracy
    e2 = payload["exp2"]
    row = {
        "run_id":           run_id,
        "model":            model,
        "seed":             seed,
        # Exp 1
        "drop_base":        e1["drop_base"],
        "drop_qtl":         e1["drop_qtl"],
        # Exp 2
        "acc_A_ansatz_A":   e2["A"]["acc_A"],
        "acc_A_ansatz_B":   e2["B"]["acc_A"],
        "acc_A_ansatz_C":   e2["C"]["acc_A"],
        "acc_B_ansatz_A":   e2["A"]["acc_B"],
        "acc_B_ansatz_B":   e2["B"]["acc_B"],
        "acc_B_ansatz_C":   e2["C"]["acc_B"],
        # Exp 3
        "scr_acc":          e3["scr_acc"],
        "qtl_acc":          e3["qtl_acc"],
        "mob_qtl_acc":      e3["mob_qtl_acc"],
        "pretrain_acc":     e3["pretrain_acc"],
        "mob_pretrain_acc": e3["mob_pretrain_acc"],
        # bookkeeping
        "train_time":       elapsed,
        "machine_id":       machine_id,
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    logger.info(f"[DONE] {run_id} | drop_qtl={row['drop_qtl']:.2f}% | "
                f"qtl_acc={row['qtl_acc']:.2f}% | t={elapsed:.1f}s")


# =============================================================================
# SLURM command export
# =============================================================================

def export_commands(runs, out_path, config_path):
    lines = []
    for r in runs:
        cmd = (
            f"python runner.py --config {config_path} --run-id {r['run_id']} "
            f"--model {r['model']} --seed {r['seed']}"
        )
        lines.append(cmd)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Exported {len(lines)} commands to {out_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--export-commands", action="store_true")
    ap.add_argument("--machine-id", default="local")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    all_runs = list(iter_runs(cfg))

    if args.export_commands:
        for phase in cfg.get("phases", []):
            filtered = apply_filter(all_runs, phase)
            export_commands(filtered, phase["file"], args.config)
        return

    if args.run_id:
        run_spec = next((r for r in all_runs if r["run_id"] == args.run_id), {
            "run_id": args.run_id,
            "model": args.model,
            "seed": args.seed,
        })
        execute_run(run_spec, cfg, args.machine_id)
        return

    runs = all_runs
    if args.model:
        runs = [r for r in runs if r["model"] == args.model]
    if args.seed is not None:
        runs = [r for r in runs if r["seed"] == args.seed]

    logger.info(f"Planned runs: {len(runs)}")
    if args.dry_run:
        for r in runs:
            print(f"  {r['run_id']}")
        return

    for r in runs:
        execute_run(r, cfg, args.machine_id)


if __name__ == "__main__":
    main()
