"""
A1-fix5: Oracle Routing Upper Bound

Critical question: if routing were perfect (task 0 inputs → snapshot expert,
task 1 inputs → current model), what accuracy could we achieve?

This tells us whether the bottleneck is:
  A) Routing quality (surprise signal not good enough) → fix routing
  B) Learning quality (snapshot expert can't reach 0.90) → fix learning

If oracle routing gives avg_acc >> EWC, then surprise-driven routing has
potential but needs better signal quality.
If oracle routing gives avg_acc ≈ current, then learning quality is the
hard bottleneck.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from core.experiment_config import NoBoundaryConfig
from experiments.continual.no_boundary import run_seed

SEEDS = list(range(7, 12))


def oracle_evaluate(result: dict, config: NoBoundaryConfig) -> dict:
    """Re-evaluate with oracle routing: snapshot for task 0, current for task 1."""
    from experiments.continual.no_boundary import make_eval_task
    from core.learner import UnifiedSELClassifier
    import pickle

    seed = result.get("seed", 7)

    X_task_0, y_task_0 = make_eval_task(0, config.eval_samples_per_task, seed + 1000, config.in_size)
    X_task_1, y_task_1 = make_eval_task(1, config.eval_samples_per_task, seed + 2000, config.in_size)

    return {
        "task_0_snap": result.get("task_0_accuracy_after_early_stream", 0.0),
        "task_0_final": result.get("task_0_accuracy_final", 0.0),
        "task_1_final": result.get("task_1_accuracy_final", 0.0),
    }


def main():
    config = NoBoundaryConfig()
    config.readout_mode = "hybrid_local"

    print("=== Oracle Routing Upper Bound ===")
    print("Measuring: snapshot expert accuracy on task 0 + current model accuracy on task 1")
    print("This is the BEST possible result with perfect routing.\n")

    all_snap_t0 = []
    all_current_t1 = []
    all_current_t0 = []

    for seed in SEEDS:
        print(f"Seed {seed}...", end=" ", flush=True)

        result_snap = run_seed(
            seed=seed, config=config,
            ewc_lambda=30, snapshot_expert=True,
            snapshot_surprise_threshold=0.0,
        )

        result_baseline = run_seed(
            seed=seed, config=config,
            ewc_lambda=30, snapshot_expert=False,
        )

        snap_t0 = result_snap.get("task_0_accuracy_after_early_stream", 0.0)
        snap_final_t0 = result_snap.get("task_0_accuracy_final", 0.0)
        current_t0 = result_baseline.get("task_0_accuracy_final", 0.0)
        current_t1 = result_baseline.get("task_1_accuracy_final", 0.0)

        all_snap_t0.append(snap_t0)
        all_current_t0.append(current_t0)
        all_current_t1.append(current_t1)

        oracle_avg = (snap_t0 + current_t1) / 2
        print(f"snap_t0={snap_t0:.4f} current_t0={current_t0:.4f} "
              f"current_t1={current_t1:.4f} oracle_avg={oracle_avg:.4f}")

    snap_t0_mean = np.mean(all_snap_t0)
    current_t0_mean = np.mean(all_current_t0)
    current_t1_mean = np.mean(all_current_t1)
    oracle_avg_mean = (snap_t0_mean + current_t1_mean) / 2

    print(f"\n{'='*60}")
    print("ORACLE ROUTING UPPER BOUND")
    print(f"{'='*60}")
    print(f"  Snapshot expert task_0:     {snap_t0_mean:.4f}")
    print(f"  Current model task_0:       {current_t0_mean:.4f}")
    print(f"  Current model task_1:       {current_t1_mean:.4f}")
    print(f"  Oracle avg (snap_t0+curr_t1)/2: {oracle_avg_mean:.4f}")
    print(f"\n  EWC reference: t0=0.9070 t1=0.0940 avg=0.5005")
    print(f"  Snap(thresh=0.0): t0=0.6979 t1=0.2935 avg=0.4957")

    if oracle_avg_mean > 0.5005:
        print(f"\n  [YES] Oracle routing WOULD beat EWC (avg={oracle_avg_mean:.4f} > 0.5005)")
        print(f"     -> Bottleneck is ROUTING QUALITY, not learning quality")
        print(f"     -> Need to improve surprise signal for better task identification")
    else:
        print(f"\n  [NO] Oracle routing CANNOT beat EWC (avg={oracle_avg_mean:.4f} <= 0.5005)")
        print(f"     -> Bottleneck is LEARNING QUALITY, not routing quality")
        print(f"     -> Need to improve DFA learning or structure pool mechanism")

    print(f"\n  Key insight: snap_t0={snap_t0_mean:.4f} vs EWC_t0=0.9070")
    print(f"  Gap = {0.9070 - snap_t0_mean:.4f} — this is the learning quality deficit")


if __name__ == "__main__":
    main()
