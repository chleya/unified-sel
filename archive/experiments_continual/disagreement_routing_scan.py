"""
A1-fix5: Prediction Disagreement Routing

Surprise signal has almost no task-discrimination power (gap = 0.006).
New approach: when current model and snapshot expert DISAGREE on prediction,
pick the one with higher confidence.

_snapshot_surprise_threshold is now a confidence ratio threshold:
  snap_conf > current_conf * threshold -> use snapshot

Configs tested:
  1. baseline: no snapshot (reference)
  2. thresh=0.0: always use snapshot when disagreeing (lower bound)
  3. thresh=0.5: snapshot needs half the confidence of current
  4. thresh=1.0: snapshot needs equal or more confidence (default)
  5. thresh=2.0: snapshot needs double the confidence
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

CONFIGS = [
    ("baseline_ewc30", 30, False, 1.0),
    ("disagree_t0.0", 30, True, 0.0),
    ("disagree_t0.5", 30, True, 0.5),
    ("disagree_t1.0", 30, True, 1.0),
    ("disagree_t1.5", 30, True, 1.5),
    ("disagree_t2.0", 30, True, 2.0),
]

SEEDS = list(range(7, 12))


def extract_metrics(result: dict) -> dict:
    final_t0 = result.get("task_0_accuracy_final", 0.0)
    final_t1 = result.get("task_1_accuracy_final", 0.0)
    ckpt_t0 = result.get("task_0_accuracy_after_early_stream", 0.0)
    forgetting = ckpt_t0 - final_t0
    return {"task_0_acc": final_t0, "task_1_acc": final_t1, "forgetting": forgetting, "avg_acc": (final_t0 + final_t1) / 2}


def main():
    config = NoBoundaryConfig()
    config.readout_mode = "hybrid_local"

    all_results = {}

    for name, ewc_lambda, use_snap, thresh in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {name} | ewc={ewc_lambda} | snap={use_snap} | thresh={thresh}")
        print(f"{'='*60}")

        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_seed(
                seed=seed, config=config,
                ewc_lambda=ewc_lambda, anchor_lambda=0.0,
                dual_path_alpha=0.0, snapshot_expert=use_snap,
                snapshot_surprise_threshold=thresh,
            )
            metrics = extract_metrics(result)
            seed_results.append(metrics)
            print(f"t0={metrics['task_0_acc']:.4f} t1={metrics['task_1_acc']:.4f} "
                  f"fg={metrics['forgetting']:.4f} avg={metrics['avg_acc']:.4f}")

        t0_mean = np.mean([r["task_0_acc"] for r in seed_results])
        t1_mean = np.mean([r["task_1_acc"] for r in seed_results])
        fg_mean = np.mean([r["forgetting"] for r in seed_results])
        avg_mean = np.mean([r["avg_acc"] for r in seed_results])
        all_results[name] = {
            "thresh": thresh,
            "task_0_acc_mean": t0_mean,
            "task_1_acc_mean": t1_mean,
            "forgetting_mean": fg_mean,
            "avg_acc_mean": avg_mean,
        }
        print(f"  MEAN: t0={t0_mean:.4f} t1={t1_mean:.4f} "
              f"fg={fg_mean:.4f} avg={avg_mean:.4f}")

    print(f"\n{'='*60}")
    print("SUMMARY - Prediction Disagreement Routing")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'thresh':>6} {'task_0':>8} {'task_1':>8} {'forget':>8} {'avg':>8}")
    print("-" * 65)
    for name, r in all_results.items():
        print(f"{name:<25} {r['thresh']:6.1f} {r['task_0_acc_mean']:8.4f} "
              f"{r['task_1_acc_mean']:8.4f} {r['forgetting_mean']:8.4f} {r['avg_acc_mean']:8.4f}")

    print(f"\nEWC baseline reference: t0=0.9070 t1=0.0940 fg=0.0250 avg=0.5005")
    print(f"Oracle routing upper bound: avg=0.7863")


if __name__ == "__main__":
    main()
