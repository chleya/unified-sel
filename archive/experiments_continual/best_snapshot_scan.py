"""
A1-fix4: Best Snapshot Expert — Snapshot at peak task_0 accuracy

Key insight from checkpoint_step scan: task_0 accuracy fluctuates during
training, and the checkpoint time is NOT the peak. By tracking the best
task_0 accuracy during task 0 training and snapshotting at the peak,
the snapshot expert should achieve higher quality.

Configs tested (all with snapshot, thresh=0.0):
  1. baseline: snapshot at checkpoint (current approach)
  2. best_snapshot: snapshot at peak task_0 accuracy during task 0
  3. best_snapshot + ckpt400: more training time + best snapshot
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
    ("snap_ckpt200", 600, 200, True, False),
    ("bestsnap_ckpt200", 600, 200, True, True),
    ("bestsnap_ckpt400", 800, 400, True, True),
    ("bestsnap_ckpt600", 1200, 600, True, True),
]

SEEDS = list(range(7, 12))


def main():
    all_results = {}

    for name, total_steps, ckpt_step, use_snap, use_best in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {name} | steps={total_steps} | ckpt={ckpt_step} | best={use_best}")
        print(f"{'='*60}")

        config = NoBoundaryConfig()
        config.steps = total_steps
        config.checkpoint_step = ckpt_step
        config.readout_mode = "hybrid_local"

        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_seed(
                seed=seed, config=config,
                ewc_lambda=30, anchor_lambda=0.0,
                dual_path_alpha=0.0, snapshot_expert=use_snap,
                snapshot_surprise_threshold=0.0,
                best_snapshot=use_best,
            )
            ckpt_t0 = result.get("task_0_accuracy_after_early_stream", 0.0)
            final_t0 = result.get("task_0_accuracy_final", 0.0)
            final_t1 = result.get("task_1_accuracy_final", 0.0)
            fg = result.get("forgetting_task_0", 0.0)
            seed_results.append({
                "ckpt_t0": ckpt_t0, "final_t0": final_t0,
                "final_t1": final_t1, "forgetting": fg,
            })
            print(f"ckpt_t0={ckpt_t0:.4f} final_t0={final_t0:.4f} "
                  f"t1={final_t1:.4f} fg={fg:.4f}")

        ckpt_t0_mean = np.mean([r["ckpt_t0"] for r in seed_results])
        final_t0_mean = np.mean([r["final_t0"] for r in seed_results])
        t1_mean = np.mean([r["final_t1"] for r in seed_results])
        fg_mean = np.mean([r["forgetting"] for r in seed_results])
        all_results[name] = {
            "steps": total_steps, "ckpt_step": ckpt_step, "best": use_best,
            "ckpt_t0_mean": ckpt_t0_mean,
            "final_t0_mean": final_t0_mean,
            "t1_mean": t1_mean,
            "forgetting_mean": fg_mean,
            "avg_acc_mean": (final_t0_mean + t1_mean) / 2,
        }
        print(f"  MEAN: ckpt_t0={ckpt_t0_mean:.4f} final_t0={final_t0_mean:.4f} "
              f"t1={t1_mean:.4f} fg={fg_mean:.4f}")

    print(f"\n{'='*60}")
    print("SUMMARY — Best Snapshot Expert")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'ckpt':>5} {'best':>5} {'ckpt_t0':>8} {'t0':>8} {'t1':>8} {'fg':>8} {'avg':>8}")
    print("-" * 80)
    for name, r in all_results.items():
        print(f"{name:<25} {r['ckpt_step']:5d} {str(r['best']):>5} {r['ckpt_t0_mean']:8.4f} "
              f"{r['final_t0_mean']:8.4f} {r['t1_mean']:8.4f} {r['forgetting_mean']:8.4f} "
              f"{r['avg_acc_mean']:8.4f}")

    print(f"\nEWC baseline reference: t0=0.9070 t1=0.0940 fg=0.0250 avg=0.5005")


if __name__ == "__main__":
    main()
