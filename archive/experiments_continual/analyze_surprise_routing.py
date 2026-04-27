"""
Analyze surprise signal routing quality.

For each test input, compute:
1. Which task it belongs to (ground truth)
2. Surprise value from the best structure
3. Whether surprise-gated routing makes the correct decision

This tells us:
- What fraction of task 0 inputs have high surprise (correctly routed to snapshot)
- What fraction of task 1 inputs have low surprise (correctly routed to current model)
- The optimal surprise threshold
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from core.experiment_config import NoBoundaryConfig
from core.learner import UnifiedSELClassifier
from experiments.continual.no_boundary import make_eval_task, stream_sample


def main():
    config = NoBoundaryConfig()
    config.readout_mode = "hybrid_local"
    seed = 9
    rng = np.random.default_rng(seed)

    X_task_0, y_task_0 = make_eval_task(0, 256, seed + 1000, config.in_size)
    X_task_1, y_task_1 = make_eval_task(1, 256, seed + 2000, config.in_size)

    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        pool_config=config.pool.to_pool_kwargs(),
        seed=seed,
        ewc_lambda=30.0,
        readout_mode=config.readout_mode,
        shared_readout_scale=config.shared_readout_scale,
        shared_readout_post_checkpoint_scale=config.shared_readout_post_checkpoint_scale,
        local_readout_lr_scale=config.local_readout_lr_scale,
    )

    for step in range(config.steps):
        if step < config.checkpoint_step:
            progress = 0.0
        else:
            progress = (step - config.checkpoint_step) / max(config.steps - config.checkpoint_step - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)

    clf.snapshot_expert(surprise_threshold=0.0)

    t0_surprises = []
    t1_surprises = []

    for i in range(len(X_task_0)):
        x = X_task_0[i]
        best_structure = clf.pool.select_best_structure(np.atleast_2d(x))
        if best_structure is not None:
            surprise = float(best_structure.current_surprise(x))
        else:
            surprise = 1.0
        t0_surprises.append(surprise)

    for i in range(len(X_task_1)):
        x = X_task_1[i]
        best_structure = clf.pool.select_best_structure(np.atleast_2d(x))
        if best_structure is not None:
            surprise = float(best_structure.current_surprise(x))
        else:
            surprise = 1.0
        t1_surprises.append(surprise)

    t0_surprises = np.array(t0_surprises)
    t1_surprises = np.array(t1_surprises)

    print("=== Surprise Signal Analysis ===")
    print(f"Task 0 inputs (should have HIGH surprise -> route to snapshot):")
    print(f"  Mean: {t0_surprises.mean():.4f}")
    print(f"  Std:  {t0_surprises.std():.4f}")
    print(f"  Min:  {t0_surprises.min():.4f}")
    print(f"  Max:  {t0_surprises.max():.4f}")
    print(f"  Median: {np.median(t0_surprises):.4f}")

    print(f"\nTask 1 inputs (should have LOW surprise -> route to current model):")
    print(f"  Mean: {t1_surprises.mean():.4f}")
    print(f"  Std:  {t1_surprises.std():.4f}")
    print(f"  Min:  {t1_surprises.min():.4f}")
    print(f"  Max:  {t1_surprises.max():.4f}")
    print(f"  Median: {np.median(t1_surprises):.4f}")

    print(f"\n=== Threshold Scan ===")
    print(f"{'Thresh':>6} {'T0->Snap':>8} {'T1->Curr':>8} {'Accuracy':>8} {'Avg_Acc':>8}")
    print("-" * 45)

    best_acc = 0
    best_thresh = 0

    for thresh in np.arange(0.0, 1.01, 0.05):
        t0_to_snap = np.mean(t0_surprises > thresh)
        t1_to_curr = np.mean(t1_surprises <= thresh)
        routing_acc = (t0_to_snap * len(t0_surprises) + t1_to_curr * len(t1_surprises)) / (len(t0_surprises) + len(t1_surprises))

        snap_t0_acc = 0.8789
        curr_t1_acc = 0.7578
        snap_t1_acc = 0.2969
        curr_t0_acc = 0.2148

        t0_correct = t0_to_snap * snap_t0_acc + (1 - t0_to_snap) * curr_t0_acc
        t1_correct = t1_to_curr * curr_t1_acc + (1 - t1_to_curr) * snap_t1_acc
        avg_acc = (t0_correct + t1_correct) / 2

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_thresh = thresh

        print(f"{thresh:6.2f} {t0_to_snap:8.4f} {t1_to_curr:8.4f} {routing_acc:8.4f} {avg_acc:8.4f}")

    print(f"\nBest threshold: {best_thresh:.2f} -> avg_acc={best_acc:.4f}")
    print(f"Oracle upper bound: 0.7863")
    print(f"Current snap(thresh=0.0): 0.4957")

    overlap = np.mean(t0_surprises) - np.mean(t1_surprises)
    if overlap < 0:
        print(f"\nPROBLEM: Task 0 surprise ({np.mean(t0_surprises):.4f}) < Task 1 surprise ({np.mean(t1_surprises):.4f})")
        print(f"  Surprise signal is INVERTED - task 0 inputs have LOWER surprise than task 1!")
        print(f"  This explains why surprise-gated routing fails.")
    else:
        print(f"\nSurprise signal direction is CORRECT (task 0 > task 1)")
        print(f"  Gap: {overlap:.4f}")


if __name__ == "__main__":
    main()
