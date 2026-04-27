"""Quick lambda scan for W_out Fisher protection."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.learner import UnifiedSELClassifier
from core.experiment_config import NoBoundaryConfig


def make_eval_task(task_id, n_samples, seed, in_size=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, in_size))
    boundary = X[:, 0] + X[:, 1]
    if task_id == 0:
        y = (boundary > 0.0).astype(int)
    else:
        y = (boundary < 0.0).astype(int)
    return X, y


def run_single(ewc_lambda, seed=7):
    config = NoBoundaryConfig()
    rng = np.random.default_rng(seed)

    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        pool_config=config.pool.to_pool_kwargs(),
        seed=seed,
        ewc_lambda=ewc_lambda,
    )

    X_task_0, y_task_0 = make_eval_task(0, 256, seed + 1000)
    X_task_1, y_task_1 = make_eval_task(1, 256, seed + 2000)

    for step in range(config.steps):
        progress = step / max(config.steps - 1, 1)
        x = rng.normal(0.0, 1.0, size=config.in_size)
        boundary = float(x[0] + x[1])
        y = int(boundary > 0.0) if rng.random() > progress else int(boundary < 0.0)
        clf.fit_one(x, y)

        if step + 1 == config.checkpoint_step:
            task_0_after_early = clf.accuracy(X_task_0, y_task_0)
            if ewc_lambda > 0 and not clf.fisher_estimated:
                clf.estimate_w_out_fisher(X_task_0, y_task_0)

    final_t0 = clf.accuracy(X_task_0, y_task_0)
    final_t1 = clf.accuracy(X_task_1, y_task_1)
    forgetting = task_0_after_early - final_t0
    avg_acc = (final_t0 + final_t1) / 2

    return {
        "ewc_lambda": ewc_lambda,
        "seed": seed,
        "task_0_after_early": round(task_0_after_early, 4),
        "task_0_final": round(final_t0, 4),
        "task_1_final": round(final_t1, 4),
        "forgetting": round(forgetting, 4),
        "avg_accuracy": round(avg_acc, 4),
    }


def main():
    print(f"{'lambda':>8} | {'T0_early':>8} | {'T0_final':>8} | {'T1_final':>8} | {'forget':>8} | {'avg_acc':>8}")
    print("-" * 70)

    for lam in [0, 10, 20, 30, 40, 50, 60, 80, 100, 120]:
        result = run_single(lam, seed=7)
        print(f"{result['ewc_lambda']:>8} | {result['task_0_after_early']:>8.4f} | {result['task_0_final']:>8.4f} | {result['task_1_final']:>8.4f} | {result['forgetting']:>8.4f} | {result['avg_accuracy']:>8.4f}")

    print("\n--- Seed 8 (high-forgetting seed) ---")
    for lam in [0, 10, 20, 30, 40, 50, 60, 80, 100]:
        result = run_single(lam, seed=8)
        print(f"{result['ewc_lambda']:>8} | {result['task_0_after_early']:>8.4f} | {result['task_0_final']:>8.4f} | {result['task_1_final']:>8.4f} | {result['forgetting']:>8.4f} | {result['avg_accuracy']:>8.4f}")

    print("\n--- Seed 9 (high-forgetting seed) ---")
    for lam in [0, 20, 40, 60, 80]:
        result = run_single(lam, seed=9)
        print(f"{result['ewc_lambda']:>8} | {result['task_0_after_early']:>8.4f} | {result['task_0_final']:>8.4f} | {result['task_1_final']:>8.4f} | {result['forgetting']:>8.4f} | {result['avg_accuracy']:>8.4f}")


if __name__ == "__main__":
    main()
