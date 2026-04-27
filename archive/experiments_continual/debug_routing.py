"""
Debug script to understand what's happening with the routing.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.continual.no_boundary import make_eval_task, run_seed, NoBoundaryConfig


def main():
    config = NoBoundaryConfig()
    config.readout_mode = "hybrid_local"
    seed = 7

    print("=== Debug Routing ===")
    print(f"Seed: {seed}\n")

    # Run a single seed to get the model
    result = run_seed(
        seed=seed, config=config, window_size=50,
        ewc_lambda=30.0, snapshot_expert=True,
        snapshot_surprise_threshold=0.0,
    )

    print("\n" + "="*60)
    print("RUN RESULTS")
    print("="*60)
    print(f"task_0_accuracy_after_early_stream: {result.get('task_0_accuracy_after_early_stream', 'N/A')}")
    print(f"task_0_accuracy_final: {result.get('task_0_accuracy_final', 'N/A')}")
    print(f"task_1_accuracy_final: {result.get('task_1_accuracy_final', 'N/A')}")

    # Now let's manually test some examples
    print("\n" + "="*60)
    print("MANUAL TESTING")
    print("="*60)

    X_task_0, y_task_0 = make_eval_task(0, 10, seed + 1000, config.in_size)
    X_task_1, y_task_1 = make_eval_task(1, 10, seed + 2000, config.in_size)

    print("\nTask 0 examples (x[0]+x[1] > 0):")
    for i in range(min(5, len(X_task_0))):
        x = X_task_0[i]
        y = y_task_0[i]
        boundary = x[0] + x[1]
        print(f"  x={x}, x[0]+x[1]={boundary:.4f}, y={y}")

    print("\nTask 1 examples (x[0]+x[1] < 0):")
    for i in range(min(5, len(X_task_1))):
        x = X_task_1[i]
        y = y_task_1[i]
        boundary = x[0] + x[1]
        print(f"  x={x}, x[0]+x[1]={boundary:.4f}, y={y}")


if __name__ == "__main__":
    main()

