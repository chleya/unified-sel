"""Freeze scan: test structure freezing at checkpoint to prevent route hijacking."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.continual.no_boundary import run_seed, NoBoundaryConfig

def main():
    configs = [
        ("no_freeze_no_anchor", 30, 0.0),
        ("anchor_no_freeze", 30, 5.0),
        ("freeze_anchor_lam1", 30, 1.0),
        ("freeze_anchor_lam5", 30, 5.0),
        ("freeze_anchor_lam10", 30, 10.0),
        ("freeze_no_anchor", 30, 0.0),
    ]
    seeds = [7, 8, 9]
    config = NoBoundaryConfig()
    config.readout_mode = "hybrid_local"
    config.steps = 600
    config.checkpoint_step = 200

    print(f"{'config':>25} | {'task0_acc':>9} | {'task1_acc':>9} | {'forgetting':>10} | {'avg_acc':>7}")
    print("-" * 70)

    for name, ewc_lam, anchor_lam in configs:
        runs = [run_seed(seed=s, config=config, ewc_lambda=ewc_lam, anchor_lambda=anchor_lam) for s in seeds]
        t0 = np.mean([r["task_0_accuracy_final"] for r in runs])
        t1 = np.mean([r["task_1_accuracy_final"] for r in runs])
        fg = np.mean([r["forgetting_task_0"] for r in runs])
        avg = (t0 + t1) / 2
        print(f"{name:>25} | {t0:9.4f} | {t1:9.4f} | {fg:10.4f} | {avg:7.4f}")

if __name__ == "__main__":
    main()
