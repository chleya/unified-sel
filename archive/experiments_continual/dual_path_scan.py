"""Dual-path readout scan: memory_path + current_path to prevent W_out overwrite."""
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
        ("baseline_no_dual", 30, 0.0, 0.0),
        ("dual_alpha0.3", 30, 0.0, 0.3),
        ("dual_alpha0.5", 30, 0.0, 0.5),
        ("dual_alpha0.7", 30, 0.0, 0.7),
        ("dual_alpha0.9", 30, 0.0, 0.9),
        ("dual0.5_ewc0", 0, 0.0, 0.5),
        ("dual0.7_ewc0", 0, 0.0, 0.7),
        ("dual0.5_anchor5", 30, 5.0, 0.5),
    ]
    seeds = [7, 8, 9]
    config = NoBoundaryConfig()
    config.readout_mode = "hybrid_local"
    config.steps = 600
    config.checkpoint_step = 200

    print(f"{'config':>22} | {'task0':>6} | {'task1':>6} | {'forget':>7} | {'avg':>5}")
    print("-" * 58)

    for name, ewc_lam, anchor_lam, dual_alpha in configs:
        runs = [run_seed(seed=s, config=config, ewc_lambda=ewc_lam,
                         anchor_lambda=anchor_lam, dual_path_alpha=dual_alpha)
                for s in seeds]
        t0 = np.mean([r["task_0_accuracy_final"] for r in runs])
        t1 = np.mean([r["task_1_accuracy_final"] for r in runs])
        fg = np.mean([r["forgetting_task_0"] for r in runs])
        avg = (t0 + t1) / 2
        print(f"{name:>22} | {t0:6.4f} | {t1:6.4f} | {fg:7.4f} | {avg:5.4f}")

if __name__ == "__main__":
    main()
