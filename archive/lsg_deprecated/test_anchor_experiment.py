"""Quick test: verify anchors are set during experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.continual.no_boundary import run_seed, NoBoundaryConfig
import numpy as np

c = NoBoundaryConfig()
c.readout_mode = "hybrid_local"
c.steps = 600
c.checkpoint_step = 200

r = run_seed(seed=7, config=c, ewc_lambda=30, anchor_lambda=5.0)
print(f"task_0_acc: {r['task_0_accuracy_final']:.4f}")
print(f"task_1_acc: {r['task_1_accuracy_final']:.4f}")
print(f"forgetting: {r['forgetting_task_0']:.4f}")
