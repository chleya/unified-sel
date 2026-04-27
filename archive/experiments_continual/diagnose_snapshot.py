"""Diagnose snapshot expert prediction quality — compare snapshot vs current model."""
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

    X_task_0, y_task_0 = make_eval_task(0, 256, seed + 1000)

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

    for step in range(200):
        progress = step / config.checkpoint_step
        x, y = stream_sample(progress, rng, config.in_size)
        clf.fit_one(x, y)

    acc_before = clf.accuracy(X_task_0, y_task_0)
    print(f"Task 0 accuracy at step 200: {acc_before:.4f}")

    clf.snapshot_expert(surprise_threshold=0.0)
    snap = clf._snapshot_experts[0]

    snap_correct = 0
    current_correct = 0
    total = len(X_task_0)

    for i in range(total):
        x = X_task_0[i]
        current_probs = clf.predict_proba_single(x)
        current_pred = np.argmax(current_probs)
        if current_pred == y_task_0[i]:
            current_correct += 1

        snap_probs = clf._predict_with_snapshot(x, snap)
        snap_pred = np.argmax(snap_probs)
        if snap_pred == y_task_0[i]:
            snap_correct += 1

    print(f"Snapshot expert accuracy on task 0: {snap_correct/total:.4f}")
    print(f"Current model accuracy on task 0:   {current_correct/total:.4f}")
    print(f"clf.accuracy() at step 200:         {acc_before:.4f}")

    print(f"\nSnapshot W_out shape: {snap['W_out'].shape}")
    print(f"Snapshot num structures: {len(snap['structures'])}")
    print(f"Current num structures: {len(clf.pool.structures)}")

    w_diff = np.linalg.norm(clf.W_out - snap['W_out'])
    print(f"W_out difference (Frobenius norm): {w_diff:.6f}")

    for s_data in snap['structures']:
        curr_s = None
        for s in clf.pool.structures:
            if s.id == s_data['id']:
                curr_s = s
                break
        if curr_s is not None:
            w_diff_s = np.linalg.norm(curr_s.weights - s_data['weights'])
            print(f"  Structure {s_data['id']}: weight_diff={w_diff_s:.6f}")

    print(f"\n=== Key question: why is snapshot accuracy ({snap_correct/total:.4f}) != clf.accuracy() ({acc_before:.4f})? ===")


if __name__ == "__main__":
    main()
