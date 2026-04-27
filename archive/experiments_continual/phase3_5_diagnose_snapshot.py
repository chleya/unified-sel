"""
Phase 3.5: 诊断 snapshot expert 为什么准确率下降

核心问题：
- Best snapshot 保存时准确率 0.8398
- 但最终 snapshot expert 准确率只有 0.6706
- 为什么会下降？

可能原因：
1. 结构池在 task 1 训练时发生了变化（新结构创建、旧结构剪枝）
2. Snapshot expert 保存的结构 ID 与当前结构池不匹配
3. Snapshot expert 无法找到对应的结构
"""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_config import NoBoundaryConfig
from core.learner import UnifiedSELClassifier
from core.runtime import get_results_path, save_json, timestamp


DEFAULT_CONFIG = NoBoundaryConfig()


def make_eval_task(
    task_id: int,
    n_samples: int,
    seed: int,
    in_size: int = DEFAULT_CONFIG.in_size,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, in_size))
    boundary = X[:, 0] + X[:, 1]
    if task_id == 0:
        y = (boundary > 0.0).astype(int)
    elif task_id == 1:
        y = (boundary < 0.0).astype(int)
    else:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return X, y


def stream_sample(
    progress: float,
    rng: np.random.Generator,
    in_size: int = DEFAULT_CONFIG.in_size,
) -> tuple[np.ndarray, int]:
    x = rng.normal(0.0, 1.0, size=in_size)
    boundary = float(x[0] + x[1])
    task_0_label = int(boundary > 0.0)
    task_1_label = int(boundary < 0.0)
    y = task_0_label if rng.random() > progress else task_1_label
    return x, y


def diagnose_snapshot_expert(seed: int, config: NoBoundaryConfig) -> dict:
    """诊断 snapshot expert 为什么准确率下降"""
    rng = np.random.default_rng(seed)
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
    )

    X_task_0, y_task_0 = make_eval_task(
        task_id=0,
        n_samples=config.eval_samples_per_task,
        seed=seed + 1000,
        in_size=config.in_size,
    )
    X_task_1, y_task_1 = make_eval_task(
        task_id=1,
        n_samples=config.eval_samples_per_task,
        seed=seed + 2000,
        in_size=config.in_size,
    )

    best_task_0_acc = 0.0
    best_snapshot_state = None
    best_snapshot_step = 0
    best_snapshot_structure_ids = []

    for step in range(config.steps):
        if step < config.checkpoint_step:
            progress = 0.0
        else:
            progress = (step - config.checkpoint_step) / max(config.steps - config.checkpoint_step - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)

        # Best snapshot 逻辑
        if step < config.checkpoint_step and (step + 1) % 25 == 0:
            current_t0_acc = clf.accuracy(X_task_0, y_task_0)
            if current_t0_acc > best_task_0_acc:
                best_task_0_acc = current_t0_acc
                best_snapshot_step = step + 1
                best_snapshot_state = {
                    "W_out": clf.W_out.copy(),
                    "structures": [],
                }
                best_snapshot_structure_ids = [s.id for s in clf.pool.structures]
                for s in clf.pool.structures:
                    best_snapshot_state["structures"].append({
                        "id": s.id,
                        "weights": s.weights.copy(),
                        "feedback": s.feedback.copy(),
                        "local_readout": s.local_readout.copy() if s.local_readout is not None else None,
                    })

        if step + 1 == config.checkpoint_step:
            task_0_accuracy_after_early_stream = clf.accuracy(X_task_0, y_task_0)
            if best_snapshot_state is not None:
                clf._snapshot_experts.append(best_snapshot_state)
                clf._snapshot_confidence_ratio_threshold = 1.0
                print(f"  [seed {seed}] Created BEST snapshot expert at step {best_snapshot_step}")
                print(f"    Snapshot structure IDs: {best_snapshot_structure_ids}")
                print(f"    Current structure IDs: {[s.id for s in clf.pool.structures]}")

    # 最终评估
    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)

    # 检查结构池变化
    final_structure_ids = [s.id for s in clf.pool.structures]
    print(f"  [seed {seed}] Final structure IDs: {final_structure_ids}")
    print(f"    Snapshot structures: {len(best_snapshot_structure_ids)}")
    print(f"    Final structures: {len(final_structure_ids)}")
    
    # 检查 snapshot expert 是否能找到对应的结构
    if clf._snapshot_experts:
        snapshot_struct_ids = [s["id"] for s in clf._snapshot_experts[0]["structures"]]
        print(f"    Snapshot structure IDs in snapshot: {snapshot_struct_ids}")
        
        # 检查是否有结构被剪枝
        pruned_ids = set(best_snapshot_structure_ids) - set(final_structure_ids)
        if pruned_ids:
            print(f"    [WARNING] Pruned structures: {pruned_ids}")
        
        # 检查是否有新结构被创建
        new_ids = set(final_structure_ids) - set(best_snapshot_structure_ids)
        if new_ids:
            print(f"    [INFO] New structures: {new_ids}")

    # 评估 snapshot expert 的质量
    snap_correct_t0 = 0
    snap_correct_t1 = 0
    
    if clf._snapshot_experts:
        for i in range(len(X_task_0)):
            x = X_task_0[i]
            y = int(y_task_0[i])
            snap_pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            if snap_pred == y:
                snap_correct_t0 += 1
        
        for i in range(len(X_task_1)):
            x = X_task_1[i]
            y = int(y_task_1[i])
            snap_pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            if snap_pred == y:
                snap_correct_t1 += 1
    
    snap_acc_t0 = snap_correct_t0 / len(X_task_0) if len(X_task_0) > 0 else 0.0
    snap_acc_t1 = snap_correct_t1 / len(X_task_1) if len(X_task_1) > 0 else 0.0

    return {
        "seed": seed,
        "best_snapshot_step": best_snapshot_step,
        "best_task_0_acc_at_snapshot": best_task_0_acc,
        "task_0_accuracy_at_checkpoint": task_0_accuracy_after_early_stream,
        "task_0_accuracy_final": final_task_0_accuracy,
        "task_1_accuracy_final": final_task_1_accuracy,
        "snap_acc_t0": snap_acc_t0,
        "snap_acc_t1": snap_acc_t1,
        "snapshot_structure_ids": best_snapshot_structure_ids,
        "final_structure_ids": final_structure_ids,
    }


def main():
    print("=" * 80)
    print("Phase 3.5: 诊断 snapshot expert 为什么准确率下降")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    config.readout_mode = "hybrid_local"
    seeds = [7, 8, 9]

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = diagnose_snapshot_expert(seed, config)
        results.append(result)

    print("\n" + "=" * 80)
    print("Phase 3.5 诊断结果")
    print("=" * 80)
    
    for r in results:
        print(f"\nSeed {r['seed']}:")
        print(f"  Best snapshot task_0 acc: {r['best_task_0_acc_at_snapshot']:.4f}")
        print(f"  Snapshot expert task_0 acc: {r['snap_acc_t0']:.4f}")
        print(f"  差异: {r['snap_acc_t0'] - r['best_task_0_acc_at_snapshot']:+.4f}")
        print(f"  Snapshot structures: {len(r['snapshot_structure_ids'])}")
        print(f"  Final structures: {len(r['final_structure_ids'])}")


if __name__ == "__main__":
    main()
