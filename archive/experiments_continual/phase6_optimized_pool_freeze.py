"""
Phase 6: 分析种子 8 和 9 的问题并优化结构池冻结策略

核心问题：
- 种子 8：task_0=0.8789, task_1=0.0977，平均=0.4883
- 种子 9：task_0=0.9648, task_1=0.0898，平均=0.5273
- 结构池冻结限制了对 task_1 的适应能力

优化策略：
- 允许在 task_1 训练时创建新结构，但不剪枝旧结构
- 这样既能保持 snapshot expert 的结构稳定，又能适应 task_1
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


def run_optimized_test(seed: int, config: NoBoundaryConfig, allow_create: bool = True) -> dict:
    """测试优化的结构池冻结策略"""
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

    for step in range(config.steps):
        if step < config.checkpoint_step:
            progress = 0.0
        else:
            progress = (step - config.checkpoint_step) / max(config.steps - config.checkpoint_step - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)

        if step + 1 == config.checkpoint_step:
            clf.snapshot_expert(confidence_ratio_threshold=0.5)
            # 优化：只冻结剪枝，允许创建新结构
            clf.freeze_pool_prune_only()
            print(f"  [seed {seed}] Created snapshot expert and FROZE pool (prune only) at step {step + 1}")

    # 评估完整系统（使用 _ensemble_predict）
    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)
    final_avg_accuracy = (final_task_0_accuracy + final_task_1_accuracy) / 2

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

    # 记录结构池状态
    final_structure_ids = [s.id for s in clf.pool.structures]
    n_structures = len(final_structure_ids)

    return {
        "seed": seed,
        "task_0_accuracy_final": final_task_0_accuracy,
        "task_1_accuracy_final": final_task_1_accuracy,
        "avg_accuracy": final_avg_accuracy,
        "snap_acc_t0": snap_acc_t0,
        "snap_acc_t1": snap_acc_t1,
        "n_structures": n_structures,
        "final_structure_ids": final_structure_ids,
    }


def main():
    print("=" * 80)
    print("Phase 6: 分析种子 8 和 9 的问题并优化结构池冻结策略")
    print("=" * 80)
    print("\n问题分析：")
    print("  - 种子 8：task_0=0.8789, task_1=0.0977，平均=0.4883")
    print("  - 种子 9：task_0=0.9648, task_1=0.0898，平均=0.5273")
    print("  - 结构池冻结限制了对 task_1 的适应能力")
    print("\n优化策略：")
    print("  - 允许在 task_1 训练时创建新结构，但不剪枝旧结构")
    print("  - 这样既能保持 snapshot expert 的结构稳定，又能适应 task_1")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    config.readout_mode = "hybrid_local"
    # 增加最大结构数，以允许更多新结构创建
    config.pool.max_structures = 16
    seeds = [8, 9]  # 重点分析这两个种子

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_optimized_test(seed, config, allow_create=True)
        results.append(result)
        
        print(f"  task_0 准确率: {result['task_0_accuracy_final']:.4f}")
        print(f"  task_1 准确率: {result['task_1_accuracy_final']:.4f}")
        print(f"  平均准确率: {result['avg_accuracy']:.4f}")
        print(f"  Snapshot expert task_0 准确率: {result['snap_acc_t0']:.4f}")
        print(f"  最终结构数: {result['n_structures']}")

    # 汇总分析
    if results:
        avg_task_0 = np.mean([r['task_0_accuracy_final'] for r in results])
        avg_task_1 = np.mean([r['task_1_accuracy_final'] for r in results])
        avg_acc = np.mean([r['avg_accuracy'] for r in results])
        avg_snap_t0 = np.mean([r['snap_acc_t0'] for r in results])

        print("\n" + "=" * 80)
        print("Phase 6 结果汇总")
        print("=" * 80)
        print(f"平均 task_0 准确率: {avg_task_0:.4f}")
        print(f"平均 task_1 准确率: {avg_task_1:.4f}")
        print(f"平均准确率: {avg_acc:.4f}")
        print(f"平均 Snapshot expert task_0 准确率: {avg_snap_t0:.4f}")
        
        print(f"\n对比之前的结果：")
        print(f"  种子 8 之前 task_1 准确率: 0.0977")
        print(f"  种子 9 之前 task_1 准确率: 0.0898")
        print(f"  现在平均 task_1 准确率: {avg_task_1:.4f}")

    output = {
        "experiment": "phase6_optimized_pool_freeze",
        "seeds": seeds,
        "results": results,
        "summary": {
            "avg_task_0": avg_task_0 if results else 0.0,
            "avg_task_1": avg_task_1 if results else 0.0,
            "avg_acc": avg_acc if results else 0.0,
            "avg_snap_t0": avg_snap_t0 if results else 0.0,
        },
    }

    results_dir = get_results_path("phase6_optimized_pool_freeze")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
