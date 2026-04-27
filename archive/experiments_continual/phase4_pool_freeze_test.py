"""
Phase 4: 验证结构池冻结的效果

核心思想：
- 在 checkpoint 后冻结结构池
- 禁止创建新结构和剪枝旧结构
- 只更新现有结构的权重
- 这样 snapshot expert 的结构就不会被剪枝

预期效果：
- Snapshot expert task_0 准确率应该高于 0.6706（Phase 2 的结果）
- 接近 Oracle 上界的 0.8438
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


def run_pool_freeze_test(seed: int, config: NoBoundaryConfig, freeze_pool: bool = True) -> dict:
    """测试结构池冻结效果"""
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

    # 记录结构池状态
    structures_at_start = []
    structures_at_checkpoint = []
    structures_at_end = []

    for step in range(config.steps):
        if step < config.checkpoint_step:
            progress = 0.0
        else:
            progress = (step - config.checkpoint_step) / max(config.steps - config.checkpoint_step - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)

        # 记录结构池状态
        if step == 0:
            structures_at_start = [s.id for s in clf.pool.structures]
        
        if step + 1 == config.checkpoint_step:
            structures_at_checkpoint = [s.id for s in clf.pool.structures]
            clf.snapshot_expert(confidence_ratio_threshold=1.0)
            if freeze_pool:
                clf.freeze_pool()
                print(f"  [seed {seed}] Created snapshot expert and FROZE pool at step {step + 1}")
            else:
                print(f"  [seed {seed}] Created snapshot expert WITHOUT freezing pool at step {step + 1}")

    # 最终记录结构池状态
    structures_at_end = [s.id for s in clf.pool.structures]

    # 最终评估
    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)

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

    # 检查结构池变化
    structures_pruned = set(structures_at_checkpoint) - set(structures_at_end)
    structures_created = set(structures_at_end) - set(structures_at_checkpoint)

    return {
        "seed": seed,
        "freeze_pool": freeze_pool,
        "task_0_accuracy_final": final_task_0_accuracy,
        "task_1_accuracy_final": final_task_1_accuracy,
        "snap_acc_t0": snap_acc_t0,
        "snap_acc_t1": snap_acc_t1,
        "structures_at_checkpoint": structures_at_checkpoint,
        "structures_at_end": structures_at_end,
        "structures_pruned": list(structures_pruned),
        "structures_created": list(structures_created),
        "n_structures_pruned": len(structures_pruned),
        "n_structures_created": len(structures_created),
    }


def main():
    print("=" * 80)
    print("Phase 4: 验证结构池冻结的效果")
    print("=" * 80)
    print("\n策略：")
    print("  - 在 checkpoint 后冻结结构池")
    print("  - 禁止创建新结构和剪枝旧结构")
    print("  - 只更新现有结构的权重")
    print("\n预期效果：")
    print("  - 结构池不变，snapshot expert 的结构不会被剪枝")
    print("  - Snapshot expert task_0 准确率应该高于 0.6706")
    print("  - 接近 Oracle 上界的 0.8438")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    config.readout_mode = "hybrid_local"
    seeds = [7, 8, 9]

    # 测试冻结和不冻结两种情况
    results_freeze = []
    results_no_freeze = []

    for seed in seeds:
        print(f"\n运行 seed {seed} (冻结结构池)...")
        result_freeze = run_pool_freeze_test(seed, config, freeze_pool=True)
        results_freeze.append(result_freeze)
        
        print(f"  Snapshot expert task_0 acc: {result_freeze['snap_acc_t0']:.4f}")
        print(f"  结构剪枝: {result_freeze['n_structures_pruned']}")
        print(f"  结构创建: {result_freeze['n_structures_created']}")
        print(f"  结构池变化: {len(result_freeze['structures_at_checkpoint'])} → {len(result_freeze['structures_at_end'])}")

        print(f"\n运行 seed {seed} (不冻结结构池)...")
        result_no_freeze = run_pool_freeze_test(seed, config, freeze_pool=False)
        results_no_freeze.append(result_no_freeze)
        
        print(f"  Snapshot expert task_0 acc: {result_no_freeze['snap_acc_t0']:.4f}")
        print(f"  结构剪枝: {result_no_freeze['n_structures_pruned']}")
        print(f"  结构创建: {result_no_freeze['n_structures_created']}")
        print(f"  结构池变化: {len(result_no_freeze['structures_at_checkpoint'])} → {len(result_no_freeze['structures_at_end'])}")

    # 汇总分析
    avg_snap_t0_freeze = np.mean([r['snap_acc_t0'] for r in results_freeze])
    avg_snap_t0_no_freeze = np.mean([r['snap_acc_t0'] for r in results_no_freeze])
    avg_pruned_freeze = np.mean([r['n_structures_pruned'] for r in results_freeze])
    avg_pruned_no_freeze = np.mean([r['n_structures_pruned'] for r in results_no_freeze])

    print("\n" + "=" * 80)
    print("Phase 4 结果汇总")
    print("=" * 80)
    print(f"\n结构池冻结:")
    print(f"  平均 Snapshot expert task_0 准确率: {avg_snap_t0_freeze:.4f}")
    print(f"  平均结构剪枝数: {avg_pruned_freeze:.2f}")
    print(f"  平均结构创建数: {np.mean([r['n_structures_created'] for r in results_freeze]):.2f}")
    
    print(f"\n结构池不冻结:")
    print(f"  平均 Snapshot expert task_0 准确率: {avg_snap_t0_no_freeze:.4f}")
    print(f"  平均结构剪枝数: {avg_pruned_no_freeze:.2f}")
    print(f"  平均结构创建数: {np.mean([r['n_structures_created'] for r in results_no_freeze]):.2f}")
    
    print(f"\n对比 Phase 2 结果：")
    print(f"  Phase 2 snapshot task_0 准确率: 0.6706")
    print(f"  结构池冻结后: {avg_snap_t0_freeze:.4f} (变化: {avg_snap_t0_freeze - 0.6706:+.4f})")

    if avg_snap_t0_freeze > avg_snap_t0_no_freeze:
        print("\n[SUCCESS] 结构池冻结有效！")
        print(f"  Snapshot expert 准确率从 {avg_snap_t0_no_freeze:.4f} 提升到 {avg_snap_t0_freeze:.4f}")
    else:
        print("\n[FAILED] 结构池冻结无效")
        print(f"  Snapshot expert 准确率没有提升")

    output = {
        "experiment": "phase4_pool_freeze_test",
        "seeds": seeds,
        "results_freeze": results_freeze,
        "results_no_freeze": results_no_freeze,
        "summary": {
            "avg_snap_t0_freeze": avg_snap_t0_freeze,
            "avg_snap_t0_no_freeze": avg_snap_t0_no_freeze,
            "avg_pruned_freeze": avg_pruned_freeze,
            "avg_pruned_no_freeze": avg_pruned_no_freeze,
        },
    }

    results_dir = get_results_path("phase4_pool_freeze")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
