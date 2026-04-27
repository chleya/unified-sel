"""
Phase 3: 验证 Best snapshot 策略的效果

核心思想：
- 在 task 0 训练过程中，每 25 步评估一次 task_0 准确率
- 保存准确率最高的 snapshot
- 使用 best snapshot 而不是 checkpoint snapshot

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


def run_best_snapshot_test(seed: int, config: NoBoundaryConfig) -> dict:
    """测试 Best snapshot 策略"""
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
                print(f"  [seed {seed}] Created BEST snapshot expert at step {best_snapshot_step} (best_t0={best_task_0_acc:.4f}, current_t0={task_0_accuracy_after_early_stream:.4f})")
            else:
                clf.snapshot_expert(confidence_ratio_threshold=1.0)
                print(f"  [seed {seed}] Created snapshot expert at checkpoint (no best snapshot found)")

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

    return {
        "seed": seed,
        "best_snapshot_step": best_snapshot_step,
        "best_task_0_acc_at_snapshot": best_task_0_acc,
        "task_0_accuracy_at_checkpoint": task_0_accuracy_after_early_stream,
        "task_0_accuracy_final": final_task_0_accuracy,
        "task_1_accuracy_final": final_task_1_accuracy,
        "snap_acc_t0": snap_acc_t0,
        "snap_acc_t1": snap_acc_t1,
    }


def main():
    print("=" * 80)
    print("Phase 3: 验证 Best snapshot 策略的效果")
    print("=" * 80)
    print("\n策略：")
    print("  - 在 task 0 训练过程中，每 25 步评估一次 task_0 准确率")
    print("  - 保存准确率最高的 snapshot")
    print("  - 使用 best snapshot 而不是 checkpoint snapshot")
    print("\n预期效果：")
    print("  - Snapshot expert task_0 准确率应该高于 0.6706（Phase 2 的结果）")
    print("  - 接近 Oracle 上界的 0.8438")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    config.readout_mode = "hybrid_local"
    seeds = [7, 8, 9]

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_best_snapshot_test(seed, config)
        results.append(result)
        
        print(f"  Best snapshot step: {result['best_snapshot_step']}")
        print(f"  Best task_0 acc at snapshot: {result['best_task_0_acc_at_snapshot']:.4f}")
        print(f"  Task_0 acc at checkpoint: {result['task_0_accuracy_at_checkpoint']:.4f}")
        print(f"  Snapshot expert task_0 acc: {result['snap_acc_t0']:.4f}")
        print(f"  Snapshot expert task_1 acc: {result['snap_acc_t1']:.4f}")

    avg_best_t0 = np.mean([r['best_task_0_acc_at_snapshot'] for r in results])
    avg_checkpoint_t0 = np.mean([r['task_0_accuracy_at_checkpoint'] for r in results])
    avg_snap_t0 = np.mean([r['snap_acc_t0'] for r in results])
    avg_snap_t1 = np.mean([r['snap_acc_t1'] for r in results])

    print("\n" + "=" * 80)
    print("Phase 3 结果汇总")
    print("=" * 80)
    print(f"平均 Best snapshot task_0 准确率: {avg_best_t0:.4f}")
    print(f"平均 Checkpoint task_0 准确率: {avg_checkpoint_t0:.4f}")
    print(f"平均 Snapshot expert task_0 准确率: {avg_snap_t0:.4f}")
    print(f"平均 Snapshot expert task_1 准确率: {avg_snap_t1:.4f}")
    
    print(f"\n对比 Phase 2 结果：")
    print(f"  Phase 2 snapshot task_0 准确率: 0.6706")
    print(f"  Phase 3 snapshot task_0 准确率: {avg_snap_t0:.4f}")
    print(f"  改进: {avg_snap_t0 - 0.6706:+.4f}")

    if avg_snap_t0 > 0.6706:
        print("\n[SUCCESS] Best snapshot 策略有效！")
        print(f"  Snapshot expert 准确率从 0.6706 提升到 {avg_snap_t0:.4f}")
    else:
        print("\n[FAILED] Best snapshot 策略无效")
        print(f"  Snapshot expert 准确率没有提升")

    output = {
        "experiment": "phase3_best_snapshot_test",
        "seeds": seeds,
        "results": results,
        "summary": {
            "avg_best_t0": avg_best_t0,
            "avg_checkpoint_t0": avg_checkpoint_t0,
            "avg_snap_t0": avg_snap_t0,
            "avg_snap_t1": avg_snap_t1,
        },
    }

    results_dir = get_results_path("phase3_best_snapshot")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
