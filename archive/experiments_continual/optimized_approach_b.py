"""
方案 B 优化版：多 snapshot expert + 智能路由

核心发现：
- 方案 B（多 snapshot expert）的最终准确率确实提高了
- seed 7: task_0=0.558, task_1=0.519, avg=0.539（两个任务都学会了！）

优化方向：
1. 保存更多 snapshot（覆盖不同阶段）
2. 改进路由策略（基于任务特异性）
3. 不冻结结构池，允许自由演化

目标：
- avg_acc > 0.55（显著超越 EWC 的 0.5005）
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


def smart_route(clf: UnifiedSELClassifier, x: np.ndarray, snapshots: list) -> int:
    """
    智能路由：基于专家特异性
    
    策略：
    1. 计算每个 expert 对两个任务的特异性
    2. 选择对当前输入最特异的 expert
    """
    if not snapshots:
        return clf.predict(x)
    
    # 获取所有 expert 的预测
    expert_preds = []
    expert_probs = []
    
    # Current model
    current_probs = clf.predict_proba(x)
    expert_preds.append(int(np.argmax(current_probs)))
    expert_probs.append(current_probs)
    
    # Snapshot experts
    for snapshot in snapshots:
        snapshot_probs = clf._predict_with_snapshot(x, snapshot)
        expert_preds.append(int(np.argmax(snapshot_probs)))
        expert_probs.append(snapshot_probs)
    
    # 计算每个 expert 的置信度
    confidences = []
    for probs in expert_probs:
        conf = float(np.max(probs) - np.min(probs))
        confidences.append(conf)
    
    # 选择置信度最高的 expert
    best_idx = int(np.argmax(confidences))
    return expert_preds[best_idx]


def run_optimized_approach_b(seed: int, config: NoBoundaryConfig) -> dict:
    """运行优化的方案 B"""
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

    snapshots = []
    saved_snapshots_at = []

    # 训练过程
    for step in range(config.steps):
        # progress 曲线：更早开始增加 task_1 比例
        if step < 100:
            progress = 0.0
        else:
            progress = (step - 100) / max(config.steps - 100 - 1, 1)
        
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)

        # 每 100 步保存一个 snapshot（不冻结）
        if step > 0 and step % 100 == 0 and len(snapshots) < 5:
            clf.snapshot_expert(confidence_ratio_threshold=0.5)
            snapshots.append(clf._snapshot_experts[-1])
            saved_snapshots_at.append(step)
            print(f"  [seed {seed}] Step {step}: 保存 snapshot #{len(snapshots)}")

    # 最终评估
    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)
    final_avg_accuracy = (final_task_0_accuracy + final_task_1_accuracy) / 2

    # 使用智能路由评估
    smart_task_0_correct = 0
    smart_task_1_correct = 0
    
    for i in range(len(X_task_0)):
        x = X_task_0[i]
        y = int(y_task_0[i])
        pred = smart_route(clf, x, snapshots)
        if pred == y:
            smart_task_0_correct += 1
    
    for i in range(len(X_task_1)):
        x = X_task_1[i]
        y = int(y_task_1[i])
        pred = smart_route(clf, x, snapshots)
        if pred == y:
            smart_task_1_correct += 1
    
    smart_task_0_accuracy = smart_task_0_correct / len(X_task_0) if len(X_task_0) > 0 else 0.0
    smart_task_1_accuracy = smart_task_1_correct / len(X_task_1) if len(X_task_1) > 0 else 0.0
    smart_avg_accuracy = (smart_task_0_accuracy + smart_task_1_accuracy) / 2

    return {
        "seed": seed,
        "saved_snapshots_at": saved_snapshots_at,
        "n_snapshots": len(snapshots),
        "final": {
            "task_0_accuracy": final_task_0_accuracy,
            "task_1_accuracy": final_task_1_accuracy,
            "avg_accuracy": final_avg_accuracy,
        },
        "smart_route": {
            "task_0_accuracy": smart_task_0_accuracy,
            "task_1_accuracy": smart_task_1_accuracy,
            "avg_accuracy": smart_avg_accuracy,
        },
    }


def main():
    print("=" * 80)
    print("方案 B 优化版：多 snapshot expert + 智能路由")
    print("=" * 80)
    print("\n优化方向：")
    print("  1. 保存更多 snapshot（覆盖不同阶段）")
    print("  2. 改进路由策略（基于专家特异性）")
    print("  3. 不冻结结构池，允许自由演化")
    print("\n目标：")
    print("  - avg_acc > 0.55（显著超越 EWC 的 0.5005）")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.readout_mode = "hybrid_local"
    config.pool.max_structures = 16
    seeds = [7, 8, 9, 10, 11]  # 5 个种子

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_optimized_approach_b(seed, config)
        results.append(result)
        
        print(f"  保存 snapshot 的 step: {result['saved_snapshots_at']}")
        print(f"  最终平均准确率：{result['final']['avg_accuracy']:.4f}")
        print(f"  智能路由平均准确率：{result['smart_route']['avg_accuracy']:.4f}")

    # 汇总
    avg_final_acc = np.mean([r['final']['avg_accuracy'] for r in results])
    avg_smart_acc = np.mean([r['smart_route']['avg_accuracy'] for r in results])
    
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    print(f"平均最终准确率：{avg_final_acc:.4f}")
    print(f"平均智能路由准确率：{avg_smart_acc:.4f}")
    
    print(f"\n对比 EWC 基线：")
    print(f"  EWC 平均准确率：0.5005")
    print(f"  最终准确率：{avg_final_acc:.4f} (差异：{avg_final_acc - 0.5005:+.4f})")
    print(f"  智能路由准确率：{avg_smart_acc:.4f} (差异：{avg_smart_acc - 0.5005:+.4f})")
    
    if avg_final_acc > 0.55:
        print("\n[SUCCESS] 最终准确率达到目标！avg_acc > 0.55")
    elif avg_smart_acc > 0.55:
        print("\n[PARTIAL SUCCESS] 智能路由达到目标！avg_acc > 0.55")
    else:
        print("\n[FAILED] 未达到目标，需要进一步优化")

    # 统计显著性
    from scipy import stats
    ewc_baseline = 0.5005
    final_accs = [r['final']['avg_accuracy'] for r in results]
    t_stat, p_value = stats.ttest_1samp(final_accs, ewc_baseline)
    print(f"\n统计检验（最终准确率）：")
    print(f"  t 统计量：{t_stat:.4f}")
    print(f"  p 值：{p_value:.4f}")
    
    if p_value < 0.05:
        print("  [STATISTICALLY SIGNIFICANT] p < 0.05")
    else:
        print("  [NOT STATISTICALLY SIGNIFICANT] p >= 0.05")

    output = {
        "experiment": "optimized_approach_b",
        "seeds": seeds,
        "results": results,
        "summary": {
            "avg_final_acc": avg_final_acc,
            "avg_smart_acc": avg_smart_acc,
            "ewc_baseline": ewc_baseline,
            "t_stat": t_stat,
            "p_value": p_value,
        },
    }

    results_dir = get_results_path("optimized_approach_b")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到：{output_path}")


if __name__ == "__main__":
    main()
