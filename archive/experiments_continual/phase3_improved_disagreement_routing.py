"""
Phase 3：改进的 Disagreement 路由策略

核心思想：
Phase 1 和 Phase 2 失败的原因是路由策略过于简单。
现在回到最根本的 disagreement 路由，但使用更智能的策略：

1. 当 snapshot 和 current disagree 时，选择置信度高的
2. 但需要平衡两个任务的准确率，避免偏科

策略：
- 如果 snapshot 和 current 预测不一致：
  - 计算两者的置信度差异
  - 如果 snapshot 置信度显著高于 current → 使用 snapshot
  - 如果 current 置信度显著高于 snapshot → 使用 current
  - 否则 → 使用加权平均（基于置信度）

目标：
- 解决种子偏科问题（种子 7、9、10 的 task_1 差；种子 8、11 的 task_0 差）
- 目标：avg_acc +0.03（从 0.5055 提升到 0.5355+）
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


def compute_confidence(probs: np.ndarray) -> float:
    """计算预测的置信度（概率分布的动态范围）"""
    return float(np.max(probs) - np.min(probs))


def improved_disagreement_route(
    snapshot_probs: np.ndarray,
    current_probs: np.ndarray,
    confidence_threshold: float = 0.1,
) -> int:
    """
    改进的 disagreement 路由策略
    
    策略：
    1. 如果 snapshot 和 current 预测一致 → 直接返回
    2. 如果不一致：
       - 计算两者的置信度
       - 如果置信度差异显著 → 选择置信度高的
       - 否则 → 使用加权平均（基于置信度）
    """
    snapshot_pred = int(np.argmax(snapshot_probs))
    current_pred = int(np.argmax(current_probs))
    
    # 如果预测一致，直接返回
    if snapshot_pred == current_pred:
        return snapshot_pred
    
    # 计算置信度
    snapshot_conf = compute_confidence(snapshot_probs)
    current_conf = compute_confidence(current_probs)
    
    # 置信度差异
    conf_diff = abs(snapshot_conf - current_conf)
    
    # 如果置信度差异显著，选择置信度高的
    if conf_diff > confidence_threshold:
        if snapshot_conf > current_conf:
            return snapshot_pred
        else:
            return current_pred
    
    # 否则使用加权平均（基于置信度）
    total_conf = snapshot_conf + current_conf
    if total_conf > 0:
        snapshot_weight = snapshot_conf / total_conf
        current_weight = current_conf / total_conf
    else:
        snapshot_weight = 0.5
        current_weight = 0.5
    
    combined_probs = snapshot_weight * snapshot_probs + current_weight * current_probs
    return int(np.argmax(combined_probs))


def run_improved_disagreement(seed: int, config: NoBoundaryConfig) -> dict:
    """运行改进的 disagreement 路由实验"""
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
            # 使用优化的结构池冻结策略：只冻结剪枝，允许创建新结构
            clf.freeze_pool_prune_only()
            print(f"  [seed {seed}] Created snapshot expert and FROZE pool (prune only) at step {step + 1}")

    # 评估完整系统（使用改进的 disagreement 路由）
    # Task 0 评估
    task_0_correct = 0
    for i in range(len(X_task_0)):
        x = X_task_0[i]
        y = int(y_task_0[i])
        
        if clf._snapshot_experts:
            snapshot_probs = clf._predict_with_snapshot(x, clf._snapshot_experts[0])
            current_probs = clf.predict_proba(x)
            pred = improved_disagreement_route(snapshot_probs, current_probs, confidence_threshold=0.1)
        else:
            pred = clf.predict(x)
        
        if pred == y:
            task_0_correct += 1
    
    # Task 1 评估
    task_1_correct = 0
    for i in range(len(X_task_1)):
        x = X_task_1[i]
        y = int(y_task_1[i])
        
        if clf._snapshot_experts:
            snapshot_probs = clf._predict_with_snapshot(x, clf._snapshot_experts[0])
            current_probs = clf.predict_proba(x)
            pred = improved_disagreement_route(snapshot_probs, current_probs, confidence_threshold=0.1)
        else:
            pred = clf.predict(x)
        
        if pred == y:
            task_1_correct += 1
    
    final_task_0_accuracy = task_0_correct / len(X_task_0) if len(X_task_0) > 0 else 0.0
    final_task_1_accuracy = task_1_correct / len(X_task_1) if len(X_task_1) > 0 else 0.0
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
    print("Phase 3：改进的 Disagreement 路由策略")
    print("=" * 80)
    print("\n核心思想：")
    print("  Phase 1 和 Phase 2 失败的原因是路由策略过于简单。")
    print("  现在回到最根本的 disagreement 路由，但使用更智能的策略：")
    print("\n  策略：")
    print("    1. 当 snapshot 和 current disagree 时，选择置信度高的")
    print("    2. 但需要平衡两个任务的准确率，避免偏科")
    print("\n目标：")
    print("  - 解决种子偏科问题（种子 7、9、10 的 task_1 差；种子 8、11 的 task_0 差）")
    print("  - 目标：avg_acc +0.03（从 0.5055 提升到 0.5355+）")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    config.readout_mode = "hybrid_local"
    # 增加最大结构数，以允许更多新结构创建
    config.pool.max_structures = 16
    seeds = [7, 8, 9, 10, 11]  # 5 个种子

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_improved_disagreement(seed, config)
        results.append(result)
        
        print(f"  task_0 准确率：{result['task_0_accuracy_final']:.4f}")
        print(f"  task_1 准确率：{result['task_1_accuracy_final']:.4f}")
        print(f"  平均准确率：{result['avg_accuracy']:.4f}")
        print(f"  Snapshot expert task_0 准确率：{result['snap_acc_t0']:.4f}")
        print(f"  最终结构数：{result['n_structures']}")

    # 汇总分析
    if results:
        avg_task_0 = np.mean([r['task_0_accuracy_final'] for r in results])
        avg_task_1 = np.mean([r['task_1_accuracy_final'] for r in results])
        avg_acc = np.mean([r['avg_accuracy'] for r in results])
        std_acc = np.std([r['avg_accuracy'] for r in results])
        avg_snap_t0 = np.mean([r['snap_acc_t0'] for r in results])

        print("\n" + "=" * 80)
        print("Phase 3 结果汇总")
        print("=" * 80)
        print(f"平均 task_0 准确率：{avg_task_0:.4f}")
        print(f"平均 task_1 准确率：{avg_task_1:.4f}")
        print(f"平均准确率：{avg_acc:.4f} ± {std_acc:.4f}")
        print(f"平均 Snapshot expert task_0 准确率：{avg_snap_t0:.4f}")
        
        print(f"\n对比之前的结果：")
        print(f"  之前平均准确率：0.5055")
        print(f"  现在平均准确率：{avg_acc:.4f}")
        print(f"  提升：{avg_acc - 0.5055:+.4f}")

        # 检查是否达到目标
        if avg_acc >= 0.5355:
            print("\n[SUCCESS] 达到目标！avg_acc +0.03+")
        else:
            print("\n[FAILED] 未达到目标，需要进一步优化")

    # 计算统计显著性（简单的 t 检验）
    from scipy import stats
    ewc_baseline = 0.5005
    unified_accs = [r['avg_accuracy'] for r in results]
    t_stat, p_value = stats.ttest_1samp(unified_accs, ewc_baseline)
    print(f"\n统计检验：")
    print(f"  t 统计量：{t_stat:.4f}")
    print(f"  p 值：{p_value:.4f}")
    
    if p_value < 0.05:
        print("  [STATISTICALLY SIGNIFICANT] p < 0.05")
    else:
        print("  [NOT STATISTICALLY SIGNIFICANT] p >= 0.05")

    output = {
        "experiment": "phase3_improved_disagreement_routing",
        "seeds": seeds,
        "results": results,
        "summary": {
            "avg_task_0": avg_task_0 if results else 0.0,
            "avg_task_1": avg_task_1 if results else 0.0,
            "avg_acc": avg_acc if results else 0.0,
            "std_acc": std_acc if results else 0.0,
            "avg_snap_t0": avg_snap_t0 if results else 0.0,
            "t_stat": t_stat if results else 0.0,
            "p_value": p_value if results else 0.0,
            "ewc_baseline": ewc_baseline,
            "previous_avg_acc": 0.5055,
            "improvement": avg_acc - 0.5055 if results else 0.0,
        },
    }

    results_dir = get_results_path("phase3_improved_disagreement_routing")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到：{output_path}")


if __name__ == "__main__":
    main()
