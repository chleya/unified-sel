"""
Phase 5：结构贝叶斯 - 用贝叶斯不确定性建模

核心思想：
当前问题：种子偏科严重，snapshot expert 只学会 task_0，current model 只学会 task_1
路由策略无法解决，因为两个专家都是"偏科"的

Phase 5 思路：
- 用贝叶斯方法建模每个结构的不确定性
- 在预测时，不仅考虑结构的输出，还考虑其不确定性
- 高不确定性的结构 → 降低其权重
- 低不确定性的结构 → 增加其权重

具体实现：
1. 对每个结构维护一个不确定性估计（基于历史预测误差）
2. 在路由时，使用不确定性加权：
   - snapshot_uncertainty 低 → 增加 snapshot 权重
   - current_uncertainty 低 → 增加 current 权重
3. 不确定性计算：滑动窗口内的预测误差方差

目标：
- 用贝叶斯不确定性建模来平衡两个专家
- 目标：avg_acc +0.05（从 0.5055 提升到 0.5555+）
- 时间限制：1-2 周
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


class BayesianStructurePool:
    """贝叶斯结构池 - 维护每个结构的不确定性估计"""
    
    def __init__(self, pool, window_size: int = 20):
        self.pool = pool
        self.window_size = window_size
        # 为每个结构维护一个预测误差历史
        self.error_history = {s.id: [] for s in pool.structures}
        # 不确定性估计（方差）
        self.uncertainty = {s.id: 1.0 for s in pool.structures}
    
    def update_uncertainty(self, structure_id: int, prediction: int, true_label: int):
        """更新结构的不确定性估计"""
        if structure_id not in self.error_history:
            self.error_history[structure_id] = []
            self.uncertainty[structure_id] = 1.0
        
        # 记录误差（0 或 1）
        error = 1 if prediction != true_label else 0
        self.error_history[structure_id].append(error)
        
        # 保持窗口大小
        if len(self.error_history[structure_id]) > self.window_size:
            self.error_history[structure_id].pop(0)
        
        # 计算不确定性（误差的方差）
        if len(self.error_history[structure_id]) >= 2:
            errors = np.array(self.error_history[structure_id])
            self.uncertainty[structure_id] = float(np.var(errors)) + 0.01  # 防止为 0
    
    def get_confidence(self, structure_id: int) -> float:
        """获取结构的置信度（1 - 不确定性）"""
        return 1.0 - self.uncertainty.get(structure_id, 1.0)
    
    def refresh_structures(self):
        """刷新结构列表（处理新创建的结构）"""
        for s in self.pool.structures:
            if s.id not in self.error_history:
                self.error_history[s.id] = []
                self.uncertainty[s.id] = 1.0  # 新结构初始不确定性高


def route_with_bayesian_uncertainty(
    clf: UnifiedSELClassifier,
    x: np.ndarray,
    snapshot: dict,
    bayesian_pool: BayesianStructurePool,
) -> int:
    """
    使用贝叶斯不确定性进行路由决策
    
    策略：
    1. 计算 snapshot expert 的加权不确定性
    2. 计算 current model 的加权不确定性
    3. 根据不确定性分配权重
    """
    snapshot_probs = clf._predict_with_snapshot(x, snapshot)
    current_probs = clf.predict_proba(x)
    
    # 计算 snapshot expert 的加权不确定性
    snapshot_uncertainty = 0.0
    total_utility = 0.0
    for s_data in snapshot["structures"]:
        utility = s_data.get("utility", 1.0)
        uncertainty = bayesian_pool.uncertainty.get(s_data["id"], 1.0)
        snapshot_uncertainty += utility * uncertainty
        total_utility += utility
    
    if total_utility > 0:
        snapshot_uncertainty /= total_utility
    
    # 计算 current model 的加权不确定性
    current_uncertainty = 0.0
    total_utility = 0.0
    for s in clf.pool.structures:
        utility = s.utility
        uncertainty = bayesian_pool.uncertainty.get(s.id, 1.0)
        current_uncertainty += utility * uncertainty
        total_utility += utility
    
    if total_utility > 0:
        current_uncertainty /= total_utility
    
    # 根据不确定性分配权重
    # 不确定性越低，权重越高
    snapshot_confidence = 1.0 - snapshot_uncertainty
    current_confidence = 1.0 - current_uncertainty
    
    total_confidence = snapshot_confidence + current_confidence
    if total_confidence > 0:
        snapshot_weight = snapshot_confidence / total_confidence
        current_weight = current_confidence / total_confidence
    else:
        snapshot_weight = 0.5
        current_weight = 0.5
    
    # 加权平均
    combined_probs = snapshot_weight * snapshot_probs + current_weight * current_probs
    return int(np.argmax(combined_probs))


def run_bayesian_validation(seed: int, config: NoBoundaryConfig) -> dict:
    """运行贝叶斯不确定性验证实验"""
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

    # 创建贝叶斯结构池
    bayesian_pool = BayesianStructurePool(clf.pool, window_size=20)

    for step in range(config.steps):
        if step < config.checkpoint_step:
            progress = 0.0
        else:
            progress = (step - config.checkpoint_step) / max(config.steps - config.checkpoint_step - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        
        # 训练前更新不确定性（使用当前模型的预测）
        if clf._snapshot_experts:
            # 获取当前模型预测
            pred = clf.predict(x)
            # 更新 current model 的不确定性
            for s in clf.pool.structures:
                bayesian_pool.update_uncertainty(s.id, pred, y)
        
        clf.fit_one(x, y)
        
        # 刷新结构列表（处理新创建的结构）
        bayesian_pool.refresh_structures()

        if step + 1 == config.checkpoint_step:
            clf.snapshot_expert(confidence_ratio_threshold=0.5)
            # 使用优化的结构池冻结策略：只冻结剪枝，允许创建新结构
            clf.freeze_pool_prune_only()
            print(f"  [seed {seed}] Created snapshot expert and FROZE pool (prune only) at step {step + 1}")

    # 评估完整系统（使用贝叶斯不确定性路由）
    # Task 0 评估
    task_0_correct = 0
    for i in range(len(X_task_0)):
        x = X_task_0[i]
        y = int(y_task_0[i])
        
        if clf._snapshot_experts:
            pred = route_with_bayesian_uncertainty(clf, x, clf._snapshot_experts[0], bayesian_pool)
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
            pred = route_with_bayesian_uncertainty(clf, x, clf._snapshot_experts[0], bayesian_pool)
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
    print("Phase 5：结构贝叶斯 - 用贝叶斯不确定性建模")
    print("=" * 80)
    print("\n核心思想：")
    print("  当前问题：种子偏科严重，snapshot expert 只学会 task_0，current model 只学会 task_1")
    print("  路由策略无法解决，因为两个专家都是'偏科'的")
    print("\n  Phase 5 思路：")
    print("    - 用贝叶斯方法建模每个结构的不确定性")
    print("    - 在预测时，不仅考虑结构的输出，还考虑其不确定性")
    print("    - 高不确定性的结构 → 降低其权重")
    print("    - 低不确定性的结构 → 增加其权重")
    print("\n  具体实现：")
    print("    1. 对每个结构维护一个不确定性估计（基于历史预测误差）")
    print("    2. 在路由时，使用不确定性加权")
    print("    3. 不确定性计算：滑动窗口内的预测误差方差")
    print("\n目标：")
    print("  - 用贝叶斯不确定性建模来平衡两个专家")
    print("  - 目标：avg_acc +0.05（从 0.5055 提升到 0.5555+）")
    print("  - 时间限制：1-2 周")
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
        result = run_bayesian_validation(seed, config)
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
        print("Phase 5 结果汇总")
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
        if avg_acc >= 0.5555:
            print("\n[SUCCESS] 达到目标！avg_acc +0.05+")
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
        "experiment": "phase5_bayesian_uncertainty",
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

    results_dir = get_results_path("phase5_bayesian_uncertainty")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到：{output_path}")


if __name__ == "__main__":
    main()
