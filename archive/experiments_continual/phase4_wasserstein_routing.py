"""
Phase 4：Wasserstein 信号 - 使用更强的拓扑信号

核心思想：
TopoMem 已经验证了 surprise + tension → decide_action 的有效性。
- surprise = 1.0 - max_similarity_to_any_adapter
- tension = mean(Wasserstein_distance(fp[i], fp[i-1]) for last N)

现在将这两个信号接入 Unified-SEL 的路由协议：
- 高 surprise + 低 tension → 使用 snapshot expert（新输入但系统稳定）
- 低 surprise + 低 tension → 使用 current model（熟悉输入且系统稳定）
- 高 tension → 系统不稳定，需要谨慎路由

目标：
- 使用 Wasserstein 漂移作为更强的路由信号
- 目标：avg_acc +0.03（从 0.5055 提升到 0.5355+）
- 时间限制：3-5 天
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


def compute_wasserstein_drift(
    clf: UnifiedSELClassifier,
    x: np.ndarray,
    window: int = 5,
) -> float:
    """
    计算当前输入的 Wasserstein 漂移（拓扑变化速率）
    
    简化版本：使用结构池的 surprise 历史来估计漂移
    tension = mean(surprise[i] - surprise[i-1] for last N)
    """
    if not hasattr(clf.pool, '_surprise_history'):
        clf.pool._surprise_history = []
    
    # 计算当前 surprise
    best_surprise = 1.0
    for s in clf.pool.structures:
        surprise = s.current_surprise(x)
        if surprise < best_surprise:
            best_surprise = surprise
    
    # 记录 surprise 历史
    clf.pool._surprise_history.append(best_surprise)
    
    # 保持窗口大小
    if len(clf.pool._surprise_history) > window:
        clf.pool._surprise_history.pop(0)
    
    # 计算漂移（surprise 的变化率）
    if len(clf.pool._surprise_history) < 2:
        return 0.0
    
    drifts = []
    for i in range(1, len(clf.pool._surprise_history)):
        drift = abs(clf.pool._surprise_history[i] - clf.pool._surprise_history[i-1])
        drifts.append(drift)
    
    return float(np.mean(drifts))


def route_with_wasserstein(
    clf: UnifiedSELClassifier,
    x: np.ndarray,
    snapshot: dict,
    surprise: float,
    tension: float,
    config: dict,
) -> int:
    """
    使用 Wasserstein 信号进行路由决策
    
    策略（基于 TopoMem 的 decide_action）：
    - tension < 0.1, surprise < 0.7 → 使用 current model（系统稳定，输入熟悉）
    - tension < 0.1, surprise >= 0.7 → 使用 snapshot expert（系统稳定，输入陌生）
    - tension >= 0.1 → 使用加权平均（系统不稳定，谨慎路由）
    """
    snapshot_probs = clf._predict_with_snapshot(x, snapshot)
    current_probs = clf.predict_proba(x)
    
    # 系统不稳定 → 使用加权平均
    if tension >= config["tension_threshold"]:
        # 权重基于 surprise：surprise 越高，越信任 snapshot
        snapshot_weight = min(1.0, surprise / config["surprise_threshold"])
        current_weight = 1.0 - snapshot_weight
        combined_probs = snapshot_weight * snapshot_probs + current_weight * current_probs
        return int(np.argmax(combined_probs))
    
    # 系统稳定
    if surprise < config["surprise_threshold"]:
        # 输入熟悉 → 使用 current model
        return int(np.argmax(current_probs))
    else:
        # 输入陌生 → 使用 snapshot expert
        return int(np.argmax(snapshot_probs))


def run_wasserstein_validation(seed: int, config: NoBoundaryConfig) -> dict:
    """运行 Wasserstein 信号验证实验"""
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

    # 评估完整系统（使用 Wasserstein 信号路由）
    route_config = {
        "surprise_threshold": 0.7,  # TopoMem 默认值
        "tension_threshold": 0.1,  # TopoMem 默认值
    }
    
    # Task 0 评估
    task_0_correct = 0
    for i in range(len(X_task_0)):
        x = X_task_0[i]
        y = int(y_task_0[i])
        
        if clf._snapshot_experts:
            # 计算 surprise 和 tension
            surprise = 1.0
            for s in clf.pool.structures:
                s_surprise = s.current_surprise(x)
                if s_surprise < surprise:
                    surprise = s_surprise
            
            tension = compute_wasserstein_drift(clf, x, window=5)
            
            pred = route_with_wasserstein(clf, x, clf._snapshot_experts[0], surprise, tension, route_config)
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
            # 计算 surprise 和 tension
            surprise = 1.0
            for s in clf.pool.structures:
                s_surprise = s.current_surprise(x)
                if s_surprise < surprise:
                    surprise = s_surprise
            
            tension = compute_wasserstein_drift(clf, x, window=5)
            
            pred = route_with_wasserstein(clf, x, clf._snapshot_experts[0], surprise, tension, route_config)
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
    print("Phase 4：Wasserstein 信号 - 使用更强的拓扑信号")
    print("=" * 80)
    print("\n核心思想：")
    print("  TopoMem 已经验证了 surprise + tension → decide_action 的有效性。")
    print("  - surprise = 1.0 - max_similarity_to_any_adapter")
    print("  - tension = mean(Wasserstein_distance(fp[i], fp[i-1]) for last N)")
    print("\n  现在将这两个信号接入 Unified-SEL 的路由协议：")
    print("    - 高 surprise + 低 tension → 使用 snapshot expert（新输入但系统稳定）")
    print("    - 低 surprise + 低 tension → 使用 current model（熟悉输入且系统稳定）")
    print("    - 高 tension → 系统不稳定，需要谨慎路由")
    print("\n目标：")
    print("  - 使用 Wasserstein 漂移作为更强的路由信号")
    print("  - 目标：avg_acc +0.03（从 0.5055 提升到 0.5355+）")
    print("  - 时间限制：3-5 天")
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
        result = run_wasserstein_validation(seed, config)
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
        print("Phase 4 结果汇总")
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
        "experiment": "phase4_wasserstein_routing",
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

    results_dir = get_results_path("phase4_wasserstein_routing")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到：{output_path}")


if __name__ == "__main__":
    main()
