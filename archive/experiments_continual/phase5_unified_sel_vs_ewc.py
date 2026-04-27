"""
Phase 5: 验证完整的 Unified-SEL 系统能否击败 EWC

核心思想：
- 使用结构池冻结和修复后的 _predict_with_snapshot 方法
- 运行 5 个种子的实验
- 与 EWC 基线对比
- 检查是否统计显著

预期效果：
- Unified-SEL 的平均准确率 > EWC（0.5005）
- 统计显著性：p < 0.05
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


def run_unified_sel_test(seed: int, config: NoBoundaryConfig) -> dict:
    """测试完整的 Unified-SEL 系统"""
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
            clf.freeze_pool()
            print(f"  [seed {seed}] Created snapshot expert and FROZE pool at step {step + 1}")

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

    return {
        "seed": seed,
        "task_0_accuracy_final": final_task_0_accuracy,
        "task_1_accuracy_final": final_task_1_accuracy,
        "avg_accuracy": final_avg_accuracy,
        "snap_acc_t0": snap_acc_t0,
        "snap_acc_t1": snap_acc_t1,
    }


def main():
    print("=" * 80)
    print("Phase 5: 验证完整的 Unified-SEL 系统能否击败 EWC")
    print("=" * 80)
    print("\n配置：")
    print("  - 结构池冻结：在 checkpoint 后冻结结构池")
    print("  - 修复的 _predict_with_snapshot：使用基于 utility 的加权平均")
    print("  - 5 个种子：7, 8, 9, 10, 11")
    print("  - EWC 基线：0.5005")
    print("\n预期效果：")
    print("  - Unified-SEL 的平均准确率 > EWC（0.5005）")
    print("  - 统计显著性：p < 0.05")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    config.readout_mode = "hybrid_local"
    seeds = [7, 8, 9, 10, 11]

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_unified_sel_test(seed, config)
        results.append(result)
        
        print(f"  task_0 准确率: {result['task_0_accuracy_final']:.4f}")
        print(f"  task_1 准确率: {result['task_1_accuracy_final']:.4f}")
        print(f"  平均准确率: {result['avg_accuracy']:.4f}")
        print(f"  Snapshot expert task_0 准确率: {result['snap_acc_t0']:.4f}")

    # 汇总分析
    avg_task_0 = np.mean([r['task_0_accuracy_final'] for r in results])
    avg_task_1 = np.mean([r['task_1_accuracy_final'] for r in results])
    avg_acc = np.mean([r['avg_accuracy'] for r in results])
    std_acc = np.std([r['avg_accuracy'] for r in results])
    avg_snap_t0 = np.mean([r['snap_acc_t0'] for r in results])

    print("\n" + "=" * 80)
    print("Phase 5 结果汇总")
    print("=" * 80)
    print(f"平均 task_0 准确率: {avg_task_0:.4f}")
    print(f"平均 task_1 准确率: {avg_task_1:.4f}")
    print(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"平均 Snapshot expert task_0 准确率: {avg_snap_t0:.4f}")
    
    print(f"\n对比 EWC 基线：")
    print(f"  EWC 平均准确率: 0.5005")
    print(f"  Unified-SEL 平均准确率: {avg_acc:.4f}")
    print(f"  差异: {avg_acc - 0.5005:+.4f}")

    if avg_acc > 0.5005:
        print("\n[SUCCESS] Unified-SEL 击败 EWC！")
        print(f"  平均准确率: {avg_acc:.4f} > 0.5005")
    else:
        print("\n[FAILED] Unified-SEL 未能击败 EWC")
        print(f"  平均准确率: {avg_acc:.4f} <= 0.5005")

    # 计算统计显著性（简单的 t 检验）
    from scipy import stats
    ewc_baseline = 0.5005
    unified_accs = [r['avg_accuracy'] for r in results]
    t_stat, p_value = stats.ttest_1samp(unified_accs, ewc_baseline)
    print(f"\n统计检验：")
    print(f"  t 统计量: {t_stat:.4f}")
    print(f"  p 值: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  [STATISTICALLY SIGNIFICANT] p < 0.05")
    else:
        print("  [NOT STATISTICALLY SIGNIFICANT] p >= 0.05")

    output = {
        "experiment": "phase5_unified_sel_vs_ewc",
        "seeds": seeds,
        "results": results,
        "summary": {
            "avg_task_0": avg_task_0,
            "avg_task_1": avg_task_1,
            "avg_acc": avg_acc,
            "std_acc": std_acc,
            "avg_snap_t0": avg_snap_t0,
            "t_stat": t_stat,
            "p_value": p_value,
            "ewc_baseline": ewc_baseline,
        },
    }

    results_dir = get_results_path("phase5_unified_sel_vs_ewc")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
