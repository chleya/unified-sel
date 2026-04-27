"""
修复后的完整实验：Unified-SEL vs EWC

目标：
1. 验证修复后的 Unified-SEL 能否击败 EWC
2. 目标：avg_acc > 0.55（显著超越 EWC 的 0.5005）
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


def run_experiment(seed: int, config: NoBoundaryConfig):
    """运行完整实验"""
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

    # 训练
    for step in range(config.steps):
        progress = 0.0 if step < config.checkpoint_step else (step - config.checkpoint_step) / max(
            config.steps - config.checkpoint_step - 1, 1
        )
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)

    # 最终评估
    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)
    final_avg_accuracy = (final_task_0_accuracy + final_task_1_accuracy) / 2

    return {
        "seed": seed,
        "final": {
            "task_0_accuracy": final_task_0_accuracy,
            "task_1_accuracy": final_task_1_accuracy,
            "avg_accuracy": final_avg_accuracy,
        },
    }


def main():
    print("=" * 80)
    print("修复后的完整实验：Unified-SEL vs EWC")
    print("=" * 80)
    print("\n目标：")
    print("  1. 验证修复后的 Unified-SEL 能否击败 EWC")
    print("  2. 目标：avg_acc > 0.55（显著超越 EWC 的 0.5005）")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.readout_mode = "hybrid_local"
    config.pool.max_structures = 16
    seeds = [7, 8, 9, 10, 11]  # 5 个种子

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_experiment(seed, config)
        results.append(result)
        
        print(f"  最终平均准确率：{result['final']['avg_accuracy']:.4f}")

    # 汇总
    avg_final_acc = np.mean([r['final']['avg_accuracy'] for r in results])
    
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    print(f"平均最终准确率：{avg_final_acc:.4f}")
    
    print(f"\n对比 EWC 基线：")
    print(f"  EWC 平均准确率：0.5005")
    print(f"  最终准确率：{avg_final_acc:.4f} (差异：{avg_final_acc - 0.5005:+.4f})")
    
    if avg_final_acc > 0.55:
        print("\n[SUCCESS] 达到目标！avg_acc > 0.55")
    elif avg_final_acc > 0.5005:
        print("\n[PARTIAL SUCCESS] 超越 EWC，但未达到 0.55")
    else:
        print("\n[FAILED] 未超越 EWC")

    # 统计显著性
    from scipy import stats
    ewc_baseline = 0.5005
    final_accs = [r['final']['avg_accuracy'] for r in results]
    t_stat, p_value = stats.ttest_1samp(final_accs, ewc_baseline)
    print(f"\n统计检验：")
    print(f"  t 统计量：{t_stat:.4f}")
    print(f"  p 值：{p_value:.4f}")
    
    if p_value < 0.05:
        print("  [STATISTICALLY SIGNIFICANT] p < 0.05")
    else:
        print("  [NOT STATISTICALLY SIGNIFICANT] p >= 0.05")

    output = {
        "experiment": "surprise_fix_full_experiment",
        "seeds": seeds,
        "results": results,
        "summary": {
            "avg_final_acc": avg_final_acc,
            "ewc_baseline": ewc_baseline,
            "t_stat": t_stat,
            "p_value": p_value,
        },
    }

    results_dir = get_results_path("surprise_fix_full_experiment")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到：{output_path}")


if __name__ == "__main__":
    main()
