"""
验证修复后的 surprise 计算

目标：
1. 验证修复后的 surprise 能否区分两个任务
2. 验证 surprise 在 task_0 和 task_1 上是否有显著差异
3. 验证 Unified-SEL 是否能击败 EWC
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


def run_validation(seed: int, config: NoBoundaryConfig):
    """验证修复后的 surprise 计算"""
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

    # 先训练一下结构池在 task_0 上
    print(f"[seed {seed}] 先在 task_0 上训练 200 step...")
    for step in range(200):
        x, y = make_eval_task(0, 1, seed + 1000 + step)
        x = x[0]
        y = y[0]
        clf.fit_one(x, y)

    # 现在验证 surprise 是否能区分两个任务
    print(f"[seed {seed}] 验证 surprise...")

    # 1. 验证 surprise 在两个任务上的差别
    print(f"\n  1. Surprise 验证：")
    surprise_task_0 = []
    surprise_task_1 = []

    for s in clf.pool.structures:
        for i in range(min(50, len(X_task_0))):  # 只测试前 50 个样本
            x = X_task_0[i]
            label = int(y_task_0[i])
            surprise_task_0.append(s.current_surprise(x, label, config.out_size))
        for i in range(min(50, len(X_task_1))):
            x = X_task_1[i]
            label = int(y_task_1[i])
            surprise_task_1.append(s.current_surprise(x, label, config.out_size))

    print(f"    task_0 平均 surprise：{np.mean(surprise_task_0):.4f}")
    print(f"    task_1 平均 surprise：{np.mean(surprise_task_1):.4f}")
    print(f"    surprise 差异：{np.mean(surprise_task_0) - np.mean(surprise_task_1):.4f}")

    # 2. 验证 surprise 是否与预测错误相关
    print(f"\n  2. Surprise 与预测错误的关系：")
    correct_surprise = []
    incorrect_surprise = []

    for s in clf.pool.structures:
        for i in range(min(50, len(X_task_0))):
            x = X_task_0[i]
            label = int(y_task_0[i])
            surprise = s.current_surprise(x, label, config.out_size)
            
            # 预测
            hidden = s.forward(np.atleast_2d(x))
            output = hidden.flatten()[:config.out_size]
            predicted_label = int(np.argmax(output))
            
            if predicted_label == label:
                correct_surprise.append(surprise)
            else:
                incorrect_surprise.append(surprise)

    print(f"    预测正确时的平均 surprise：{np.mean(correct_surprise) if correct_surprise else 0:.4f}")
    print(f"    预测错误时的平均 surprise：{np.mean(incorrect_surprise) if incorrect_surprise else 0:.4f}")
    if correct_surprise and incorrect_surprise:
        print(f"    差异：{np.mean(incorrect_surprise) - np.mean(correct_surprise):.4f}")

    return {
        "seed": seed,
        "surprise_stats": {
            "task_0_mean": float(np.mean(surprise_task_0)),
            "task_1_mean": float(np.mean(surprise_task_1)),
            "difference": float(np.mean(surprise_task_0) - np.mean(surprise_task_1)),
        },
        "prediction_surprise": {
            "correct_mean": float(np.mean(correct_surprise)) if correct_surprise else 0.0,
            "incorrect_mean": float(np.mean(incorrect_surprise)) if incorrect_surprise else 0.0,
            "difference": float(np.mean(incorrect_surprise) - np.mean(correct_surprise)) if correct_surprise and incorrect_surprise else 0.0,
        },
    }


def main():
    print("=" * 80)
    print("验证修复后的 surprise 计算")
    print("=" * 80)
    print("\n目标：")
    print("  1. 验证修复后的 surprise 能否区分两个任务")
    print("  2. 验证 surprise 在 task_0 和 task_1 上是否有显著差异")
    print("  3. 验证 surprise 是否与预测错误相关")
    print("=" * 80)

    config = NoBoundaryConfig()
    seeds = [7]

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_validation(seed, config)
        results.append(result)

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("\n关键发现：")
    print("  1. 修复后的 surprise 应该能区分两个任务")
    print("  2. task_1 的 surprise 应该高于 task_0（因为结构池在 task_0 上训练）")
    print("  3. 预测错误时的 surprise 应该高于预测正确时")

    output = {
        "experiment": "surprise_fix_validation",
        "seeds": seeds,
        "results": results,
    }

    results_dir = get_results_path("surprise_fix_validation")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到：{output_path}")


if __name__ == "__main__":
    main()
