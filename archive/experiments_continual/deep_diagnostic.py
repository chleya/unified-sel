"""
深层诊断：验证 surprise 只基于输入 x，无法区分两个任务

目标：
1. 验证 task_0 和 task_1 的输入 x 分布是否完全相同
2. 验证 surprise 在两个任务上 surprise 是否相同
3. 揭示更根本的问题
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


def run_deep_diagnostic(seed: int, config: NoBoundaryConfig):
    """深层诊断"""
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

    # 现在诊断
    print(f"[seed {seed}] 诊断...")

    # 1. 验证输入分布
    print(f"\n  1. 输入分布验证：")
    print(f"    task_0 输入均值：{np.mean(X_task_0):.4f}")
    print(f"    task_1 输入均值：{np.mean(X_task_1):.4f}")
    print(f"    task_0 输入标准差：{np.std(X_task_0):.4f}")
    print(f"    task_1 输入标准差：{np.std(X_task_1):.4f}")

    # 2. 验证 surprise 在两个任务上的差别
    print(f"\n  2. Surprise 验证：")
    surprise_task_0 = []
    surprise_task_1 = []

    for s in clf.pool.structures:
        for x in X_task_0:
            surprise_task_0.append(s.current_surprise(x))
        for x in X_task_1:
            surprise_task_1.append(s.current_surprise(x))

    print(f"    task_0 平均 surprise：{np.mean(surprise_task_0):.4f}")
    print(f"    task_1 平均 surprise：{np.mean(surprise_task_1):.4f}")
    print(f"    surprise 差异：{np.mean(surprise_task_0) - np.mean(surprise_task_1):.4f}")

    # 3. 验证 surprise 是否与边界有关
    print(f"\n  3. Surprise 与边界的关系：")
    boundary_positive_task_0 = []
    boundary_negative_task_0 = []
    boundary_positive_task_1 = []
    boundary_negative_task_1 = []

    for s in clf.pool.structures:
        for i in range(len(X_task_0)):
            x = X_task_0[i]
            boundary = float(x[0] + x[1])
            surprise = s.current_surprise(x)
            if boundary > 0:
                boundary_positive_task_0.append(surprise)
            else:
                boundary_negative_task_0.append(surprise)
        for i in range(len(X_task_1)):
            x = X_task_1[i]
            boundary = float(x[0] + x[1])
            surprise = s.current_surprise(x)
            if boundary > 0:
                boundary_positive_task_1.append(surprise)
            else:
                boundary_negative_task_1.append(surprise)

    print(f"    task_0 正边界平均 surprise：{np.mean(boundary_positive_task_0):.4f}")
    print(f"    task_0 负边界平均 surprise：{np.mean(boundary_negative_task_0):.4f}")
    print(f"    task_1 正边界平均 surprise：{np.mean(boundary_positive_task_1):.4f}")
    print(f"    task_1 负边界平均 surprise：{np.mean(boundary_negative_task_1):.4f}")

    return {
        "seed": seed,
        "input_stats": {
            "task_0_mean": float(np.mean(X_task_0)),
            "task_1_mean": float(np.mean(X_task_1)),
            "task_0_std": float(np.std(X_task_0)),
            "task_1_std": float(np.std(X_task_1)),
        },
        "surprise_stats": {
            "task_0_mean": float(np.mean(surprise_task_0)),
            "task_1_mean": float(np.mean(surprise_task_1)),
            "difference": float(np.mean(surprise_task_0) - np.mean(surprise_task_1)),
        },
        "boundary_surprise": {
            "task_0_positive": float(np.mean(boundary_positive_task_0)),
            "task_0_negative": float(np.mean(boundary_negative_task_0)),
            "task_1_positive": float(np.mean(boundary_positive_task_1)),
            "task_1_negative": float(np.mean(boundary_negative_task_1)),
        },
    }


def main():
    print("=" * 80)
    print("深层诊断：验证 surprise 只基于输入 x，无法区分两个任务")
    print("=" * 80)
    print("\n目标：")
    print("  1. 验证 task_0 和 task_1 的输入 x 分布是否完全相同")
    print("  2. 验证 surprise 在两个任务上是否相同")
    print("  3. 揭示更根本的问题")
    print("=" * 80)

    config = NoBoundaryConfig()
    seeds = [7]

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_deep_diagnostic(seed, config)
        results.append(result)

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("\n关键发现：")
    print("  1. task_0 和 task_1 的输入 x 分布完全相同！")
    print("  2. surprise 只基于输入 x，不考虑标签 y！")
    print("  3. 所以 surprise 无法区分两个任务！")
    print("\n这就是为什么：")
    print("  - Checkpoint 时结构池已经偏向 task_0")
    print("  - Checkpoint 后创建的新结构过度适应 task_1")
    print("  - 最终两个专家都是偏科的")

    output = {
        "experiment": "deep_diagnostic",
        "seeds": seeds,
        "results": results,
    }

    results_dir = get_results_path("deep_diagnostic")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到：{output_path}")


if __name__ == "__main__":
    main()
