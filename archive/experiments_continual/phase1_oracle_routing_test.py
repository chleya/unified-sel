"""
Phase 1: 实现 Oracle 级路由信号（x[0]+x[1] 符号）验证上限
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


def run_oracle_routing_test(seed: int, config: NoBoundaryConfig) -> dict:
    """测试 Oracle 级路由（x[0]+x[1] 符号）"""
    rng = np.random.default_rng(seed)
    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        pool_config=config.pool.to_pool_kwargs(),
        seed=seed,
        ewc_lambda=0.0,
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
            print(f"  [seed {seed}] Created snapshot expert at step {step + 1}")

    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)

    oracle_task_0_correct = 0
    oracle_task_1_correct = 0
    
    if clf._snapshot_experts:
        for i in range(len(X_task_0)):
            x = X_task_0[i]
            y = int(y_task_0[i])
            boundary = x[0] + x[1]
            if boundary > 0.0:
                pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            else:
                pred = int(np.argmax(clf.predict_proba_single(x)))
            if pred == y:
                oracle_task_0_correct += 1
        
        for i in range(len(X_task_1)):
            x = X_task_1[i]
            y = int(y_task_1[i])
            boundary = x[0] + x[1]
            if boundary > 0.0:
                pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            else:
                pred = int(np.argmax(clf.predict_proba_single(x)))
            if pred == y:
                oracle_task_1_correct += 1
    
    oracle_task_0_acc = oracle_task_0_correct / len(X_task_0) if len(X_task_0) > 0 else 0.0
    oracle_task_1_acc = oracle_task_1_correct / len(X_task_1) if len(X_task_1) > 0 else 0.0

    return {
        "seed": seed,
        "task_0_accuracy_final": final_task_0_accuracy,
        "task_1_accuracy_final": final_task_1_accuracy,
        "oracle_task_0_accuracy": oracle_task_0_acc,
        "oracle_task_1_accuracy": oracle_task_1_acc,
        "oracle_avg_accuracy": (oracle_task_0_acc + oracle_task_1_acc) / 2,
        "forgetting_task_0": 0.0,
    }


def main():
    print("=" * 80)
    print("Phase 1: Oracle 级路由信号验证")
    print("=" * 80)
    print("\n路由策略：")
    print("  - 当 x[0]+x[1] > 0 时，用快照专家（任务 0 专家）")
    print("  - 当 x[0]+x[1] < 0 时，用当前模型（任务 1 专家）")
    print("\n这是 Oracle 级路由信号，应该接近 Oracle 上界 0.7863")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    seeds = [7, 8, 9]

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_oracle_routing_test(seed, config)
        results.append(result)
        print(f"  task_0_acc: {result['task_0_accuracy_final']:.4f}")
        print(f"  task_1_acc: {result['task_1_accuracy_final']:.4f}")
        print(f"  oracle_task_0_acc: {result['oracle_task_0_accuracy']:.4f}")
        print(f"  oracle_task_1_acc: {result['oracle_task_1_accuracy']:.4f}")
        print(f"  oracle_avg_acc: {result['oracle_avg_accuracy']:.4f}")

    avg_oracle_acc = np.mean([r["oracle_avg_accuracy"] for r in results])
    std_oracle_acc = np.std([r["oracle_avg_accuracy"] for r in results])
    avg_task_0 = np.mean([r["task_0_accuracy_final"] for r in results])
    avg_task_1 = np.mean([r["task_1_accuracy_final"] for r in results])

    print("\n" + "=" * 80)
    print("Phase 1 结果汇总")
    print("=" * 80)
    print(f"平均 Oracle 路由准确率: {avg_oracle_acc:.4f} ± {std_oracle_acc:.4f}")
    print(f"平均 task_0 准确率: {avg_task_0:.4f}")
    print(f"平均 task_1 准确率: {avg_task_1:.4f}")
    print(f"平均准确率（无路由）: {(avg_task_0 + avg_task_1) / 2:.4f}")
    print(f"\n对比：")
    print(f"  EWC baseline: 0.5005")
    print(f"  Oracle 上界: 0.7863")
    print(f"  当前结果: {avg_oracle_acc:.4f}")

    if avg_oracle_acc > 0.70:
        print("\n✅ 成功！Oracle 级路由信号显著提升准确率！")
        print("   这证明了路由质量是瓶颈，而不是学习质量。")
    else:
        print("\n❌ 失败！Oracle 级路由信号没有达到预期效果。")
        print("   需要进一步分析原因。")

    output = {
        "experiment": "phase1_oracle_routing_test",
        "seeds": seeds,
        "results": results,
        "summary": {
            "avg_oracle_acc": avg_oracle_acc,
            "std_oracle_acc": std_oracle_acc,
            "avg_task_0": avg_task_0,
            "avg_task_1": avg_task_1,
            "baseline_acc_no_routing": (avg_task_0 + avg_task_1) / 2,
        },
    }

    results_dir = get_results_path("phase1_oracle_routing")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
