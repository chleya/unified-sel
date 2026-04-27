"""
Phase 2: 深入分析为什么 prediction disagreement 效果不明显
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


def analyze_expert_quality(seed: int, config: NoBoundaryConfig) -> dict:
    """分析 snapshot expert 和 current model 的质量"""
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
            print(f"  [seed {seed}] Created snapshot expert at step {step + 1}")

    # 分析 snapshot expert 的质量
    snap_correct_t0 = 0
    snap_correct_t1 = 0
    current_correct_t0 = 0
    current_correct_t1 = 0
    
    if clf._snapshot_experts:
        for i in range(len(X_task_0)):
            x = X_task_0[i]
            y = int(y_task_0[i])
            
            # snapshot expert 预测
            snap_pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            if snap_pred == y:
                snap_correct_t0 += 1
            
            # current model 预测
            current_pred = int(np.argmax(clf.predict_proba_single(x)))
            if current_pred == y:
                current_correct_t0 += 1
        
        for i in range(len(X_task_1)):
            x = X_task_1[i]
            y = int(y_task_1[i])
            
            # snapshot expert 预测
            snap_pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            if snap_pred == y:
                snap_correct_t1 += 1
            
            # current model 预测
            current_pred = int(np.argmax(clf.predict_proba_single(x)))
            if current_pred == y:
                current_correct_t1 += 1
    
    snap_acc_t0 = snap_correct_t0 / len(X_task_0) if len(X_task_0) > 0 else 0.0
    snap_acc_t1 = snap_correct_t1 / len(X_task_1) if len(X_task_1) > 0 else 0.0
    current_acc_t0 = current_correct_t0 / len(X_task_0) if len(X_task_0) > 0 else 0.0
    current_acc_t1 = current_correct_t1 / len(X_task_1) if len(X_task_1) > 0 else 0.0

    # 分析 disagreement 情况
    disagree_count_t0 = 0
    disagree_correct_snap_t0 = 0
    disagree_correct_current_t0 = 0
    disagree_count_t1 = 0
    disagree_correct_snap_t1 = 0
    disagree_correct_current_t1 = 0
    
    if clf._snapshot_experts:
        for i in range(len(X_task_0)):
            x = X_task_0[i]
            y = int(y_task_0[i])
            
            snap_pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            current_pred = int(np.argmax(clf.predict_proba_single(x)))
            
            if snap_pred != current_pred:
                disagree_count_t0 += 1
                if snap_pred == y:
                    disagree_correct_snap_t0 += 1
                if current_pred == y:
                    disagree_correct_current_t0 += 1
        
        for i in range(len(X_task_1)):
            x = X_task_1[i]
            y = int(y_task_1[i])
            
            snap_pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            current_pred = int(np.argmax(clf.predict_proba_single(x)))
            
            if snap_pred != current_pred:
                disagree_count_t1 += 1
                if snap_pred == y:
                    disagree_correct_snap_t1 += 1
                if current_pred == y:
                    disagree_correct_current_t1 += 1

    return {
        "seed": seed,
        "snap_acc_t0": snap_acc_t0,
        "snap_acc_t1": snap_acc_t1,
        "current_acc_t0": current_acc_t0,
        "current_acc_t1": current_acc_t1,
        "disagree_count_t0": disagree_count_t0,
        "disagree_correct_snap_t0": disagree_correct_snap_t0,
        "disagree_correct_current_t0": disagree_correct_current_t0,
        "disagree_count_t1": disagree_count_t1,
        "disagree_correct_snap_t1": disagree_correct_snap_t1,
        "disagree_correct_current_t1": disagree_correct_current_t1,
    }


def main():
    print("=" * 80)
    print("Phase 2: 深入分析为什么 prediction disagreement 效果不明显")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    config.readout_mode = "hybrid_local"
    seeds = [7, 8, 9]

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = analyze_expert_quality(seed, config)
        results.append(result)
        
        print(f"  Snapshot expert:")
        print(f"    task_0 准确率: {result['snap_acc_t0']:.4f}")
        print(f"    task_1 准确率: {result['snap_acc_t1']:.4f}")
        print(f"  Current model:")
        print(f"    task_0 准确率: {result['current_acc_t0']:.4f}")
        print(f"    task_1 准确率: {result['current_acc_t1']:.4f}")
        print(f"  Disagreement 分析:")
        print(f"    task_0: {result['disagree_count_t0']} 个 disagreement")
        print(f"      snapshot 正确: {result['disagree_correct_snap_t0']}")
        print(f"      current 正确: {result['disagree_correct_current_t0']}")
        print(f"    task_1: {result['disagree_count_t1']} 个 disagreement")
        print(f"      snapshot 正确: {result['disagree_correct_snap_t1']}")
        print(f"      current 正确: {result['disagree_correct_current_t1']}")

    # 汇总分析
    avg_snap_t0 = np.mean([r['snap_acc_t0'] for r in results])
    avg_snap_t1 = np.mean([r['snap_acc_t1'] for r in results])
    avg_current_t0 = np.mean([r['current_acc_t0'] for r in results])
    avg_current_t1 = np.mean([r['current_acc_t1'] for r in results])

    print("\n" + "=" * 80)
    print("Phase 2 结果汇总")
    print("=" * 80)
    print(f"平均 Snapshot expert 准确率:")
    print(f"  task_0: {avg_snap_t0:.4f}")
    print(f"  task_1: {avg_snap_t1:.4f}")
    print(f"平均 Current model 准确率:")
    print(f"  task_0: {avg_current_t0:.4f}")
    print(f"  task_1: {avg_current_t1:.4f}")
    
    print(f"\n关键洞察:")
    print(f"  1. Snapshot expert 在 task_0 上的准确率: {avg_snap_t0:.4f}")
    print(f"     - 这远低于 Oracle 上界实验中的 0.8438")
    print(f"     - 说明 snapshot expert 在训练后已经发生了遗忘")
    print(f"  2. Snapshot expert 在 task_1 上的准确率: {avg_snap_t1:.4f}")
    print(f"     - 这说明 snapshot expert 对 task_1 几乎没有知识")
    print(f"  3. Current model 在 task_0 上的准确率: {avg_current_t0:.4f}")
    print(f"     - 这说明 current model 已经遗忘了 task_0")
    print(f"  4. Current model 在 task_1 上的准确率: {avg_current_t1:.4f}")
    print(f"     - 这说明 current model 对 task_1 有一定知识")

    print(f"\n结论:")
    print(f"  - Prediction disagreement 效果不明显的原因是：")
    print(f"    - Snapshot expert 的准确率不够高（只有 {avg_snap_t0:.4f}）")
    print(f"    - 当 snapshot 和 current 都预测错误时，disagreement 无法挽救")
    print(f"  - 需要更强的专家保护机制，提高 snapshot expert 的准确率")

    output = {
        "experiment": "phase2_expert_quality_analysis",
        "seeds": seeds,
        "results": results,
        "summary": {
            "avg_snap_t0": avg_snap_t0,
            "avg_snap_t1": avg_snap_t1,
            "avg_current_t0": avg_current_t0,
            "avg_current_t1": avg_current_t1,
        },
    }

    results_dir = get_results_path("phase2_expert_quality")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
