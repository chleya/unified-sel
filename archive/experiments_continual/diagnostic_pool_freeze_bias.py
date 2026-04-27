"""
诊断实验：分析结构池冻结导致偏科的根本原因

目标：
1. 观察 checkpoint 前后结构池的变化
2. 分析为什么 snapshot expert 只学会 task_0
3. 分析为什么 current model 只学会 task_1
4. 找到真正的解决方案

方法：
- 在 checkpoint 前后分别评估结构池对 task_0 和 task_1 的适应能力
- 分析每个结构的 surprise、tension、utility 等指标
- 观察结构创建和剪枝的模式
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


def evaluate_structure_pool(clf: UnifiedSELClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """评估结构池对给定数据的适应能力"""
    correct = 0
    surprises = []
    tensions = []
    utilities = []
    
    for i in range(len(X)):
        x = X[i]
        label = int(y[i])
        pred = clf.predict(x)
        if pred == label:
            correct += 1
        
        # 计算结构池的平均 surprise
        best_surprise = 1.0
        for s in clf.pool.structures:
            surprise = s.current_surprise(x)
            if surprise < best_surprise:
                best_surprise = surprise
        surprises.append(best_surprise)
        
        # 记录结构的 utility
        for s in clf.pool.structures:
            utilities.append(s.utility)
    
    accuracy = correct / len(X) if len(X) > 0 else 0.0
    avg_surprise = float(np.mean(surprises)) if surprises else 0.0
    avg_utility = float(np.mean(utilities)) if utilities else 0.0
    
    return {
        "accuracy": accuracy,
        "avg_surprise": avg_surprise,
        "avg_utility": avg_utility,
        "n_structures": len(clf.pool.structures),
    }


def analyze_structure_ids(before_ids: list, after_ids: list) -> dict:
    """分析结构 ID 的变化"""
    before_set = set(before_ids)
    after_set = set(after_ids)
    
    pruned = before_set - after_set
    created = after_set - before_set
    preserved = before_set & after_set
    
    return {
        "pruned_count": len(pruned),
        "pruned_ids": sorted(list(pruned)),
        "created_count": len(created),
        "created_ids": sorted(list(created)),
        "preserved_count": len(preserved),
        "preserved_ids": sorted(list(preserved)),
    }


def run_diagnostic(seed: int, config: NoBoundaryConfig) -> dict:
    """运行诊断实验"""
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

    # 记录 checkpoint 前的状态
    checkpoint_before_ids = [s.id for s in clf.pool.structures]
    
    print(f"  [seed {seed}] Step 0: 初始结构数 = {len(checkpoint_before_ids)}")

    for step in range(config.steps):
        if step < config.checkpoint_step:
            progress = 0.0
        else:
            progress = (step - config.checkpoint_step) / max(config.steps - config.checkpoint_step - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)

        # 在 checkpoint 时记录状态
        if step + 1 == config.checkpoint_step:
            # 评估 checkpoint 前的结构池
            task_0_eval_before = evaluate_structure_pool(clf, X_task_0, y_task_0)
            task_1_eval_before = evaluate_structure_pool(clf, X_task_1, y_task_1)
            
            checkpoint_after_ids = [s.id for s in clf.pool.structures]
            
            print(f"  [seed {seed}] Step {step + 1}: checkpoint 时结构数 = {len(checkpoint_after_ids)}")
            print(f"    Task 0 准确率：{task_0_eval_before['accuracy']:.4f}")
            print(f"    Task 1 准确率：{task_1_eval_before['accuracy']:.4f}")
            
            # 保存 snapshot expert
            clf.snapshot_expert(confidence_ratio_threshold=0.5)
            
            # 冻结结构池（只冻结剪枝）
            clf.freeze_pool_prune_only()
            print(f"  [seed {seed}] Created snapshot expert and FROZE pool (prune only)")

    # 最终评估
    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)
    final_avg_accuracy = (final_task_0_accuracy + final_task_1_accuracy) / 2
    
    final_ids = [s.id for s in clf.pool.structures]
    
    # 分析结构 ID 变化
    id_changes = analyze_structure_ids(checkpoint_after_ids, final_ids)
    
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
        "checkpoint_before": {
            "n_structures": len(checkpoint_before_ids),
            "structure_ids": checkpoint_before_ids,
        },
        "checkpoint_after": {
            "n_structures": len(checkpoint_after_ids),
            "structure_ids": checkpoint_after_ids,
            "task_0_accuracy": task_0_eval_before['accuracy'],
            "task_1_accuracy": task_1_eval_before['accuracy'],
            "task_0_surprise": task_0_eval_before['avg_surprise'],
            "task_1_surprise": task_1_eval_before['avg_surprise'],
        },
        "final": {
            "n_structures": len(final_ids),
            "structure_ids": final_ids,
            "task_0_accuracy": final_task_0_accuracy,
            "task_1_accuracy": final_task_1_accuracy,
            "avg_accuracy": final_avg_accuracy,
            "snap_acc_t0": snap_acc_t0,
            "snap_acc_t1": snap_acc_t1,
        },
        "id_changes": id_changes,
    }


def main():
    print("=" * 80)
    print("诊断实验：分析结构池冻结导致偏科的根本原因")
    print("=" * 80)
    print("\n目标：")
    print("  1. 观察 checkpoint 前后结构池的变化")
    print("  2. 分析为什么 snapshot expert 只学会 task_0")
    print("  3. 分析为什么 current model 只学会 task_1")
    print("  4. 找到真正的解决方案")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    config.readout_mode = "hybrid_local"
    config.pool.max_structures = 16
    seeds = [7]  # 先分析一个种子

    results = []
    for seed in seeds:
        print(f"\n运行 seed {seed}...")
        result = run_diagnostic(seed, config)
        results.append(result)
        
        print(f"\n  最终结果：")
        print(f"    Task 0 准确率：{result['final']['task_0_accuracy']:.4f}")
        print(f"    Task 1 准确率：{result['final']['task_1_accuracy']:.4f}")
        print(f"    平均准确率：{result['final']['avg_accuracy']:.4f}")
        print(f"    Snapshot expert task_0 准确率：{result['final']['snap_acc_t0']:.4f}")
        print(f"    Snapshot expert task_1 准确率：{result['final']['snap_acc_t1']:.4f}")
        
        print(f"\n  结构 ID 变化：")
        print(f"    Checkpoint 时结构数：{result['checkpoint_after']['n_structures']}")
        print(f"    最终结构数：{result['final']['n_structures']}")
        print(f"    被剪枝：{result['id_changes']['pruned_count']} 个")
        print(f"    新创建：{result['id_changes']['created_count']} 个")
        print(f"    保留：{result['id_changes']['preserved_count']} 个")

    # 汇总分析
    if results:
        result = results[0]
        print("\n" + "=" * 80)
        print("诊断结果分析")
        print("=" * 80)
        
        print(f"\nCheckpoint 时的表现：")
        print(f"  Task 0 准确率：{result['checkpoint_after']['task_0_accuracy']:.4f}")
        print(f"  Task 1 准确率：{result['checkpoint_after']['task_1_accuracy']:.4f}")
        
        print(f"\n最终表现：")
        print(f"  Task 0 准确率：{result['final']['task_0_accuracy']:.4f}")
        print(f"  Task 1 准确率：{result['final']['task_1_accuracy']:.4f}")
        
        print(f"\nSnapshot expert 表现：")
        print(f"  Task 0 准确率：{result['final']['snap_acc_t0']:.4f}")
        print(f"  Task 1 准确率：{result['final']['snap_acc_t1']:.4f}")
        
        print(f"\n关键发现：")
        if result['checkpoint_after']['task_0_accuracy'] > result['checkpoint_after']['task_1_accuracy']:
            print("  [FOUND] Checkpoint 时结构池已经偏向 task_0")
            print("    -> 这是 snapshot expert 只学会 task_0 的原因")
        
        if result['id_changes']['created_count'] > 0:
            print(f"  [FOUND] Checkpoint 后创建了 {result['id_changes']['created_count']} 个新结构")
            print("    -> 这些新结构可能过度适应 task_1")
        
        if result['final']['snap_acc_t0'] > result['final']['snap_acc_t1']:
            print("  [FOUND] Snapshot expert 确实只学会了 task_0")
            print("    -> 需要改变 checkpoint 策略")

    output = {
        "experiment": "diagnostic_pool_freeze_bias",
        "seeds": seeds,
        "results": results,
    }

    results_dir = get_results_path("diagnostic_pool_freeze_bias")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到：{output_path}")


if __name__ == "__main__":
    main()
