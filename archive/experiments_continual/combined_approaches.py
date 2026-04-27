"""
综合实验：测试三种新方案

方案 A：动态 checkpoint 策略
- 不固定在 step 200
- 持续监控两个任务的准确率
- 当结构池同时学会两个任务时再保存 snapshot expert

方案 B：多 snapshot expert
- 保存多个 snapshot expert
- 一个在 task_0 表现好时保存
- 一个在 task_1 表现好时保存
- 路由时选择最合适的 expert

方案 C：改变训练策略
- 在 checkpoint 之前就强制结构池接触两个任务
- 在 step 100-200 之间就开始增加 task_1 的比例

目标：
- 找到真正能解决种子偏科的方法
- 目标：avg_acc > 0.55（显著超越 EWC 的 0.5005）
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


def evaluate_on_both_tasks(clf: UnifiedSELClassifier, X_task_0, y_task_0, X_task_1, y_task_1) -> dict:
    """评估当前模型在两个任务上的表现"""
    task_0_correct = sum(1 for i in range(len(X_task_0)) if clf.predict(X_task_0[i]) == int(y_task_0[i]))
    task_1_correct = sum(1 for i in range(len(X_task_1)) if clf.predict(X_task_1[i]) == int(y_task_1[i]))
    
    return {
        "task_0_accuracy": task_0_correct / len(X_task_0) if len(X_task_0) > 0 else 0.0,
        "task_1_accuracy": task_1_correct / len(X_task_1) if len(X_task_1) > 0 else 0.0,
    }


def route_with_multiple_experts(clf: UnifiedSELClassifier, x: np.ndarray, snapshots: list) -> int:
    """使用多个 snapshot expert 进行路由"""
    if not snapshots:
        return clf.predict(x)
    
    # 计算每个 expert 的置信度
    expert_preds = []
    expert_confs = []
    
    # Current model
    current_probs = clf.predict_proba(x)
    expert_preds.append(int(np.argmax(current_probs)))
    expert_confs.append(float(np.max(current_probs) - np.min(current_probs)))
    
    # Snapshot experts
    for snapshot in snapshots:
        snapshot_probs = clf._predict_with_snapshot(x, snapshot)
        expert_preds.append(int(np.argmax(snapshot_probs)))
        expert_confs.append(float(np.max(snapshot_probs) - np.min(snapshot_probs)))
    
    # 选择置信度最高的 expert
    best_idx = int(np.argmax(expert_confs))
    return expert_preds[best_idx]


def run_combined_approach(seed: int, config: NoBoundaryConfig, approach: str) -> dict:
    """运行综合方案"""
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

    snapshots = []  # 多 snapshot expert（方案 B）
    saved_snapshots_at = []  # 记录保存 snapshot 的 step
    
    # 方案 C：改变 training progress 曲线
    if approach == "C" or approach == "AC" or approach == "BC" or approach == "ABC":
        # 在 step 100 就开始增加 task_1 比例
        early_start_step = 100
    else:
        early_start_step = config.checkpoint_step

    for step in range(config.steps):
        # 方案 C：更早开始增加 task_1 比例
        if step < early_start_step:
            progress = 0.0
        else:
            progress = (step - early_start_step) / max(config.steps - early_start_step - 1, 1)
        
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)

        # 方案 A：动态 checkpoint 策略
        if approach == "A" or approach == "AC" or approach == "ABC":
            # 每 50 步检查一次两个任务的准确率
            if step > 0 and step % 50 == 0:
                eval_result = evaluate_on_both_tasks(clf, X_task_0[:100], y_task_0[:100], X_task_1[:100], y_task_1[:100])
                
                # 如果两个任务都学会（准确率都 > 0.6），保存 snapshot
                if eval_result["task_0_accuracy"] > 0.6 and eval_result["task_1_accuracy"] > 0.6:
                    if len(snapshots) == 0:  # 只保存第一个
                        clf.snapshot_expert(confidence_ratio_threshold=0.5)
                        snapshots.append(clf._snapshot_experts[-1])
                        saved_snapshots_at.append(step)
                        print(f"  [seed {seed}] Step {step}: 保存 snapshot (task_0={eval_result['task_0_accuracy']:.3f}, task_1={eval_result['task_1_accuracy']:.3f})")
        
        # 方案 B：多 snapshot expert（固定间隔保存）
        elif approach == "B" or approach == "BC":
            if step > 0 and step % 100 == 0 and len(snapshots) < 3:  # 最多保存 3 个
                clf.snapshot_expert(confidence_ratio_threshold=0.5)
                snapshots.append(clf._snapshot_experts[-1])
                saved_snapshots_at.append(step)
                print(f"  [seed {seed}] Step {step}: 保存 snapshot #{len(snapshots)}")
        
        # 原始方案：固定在 checkpoint_step 保存
        else:
            if step + 1 == config.checkpoint_step:
                clf.snapshot_expert(confidence_ratio_threshold=0.5)
                # 不冻结结构池，允许自由演化
                print(f"  [seed {seed}] Step {step + 1}: 保存 snapshot (不冻结)")

    # 最终评估
    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)
    final_avg_accuracy = (final_task_0_accuracy + final_task_1_accuracy) / 2

    # 使用多 expert 路由评估（如果有多个 snapshot）
    if len(snapshots) > 0:
        multi_task_0_correct = 0
        multi_task_1_correct = 0
        
        for i in range(len(X_task_0)):
            x = X_task_0[i]
            y = int(y_task_0[i])
            pred = route_with_multiple_experts(clf, x, snapshots)
            if pred == y:
                multi_task_0_correct += 1
        
        for i in range(len(X_task_1)):
            x = X_task_1[i]
            y = int(y_task_1[i])
            pred = route_with_multiple_experts(clf, x, snapshots)
            if pred == y:
                multi_task_1_correct += 1
        
        multi_task_0_accuracy = multi_task_0_correct / len(X_task_0) if len(X_task_0) > 0 else 0.0
        multi_task_1_accuracy = multi_task_1_correct / len(X_task_1) if len(X_task_1) > 0 else 0.0
    else:
        multi_task_0_accuracy = final_task_0_accuracy
        multi_task_1_accuracy = final_task_1_accuracy

    return {
        "seed": seed,
        "approach": approach,
        "saved_snapshots_at": saved_snapshots_at,
        "n_snapshots": len(snapshots),
        "final": {
            "task_0_accuracy": final_task_0_accuracy,
            "task_1_accuracy": final_task_1_accuracy,
            "avg_accuracy": final_avg_accuracy,
        },
        "multi_expert": {
            "task_0_accuracy": multi_task_0_accuracy,
            "task_1_accuracy": multi_task_1_accuracy,
            "avg_accuracy": multi_task_0_accuracy + multi_task_1_accuracy / 2,
        },
    }


def main():
    print("=" * 80)
    print("综合实验：测试三种新方案")
    print("=" * 80)
    print("\n方案 A：动态 checkpoint 策略")
    print("  - 不固定在 step 200")
    print("  - 持续监控两个任务的准确率")
    print("  - 当结构池同时学会两个任务时再保存 snapshot expert")
    print("\n方案 B：多 snapshot expert")
    print("  - 保存多个 snapshot expert")
    print("  - 一个在 task_0 表现好时保存")
    print("  - 一个在 task_1 表现好时保存")
    print("  - 路由时选择最合适的 expert")
    print("\n方案 C：改变训练策略")
    print("  - 在 checkpoint 之前就强制结构池接触两个任务")
    print("  - 在 step 100-200 之间就开始增加 task_1 的比例")
    print("\n目标：")
    print("  - 找到真正能解决种子偏科的方法")
    print("  - 目标：avg_acc > 0.55（显著超越 EWC 的 0.5005）")
    print("=" * 80)

    config = NoBoundaryConfig()
    config.steps = 600
    config.checkpoint_step = 200
    config.readout_mode = "hybrid_local"
    config.pool.max_structures = 16
    
    # 测试不同方案
    approaches = {
        "Base": "原始方案（固定 checkpoint，不冻结）",
        "A": "动态 checkpoint",
        "B": "多 snapshot expert",
        "C": "改变训练策略",
        "AC": "动态 checkpoint + 改变训练策略",
        "BC": "多 snapshot + 改变训练策略",
        "ABC": "全部方案",
    }
    
    seeds = [7, 8, 9]  # 先测试 3 个种子
    
    all_results = {}
    
    for approach_name, approach_desc in approaches.items():
        print(f"\n{'='*80}")
        print(f"运行方案 {approach_name}: {approach_desc}")
        print(f"{'='*80}")
        
        results = []
        for seed in seeds:
            print(f"\n  运行 seed {seed}...")
            result = run_combined_approach(seed, config, approach_name)
            results.append(result)
            
            print(f"    保存 snapshot 的 step: {result['saved_snapshots_at']}")
            print(f"    最终平均准确率：{result['final']['avg_accuracy']:.4f}")
            print(f"    多 expert 平均准确率：{result['multi_expert']['avg_accuracy']:.4f}")
        
        # 汇总
        avg_final_acc = np.mean([r['final']['avg_accuracy'] for r in results])
        avg_multi_acc = np.mean([r['multi_expert']['avg_accuracy'] for r in results])
        
        print(f"\n  方案 {approach_name} 汇总：")
        print(f"    平均最终准确率：{avg_final_acc:.4f}")
        print(f"    平均多 expert 准确率：{avg_multi_acc:.4f}")
        
        all_results[approach_name] = {
            "description": approach_desc,
            "results": results,
            "avg_final_acc": avg_final_acc,
            "avg_multi_acc": avg_multi_acc,
        }

    # 最终对比
    print("\n" + "=" * 80)
    print("最终对比")
    print("=" * 80)
    
    print(f"\n{'方案':<10} {'描述':<30} {'最终准确率':<10} {'多 expert 准确率':<10}")
    print("-" * 80)
    for approach_name, data in all_results.items():
        desc = data['description'][:30]
        print(f"{approach_name:<10} {desc:<30} {data['avg_final_acc']:.4f}      {data['avg_multi_acc']:.4f}")
    
    # 找出最佳方案
    best_approach = max(all_results.items(), key=lambda x: x[1]['avg_final_acc'])
    print(f"\n最佳方案：{best_approach[0]} (平均准确率：{best_approach[1]['avg_final_acc']:.4f})")
    
    if best_approach[1]['avg_final_acc'] > 0.55:
        print("[SUCCESS] 达到目标！avg_acc > 0.55")
    else:
        print("[FAILED] 未达到目标，需要进一步优化")

    output = {
        "experiment": "combined_approaches",
        "seeds": seeds,
        "approaches": all_results,
        "best_approach": best_approach[0],
        "best_avg_acc": best_approach[1]['avg_final_acc'],
    }

    results_dir = get_results_path("combined_approaches")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果保存到：{output_path}")


if __name__ == "__main__":
    main()
