"""
experiments/fusion_threshold_optimization.py — 融合阈值参数优化

目标：
- 找到最优的 accept/verify 阈值组合
- 在保持100%成功率的前提下进一步降低成本
- 测试不同的阈值配置，找到Pareto前沿

当前默认阈值：
- accept_threshold: 0.3
- verify_threshold: 0.7

我们将测试：
- accept_threshold: 0.2, 0.25, 0.3, 0.35, 0.4
- verify_threshold: 0.5, 0.55, 0.6, 0.65, 0.7
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.capability_benchmark import run_capability_benchmark
from core.experiment_utils import run_multiple_seeds_with_cache


def run_threshold_optimization(
    suite: str,
    num_tasks: int,
    seeds: List[int],
    local_solver_name: str = "search",
    strong_monitor: str = "semantic",
    weak_monitor: str = "external",
) -> Dict:
    """运行融合阈值优化实验。
    
    Args:
        suite: 测试套件名称
        num_tasks: 任务数量
        seeds: 随机种子列表
        local_solver_name: 本地求解器名称
        strong_monitor: 强监控器名称
        weak_monitor: 弱监控器名称
    
    Returns:
        实验结果字典
    """
    # 阈值网格搜索
    accept_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
    verify_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    
    print(f"\n{'='*60}")
    print(f"融合阈值参数优化")
    print(f"{'='*60}")
    print(f"套件：{suite}")
    print(f"任务数：{num_tasks}")
    print(f"种子：{seeds}")
    print(f"强监控：{strong_monitor}")
    print(f"弱监控：{weak_monitor}")
    print(f"Accept thresholds: {accept_thresholds}")
    print(f"Verify thresholds: {verify_thresholds}")
    print(f"{'='*60}\n")
    
    # 第一步：运行两个监控器
    print("步骤 1: 运行监控器基线...")
    single_results = {}
    
    for monitor in [strong_monitor, weak_monitor]:
        print(f"  运行监控器：{monitor}")
        
        def run_single(seed):
            results = run_capability_benchmark(
                suite=suite,
                protocol="monitor_repair_triage",
                num_tasks=num_tasks,
                seed=seed,
                local_solver_name=local_solver_name,
                routing_monitor_name=monitor,
            )
            return results
        
        results, num_cached = run_multiple_seeds_with_cache(
            seed_fn=run_single,
            seeds=seeds,
            experiment_name=f"thresh_opt_{monitor}",
            cache_prefix=f"thresh_opt_{monitor}",
        )
        
        single_results[monitor] = results
    
    print(f"  完成 2 个监控器\n")
    
    # 第二步：分析基线表现
    print("步骤 2: 分析基线表现...")
    baseline_stats = {}
    
    for monitor in [strong_monitor, weak_monitor]:
        all_successes = []
        all_costs = []
        
        for result in single_results[monitor]:
            for task_result in result["results"]:
                all_successes.append(task_result.get("success", False))
                all_costs.append(task_result.get("cost_units", 1.0))
        
        baseline_stats[monitor] = {
            "success_rate": np.mean(all_successes),
            "mean_cost": np.mean(all_costs),
        }
        
        print(f"  {monitor}:")
        print(f"    成功率：{baseline_stats[monitor]['success_rate']:.4f}")
        print(f"    平均成本：{baseline_stats[monitor]['mean_cost']:.2f}")
    
    print()
    
    # 第三步：阈值网格搜索
    print("步骤 3: 阈值网格搜索...")
    threshold_results = {}
    
    for accept_thresh in accept_thresholds:
        for verify_thresh in verify_thresholds:
            threshold_key = f"accept_{accept_thresh}_verify_{verify_thresh}"
            print(f"  测试阈值组合：accept={accept_thresh}, verify={verify_thresh}")
            
            def run_with_threshold(seed):
                # 获取该种子下的结果
                strong_result = None
                weak_result = None
                
                for result in single_results[strong_monitor]:
                    if result["seed"] == seed:
                        strong_result = result
                        break
                
                for result in single_results[weak_monitor]:
                    if result["seed"] == seed:
                        weak_result = result
                        break
                
                if strong_result is None or weak_result is None:
                    raise ValueError(f"种子 {seed} 缺少监控器结果")
                
                # 融合策略（使用当前阈值）
                fused_decisions = []
                task_costs = []
                task_successes = []
                
                for task_idx in range(num_tasks):
                    strong_task_result = strong_result["results"][task_idx]
                    weak_task_result = weak_result["results"][task_idx]
                    
                    strong_signal = strong_task_result.get("routing_signal", 0.0)
                    strong_decision = strong_task_result.get("decision", "accept")
                    weak_signal = weak_task_result.get("routing_signal", 0.0)
                    
                    # 使用当前阈值的融合决策逻辑
                    if strong_decision == "escalate":
                        fused_decision = "escalate"
                    elif strong_decision == "verify":
                        if weak_signal < accept_thresh:
                            fused_decision = "accept"
                        else:
                            fused_decision = "verify"
                    else:  # strong_decision == "accept"
                        if weak_signal > verify_thresh:
                            fused_decision = "verify"
                        else:
                            fused_decision = "accept"
                    
                    fused_decisions.append(fused_decision)
                    
                    # 根据融合决策模拟成本和成功率
                    if fused_decision == "accept":
                        task_success = strong_task_result.get("success", False)
                        task_cost = 1.0
                    elif fused_decision == "verify":
                        task_success = strong_task_result.get("success", False)
                        task_cost = 1.5
                    else:  # escalate
                        task_success = True
                        task_cost = 2.0
                    
                    task_costs.append(task_cost)
                    task_successes.append(task_success)
                
                success_rate = np.mean(task_successes) if task_successes else 0.0
                mean_cost = np.mean(task_costs) if task_costs else 0.0
                
                return {
                    "seed": seed,
                    "accept_threshold": accept_thresh,
                    "verify_threshold": verify_thresh,
                    "fused_decisions": fused_decisions,
                    "success_rate": success_rate,
                    "mean_cost": mean_cost,
                    "num_tasks": num_tasks,
                }
            
            # 运行所有种子
            results = []
            for seed in seeds:
                result = run_with_threshold(seed)
                results.append(result)
                print(f"    种子 {seed}: 成功率 {result['success_rate']:.4f}, 成本 {result['mean_cost']:.2f}")
            
            threshold_results[threshold_key] = results
            print()
    
    # 第四步：分析结果，找到最优阈值
    print("步骤 4: 分析最优阈值...")
    summary = {}
    
    for threshold_key, results in threshold_results.items():
        success_rates = [r["success_rate"] for r in results]
        costs = [r["mean_cost"] for r in results]
        
        summary[threshold_key] = {
            "mean_success_rate": float(np.mean(success_rates)),
            "std_success_rate": float(np.std(success_rates)),
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
        }
    
    # 筛选出成功率 1.0 的阈值，然后找成本最低的
    valid_thresholds = []
    for threshold_key, perf in summary.items():
        if perf["mean_success_rate"] >= 1.0:
            valid_thresholds.append((threshold_key, perf["mean_cost"]))
    
    # 按成本排序
    valid_thresholds.sort(key=lambda x: x[1])
    
    print("\n最优阈值分析 (成功率 1.0 的阈值按成本排序):")
    for i, (threshold_key, cost) in enumerate(valid_thresholds[:10]):  # 显示前10个
        print(f"  {i+1}. {threshold_key}: 成本 {cost:.4f}")
    
    if valid_thresholds:
        best_threshold, best_cost = valid_thresholds[0]
        print(f"\n最优阈值: {best_threshold}")
        print(f"最优成本: {best_cost:.4f}")
        
        # 对比基线
        strong_cost = baseline_stats[strong_monitor]["mean_cost"]
        improvement = ((strong_cost - best_cost) / strong_cost) * 100
        print(f"相比单一监控成本降低: {improvement:.1f}%")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/fusion_threshold_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"optimization_results_{suite}_{timestamp}.json"
    
    output_data = {
        "config": {
            "suite": suite,
            "num_tasks": num_tasks,
            "seeds": seeds,
            "strong_monitor": strong_monitor,
            "weak_monitor": weak_monitor,
            "accept_thresholds": accept_thresholds,
            "verify_thresholds": verify_thresholds,
        },
        "baseline_stats": baseline_stats,
        "threshold_results": threshold_results,
        "summary": summary,
        "valid_thresholds": valid_thresholds,
        "best_threshold": best_threshold if valid_thresholds else None,
        "best_cost": best_cost if valid_thresholds else None,
        "timestamp": timestamp,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到：{output_file}")
    print(f"{'='*60}\n")
    
    return output_data


def main():
    """主函数。"""
    # 在 code 和 mixed 套件上优化阈值
    suites = ["code", "mixed"]
    num_tasks_map = {"code": 20, "mixed": 40}
    seeds = [7, 8, 9]
    
    all_results = {}
    
    for suite in suites:
        print(f"\n{'='*60}")
        print(f"运行套件：{suite}")
        print(f"{'='*60}")
        
        results = run_threshold_optimization(
            suite=suite,
            num_tasks=num_tasks_map[suite],
            seeds=seeds,
            strong_monitor="semantic",
            weak_monitor="external",
        )
        
        all_results[suite] = results
    
    return all_results


if __name__ == "__main__":
    main()
