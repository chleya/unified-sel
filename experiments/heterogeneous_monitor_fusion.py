"""
experiments/heterogeneous_monitor_fusion.py — 异质监控信号融合实验

核心思想：
- 异质监控组合 = 强监控（semantic） + 弱监控（external）
- 在高压力套件上测试，这里监控表现差距大，融合才能体现价值
- 目标：能否用弱监控补全强监控，在保持成功率的前提下降低平均成本

异质组合：
1. semantic（强） + external（弱）
2. semantic（强） + surface（弱）
3. counterfactual（强） + external（弱）

高压力套件：
- code-16（有质数/除数歧义任务）
- mixed-32（推理+代码混合，压力更大）
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.capability_benchmark import run_capability_benchmark
from core.experiment_utils import run_multiple_seeds_with_cache


def run_heterogeneous_fusion(
    suite: str,
    num_tasks: int,
    seeds: List[int],
    local_solver_name: str = "search",
) -> Dict:
    """运行异质监控融合实验。
    
    Args:
        suite: 测试套件名称 (code-16, mixed-32)
        num_tasks: 任务数量
        seeds: 随机种子列表
        local_solver_name: 本地求解器名称
    
    Returns:
        实验结果字典
    """
    # 异质监控组合
    fusion_combinations = [
        ("semantic", "external"),
        ("semantic", "surface"),
        ("counterfactual", "external"),
    ]
    
    # 单一监控基线
    single_monitors = ["semantic", "counterfactual", "external", "surface"]
    
    print(f"\n{'='*60}")
    print(f"异质监控信号融合实验")
    print(f"{'='*60}")
    print(f"套件：{suite}")
    print(f"任务数：{num_tasks}")
    print(f"种子：{seeds}")
    print(f"异质组合：{fusion_combinations}")
    print(f"单一监控基线：{single_monitors}")
    print(f"{'='*60}\n")
    
    # 第一步：运行所有单一监控
    print("步骤 1: 运行单一监控基线...")
    single_results = {}
    
    for monitor in single_monitors:
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
            experiment_name=f"hetero_single_{monitor}",
            cache_prefix=f"hetero_single_{monitor}",
        )
        
        single_results[monitor] = results
    
    print(f"  完成 {len(single_monitors)} 个单一监控\n")
    
    # 第二步：分析单一监控表现
    print("步骤 2: 分析单一监控表现...")
    single_stats = {}
    
    for monitor in single_monitors:
        all_successes = []
        all_costs = []
        
        for result in single_results[monitor]:
            for task_result in result["results"]:
                all_successes.append(task_result.get("success", False))
                all_costs.append(task_result.get("cost_units", 1.0))
        
        single_stats[monitor] = {
            "success_rate": np.mean(all_successes),
            "mean_cost": np.mean(all_costs),
        }
        
        print(f"  {monitor}:")
        print(f"    成功率：{single_stats[monitor]['success_rate']:.4f}")
        print(f"    平均成本：{single_stats[monitor]['mean_cost']:.2f}")
    
    print()
    
    # 第三步：运行异质监控融合
    print("步骤 3: 运行异质监控融合...")
    fusion_results = {}
    
    for (strong_monitor, weak_monitor) in fusion_combinations:
        print(f"  异质组合：{strong_monitor} + {weak_monitor}")
        
        def run_fusion(seed):
            # 获取该种子下两个监控器的结果
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
            
            # 融合策略：
            # - 如果强监控决策是 escalate，直接 escalate（信任强监控）
            # - 如果强监控决策是 verify，用弱监控信号辅助判断：
            #   - 弱监控信号低（<0.3）说明安全，直接 accept，节省成本
            #   - 否则仍然 verify
            # - 如果强监控决策是 accept，用弱监控信号 double check：
            #   - 弱监控信号高（>0.7）说明可能有问题，升级到 verify
            #   - 否则保持 accept
            fused_decisions = []
            task_costs = []
            task_successes = []
            
            n_tasks = min(len(strong_result.get("results", [])), len(weak_result.get("results", [])))
            for task_idx in range(n_tasks):
                strong_task_result = strong_result["results"][task_idx]
                weak_task_result = weak_result["results"][task_idx]
                
                strong_signal = strong_task_result.get("routing_signal", 0.0)
                strong_decision = strong_task_result.get("decision", "accept")
                weak_signal = weak_task_result.get("routing_signal", 0.0)
                
                # 融合决策逻辑
                if strong_decision == "escalate":
                    fused_decision = "escalate"
                elif strong_decision == "verify":
                    if weak_signal < 0.3:
                        fused_decision = "accept"
                    else:
                        fused_decision = "verify"
                else:  # strong_decision == "accept"
                    if weak_signal > 0.7:
                        fused_decision = "verify"
                    else:
                        fused_decision = "accept"
                
                fused_decisions.append(fused_decision)
                
                if fused_decision == "accept":
                    task_success = strong_task_result.get("success", False)
                    task_cost = strong_task_result.get("cost_units", 1.0)
                elif fused_decision == "verify":
                    task_success = strong_task_result.get("success", False)
                    task_cost = strong_task_result.get("cost_units", 1.0) + 0.2
                else:
                    task_success = strong_task_result.get("success", False)
                    task_cost = 5.3
                
                task_costs.append(task_cost)
                task_successes.append(task_success)
            
            success_rate = np.mean(task_successes) if task_successes else 0.0
            mean_cost = np.mean(task_costs) if task_costs else 0.0
            
            return {
                "seed": seed,
                "strong_monitor": strong_monitor,
                "weak_monitor": weak_monitor,
                "fused_decisions": fused_decisions,
                "success_rate": success_rate,
                "mean_cost": mean_cost,
                "num_tasks": num_tasks,
            }
        
        # 运行所有种子
        results = []
        for seed in seeds:
            result = run_fusion(seed)
            results.append(result)
            print(f"    种子 {seed}: 成功率 {result['success_rate']:.4f}, 成本 {result['mean_cost']:.2f}")
        
        fusion_results[f"{strong_monitor}+{weak_monitor}"] = results
        print()
    
    # 第四步：对比分析
    print("步骤 4: 对比分析...")
    
    print("\n最终对比:")
    print("  单一监控:")
    for monitor, stats in single_stats.items():
        print(f"    {monitor}: 成功率 {stats['success_rate']:.4f}, 成本 {stats['mean_cost']:.2f}")
    
    print("\n  异质融合:")
    for combo_name, results in fusion_results.items():
        success_rates = [r["success_rate"] for r in results]
        costs = [r["mean_cost"] for r in results]
        print(f"    {combo_name}: 成功率 {np.mean(success_rates):.4f}, 成本 {np.mean(costs):.2f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/heterogeneous_monitor_fusion")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"fusion_results_{suite}_{timestamp}.json"
    
    output_data = {
        "config": {
            "suite": suite,
            "num_tasks": num_tasks,
            "seeds": seeds,
            "fusion_combinations": fusion_combinations,
            "single_monitors": single_monitors,
        },
        "single_stats": single_stats,
        "fusion_results": fusion_results,
        "timestamp": timestamp,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到：{output_file}")
    print(f"{'='*60}\n")
    
    return output_data


def main():
    """主函数。"""
    # 在高压力套件上测试
    suites = ["code", "mixed"]
    num_tasks_map = {"code": 20, "mixed": 40}
    seeds = [7, 8, 9]
    
    all_results = {}
    
    for suite in suites:
        print(f"\n{'='*60}")
        print(f"运行套件：{suite}")
        print(f"{'='*60}")
        
        results = run_heterogeneous_fusion(
            suite=suite,
            num_tasks=num_tasks_map[suite],
            seeds=seeds,
        )
        
        all_results[suite] = results
    
    return all_results


if __name__ == "__main__":
    main()
