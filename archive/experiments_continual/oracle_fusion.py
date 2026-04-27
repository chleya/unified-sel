"""
experiments/oracle_fusion.py — 基于真实结果的融合评估

核心思想：
- 不使用模拟，直接使用监控器的真实成功/失败结果
- 评估融合决策是否能改进单个监控器的表现
- 重点：external 监控器在压力下成功率只有 0.67，能否通过融合改进？

实验逻辑：
1. 运行多个监控器，获取它们的真实决策和结果
2. 对于每个任务，比较融合决策 vs 单一监控器决策
3. 评估融合是否能：
   - 提高成功率
   - 降低成本
   - 在保持成功率的同时降低成本
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


def run_oracle_fusion(
    suite: str,
    num_tasks: int,
    seeds: List[int],
    local_solver_name: str = "search",
) -> Dict:
    """运行基于真实结果的融合评估。
    
    Args:
        suite: 测试套件名称
        num_tasks: 任务数量
        seeds: 随机种子列表
        local_solver_name: 本地求解器名称
    
    Returns:
        实验结果字典
    """
    # 异质监控器组合
    monitors = ["external", "counterfactual", "behavioral", "surface"]
    
    print(f"\n{'='*60}")
    print(f"基于真实结果的融合评估")
    print(f"{'='*60}")
    print(f"套件：{suite}")
    print(f"任务数：{num_tasks}")
    print(f"种子：{seeds}")
    print(f"监控器：{monitors}")
    print(f"{'='*60}\n")
    
    # 第一步：运行所有监控器
    print("步骤 1: 运行各个监控器...")
    monitor_results = {}
    
    for monitor in monitors:
        print(f"  运行监控器：{monitor}")
        
        def run_monitor(seed):
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
            seed_fn=run_monitor,
            seeds=seeds,
            experiment_name=f"oracle_fusion_{monitor}",
            cache_prefix=f"oracle_fusion_{monitor}",
        )
        
        monitor_results[monitor] = results
        print(f"    缓存：{num_cached} 个结果")
    
    print(f"  完成 {len(monitors)} 个监控器\n")
    
    # 第二步：分析各监控器的真实表现
    print("步骤 2: 分析各监控器的真实表现...")
    monitor_stats = {}
    
    for monitor in monitors:
        all_successes = []
        all_costs = []
        all_escalations = []
        
        for result in monitor_results[monitor]:
            for task_result in result["results"]:
                all_successes.append(task_result.get("success", False))
                all_costs.append(task_result.get("cost_units", 1.0))
                all_escalations.append(task_result.get("escalated", False))
        
        monitor_stats[monitor] = {
            "success_rate": np.mean(all_successes),
            "mean_cost": np.mean(all_costs),
            "escalation_rate": np.mean(all_escalations),
            "total_tasks": len(all_successes),
        }
        
        print(f"  {monitor}:")
        print(f"    成功率：{monitor_stats[monitor]['success_rate']:.4f}")
        print(f"    平均成本：{monitor_stats[monitor]['mean_cost']:.2f}")
        print(f"    升级率：{monitor_stats[monitor]['escalation_rate']:.4f}")
    
    print()
    
    # 第三步：评估不同融合策略
    print("步骤 3: 评估融合策略...")
    
    # 评估策略：多数投票（基于成功/失败）
    fusion_results = {}
    
    for seed in seeds:
        print(f"  种子 {seed}:")
        
        # 收集该种子下所有监控器的结果
        seed_results = {monitor: None for monitor in monitors}
        for monitor in monitors:
            for result in monitor_results[monitor]:
                if result["seed"] == seed:
                    seed_results[monitor] = result
                    break
        
        # 分析每个任务
        task_analysis = []
        
        for task_idx in range(num_tasks):
            task_data = {"task_idx": task_idx}
            
            # 收集各监控器的结果
            for monitor in monitors:
                task_result = seed_results[monitor]["results"][task_idx]
                task_data[monitor] = {
                    "success": task_result.get("success", False),
                    "cost": task_result.get("cost_units", 1.0),
                    "escalated": task_result.get("escalated", False),
                    "signal": task_result.get("routing_signal", 0.0),
                }
            
            # 评估多数投票融合
            successes = [task_data[m]["success"] for m in monitors]
            costs = [task_data[m]["cost"] for m in monitors]
            
            # 多数投票：如果超过一半监控器成功，则融合成功
            majority_success = sum(successes) > len(monitors) / 2
            # 融合成本：取平均
            fusion_cost = np.mean(costs)
            
            task_data["fusion"] = {
                "success": majority_success,
                "cost": fusion_cost,
                "agreement": sum(successes),  # 多少个监控器成功
            }
            
            task_analysis.append(task_data)
        
        # 计算融合的表现
        fusion_successes = [t["fusion"]["success"] for t in task_analysis]
        fusion_costs = [t["fusion"]["cost"] for t in task_analysis]
        
        fusion_results[seed] = {
            "success_rate": np.mean(fusion_successes),
            "mean_cost": np.mean(fusion_costs),
            "task_analysis": task_analysis,
        }
        
        print(f"    融合成功率：{fusion_results[seed]['success_rate']:.4f}")
        print(f"    融合平均成本：{fusion_results[seed]['mean_cost']:.2f}")
    
    print()
    
    # 第四步：对比分析
    print("步骤 4: 对比分析...")
    
    # 计算平均融合表现
    avg_fusion_success = np.mean([r["success_rate"] for r in fusion_results.values()])
    avg_fusion_cost = np.mean([r["mean_cost"] for r in fusion_results.values()])
    
    print("\n最终对比:")
    print("  单一监控器:")
    for monitor, stats in monitor_stats.items():
        print(f"    {monitor}: 成功率 {stats['success_rate']:.4f}, 成本 {stats['mean_cost']:.2f}")
    
    print(f"  融合策略 (多数投票):")
    print(f"    成功率：{avg_fusion_success:.4f}")
    print(f"    成本：{avg_fusion_cost:.2f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/oracle_fusion")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"fusion_results_{timestamp}.json"
    
    output_data = {
        "config": {
            "suite": suite,
            "num_tasks": num_tasks,
            "seeds": seeds,
            "monitors": monitors,
        },
        "monitor_stats": monitor_stats,
        "fusion_results": fusion_results,
        "summary": {
            "avg_fusion_success_rate": float(avg_fusion_success),
            "avg_fusion_cost": float(avg_fusion_cost),
        },
        "timestamp": timestamp,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到：{output_file}")
    print(f"{'='*60}\n")
    
    return output_data


def main():
    """主函数。"""
    # 实验配置
    suite = "mixed"
    num_tasks = 20
    seeds = [7, 8, 9]
    
    # 运行实验
    results = run_oracle_fusion(
        suite=suite,
        num_tasks=num_tasks,
        seeds=seeds,
    )
    
    return results


if __name__ == "__main__":
    main()
