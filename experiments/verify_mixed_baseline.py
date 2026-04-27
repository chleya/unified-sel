"""
experiments/verify_mixed_baseline.py — 在 mixed 套件上验证新基线

只在 mixed-40 上验证 semantic+external 融合策略的表现
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.capability_benchmark import run_capability_benchmark
from core.experiment_utils import run_multiple_seeds_with_cache


def run_mixed_verification():
    """在 mixed 套件上验证"""
    suite = "mixed"
    num_tasks = 40
    seeds = [7, 8, 9]
    strong_monitor = "semantic"
    weak_monitor = "external"
    accept_threshold = 0.2
    verify_threshold = 0.55
    
    print(f"\n{'='*60}")
    print(f"在 mixed-40 上验证新基线")
    print(f"{'='*60}")
    
    # 第一步：运行两个监控器
    print("\n步骤 1: 运行监控器基线...")
    single_results = {}
    
    for monitor in [strong_monitor, weak_monitor]:
        print(f"  运行监控器：{monitor}")
        
        def run_single(seed):
            results = run_capability_benchmark(
                suite=suite,
                protocol="monitor_repair_triage",
                num_tasks=num_tasks,
                seed=seed,
                local_solver_name="search",
                routing_monitor_name=monitor,
            )
            return results
        
        results, num_cached = run_multiple_seeds_with_cache(
            seed_fn=run_single,
            seeds=seeds,
            experiment_name=f"mixed_verify_{monitor}",
            cache_prefix=f"mixed_verify_{monitor}",
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
    
    # 第三步：运行融合策略
    print("步骤 3: 运行融合策略...")
    fusion_results = []
    
    for seed in seeds:
        print(f"  种子 {seed}...")
        
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
        
        # 融合策略
        fused_decisions = []
        task_successes = []
        task_costs = []
        
        for task_idx in range(num_tasks):
            strong_task_result = strong_result["results"][task_idx]
            weak_task_result = weak_result["results"][task_idx]
            
            strong_signal = strong_task_result.get("routing_signal", 0.0)
            strong_decision = strong_task_result.get("decision", "accept")
            weak_signal = weak_task_result.get("routing_signal", 0.0)
            
            # 融合决策逻辑
            if strong_decision == "escalate":
                fused_decision = "escalate"
            elif strong_decision == "verify":
                if weak_signal < accept_threshold:
                    fused_decision = "accept"
                else:
                    fused_decision = "verify"
            else:  # strong_decision == "accept"
                if weak_signal > verify_threshold:
                    fused_decision = "verify"
                else:
                    fused_decision = "accept"
            
            fused_decisions.append(fused_decision)
            
            # 根据融合决策模拟结果
            if fused_decision == "accept":
                task_success = strong_task_result.get("success", False)
                task_cost = 1.0
            elif fused_decision == "verify":
                task_success = strong_task_result.get("success", False)
                task_cost = 1.5
            else:  # escalate
                task_success = True
                task_cost = 2.0
            
            task_successes.append(task_success)
            task_costs.append(task_cost)
        
        success_rate = np.mean(task_successes) if task_successes else 0.0
        mean_cost = np.mean(task_costs) if task_costs else 0.0
        
        fusion_results.append({
            "seed": seed,
            "success_rate": success_rate,
            "mean_cost": mean_cost,
            "num_tasks": num_tasks,
        })
        
        print(f"    成功率：{success_rate:.4f}, 成本：{mean_cost:.2f}")
    
    # 第四步：分析结果
    print("\n步骤 4: 分析结果...")
    fusion_success_rates = [r["success_rate"] for r in fusion_results]
    fusion_costs = [r["mean_cost"] for r in fusion_results]
    
    mean_fusion_success = np.mean(fusion_success_rates)
    mean_fusion_cost = np.mean(fusion_costs)
    
    print("\n最终对比:")
    print(f"  {strong_monitor} (单一): 成功率 {baseline_stats[strong_monitor]['success_rate']:.4f}, 成本 {baseline_stats[strong_monitor]['mean_cost']:.2f}")
    print(f"  {strong_monitor}+{weak_monitor} (融合): 成功率 {mean_fusion_success:.4f}, 成本 {mean_fusion_cost:.2f}")
    
    # 计算成本降低
    strong_cost = baseline_stats[strong_monitor]['mean_cost']
    cost_reduction = ((strong_cost - mean_fusion_cost) / strong_cost) * 100
    print(f"\n成本降低：{cost_reduction:.1f}%")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/mixed_verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"verification_results_{timestamp}.json"
    
    output_data = {
        "config": {
            "suite": suite,
            "num_tasks": num_tasks,
            "seeds": seeds,
            "strong_monitor": strong_monitor,
            "weak_monitor": weak_monitor,
            "accept_threshold": accept_threshold,
            "verify_threshold": verify_threshold,
        },
        "baseline_stats": baseline_stats,
        "fusion_results": fusion_results,
        "summary": {
            "mean_fusion_success_rate": float(mean_fusion_success),
            "mean_fusion_cost": float(mean_fusion_cost),
            "cost_reduction_percent": float(cost_reduction),
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
    results = run_mixed_verification()
    return results


if __name__ == "__main__":
    main()
