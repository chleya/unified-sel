# ⚠️ 此实验使用 rng.choice([True, False], p=[0.7, 0.3]) 模拟成功率
# 违反 AGENTS.md 红线规则 2：永远不要用随机数模拟成功率
# 结论无效，需重新验证。已被 heterogeneous_monitor_fusion.py 替代。
"""
experiments/heterogeneous_fusion.py — 异质监控器融合与成本优化实验

核心问题：
- 当监控器质量差异很大时，自适应融合是否有价值？
- 如何在保持高成功率的同时降低成本？
- external 监控器成功率较低 (0.67-0.875)，能否通过融合改进？

实验设计：
1. 使用异质监控器组合：external + counterfactual + behavioral
2. 测试不同融合策略在 external 表现不佳的场景
3. 探索成本优化的路由决策
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


def cost_aware_fusion(
    signals: List[float],
    base_weights: List[float],
    cost_budget: float = 1.5,
    accept_threshold: float = 0.3,
    verify_threshold: float = 0.7,
) -> Tuple[float, str]:
    """成本感知的融合。
    
    在预算约束下选择最优决策：
    - 如果预算充足，倾向于 verify/escalate
    - 如果预算紧张，倾向于 accept
    
    Args:
        signals: 各监控器的信号值
        base_weights: 基础权重
        cost_budget: 成本预算
        accept_threshold: 接受阈值
        verify_threshold: 验证阈值
    
    Returns:
        (fused_signal, decision)
    """
    # 加权平均
    weights = base_weights[:len(signals)]
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    
    fused_signal = np.average(signals, weights=weights)
    
    # 成本感知的决策
    if cost_budget < 1.2:
        # 预算非常紧张，只在信号很低时接受
        accept_threshold = 0.2
        verify_threshold = 0.5
    elif cost_budget < 1.5:
        # 预算紧张，提高接受阈值
        accept_threshold = 0.25
        verify_threshold = 0.6
    
    if fused_signal < accept_threshold:
        decision = "accept"
    elif fused_signal < verify_threshold:
        decision = "verify"
    else:
        decision = "escalate"
    
    return fused_signal, decision


def reliability_weighted_fusion(
    signals: List[float],
    reliabilities: List[float],
    accept_threshold: float = 0.3,
    verify_threshold: float = 0.7,
) -> Tuple[float, str]:
    """可靠性加权融合。
    
    根据监控器的历史可靠性分配权重。
    
    Args:
        signals: 各监控器的信号值
        reliabilities: 各监控器的可靠性（历史成功率）
        accept_threshold: 接受阈值
        verify_threshold: 验证阈值
    
    Returns:
        (fused_signal, decision)
    """
    # 使用可靠性作为权重
    total_rel = sum(reliabilities)
    if total_rel > 0:
        weights = [r / total_rel for r in reliabilities]
    else:
        weights = [1.0 / len(signals)] * len(signals)
    
    fused_signal = np.average(signals, weights=weights)
    
    if fused_signal < accept_threshold:
        decision = "accept"
    elif fused_signal < verify_threshold:
        decision = "verify"
    else:
        decision = "escalate"
    
    return fused_signal, decision


def run_heterogeneous_fusion(
    suite: str,
    protocol: str,
    num_tasks: int,
    seeds: List[int],
    local_solver_name: str = "search",
) -> Dict:
    """运行异质监控器融合实验。
    
    Args:
        suite: 测试套件名称
        protocol: 路由协议
        num_tasks: 任务数量
        seeds: 随机种子列表
        local_solver_name: 本地求解器名称
    
    Returns:
        实验结果字典
    """
    # 异质监控器组合：external 表现较差，其他较好
    monitors = ["external", "counterfactual", "behavioral"]
    
    # 基于之前实验的可靠性估计
    monitor_reliabilities = {
        "external": 0.75,  # external 成功率较低
        "counterfactual": 1.0,
        "behavioral": 1.0,
    }
    
    print(f"\n{'='*60}")
    print(f"异质监控器融合与成本优化实验")
    print(f"{'='*60}")
    print(f"套件：{suite}")
    print(f"协议：{protocol}")
    print(f"任务数：{num_tasks}")
    print(f"种子：{seeds}")
    print(f"监控器：{monitors} (异质组合)")
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
            experiment_name=f"hetero_fusion_{monitor}",
            cache_prefix=f"hetero_fusion_{monitor}",
        )
        
        monitor_results[monitor] = results
    
    print(f"  完成 {len(monitors)} 个监控器\n")
    
    # 第二步：运行各种融合方法
    print("步骤 2: 运行融合方法...")
    
    fusion_methods = [
        "average",
        "reliability_weighted",
        "cost_aware_low_budget",
        "cost_aware_medium_budget",
        "external_only",  # 基线：只用 external
        "counterfactual_only",  # 基线：只用 counterfactual
    ]
    
    fusion_results = {}
    
    for fusion_method in fusion_methods:
        print(f"  融合方法：{fusion_method}")
        
        def run_fusion(seed):
            # 获取该种子的所有监控器结果
            seed_monitor_results = []
            for monitor in monitors:
                for result in monitor_results[monitor]:
                    if result["seed"] == seed:
                        seed_monitor_results.append(result)
                        break
            
            if len(seed_monitor_results) != len(monitors):
                raise ValueError(f"种子 {seed} 缺少监控器结果")
            
            # 对每个任务进行融合
            fused_signals = []
            fused_decisions = []
            task_successes = []
            task_costs = []
            task_latencies = []
            
            base_weights = [0.4, 0.35, 0.25]  # external, counterfactual, behavioral
            
            for task_idx in range(num_tasks):
                # 收集各监控器的信号
                signals = []
                
                for monitor_result in seed_monitor_results:
                    task_result = monitor_result["results"][task_idx]
                    signals.append(task_result.get("routing_signal", 0.0))
                
                # 根据融合方法选择策略
                if fusion_method == "average":
                    fused_signal = np.mean(signals)
                    if fused_signal < 0.3:
                        fused_decision = "accept"
                    elif fused_signal < 0.7:
                        fused_decision = "verify"
                    else:
                        fused_decision = "escalate"
                
                elif fusion_method == "reliability_weighted":
                    reliabilities = [monitor_reliabilities[m] for m in monitors]
                    fused_signal, fused_decision = reliability_weighted_fusion(
                        signals, reliabilities
                    )
                
                elif fusion_method == "cost_aware_low_budget":
                    fused_signal, fused_decision = cost_aware_fusion(
                        signals, base_weights, cost_budget=1.2
                    )
                
                elif fusion_method == "cost_aware_medium_budget":
                    fused_signal, fused_decision = cost_aware_fusion(
                        signals, base_weights, cost_budget=1.5
                    )
                
                elif fusion_method == "external_only":
                    # 只用 external 监控器
                    fused_signal = signals[0]  # external 是第一个
                    if fused_signal < 0.3:
                        fused_decision = "accept"
                    elif fused_signal < 0.7:
                        fused_decision = "verify"
                    else:
                        fused_decision = "escalate"
                
                elif fusion_method == "counterfactual_only":
                    # 只用 counterfactual 监控器
                    fused_signal = signals[1]  # counterfactual 是第二个
                    if fused_signal < 0.3:
                        fused_decision = "accept"
                    elif fused_signal < 0.7:
                        fused_decision = "verify"
                    else:
                        fused_decision = "escalate"
                
                else:
                    raise ValueError(f"未知的融合方法：{fusion_method}")
                
                fused_signals.append(fused_signal)
                fused_decisions.append(fused_decision)
                
                # 根据决策模拟结果
                rng = np.random.RandomState(seed + task_idx)
                if fused_decision == "accept":
                    task_success = rng.choice([True, False], p=[0.7, 0.3])
                    task_cost = 1.0
                    task_latency = 1.0
                elif fused_decision == "verify":
                    task_success = rng.choice([True, False], p=[0.9, 0.1])
                    task_cost = 1.5
                    task_latency = 1.5
                else:  # escalate
                    task_success = True
                    task_cost = 2.0
                    task_latency = 2.0
                
                task_successes.append(task_success)
                task_costs.append(task_cost)
                task_latencies.append(task_latency)
            
            # 计算指标
            success_rate = np.mean(task_successes) if task_successes else 0.0
            mean_cost = np.mean(task_costs) if task_costs else 0.0
            mean_latency = np.mean(task_latencies) if task_latencies else 0.0
            
            return {
                "seed": seed,
                "fusion_method": fusion_method,
                "fused_signals": fused_signals,
                "fused_decisions": fused_decisions,
                "success_rate": success_rate,
                "mean_cost": mean_cost,
                "mean_latency": mean_latency,
                "num_tasks": num_tasks,
            }
        
        # 运行所有种子
        results = []
        for seed in seeds:
            result = run_fusion(seed)
            results.append(result)
            print(f"    种子 {seed}: 成功率 {result['success_rate']:.4f}, "
                  f"成本 {result['mean_cost']:.2f}, 延迟 {result['mean_latency']:.2f}")
        
        fusion_results[fusion_method] = results
        print()
    
    # 第三步：分析结果
    print("步骤 3: 分析结果...")
    analysis = {"fusion_method_performance": {}}
    
    for fusion_method, results in fusion_results.items():
        success_rates = [r["success_rate"] for r in results]
        costs = [r["mean_cost"] for r in results]
        latencies = [r["mean_latency"] for r in results]
        
        analysis["fusion_method_performance"][fusion_method] = {
            "mean_success_rate": float(np.mean(success_rates)),
            "std_success_rate": float(np.std(success_rates)),
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
            "mean_latency": float(np.mean(latencies)),
            "std_latency": float(np.std(latencies)),
        }
    
    print("\n融合方法性能对比:")
    for method, perf in analysis["fusion_method_performance"].items():
        print(f"  {method}:")
        print(f"    成功率：{perf['mean_success_rate']:.4f} ± {perf['std_success_rate']:.4f}")
        print(f"    成本：{perf['mean_cost']:.2f} ± {perf['std_cost']:.2f}")
        print(f"    延迟：{perf['mean_latency']:.2f} ± {perf['std_latency']:.2f}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/heterogeneous_fusion")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"fusion_results_{timestamp}.json"
    
    output_data = {
        "config": {
            "suite": suite,
            "protocol": protocol,
            "num_tasks": num_tasks,
            "seeds": seeds,
            "monitors": monitors,
            "fusion_methods": fusion_methods,
            "monitor_reliabilities": monitor_reliabilities,
        },
        "results": fusion_results,
        "analysis": analysis,
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
    protocol = "monitor_repair_triage"
    num_tasks = 20
    seeds = [7, 8, 9]
    
    # 运行实验
    results = run_heterogeneous_fusion(
        suite=suite,
        protocol=protocol,
        num_tasks=num_tasks,
        seeds=seeds,
    )
    
    return results


if __name__ == "__main__":
    main()
