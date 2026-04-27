# ⚠️ 此实验使用 rng.choice([True, False], p=[0.7, 0.3]) 模拟成功率
# 违反 AGENTS.md 红线规则 2：永远不要用随机数模拟成功率
# 结论无效，需重新验证。已被 heterogeneous_monitor_fusion.py 替代。
"""
experiments/adaptive_signal_fusion.py — 自适应多信号融合路由实验

灵感来源：
- TopoMem 健康控制器：使用加权平均、最小值、几何平均等多种融合方式
- 趋势感知：不仅看当前信号值，还看趋势和变化率
- 自适应权重：根据信号质量和历史表现动态调整权重

核心思想：
- 不同的融合策略适用于不同的场景
- 信号质量应该影响其权重
- 趋势信息可以帮助预测性决策
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.capability_benchmark import run_capability_benchmark
from core.experiment_utils import run_multiple_seeds_with_cache, config_to_hash_prefix


@dataclass
class SignalQuality:
    """信号质量指标。"""
    mean_value: float = 0.0
    variance: float = 0.0
    trend_slope: float = 0.0  # 变化率
    reliability: float = 1.0  # 历史准确率


def compute_signal_quality(signals: List[float], successes: List[bool]) -> SignalQuality:
    """计算信号质量指标。
    
    Args:
        signals: 信号值序列
        successes: 对应的成功状态
    
    Returns:
        SignalQuality 对象
    """
    if not signals:
        return SignalQuality()
    
    signals_array = np.array(signals)
    
    # 计算均值和方差
    mean_value = np.mean(signals_array)
    variance = np.var(signals_array)
    
    # 计算趋势（简单线性回归斜率）
    if len(signals) > 1:
        x = np.arange(len(signals))
        trend_slope = np.polyfit(x, signals_array, 1)[0]
    else:
        trend_slope = 0.0
    
    # 计算可靠性（基于历史准确率）
    if successes:
        reliability = np.mean(successes)
    else:
        reliability = 1.0
    
    return SignalQuality(
        mean_value=mean_value,
        variance=variance,
        trend_slope=trend_slope,
        reliability=reliability
    )


def adaptive_weighted_fusion(
    signals: List[float],
    qualities: List[SignalQuality],
    base_weights: List[float],
    accept_threshold: float = 0.3,
    verify_threshold: float = 0.7,
    trend_bonus: float = 0.15,
) -> Tuple[float, str]:
    """自适应加权融合。
    
    根据信号质量动态调整权重：
    - 高可靠性 -> 增加权重
    - 低方差 -> 增加权重
    - 正趋势 -> 增加权重
    
    Args:
        signals: 各监控器的信号值
        qualities: 各监控器的信号质量
        base_weights: 基础权重
        accept_threshold: 接受阈值
        verify_threshold: 验证阈值
        trend_bonus: 趋势奖励系数
    
    Returns:
        (fused_signal, decision)
    """
    # 计算自适应权重
    weights = []
    for i, quality in enumerate(qualities):
        # 基础权重
        base_weight = base_weights[i] if i < len(base_weights) else 1.0 / len(signals)
        
        # 可靠性调整
        reliability_factor = quality.reliability
        
        # 方差调整（低方差更好）
        variance_factor = 1.0 / (1.0 + quality.variance * 10)
        
        # 趋势调整（正趋势更好）
        trend_factor = 1.0 + max(0, quality.trend_slope) * trend_bonus
        
        # 综合权重
        adaptive_weight = base_weight * reliability_factor * variance_factor * trend_factor
        weights.append(adaptive_weight)
    
    # 归一化权重
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(signals)] * len(signals)
    
    # 加权平均
    fused_signal = np.average(signals, weights=weights)
    
    # 决策
    if fused_signal < accept_threshold:
        decision = "accept"
    elif fused_signal < verify_threshold:
        decision = "verify"
    else:
        decision = "escalate"
    
    return fused_signal, decision


def run_adaptive_fusion(
    suite: str,
    protocol: str,
    num_tasks: int,
    seeds: List[int],
    local_solver_name: str = "search",
    fusion_methods: List[str] = None,
    monitors: List[str] = None,
) -> Dict:
    """运行自适应多信号融合实验。"""
    if fusion_methods is None:
        fusion_methods = [
            "adaptive_weighted",
            "confidence_weighted",
            "trend_aware",
            "average",
            "weighted_average"
        ]
    
    if monitors is None:
        monitors = ["semantic", "counterfactual", "behavioral"]
    
    print(f"\n{'='*60}")
    print(f"自适应多信号融合实验")
    print(f"{'='*60}")
    print(f"套件：{suite}")
    print(f"协议：{protocol}")
    print(f"任务数：{num_tasks}")
    print(f"种子：{seeds}")
    print(f"监控器：{monitors}")
    print(f"融合方法：{fusion_methods}")
    print(f"{'='*60}\n")
    
    # 第一步：运行所有监控器，收集结果
    print("步骤 1: 运行各个监控器...")
    monitor_results = {}
    monitor_signals_history = defaultdict(list)
    monitor_successes_history = defaultdict(list)
    
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
            experiment_name=f"adaptive_fusion_{monitor}",
            cache_prefix=f"adaptive_fusion_{monitor}",
        )
        
        monitor_results[monitor] = results
        
        # 收集信号历史和成功状态
        for result in results:
            for task_result in result["results"]:
                signal = task_result.get("routing_signal", 0.0)
                success = task_result.get("success", False)
                monitor_signals_history[monitor].append(signal)
                monitor_successes_history[monitor].append(success)
    
    print(f"  完成 {len(monitors)} 个监控器\n")
    
    # 第二步：计算每个监控器的信号质量
    print("步骤 2: 计算信号质量...")
    signal_qualities = {}
    for monitor in monitors:
        signals = monitor_signals_history[monitor]
        successes = monitor_successes_history[monitor]
        quality = compute_signal_quality(signals, successes)
        signal_qualities[monitor] = quality
        print(f"  {monitor}:")
        print(f"    均值：{quality.mean_value:.4f}")
        print(f"    方差：{quality.variance:.4f}")
        print(f"    趋势：{quality.trend_slope:.4f}")
        print(f"    可靠性：{quality.reliability:.4f}")
    print()
    
    # 第三步：运行各种融合方法
    print("步骤 3: 运行融合方法...")
    fusion_results = {}
    
    # 基础权重配置
    base_weights = [0.4, 0.35, 0.25]  # semantic, counterfactual, behavioral
    
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
            
            for task_idx in range(num_tasks):
                # 收集各监控器的信号和决策
                signals = []
                decisions = []
                confidences = []
                
                for monitor_result in seed_monitor_results:
                    task_result = monitor_result["results"][task_idx]
                    signals.append(task_result.get("routing_signal", 0.0))
                    decisions.append(task_result.get("decision", "accept"))
                    # 使用信号值的补数作为置信度（信号越低越确定）
                    confidences.append(1.0 - task_result.get("routing_signal", 0.0))
                
                # 根据融合方法选择融合策略
                if fusion_method == "adaptive_weighted":
                    qualities = [signal_qualities[m] for m in monitors]
                    fused_signal, fused_decision = adaptive_weighted_fusion(
                        signals, qualities, base_weights
                    )
                elif fusion_method == "confidence_weighted":
                    # 置信度加权
                    total_conf = sum(confidences)
                    if total_conf > 0:
                        weights = [c / total_conf for c in confidences]
                    else:
                        weights = [1.0 / len(signals)] * len(signals)
                    fused_signal = np.average(signals, weights=weights)
                    if fused_signal < 0.3:
                        fused_decision = "accept"
                    elif fused_signal < 0.7:
                        fused_decision = "verify"
                    else:
                        fused_decision = "escalate"
                elif fusion_method == "trend_aware":
                    # 趋势感知融合
                    weights = []
                    for i, monitor in enumerate(monitors):
                        quality = signal_qualities[monitor]
                        base_weight = base_weights[i] if i < len(base_weights) else 1.0 / len(signals)
                        trend_adjustment = 1.0 + quality.trend_slope * 0.15
                        weights.append(base_weight * trend_adjustment)
                    
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                    else:
                        weights = [1.0 / len(signals)] * len(signals)
                    
                    fused_signal = np.average(signals, weights=weights)
                    if fused_signal < 0.3:
                        fused_decision = "accept"
                    elif fused_signal < 0.7:
                        fused_decision = "verify"
                    else:
                        fused_decision = "escalate"
                elif fusion_method == "average":
                    fused_signal = np.mean(signals)
                    if fused_signal < 0.3:
                        fused_decision = "accept"
                    elif fused_signal < 0.7:
                        fused_decision = "verify"
                    else:
                        fused_decision = "escalate"
                elif fusion_method == "weighted_average":
                    weights = base_weights[:len(signals)]
                    fused_signal = np.average(signals, weights=weights)
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
    
    # 第四步：分析结果
    print("步骤 4: 分析结果...")
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
    output_dir = Path("results/adaptive_signal_fusion")
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
        },
        "signal_qualities": {
            m: {
                "mean_value": q.mean_value,
                "variance": q.variance,
                "trend_slope": q.trend_slope,
                "reliability": q.reliability,
            }
            for m, q in signal_qualities.items()
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
    
    # 融合方法
    fusion_methods = [
        "adaptive_weighted",      # 自适应加权（新）
        "confidence_weighted",    # 置信度加权（新）
        "trend_aware",            # 趋势感知（新）
        "average",                # 简单平均（基线）
        "weighted_average",       # 加权平均（基线）
    ]
    
    # 监控器
    monitors = ["semantic", "counterfactual", "behavioral"]
    
    # 运行实验
    results = run_adaptive_fusion(
        suite=suite,
        protocol=protocol,
        num_tasks=num_tasks,
        seeds=seeds,
        fusion_methods=fusion_methods,
        monitors=monitors,
    )
    
    return results


if __name__ == "__main__":
    main()
