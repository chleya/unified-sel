"""
路由阈值验证实验

验证最优阈值在不同协议上的表现：
- monitor_triage
- monitor_repair_triage

测试监控器+阈值组合：
- diagnostic @ 0.70
- external @ 0.60
- counterfactual @ 0.70
- surface @ 0.60
- behavioral @ 0.60
- semantic @ 0.70

测试协议：monitor_triage, monitor_repair_triage
测试套件：code-20
种子：7, 8, 9

用法：
    python experiments/routing_threshold_validation.py

输出：
    results/capability_routing_threshold/validation_<timestamp>.json
    results/capability_routing_threshold/validation_latest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.capability_benchmark import run_capability_benchmark
from core.experiment_utils import run_seed_with_cache, config_to_hash_prefix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "capability_routing_threshold"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 最优配置（从扫描结果中得到）
OPTIMAL_CONFIGS = [
    {"monitor": "diagnostic", "threshold": 0.70},
    {"monitor": "external", "threshold": 0.60},
    {"monitor": "counterfactual", "threshold": 0.70},
    {"monitor": "surface", "threshold": 0.60},
    {"monitor": "behavioral", "threshold": 0.60},
    {"monitor": "semantic", "threshold": 0.70},
]

# 测试协议
PROTOCOLS = ["monitor_triage", "monitor_repair_triage"]

# 其他参数
NUM_TASKS = 20
SEEDS = [7, 8, 9]  # 3个种子验证鲁棒性
ESCALATION_THRESHOLD = 0.9


def run_single_validation(
    protocol: str,
    monitor: str,
    threshold: float,
    seed: int,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """运行单个验证用例"""
    
    def _benchmark_fn(_):
        return run_capability_benchmark(
            suite="code",
            protocol=protocol,
            num_tasks=NUM_TASKS,
            seed=seed,
            local_solver_name="search",
            routing_monitor_name=monitor,
            routing_signal_threshold=threshold,
            escalation_signal_threshold=ESCALATION_THRESHOLD,
        )
    
    cache_prefix = f"{protocol}_{monitor}_{threshold:.2f}_{seed}"
    result, cached = run_seed_with_cache(
        seed_fn=_benchmark_fn,
        seed=seed,
        experiment_name="capability_routing_threshold_validation",
        cache_prefix=cache_prefix,
        force_rerun=force_rerun,
    )
    
    result["monitor"] = monitor
    result["threshold"] = threshold
    result["protocol"] = protocol
    result["seed"] = seed
    result["cached"] = cached
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help="强制重新运行所有用例，忽略缓存",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("路由阈值验证实验")
    print("=" * 60)
    print(f"测试协议: {PROTOCOLS}")
    print(f"测试监控器+阈值: {[f'{c["monitor"]}@{c["threshold"]:.2f}' for c in OPTIMAL_CONFIGS]}")
    print(f"任务数: {NUM_TASKS}")
    print(f"种子: {SEEDS}")
    print()
    
    all_results = []
    total_cells = len(PROTOCOLS) * len(OPTIMAL_CONFIGS) * len(SEEDS)
    completed = 0
    
    for protocol in PROTOCOLS:
        print(f"\n=== 协议: {protocol} ===")
        for config in OPTIMAL_CONFIGS:
            monitor = config["monitor"]
            threshold = config["threshold"]
            print(f"\n  监控器: {monitor} @ {threshold:.2f}")
            
            for seed in SEEDS:
                print(f"    种子: {seed} ... ", end="", flush=True)
                
                try:
                    result = run_single_validation(
                        protocol=protocol,
                        monitor=monitor,
                        threshold=threshold,
                        seed=seed,
                        force_rerun=args.force_rerun,
                    )
                    all_results.append(result)
                    
                    sr = result["summary"]["success_rate"]
                    cost = result["summary"]["mean_cost_units"]
                    cached_note = " (缓存)" if result["cached"] else ""
                    print(f"完成 - 成功率: {sr:.2%}, 平均成本: {cost:.3f}{cached_note}")
                except Exception as e:
                    print(f"失败: {e}")
                
                completed += 1
                print(f"    进度: {completed}/{total_cells}")
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)
    
    # 按协议和监控器分组
    results_by_protocol = {}
    for result in all_results:
        protocol = result["protocol"]
        if protocol not in results_by_protocol:
            results_by_protocol[protocol] = {}
        
        monitor = result["monitor"]
        if monitor not in results_by_protocol[protocol]:
            results_by_protocol[protocol][monitor] = []
        results_by_protocol[protocol][monitor].append(result)
    
    # 计算每个协议+监控器的平均表现
    print("\n--- 每个协议+监控器的平均表现 ---")
    avg_results = []
    
    for protocol in PROTOCOLS:
        print(f"\n协议: {protocol}")
        if protocol not in results_by_protocol:
            continue
        
        protocol_results = results_by_protocol[protocol]
        for monitor, monitor_results in protocol_results.items():
            if not monitor_results:
                continue
            
            # 找到对应的阈值
            threshold = None
            for config in OPTIMAL_CONFIGS:
                if config["monitor"] == monitor:
                    threshold = config["threshold"]
                    break
            
            # 计算平均值
            success_rates = [r["summary"]["success_rate"] for r in monitor_results]
            costs = [r["summary"]["mean_cost_units"] for r in monitor_results]
            
            avg_sr = np.mean(success_rates)
            avg_cost = np.mean(costs)
            std_cost = np.std(costs)
            
            avg_results.append({
                "protocol": protocol,
                "monitor": monitor,
                "threshold": threshold,
                "avg_success_rate": avg_sr,
                "avg_cost": avg_cost,
                "std_cost": std_cost,
                "seed_count": len(monitor_results),
            })
            
            print(f"  {monitor} @ {threshold:.2f}:")
            print(f"    平均成功率: {avg_sr:.2%}")
            print(f"    平均成本: {avg_cost:.3f} ± {std_cost:.3f}")
    
    # 找到每个协议的最佳配置
    print("\n--- 每个协议的最佳配置 ---")
    best_configs = {}
    for protocol in PROTOCOLS:
        protocol_avg = [r for r in avg_results if r["protocol"] == protocol]
        if not protocol_avg:
            continue
        
        # 按成功率降序，成本升序排序
        sorted_configs = sorted(
            protocol_avg,
            key=lambda x: (-x["avg_success_rate"], x["avg_cost"])
        )
        best = sorted_configs[0]
        best_configs[protocol] = best
        
        print(f"\n{protocol} 最佳配置:")
        print(f"  {best['monitor']} @ {best['threshold']:.2f}")
        print(f"    平均成功率: {best['avg_success_rate']:.2%}")
        print(f"    平均成本: {best['avg_cost']:.3f} ± {best['std_cost']:.3f}")
    
    # 保存完整结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    full_result = {
        "timestamp": timestamp,
        "optimal_configs": OPTIMAL_CONFIGS,
        "protocols": PROTOCOLS,
        "num_tasks": NUM_TASKS,
        "seeds": SEEDS,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "all_results": all_results,
        "results_by_protocol": results_by_protocol,
        "avg_results": avg_results,
        "best_configs": best_configs,
    }
    
    output_path = RESULTS_DIR / f"validation_{timestamp}.json"
    latest_path = RESULTS_DIR / "validation_latest.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, default=str)
    
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, default=str)
    
    print(f"\n完整结果已保存到:")
    print(f"  {output_path}")
    print(f"  {latest_path}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    print("\n所有协议的最佳配置:")
    for protocol, best in best_configs.items():
        print(f"  {protocol}:")
        print(f"    {best['monitor']} @ {best['threshold']:.2f}")
        print(f"    成功率: {best['avg_success_rate']:.2%}, 成本: {best['avg_cost']:.3f}")


if __name__ == "__main__":
    main()
