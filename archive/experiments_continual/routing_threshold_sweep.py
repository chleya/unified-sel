"""
路由阈值扫描实验

为不同路由监控器寻找最优的 routing_signal_threshold：
- semantic
- counterfactual
- diagnostic
- surface
- behavioral
- external

扫描范围：0.30 到 0.70，步长 0.05
测试协议：monitor_gate（最简单的协议，方便对比）
测试套件：code-20（当前顶级基准）
种子：7（单个种子快速验证）

用法：
    python experiments/routing_threshold_sweep.py

输出：
    results/capability_routing_threshold/sweep_<timestamp>.json
    results/capability_routing_threshold/sweep_latest.json
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

# 扫描配置
MONITORS = [
    "semantic",
    "counterfactual",
    "diagnostic",
    "surface",
    "behavioral",
    "external",
]

THRESHOLDS = [round(0.30 + i * 0.05, 2) for i in range(9)]  # 0.30 to 0.70 step 0.05

PROTOCOL = "monitor_gate"
NUM_TASKS = 20
SEED = 7
ESCALATION_THRESHOLD = 0.9  # 固定升级阈值


def run_single_cell(
    monitor: str,
    threshold: float,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """运行单个监控器+阈值的测试用例"""
    
    def _benchmark_fn(_):
        return run_capability_benchmark(
            suite="code",
            protocol=PROTOCOL,
            num_tasks=NUM_TASKS,
            seed=SEED,
            local_solver_name="search",
            routing_monitor_name=monitor,
            routing_signal_threshold=threshold,
            escalation_signal_threshold=ESCALATION_THRESHOLD,
        )
    
    cache_prefix = f"{monitor}_{threshold:.2f}"
    result, cached = run_seed_with_cache(
        seed_fn=_benchmark_fn,
        seed=SEED,
        experiment_name="capability_routing_threshold",
        cache_prefix=cache_prefix,
        force_rerun=force_rerun,
    )
    
    result["monitor"] = monitor
    result["threshold"] = threshold
    result["cached"] = cached
    
    return result


def find_pareto_frontier(results: List[Dict]) -> List[Dict]:
    """找到帕累托前沿（成功率最高的前提下，成本最低）"""
    # 按成功率降序，成本升序排序
    sorted_results = sorted(
        results,
        key=lambda x: (-x["summary"]["success_rate"], x["summary"]["mean_cost_units"])
    )
    
    pareto_front = []
    best_cost = float("inf")
    
    for result in sorted_results:
        cost = result["summary"]["mean_cost_units"]
        if cost < best_cost:
            pareto_front.append(result)
            best_cost = cost
    
    return pareto_front


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
    print("路由阈值扫描实验")
    print("=" * 60)
    print(f"监控器: {MONITORS}")
    print(f"阈值范围: {THRESHOLDS[0]} 到 {THRESHOLDS[-1]} (步长 0.05)")
    print(f"协议: {PROTOCOL}")
    print(f"任务数: {NUM_TASKS}")
    print(f"种子: {SEED}")
    print()
    
    all_results = []
    total_cells = len(MONITORS) * len(THRESHOLDS)
    completed = 0
    
    for monitor in MONITORS:
        print(f"\n--- 监控器: {monitor} ---")
        for threshold in THRESHOLDS:
            print(f"  阈值: {threshold:.2f} ... ", end="", flush=True)
            
            try:
                result = run_single_cell(monitor, threshold, force_rerun=args.force_rerun)
                all_results.append(result)
                
                sr = result["summary"]["success_rate"]
                cost = result["summary"]["mean_cost_units"]
                cached_note = " (缓存)" if result["cached"] else ""
                print(f"完成 - 成功率: {sr:.2%}, 平均成本: {cost:.3f}{cached_note}")
            except Exception as e:
                print(f"失败: {e}")
            
            completed += 1
            print(f"  进度: {completed}/{total_cells}")
    
    print("\n" + "=" * 60)
    print("扫描完成！")
    print("=" * 60)
    
    # 按监控器分组
    results_by_monitor = {}
    for result in all_results:
        monitor = result["monitor"]
        if monitor not in results_by_monitor:
            results_by_monitor[monitor] = []
        results_by_monitor[monitor].append(result)
    
    # 为每个监控器找最优配置
    print("\n--- 每个监控器的最优配置 ---")
    optimal_configs = []
    
    for monitor in MONITORS:
        if monitor not in results_by_monitor:
            continue
        
        monitor_results = results_by_monitor[monitor]
        pareto = find_pareto_frontier(monitor_results)
        
        if pareto:
            optimal = pareto[0]
            optimal_configs.append(optimal)
            print(f"\n{monitor}:")
            print(f"  最优阈值: {optimal['threshold']:.2f}")
            print(f"  成功率: {optimal['summary']['success_rate']:.2%}")
            print(f"  平均成本: {optimal['summary']['mean_cost_units']:.3f}")
            
            # 打印所有帕累托点
            if len(pareto) > 1:
                print(f"  帕累托前沿 ({len(pareto)} 个点):")
                for i, p in enumerate(pareto):
                    print(f"    #{i+1}: 阈值={p['threshold']:.2f}, 成功率={p['summary']['success_rate']:.2%}, 成本={p['summary']['mean_cost_units']:.3f}")
    
    # 保存完整结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    full_result = {
        "timestamp": timestamp,
        "monitors": MONITORS,
        "thresholds": THRESHOLDS,
        "protocol": PROTOCOL,
        "num_tasks": NUM_TASKS,
        "seed": SEED,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "all_results": all_results,
        "results_by_monitor": results_by_monitor,
        "optimal_configs": optimal_configs,
    }
    
    output_path = RESULTS_DIR / f"sweep_{timestamp}.json"
    latest_path = RESULTS_DIR / "sweep_latest.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, default=str)
    
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, default=str)
    
    print(f"\n完整结果已保存到:")
    print(f"  {output_path}")
    print(f"  {latest_path}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("快速总结")
    print("=" * 60)
    
    print("\n所有监控器在 0.5 阈值下的表现（默认值）:")
    for monitor in MONITORS:
        if monitor not in results_by_monitor:
            continue
        for result in results_by_monitor[monitor]:
            if abs(result["threshold"] - 0.5) < 0.01:
                sr = result["summary"]["success_rate"]
                cost = result["summary"]["mean_cost_units"]
                print(f"  {monitor}: 成功率 {sr:.2%}, 成本 {cost:.3f}")
                break
    
    print("\n最优监控器+阈值组合:")
    if optimal_configs:
        best = max(optimal_configs, key=lambda x: (x["summary"]["success_rate"], -x["summary"]["mean_cost_units"]))
        print(f"  {best['monitor']} @ {best['threshold']:.2f}:")
        print(f"    成功率: {best['summary']['success_rate']:.2%}")
        print(f"    平均成本: {best['summary']['mean_cost_units']:.3f}")


if __name__ == "__main__":
    main()
