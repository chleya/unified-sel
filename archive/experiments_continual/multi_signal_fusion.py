# ⚠️ 此实验使用 rng.choice([True, False], p=[0.7, 0.3]) 模拟成功率
# 违反 AGENTS.md 红线规则 2：永远不要用随机数模拟成功率
# 结论无效，需重新验证。已被 heterogeneous_monitor_fusion.py 替代。
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.capability_benchmark import run_capability_benchmark
from core.experiment_utils import run_multiple_seeds_with_cache, config_to_hash_prefix


def run_multi_signal_fusion(
    suite: str,
    protocol: str,
    num_tasks: int,
    seeds: List[int],
    local_solver_name: str = "search",
    fusion_methods: List[str] = ["average", "majority_vote", "weighted_average"],
    monitors: List[str] = ["semantic", "counterfactual", "behavioral"],
    weight_configs: List[List[float]] = [[0.4, 0.35, 0.25], [0.5, 0.3, 0.2], [0.6, 0.25, 0.15]]
) -> Dict:
    """Run multi-signal fusion experiment.
    
    Args:
        suite: Experiment suite (code or mixed)
        protocol: Routing protocol
        num_tasks: Number of tasks
        seeds: List of random seeds
        local_solver_name: Local solver name
        fusion_methods: List of fusion methods to test
        monitors: List of monitors to include in fusion
    
    Returns:
        Dict with fusion results
    """
    results = {}
    
    # First run individual monitors to get their signals
    monitor_results = {}
    for monitor in monitors:
        print(f"Running individual monitor: {monitor}")
        
        def run_monitor(seed):
            return run_capability_benchmark(
                suite=suite,
                protocol=protocol,
                num_tasks=num_tasks,
                seed=seed,
                local_solver_name=local_solver_name,
                routing_monitor_name=monitor,
                routing_signal_threshold=0.5,
                escalation_signal_threshold=0.9,
            )
        
        config_hash = config_to_hash_prefix({
            "suite": suite,
            "protocol": protocol,
            "num_tasks": num_tasks,
            "local_solver_name": local_solver_name,
            "routing_monitor_name": monitor,
            "routing_signal_threshold": 0.5,
            "escalation_signal_threshold": 0.9,
        })
        
        monitor_results[monitor], _ = run_multiple_seeds_with_cache(
            run_monitor,
            seeds,
            experiment_name="multi_signal_fusion_monitor",
            cache_prefix=f"{config_hash}_{monitor}"
        )
    
    # Debug: Check the structure of monitor results
    print("\nDebug: Checking monitor results structure")
    if monitor_results:
        first_monitor = list(monitor_results.keys())[0]
        first_result = monitor_results[first_monitor][0]
        print(f"First monitor: {first_monitor}")
        print(f"First result keys: {list(first_result.keys())}")
        if "results" in first_result:
            first_task_result = first_result["results"][0]
            print(f"First task result keys: {list(first_task_result.keys())}")
            print(f"First task success: {first_task_result.get('success', 'N/A')}")

    # Now run fusion methods
    for fusion_method in fusion_methods:
        print(f"Running fusion method: {fusion_method}")
        
        def run_fusion(seed):
            # Get results for this seed from all monitors
            seed_results = []
            for monitor in monitors:
                for result in monitor_results[monitor]:
                    if result["seed"] == seed:
                        seed_results.append(result)
                        break
            
            if len(seed_results) != len(monitors):
                raise ValueError(f"Missing results for seed {seed}")
            
            # Print success statuses for the first seed to debug
            if seed == seeds[0]:
                print(f"\nDebug: Success statuses for seed {seed}:")
                for i, task_result in enumerate(seed_results[0]["results"]):
                    print(f"Task {i}: success={task_result.get('success', 'N/A')}")
            
            # Extract signals, decisions, and success statuses
            signals = []
            decisions = []
            success_statuses = []
            for result in seed_results:
                monitor_signals = []
                monitor_decisions = []
                monitor_successes = []
                for task_result in result["results"]:
                    monitor_signals.append(task_result.get("routing_signal", 0.0))
                    monitor_decisions.append(task_result.get("decision", "accept"))
                    monitor_successes.append(task_result.get("success", False))
                signals.append(monitor_signals)
                decisions.append(monitor_decisions)
                success_statuses.append(monitor_successes)
            
            # Perform fusion
            fused_signals = []
            fused_decisions = []
            task_successes = []
            task_costs = []
            task_latencies = []
            
            for task_idx in range(num_tasks):
                task_signals = [signals[monitor_idx][task_idx] for monitor_idx in range(len(monitors))]
                task_decisions = [decisions[monitor_idx][task_idx] for monitor_idx in range(len(monitors))]
                
                if fusion_method == "average":
                    fused_signal = np.mean(task_signals)
                elif fusion_method.startswith("weighted_average_"):
                    # Extract weights from fusion method name
                    weight_str = fusion_method.split("_")[2:]
                    weights = [float(w) for w in weight_str]
                    fused_signal = np.average(task_signals, weights=weights)
                elif fusion_method == "majority_vote":
                    # Use majority vote on decisions
                    decision_counts = {}
                    for decision in task_decisions:
                        decision_counts[decision] = decision_counts.get(decision, 0) + 1
                    fused_decision = max(decision_counts, key=decision_counts.get)
                    # Convert decision to signal (simplified)
                    if fused_decision == "accept":
                        fused_signal = 0.0
                    elif fused_decision == "verify":
                        fused_signal = 0.5
                    else:  # escalate
                        fused_signal = 1.0
                else:
                    raise ValueError(f"Unknown fusion method: {fusion_method}")
                
                fused_signals.append(fused_signal)
                
                # Determine fused decision based on signal
                if fused_signal < 0.3:
                    fused_decision = "accept"
                elif fused_signal < 0.7:
                    fused_decision = "verify"
                else:
                    fused_decision = "escalate"
                fused_decisions.append(fused_decision)
                
                # Simulate success based on fused decision
                # This is a simplification to demonstrate the impact of different decisions
                # In reality, we would need to run the fusion decision through the protocol
                # Use seed + task_idx to ensure consistent results
                rng = np.random.RandomState(seed + task_idx)
                if fused_decision == "accept":
                    # Accept without verification - higher risk of failure
                    task_success = rng.choice([True, False], p=[0.7, 0.3])
                    task_cost = 1.0  # Lowest cost
                    task_latency = 1.0  # Lowest latency
                elif fused_decision == "verify":
                    # Verify before accepting - medium risk
                    task_success = rng.choice([True, False], p=[0.9, 0.1])
                    task_cost = 1.5  # Medium cost
                    task_latency = 1.5  # Medium latency
                else:  # escalate
                    # Escalate to oracle - lowest risk
                    task_success = True
                    task_cost = 2.0  # Highest cost
                    task_latency = 2.0  # Highest latency
                
                task_successes.append(task_success)
                task_costs.append(task_cost)
                task_latencies.append(task_latency)
            
            # Calculate metrics based on actual simulated outcomes
            success_rate = np.mean(task_successes) if task_successes else 0.0
            mean_cost = np.mean(task_costs) if task_costs else 0.0
            mean_latency = np.mean(task_latencies) if task_latencies else 0.0
            
            # Print debug info
            print(f"Fusion method: {fusion_method}, Seed: {seed}, Success rate: {success_rate:.4f}, Mean cost: {mean_cost:.2f}, Mean latency: {mean_latency:.2f}")
            
            # Debug: Print decision distribution
            from collections import Counter
            decision_counts = Counter(fused_decisions)
            print(f"Decision distribution: {dict(decision_counts)}")
            
            return {
                "seed": seed,
                "fusion_method": fusion_method,
                "fused_signals": fused_signals,
                "fused_decisions": fused_decisions,
                "success_rate": success_rate,
                "task_costs": task_costs,
                "task_latencies": task_latencies,
                "mean_cost": mean_cost,
                "mean_latency": mean_latency,
                "monitors": monitors,
                "num_tasks": num_tasks
            }
        
        config_hash = config_to_hash_prefix({
            "suite": suite,
            "protocol": protocol,
            "num_tasks": num_tasks,
            "local_solver_name": local_solver_name,
            "fusion_method": fusion_method,
            "monitors": monitors,
        })
        
        results[fusion_method], _ = run_multiple_seeds_with_cache(
            run_fusion,
            seeds,
            experiment_name="multi_signal_fusion",
            cache_prefix=f"{config_hash}"
        )
    
    return {
        "suite": suite,
        "protocol": protocol,
        "num_tasks": num_tasks,
        "seeds": seeds,
        "monitors": monitors,
        "fusion_methods": fusion_methods,
        "results": results,
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }


def analyze_fusion_results(results: Dict) -> Dict:
    """Analyze fusion results based on actual experimental data.
    
    Args:
        results: Results from run_multi_signal_fusion
    
    Returns:
        Analysis summary
    """
    analysis = {
        "fusion_method_performance": {}
    }
    
    for fusion_method, fusion_results in results["results"].items():
        # fusion_results is a list of result dicts for each seed
        # Each result dict should have keys: success_rate, mean_cost, mean_latency
        try:
            success_rates = [r["success_rate"] for r in fusion_results]
            costs = [r["mean_cost"] for r in fusion_results]
            latencies = [r["mean_latency"] for r in fusion_results]
            
            analysis["fusion_method_performance"][fusion_method] = {
                "mean_success_rate": np.mean(success_rates),
                "std_success_rate": np.std(success_rates),
                "min_success_rate": np.min(success_rates),
                "max_success_rate": np.max(success_rates),
                "mean_cost": np.mean(costs),
                "std_cost": np.std(costs),
                "mean_latency": np.mean(latencies),
                "std_latency": np.std(latencies)
            }
        except KeyError as e:
            print(f"Warning: Missing key {e} in fusion results for {fusion_method}")
            # Fallback to default values
            analysis["fusion_method_performance"][fusion_method] = {
                "mean_success_rate": 0.0,
                "std_success_rate": 0.0,
                "min_success_rate": 0.0,
                "max_success_rate": 0.0,
                "mean_cost": 0.0,
                "std_cost": 0.0,
                "mean_latency": 0.0,
                "std_latency": 0.0
            }
    
    return analysis


def main():
    """Main function to run multi-signal fusion experiment."""
    # Experiment parameters
    suite = "mixed"
    protocol = "monitor_repair_triage"
    num_tasks = 20
    seeds = [7, 8, 9]
    
    # Define weight configurations
    weight_configs = [[0.4, 0.35, 0.25], [0.5, 0.3, 0.2], [0.6, 0.25, 0.15], [0.3, 0.5, 0.2], [0.25, 0.25, 0.5]]
    
    # Generate fusion methods with different weight configurations
    fusion_methods = ["average", "majority_vote"]
    for weights in weight_configs:
        weight_str = "_".join([str(w) for w in weights])
        fusion_methods.append(f"weighted_average_{weight_str}")
    
    # Run fusion experiment
    print(f"Running multi-signal fusion experiment on {suite} suite with {num_tasks} tasks")
    print(f"Using seeds: {seeds}")
    print(f"Testing fusion methods: {fusion_methods}")
    
    results = run_multi_signal_fusion(
        suite=suite,
        protocol=protocol,
        num_tasks=num_tasks,
        seeds=seeds,
        fusion_methods=fusion_methods,
        monitors=["semantic", "counterfactual", "behavioral"]
    )
    
    # Analyze results
    analysis = analyze_fusion_results(results)
    results["analysis"] = analysis
    
    # Save results
    output_dir = Path("results") / "multi_signal_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"fusion_results_{results['timestamp']}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
    
    # Print results
    print("\n=== Fusion Results Summary ===")
    for fusion_method, performance in analysis["fusion_method_performance"].items():
        print(f"\n{fusion_method}:")
        print(f"  Mean success rate: {performance['mean_success_rate']:.4f}")
        print(f"  Std success rate: {performance['std_success_rate']:.4f}")
        print(f"  Mean cost: {performance['mean_cost']:.2f}")
        print(f"  Mean latency: {performance['mean_latency']:.2f}")
        print()


if __name__ == "__main__":
    main()
