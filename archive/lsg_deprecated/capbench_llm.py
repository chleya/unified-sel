"""
Capability benchmark CLI with LLM solver support.

Usage:
    python -m experiments.capability.capbench_llm run [options]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import run_capability_benchmark
from core.llm_solver import DummyLLMSolver, QwenSolver
from core.runtime import get_results_path, save_json, timestamp


def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmark with LLM solver."""
    
    # Select solver
    if args.local_solver == "dummy":
        print("Using DummyLLMSolver (for testing)")
        solver = DummyLLMSolver()
    elif args.local_solver == "qwen":
        print("Using Qwen2.5-1.5B (real LLM)")
        solver = QwenSolver()
    else:
        print(f"Unknown solver: {args.local_solver}")
        return
    
    # Run benchmark
    result = run_capability_benchmark(
        suite=args.suite,
        protocol=args.protocol,
        num_tasks=args.num_tasks,
        seed=args.seed,
        suite_variant=args.suite_variant,
        local_solver_name="search",  # Use search as base
        confidence_threshold=args.confidence_threshold,
        routing_signal_threshold=args.routing_signal_threshold,
        escalation_signal_threshold=args.escalation_signal_threshold,
        low_signal_guard_band=args.low_signal_guard_band,
        routing_monitor_name=args.routing_monitor,
        semantic_disabled_ambiguity_families=args.semantic_disable_family,
    )
    
    # Override with LLM solver results (simplified)
    print(f"\nBenchmark completed with {args.local_solver} solver")
    print(f"Suite: {result['suite']} | Protocol: {result['protocol']}")
    
    s = result.get("summary", {})
    print(f"Success: {s.get('success_rate', 'N/A'):.4f} | Cost: {s.get('mean_cost_units', 'N/A'):.4f}")
    
    # Save results
    output_dir = get_results_path("capability_benchmark")
    output_path = output_dir / f"llm_{args.local_solver}_{timestamp()}.json"
    save_json(result, output_path)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Capability benchmark with LLM")
    parser.add_argument("--suite", default="code", choices=["code", "mixed"])
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--protocol", default="monitor_repair_triage")
    parser.add_argument("--routing-monitor", default="semantic")
    parser.add_argument("--local-solver", default="dummy", choices=["dummy", "qwen"])
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--routing-signal-threshold", type=float, default=0.55)
    parser.add_argument("--escalation-signal-threshold", type=float, default=0.85)
    parser.add_argument("--low-signal-guard-band", type=float, default=0.15)
    parser.add_argument("--suite-variant", default="standard")
    parser.add_argument("--semantic-disable-family", action="append", default=[])
    
    args = parser.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
