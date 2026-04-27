"""
Cross-domain LLM Routing Experiment

Tests monitor_no_revision_triage on mixed code+reasoning tasks
to validate cross-domain robustness.

Usage:
    python experiments/capability/cross_domain_llm_routing.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.capability_benchmark import build_task_suite
from core.llm_solver import LlamaCppSolver
from experiments.capability.llm_routing_experiment import run_routing_experiment


def main():
    print("=" * 70)
    print("Cross-Domain LLM Routing Experiment")
    print("=" * 70)

    print("\n[1/3] Loading mixed tasks...")
    code_tasks = build_task_suite("code", 20, seed=42)
    reasoning_tasks = build_task_suite("reasoning", 20, seed=42)
    mixed_tasks = code_tasks + reasoning_tasks
    print(f"  Code: {len(code_tasks)}, Reasoning: {len(reasoning_tasks)}, Total: {len(mixed_tasks)}")

    print("\n[2/3] Testing LLM solver...")
    solver = LlamaCppSolver(base_url="http://127.0.0.1:8081")

    monitor_names = ["semantic", "counterfactual"]
    protocols = ["local_only", "monitor_no_revision_triage", "verify_escalate_no_revision", "verifier_first"]

    print("\n[3/3] Running experiments...")
    all_results = {}

    for monitor_name in monitor_names:
        for protocol in protocols:
            label = f"{monitor_name}/{protocol}"
            print(f"\n  Running {label}...")
            try:
                per_task = run_routing_experiment(
                    mixed_tasks, solver, monitor_name, protocol
                )
                all_results[label] = per_task

                initial = sum(1 for r in per_task if r.get("initial_correct", False))
                final = sum(1 for r in per_task if r.get("final_correct", False))
                total = len(per_task)
                rev = sum(1 for r in per_task if r.get("revision_used", False))
                esc = sum(1 for r in per_task if r.get("escalation_used", False))
                print(f"  === {label} ===")
                print(f"    Initial: {initial}/{total} ({initial/total*100:.0f}%)")
                print(f"    Final:   {final}/{total} ({final/total*100:.0f}%)")
                print(f"    Revisions: {rev}, Escalations: {esc}")

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results[label] = {"error": str(e)}

    print("\n" + "=" * 70)
    print("CROSS-DOMAIN COMPARISON")
    print("=" * 70)

    for monitor_name in monitor_names:
        print(f"\n  Monitor: {monitor_name}")
        for protocol in protocols:
            label = f"{monitor_name}/{protocol}"
            r = all_results.get(label, {})
            if isinstance(r, dict) and "error" in r:
                print(f"    {protocol}: ERROR - {r['error']}")
                continue
            if not isinstance(r, list):
                print(f"    {protocol}: unexpected type {type(r)}")
                continue
            initial = sum(1 for t in r if t.get("initial_correct", False))
            final = sum(1 for t in r if t.get("final_correct", False))
            total = len(r)
            rev = sum(1 for t in r if t.get("revision_used", False))
            esc = sum(1 for t in r if t.get("escalation_used", False))
            print(f"    {protocol:35s}: {final}/{total} ({final/total*100:.0f}%) rev={rev} esc={esc}")

    output = {
        "experiment": "cross_domain_llm_routing",
        "model": "qwen2.5-0.5b-instruct-q4_k_m",
        "task_mix": "code-20 + reasoning-20",
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = Path("results/capability_benchmark")
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"cross_domain_llm_{int(time.time())}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved to: {fname}")


if __name__ == "__main__":
    main()
