"""
LLM Routing Experiment: Test Capability Router with real LLM solver.

Key question: Can the routing monitors detect when the LLM's output
is wrong despite high confidence?

Runs multiple routing policies with LlamaCppSolver on code-20.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    BenchmarkVerifier,
    SolverAttempt,
    VerificationResult,
    SearchLocalSolver,
    OracleSolver,
    BatchHealthMonitor,
    build_routing_monitor,
    estimate_routing_signal,
    _run_code_task,
    _task_verifier_tests,
    _task_search_tests,
)
from core.llm_solver import LlamaCppSolver


def load_code20_tasks(path: str = "data/capability_boundary_bench/code-20.public.jsonl") -> list[BenchmarkTask]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks.append(BenchmarkTask(
                task_id=d["task_id"],
                family="code",
                prompt=d["prompt"],
                expected_answer=d.get("buggy_code", ""),
                metadata={
                    "function_name": d.get("function_name", "solve"),
                    "bug_type": d.get("bug_type", "unknown"),
                    "difficulty": d.get("difficulty", ""),
                    "visible_tests": d.get("visible_tests", []),
                    "hidden_tests": d.get("hidden_tests", []),
                    "ambiguity_signals": d.get("ambiguity_signals", []),
                    "buggy_code": d.get("buggy_code", ""),
                }
            ))
    return tasks


def run_routing_experiment(
    tasks: list[BenchmarkTask],
    solver,
    monitor_name: str,
    protocol: str,
    confidence_threshold: float = 0.95,
    routing_signal_threshold: float = 0.5,
    escalation_signal_threshold: float = 0.9,
) -> list[Dict[str, Any]]:
    verifier = BenchmarkVerifier()
    oracle = OracleSolver()
    monitor = build_routing_monitor(monitor_name)
    batch_health = BatchHealthMonitor(window_size=5) if protocol == "verify_escalate_no_revision" else None
    drift_latch = False

    results = []
    for task in tasks:
        t0 = time.time()

        local_attempt = solver.solve(task)
        verification = verifier.verify(task, local_attempt)
        routing_signal = monitor.score(task, local_attempt)

        if batch_health is not None:
            health_obs = batch_health.observe(task)
            domain_drift_detected = health_obs.get("status") in (
                "domain_shift_detected",
                "gradual_drift",
            )
            if domain_drift_detected:
                drift_latch = True

        decision = "accept"
        final_correct = verification.passed
        final_answer = local_attempt.answer
        revision_used = False
        escalation_used = False

        if protocol == "local_only":
            decision = "accept"

        elif protocol == "monitor_gate":
            if routing_signal >= routing_signal_threshold:
                if verification.passed:
                    decision = "verify_accept"
                else:
                    revised = solver.revise(task, local_attempt, verification.feedback)
                    rev_ver = verifier.verify(task, revised)
                    revision_used = True
                    if rev_ver.passed:
                        decision = "revise_accept"
                        final_correct = True
                        final_answer = revised.answer
                    else:
                        decision = "escalate"
                        escalation_used = True
                        final_correct = True
                        final_answer = oracle.solve(task).answer
            else:
                decision = "accept_low_signal"

        elif protocol == "monitor_repair_triage":
            low_floor = routing_signal_threshold - 0.15
            if routing_signal < low_floor:
                decision = "accept_low_signal"
            elif routing_signal >= escalation_signal_threshold:
                if solver.supports_feedback_revision(task):
                    if verification.passed:
                        decision = "verify_accept"
                    else:
                        revised = solver.revise(task, local_attempt, verification.feedback)
                        rev_ver = verifier.verify(task, revised)
                        revision_used = True
                        if rev_ver.passed:
                            decision = "revise_accept"
                            final_correct = True
                            final_answer = revised.answer
                        else:
                            decision = "escalate_after_revise"
                            escalation_used = True
                            final_correct = True
                            final_answer = oracle.solve(task).answer
                else:
                    decision = "direct_escalate"
                    escalation_used = True
                    final_correct = True
                    final_answer = oracle.solve(task).answer
            else:
                if verification.passed:
                    decision = "verify_accept"
                else:
                    revised = solver.revise(task, local_attempt, verification.feedback)
                    rev_ver = verifier.verify(task, revised)
                    revision_used = True
                    if rev_ver.passed:
                        decision = "revise_accept"
                        final_correct = True
                        final_answer = revised.answer
                    else:
                        decision = "escalate"
                        escalation_used = True
                        final_correct = True
                        final_answer = oracle.solve(task).answer

        elif protocol == "monitor_no_revision_triage":
            low_floor = routing_signal_threshold - 0.15
            if routing_signal < low_floor:
                decision = "accept_low_signal"
            elif routing_signal >= escalation_signal_threshold:
                decision = "direct_escalate"
                escalation_used = True
                final_correct = True
                final_answer = oracle.solve(task).answer
            else:
                if verification.passed:
                    decision = "verify_accept"
                else:
                    decision = "escalate_after_verify"
                    escalation_used = True
                    final_correct = True
                    final_answer = oracle.solve(task).answer

        elif protocol == "verify_escalate_no_revision":
            if verification.passed:
                decision = "verify_accept"
            else:
                decision = "domain_aware_escalate" if drift_latch else "pre_latch_escalate"
                escalation_used = True
                final_correct = True
                final_answer = oracle.solve(task).answer

        elif protocol == "verifier_first":
            if verification.passed:
                decision = "verify_accept"
            else:
                revised = solver.revise(task, local_attempt, verification.feedback)
                rev_ver = verifier.verify(task, revised)
                revision_used = True
                if rev_ver.passed:
                    decision = "revise_accept"
                    final_correct = True
                    final_answer = revised.answer
                else:
                    decision = "escalate"
                    escalation_used = True
                    final_correct = True
                    final_answer = oracle.solve(task).answer

        elapsed = time.time() - t0

        results.append({
            "task_id": task.task_id,
            "bug_type": task.metadata.get("bug_type", ""),
            "difficulty": task.metadata.get("difficulty", ""),
            "confidence": round(local_attempt.confidence, 3),
            "routing_signal": round(routing_signal, 3),
            "initial_correct": verification.passed,
            "final_correct": final_correct,
            "decision": decision,
            "revision_used": revision_used,
            "escalation_used": escalation_used,
            "latency_s": round(elapsed, 2),
            "answer_preview": local_attempt.answer[:80],
        })

    return results


def print_experiment_summary(results: list[Dict], label: str):
    total = len(results)
    initial_correct = sum(1 for r in results if r["initial_correct"])
    final_correct = sum(1 for r in results if r["final_correct"])
    revisions = sum(1 for r in results if r["revision_used"])
    escalations = sum(1 for r in results if r["escalation_used"])

    print(f"\n  === {label} ===")
    print(f"  Initial correct:  {initial_correct}/{total} ({initial_correct/total*100:.0f}%)")
    print(f"  Final correct:    {final_correct}/{total} ({final_correct/total*100:.0f}%)")
    print(f"  Revisions:        {revisions} ({revisions/total*100:.0f}%)")
    print(f"  Escalations:      {escalations} ({escalations/total*100:.0f}%)")

    wrong_but_high_conf = [r for r in results if not r["initial_correct"] and r["confidence"] >= 0.8]
    print(f"  Wrong + High Conf: {len(wrong_but_high_conf)} (confidence gap)")

    if wrong_but_high_conf:
        detected_by_signal = [r for r in wrong_but_high_conf if r["routing_signal"] >= 0.5]
        print(f"  Detected by monitor (signal>=0.5): {len(detected_by_signal)}/{len(wrong_but_high_conf)} ({len(detected_by_signal)/len(wrong_but_high_conf)*100:.0f}%)")


def main():
    print("=" * 70)
    print("LLM Routing Experiment: Capability Router with Qwen2.5-0.5B")
    print("=" * 70)

    print("\n[1/4] Loading code-20 tasks...")
    tasks = load_code20_tasks()
    print(f"  Loaded {len(tasks)} tasks")

    print("\n[2/4] Testing LLM solver + routing monitors...")
    solver = LlamaCppSolver(base_url="http://127.0.0.1:8081")

    monitor_names = ["confidence", "diagnostic", "semantic", "counterfactual"]
    protocols = ["local_only", "monitor_gate", "monitor_repair_triage", "monitor_no_revision_triage", "verifier_first"]

    print("\n[3/4] Running experiments...")
    all_results = {}

    for monitor_name in monitor_names:
        for protocol in protocols:
            label = f"{monitor_name}/{protocol}"
            print(f"\n  Running {label}...")
            try:
                results = run_routing_experiment(
                    tasks, solver, monitor_name, protocol
                )
                all_results[label] = results
                print_experiment_summary(results, label)
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results[label] = []

    print("\n[4/4] Cross-comparison...")
    print("\n" + "=" * 70)
    print("MONITOR SIGNAL ANALYSIS (key finding)")
    print("=" * 70)

    for monitor_name in monitor_names:
        label = f"{monitor_name}/local_only"
        if label not in all_results or not all_results[label]:
            continue
        results = all_results[label]

        correct_tasks = [r for r in results if r["initial_correct"]]
        wrong_tasks = [r for r in results if not r["initial_correct"]]

        if correct_tasks:
            avg_signal_correct = sum(r["routing_signal"] for r in correct_tasks) / len(correct_tasks)
        else:
            avg_signal_correct = 0

        if wrong_tasks:
            avg_signal_wrong = sum(r["routing_signal"] for r in wrong_tasks) / len(wrong_tasks)
        else:
            avg_signal_wrong = 0

        separation = avg_signal_wrong - avg_signal_correct

        print(f"\n  Monitor: {monitor_name}")
        print(f"    Correct tasks (n={len(correct_tasks)}): avg signal = {avg_signal_correct:.3f}")
        print(f"    Wrong tasks   (n={len(wrong_tasks)}): avg signal = {avg_signal_wrong:.3f}")
        print(f"    Separation: {separation:+.3f} {'GOOD' if separation > 0.1 else 'WEAK' if separation > 0 else 'INVERTED'}")

    print("\n" + "=" * 70)
    print("POLICY COMPARISON (final success rate)")
    print("=" * 70)

    for monitor_name in monitor_names:
        print(f"\n  Monitor: {monitor_name}")
        for protocol in protocols:
            label = f"{monitor_name}/{protocol}"
            if label not in all_results or not all_results[label]:
                continue
            results = all_results[label]
            final = sum(1 for r in results if r["final_correct"])
            rev = sum(1 for r in results if r["revision_used"])
            esc = sum(1 for r in results if r["escalation_used"])
            print(f"    {protocol:30s}: {final}/{len(results)} ({final/len(results)*100:.0f}%) rev={rev} esc={esc}")

    output_path = PROJECT_ROOT / "results" / "capability_benchmark" / f"llm_routing_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "llm_routing",
            "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "monitors_tested": monitor_names,
            "protocols_tested": protocols,
            "results": {k: v for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
