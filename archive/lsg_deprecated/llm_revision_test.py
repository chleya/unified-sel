"""
LLM Revision Capability Test.

Tests whether Qwen2.5-0.5B can fix its own mistakes when given
verification feedback. This is critical for the routing pipeline:
if revision works, we can avoid expensive escalation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    BenchmarkVerifier,
    SolverAttempt,
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
                    "ambiguity_signals": d.get("ambiguity_signals", []),
                    "buggy_code": d.get("buggy_code", ""),
                }
            ))
    return tasks


def main():
    print("=" * 70)
    print("LLM Revision Capability Test: Qwen2.5-0.5B")
    print("Can the LLM fix its own mistakes when given feedback?")
    print("=" * 70)

    print("\n[1/3] Loading code-20 tasks...")
    tasks = load_code20_tasks()
    print(f"  Loaded {len(tasks)} tasks")

    solver = LlamaCppSolver(base_url="http://127.0.0.1:8081")
    verifier = BenchmarkVerifier()

    print("\n[2/3] Phase 1: Initial solve...")
    initial_results = []
    failed_tasks = []

    for i, task in enumerate(tasks):
        t0 = time.time()
        attempt = solver.solve(task)
        try:
            vr = verifier.verify(task, attempt)
        except Exception:
            vr = type('VR', (), {'passed': False, 'feedback': 'verify_error'})()
        elapsed = time.time() - t0

        initial_results.append({
            "task_id": task.task_id,
            "bug_type": task.metadata["bug_type"],
            "difficulty": task.metadata["difficulty"],
            "correct": vr.passed,
            "confidence": round(attempt.confidence, 3),
            "feedback": vr.feedback,
            "latency_s": round(elapsed, 2),
        })

        status = "OK" if vr.passed else "FAIL"
        print(f"  [{i+1:2d}/20] {task.task_id:12s} [{status}] conf={attempt.confidence:.2f}")

        if not vr.passed:
            failed_tasks.append((task, attempt, vr))

    initial_correct = sum(1 for r in initial_results if r["correct"])
    print(f"\n  Initial: {initial_correct}/20 ({initial_correct/20*100:.0f}%)")

    print(f"\n[3/3] Phase 2: Revision on {len(failed_tasks)} failed tasks...")
    revision_results = []

    for i, (task, prev_attempt, prev_vr) in enumerate(failed_tasks):
        t0 = time.time()
        revised = solver.revise(task, prev_attempt, prev_vr.feedback)
        try:
            rev_vr = verifier.verify(task, revised)
        except Exception:
            rev_vr = type('VR', (), {'passed': False, 'feedback': 'verify_error'})()
        elapsed = time.time() - t0

        revision_results.append({
            "task_id": task.task_id,
            "bug_type": task.metadata["bug_type"],
            "difficulty": task.metadata["difficulty"],
            "initial_correct": False,
            "revision_correct": rev_vr.passed,
            "initial_confidence": round(prev_attempt.confidence, 3),
            "revision_confidence": round(revised.confidence, 3),
            "feedback": prev_vr.feedback,
            "revision_feedback": rev_vr.feedback,
            "latency_s": round(elapsed, 2),
            "initial_answer_preview": prev_attempt.answer[:80],
            "revision_answer_preview": revised.answer[:80],
        })

        status = "FIXED" if rev_vr.passed else "STILL FAIL"
        print(f"  [{i+1:2d}/{len(failed_tasks)}] {task.task_id:12s} bug={task.metadata['bug_type']:28s} [{status}]")

    revision_fixed = sum(1 for r in revision_results if r["revision_correct"])
    total_after_revision = initial_correct + revision_fixed

    print("\n" + "=" * 70)
    print("REVISION ANALYSIS")
    print("=" * 70)

    print(f"\n  Initial correct:     {initial_correct}/20 ({initial_correct/20*100:.0f}%)")
    print(f"  Failed tasks:        {len(failed_tasks)}")
    print(f"  Fixed by revision:   {revision_fixed}/{len(failed_tasks)} ({revision_fixed/max(len(failed_tasks),1)*100:.0f}%)")
    print(f"  Total after revision: {total_after_revision}/20 ({total_after_revision/20*100:.0f}%)")

    if revision_results:
        print(f"\n  Revision by Difficulty:")
        for diff in ["trivial", "easy", "medium"]:
            subset = [r for r in revision_results if r["difficulty"] == diff]
            if subset:
                fixed = sum(1 for r in subset if r["revision_correct"])
                print(f"    {diff:8s}: {fixed}/{len(subset)} ({fixed/len(subset)*100:.0f}%)")

        print(f"\n  Revision by Bug Type:")
        for r in revision_results:
            status = "FIXED" if r["revision_correct"] else "STILL FAIL"
            print(f"    [{status:10s}] {r['bug_type']:30s}")

        still_failing = [r for r in revision_results if not r["revision_correct"]]
        if still_failing:
            print(f"\n  Still failing after revision ({len(still_failing)} tasks):")
            for r in still_failing:
                print(f"    {r['task_id']:12s} {r['bug_type']:28s} diff={r['difficulty']}")

    output_path = PROJECT_ROOT / "results" / "capability_benchmark" / f"llm_revision_test_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "llm_revision_test",
            "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "initial_correct": initial_correct,
            "initial_total": 20,
            "revision_fixed": revision_fixed,
            "revision_attempted": len(failed_tasks),
            "total_after_revision": total_after_revision,
            "initial_results": initial_results,
            "revision_results": revision_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
