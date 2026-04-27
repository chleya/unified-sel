"""
Full code-20 benchmark with LLM solver via llama.cpp.
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
                    "function_name": d["function_name"],
                    "bug_type": d["bug_type"],
                    "difficulty": d["difficulty"],
                    "visible_tests": d.get("visible_tests", []),
                    "ambiguity_signals": d.get("ambiguity_signals", []),
                }
            ))
    return tasks


def run_full_benchmark():
    print("=" * 70)
    print("Full code-20 Benchmark with Qwen2.5-0.5B via llama.cpp")
    print("=" * 70)

    # Step 1: Load tasks
    print("\n[1/3] Loading code-20 tasks...")
    tasks = load_code20_tasks()
    print(f"  Loaded {len(tasks)} tasks")

    # Step 2: Run LLM solver
    print("\n[2/3] Running Qwen2.5-0.5B on all tasks...")
    solver = LlamaCppSolver(base_url="http://127.0.0.1:8081")
    verifier = BenchmarkVerifier()

    results = []
    total_start = time.time()

    for i, task in enumerate(tasks):
        t0 = time.time()
        print(f"\n  [{i+1:2d}/20] {task.task_id} (bug={task.metadata['bug_type']}, diff={task.metadata['difficulty']})")

        attempt = solver.solve(task)
        vr = verifier.verify(task, attempt)
        elapsed = time.time() - t0

        result = {
            "task_id": task.task_id,
            "bug_type": task.metadata["bug_type"],
            "difficulty": task.metadata["difficulty"],
            "confidence": attempt.confidence,
            "correct": vr.passed,
            "feedback": vr.feedback,
            "latency_s": round(elapsed, 1),
            "answer_preview": attempt.answer[:120],
            "tokens": attempt.metadata.get("completion_tokens", 0),
        }
        results.append(result)

        status = "OK" if vr.passed else "FAIL"
        print(f"         [{status}] conf={attempt.confidence:.2f} time={elapsed:.1f}s tokens={result['tokens']} fb={vr.feedback}")

    total_elapsed = time.time() - total_start

    # Step 3: Summary
    print("\n" + "=" * 70)
    print("[3/3] Summary")
    print("=" * 70)

    correct = sum(1 for r in results if r["correct"])
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    avg_time = sum(r["latency_s"] for r in results) / len(results)
    total_tokens = sum(r["tokens"] for r in results)

    print(f"\n  Tasks:        {len(results)}")
    print(f"  Correct:      {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
    print(f"  Avg Conf:     {avg_conf:.2f}")
    print(f"  Avg Latency:  {avg_time:.1f}s")
    print(f"  Total Time:   {total_elapsed:.0f}s")
    print(f"  Total Tokens: {total_tokens}")

    # Breakdown by difficulty
    print("\n  By Difficulty:")
    for diff in ["trivial", "easy", "medium"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            c = sum(1 for r in subset if r["correct"])
            print(f"    {diff:8s}: {c}/{len(subset)} ({c/len(subset)*100:.0f}%)")

    # Breakdown by bug_type
    print("\n  By Bug Type:")
    bug_types = sorted(set(r["bug_type"] for r in results))
    for bt in bug_types:
        subset = [r for r in results if r["bug_type"] == bt]
        c = sum(1 for r in subset if r["correct"])
        status = "OK" if c == len(subset) else "PARTIAL" if c > 0 else "FAIL"
        print(f"    [{status:7s}] {bt:30s}: {c}/{len(subset)}")

    # Failed tasks detail
    failed = [r for r in results if not r["correct"]]
    if failed:
        print(f"\n  Failed Tasks ({len(failed)}):")
        for r in failed:
            print(f"    {r['task_id']}: bug={r['bug_type']} conf={r['confidence']:.2f} fb={r['feedback']}")
            print(f"      answer: {r['answer_preview'][:100]}")

    # Save results
    output_path = PROJECT_ROOT / "results" / "capability_benchmark" / f"llm_code20_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
            "solver": "LlamaCppSolver",
            "suite": "code-20",
            "summary": {
                "correct": correct,
                "total": len(results),
                "success_rate": correct / len(results),
                "avg_confidence": avg_conf,
                "avg_latency_s": avg_time,
                "total_time_s": total_elapsed,
                "total_tokens": total_tokens,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_full_benchmark()
