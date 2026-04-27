"""
Solver comparison: SearchLocalSolver vs LlamaCppSolver on mixed-40 and code-20.

Compares synthetic search solver against real LLM on the same benchmark.
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
    SearchLocalSolver,
)
from core.llm_solver import LlamaCppSolver


def load_jsonl_tasks(path: str) -> list[BenchmarkTask]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            family = d.get("family", "code")
            if family == "code":
                expected = d.get("buggy_code", "")
                metadata = {
                    "function_name": d.get("function_name", "solve"),
                    "bug_type": d.get("bug_type", "unknown"),
                    "difficulty": d.get("difficulty", ""),
                    "visible_tests": d.get("visible_tests", []),
                    "ambiguity_signals": d.get("ambiguity_signals", []),
                    "buggy_code": d.get("buggy_code", ""),
                }
            else:
                expression = d.get("expression", "")
                if expression and not d.get("expected_answer"):
                    from core.capability_benchmark import _safe_eval_expression
                    expected = str(_safe_eval_expression(expression))
                else:
                    expected = d.get("expected_answer", "")
                metadata = {
                    "expression": expression,
                    "ops": d.get("ops", []),
                }
            tasks.append(BenchmarkTask(
                task_id=d["task_id"],
                family=family,
                prompt=d["prompt"],
                expected_answer=expected,
                metadata=metadata,
            ))
    return tasks


def run_solver_on_tasks(solver, solver_name: str, tasks: list[BenchmarkTask]) -> list[dict]:
    verifier = BenchmarkVerifier()
    results = []
    total_start = time.time()

    for i, task in enumerate(tasks):
        t0 = time.time()
        attempt = solver.solve(task)
        vr = verifier.verify(task, attempt)
        elapsed = time.time() - t0

        result = {
            "task_id": task.task_id,
            "family": task.family,
            "bug_type": task.metadata.get("bug_type", ""),
            "difficulty": task.metadata.get("difficulty", ""),
            "confidence": round(attempt.confidence, 3),
            "correct": vr.passed,
            "feedback": vr.feedback,
            "latency_s": round(elapsed, 2),
            "answer_preview": attempt.answer[:120],
            "solver": solver_name,
        }
        results.append(result)

        status = "OK" if vr.passed else "FAIL"
        print(f"  [{i+1:2d}/{len(tasks)}] {task.task_id:20s} [{status:4s}] conf={attempt.confidence:.2f} time={elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    return results, total_elapsed


def print_summary(results: list[dict], solver_name: str, total_time: float):
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    avg_conf = sum(r["confidence"] for r in results) / total if total else 0

    print(f"\n  === {solver_name} ===")
    print(f"  Total:    {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  Avg Conf: {avg_conf:.3f}")
    print(f"  Time:     {total_time:.1f}s")

    code_tasks = [r for r in results if r["family"] == "code"]
    reas_tasks = [r for r in results if r["family"] == "reasoning"]

    if code_tasks:
        c = sum(1 for r in code_tasks if r["correct"])
        print(f"  Code:     {c}/{len(code_tasks)} ({c/len(code_tasks)*100:.0f}%)")
    if reas_tasks:
        c = sum(1 for r in reas_tasks if r["correct"])
        print(f"  Reason:   {c}/{len(reas_tasks)} ({c/len(reas_tasks)*100:.0f}%)")

    if code_tasks:
        print(f"\n  By Difficulty ({solver_name}):")
        for diff in ["trivial", "easy", "medium"]:
            subset = [r for r in code_tasks if r["difficulty"] == diff]
            if subset:
                c = sum(1 for r in subset if r["correct"])
                print(f"    {diff:8s}: {c}/{len(subset)} ({c/len(subset)*100:.0f}%)")

        print(f"\n  By Bug Type ({solver_name}):")
        bug_types = sorted(set(r["bug_type"] for r in code_tasks if r["bug_type"]))
        for bt in bug_types:
            subset = [r for r in code_tasks if r["bug_type"] == bt]
            c = sum(1 for r in subset if r["correct"])
            status = "OK" if c == len(subset) else "PARTIAL" if c > 0 else "FAIL"
            print(f"    [{status:7s}] {bt:30s}: {c}/{len(subset)}")


def compare_solvers(search_results: list[dict], llm_results: list[dict]):
    print("\n" + "=" * 70)
    print("COMPARISON: SearchLocalSolver vs LlamaCppSolver")
    print("=" * 70)

    search_by_id = {r["task_id"]: r for r in search_results}
    llm_by_id = {r["task_id"]: r for r in llm_results}

    both_correct = 0
    search_only = 0
    llm_only = 0
    both_fail = 0

    task_ids = sorted(set(search_by_id.keys()) & set(llm_by_id.keys()))

    for tid in task_ids:
        s = search_by_id[tid]
        l = llm_by_id[tid]
        if s["correct"] and l["correct"]:
            both_correct += 1
        elif s["correct"] and not l["correct"]:
            search_only += 1
        elif not s["correct"] and l["correct"]:
            llm_only += 1
        else:
            both_fail += 1

    total = len(task_ids)
    print(f"\n  Agreement Matrix (n={total}):")
    print(f"    Both Correct:   {both_correct:3d} ({both_correct/total*100:.0f}%)")
    print(f"    Search Only:    {search_only:3d} ({search_only/total*100:.0f}%)")
    print(f"    LLM Only:       {llm_only:3d} ({llm_only/total*100:.0f}%)")
    print(f"    Both Fail:      {both_fail:3d} ({both_fail/total*100:.0f}%)")

    print(f"\n  Per-Task Detail:")
    print(f"  {'Task ID':20s} {'Family':10s} {'Search':8s} {'LLM':8s} {'Delta':8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for tid in task_ids:
        s = search_by_id[tid]
        l = llm_by_id[tid]
        s_status = "OK" if s["correct"] else "FAIL"
        l_status = "OK" if l["correct"] else "FAIL"
        if s["correct"] and not l["correct"]:
            delta = "S>L"
        elif not s["correct"] and l["correct"]:
            delta = "L>S"
        elif s["correct"] and l["correct"]:
            delta = "="
        else:
            delta = "both X"
        print(f"  {tid:20s} {s['family']:10s} {s_status:8s} {l_status:8s} {delta:8s}")

    code_ids = [tid for tid in task_ids if search_by_id[tid]["family"] == "code"]
    reas_ids = [tid for tid in task_ids if search_by_id[tid]["family"] == "reasoning"]

    if code_ids:
        s_code = sum(1 for tid in code_ids if search_by_id[tid]["correct"])
        l_code = sum(1 for tid in code_ids if llm_by_id[tid]["correct"])
        print(f"\n  Code tasks: Search {s_code}/{len(code_ids)} vs LLM {l_code}/{len(code_ids)}")

    if reas_ids:
        s_reas = sum(1 for tid in reas_ids if search_by_id[tid]["correct"])
        l_reas = sum(1 for tid in reas_ids if llm_by_id[tid]["correct"])
        print(f"  Reasoning:  Search {s_reas}/{len(reas_ids)} vs LLM {l_reas}/{len(reas_ids)}")


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "mixed-40"

    if suite == "mixed-40":
        task_path = str(PROJECT_ROOT / "data" / "capability_boundary_bench" / "mixed-40.public.jsonl")
    elif suite == "code-20":
        task_path = str(PROJECT_ROOT / "data" / "capability_boundary_bench" / "code-20.public.jsonl")
    else:
        print(f"Unknown suite: {suite}. Use 'mixed-40' or 'code-20'.")
        return

    print("=" * 70)
    print(f"Solver Comparison: SearchLocalSolver vs LlamaCppSolver")
    print(f"Suite: {suite}")
    print("=" * 70)

    print(f"\n[1/3] Loading tasks from {task_path}...")
    tasks = load_jsonl_tasks(task_path)
    print(f"  Loaded {len(tasks)} tasks")

    print(f"\n[2/3] Running SearchLocalSolver...")
    search_solver = SearchLocalSolver()
    search_results, search_time = run_solver_on_tasks(search_solver, "SearchLocalSolver", tasks)
    print_summary(search_results, "SearchLocalSolver", search_time)

    print(f"\n[3/3] Running LlamaCppSolver (Qwen2.5-0.5B)...")
    llm_solver = LlamaCppSolver(base_url="http://127.0.0.1:8081")
    llm_results, llm_time = run_solver_on_tasks(llm_solver, "LlamaCppSolver", tasks)
    print_summary(llm_results, "LlamaCppSolver", llm_time)

    compare_solvers(search_results, llm_results)

    output_path = PROJECT_ROOT / "results" / "capability_benchmark" / f"solver_compare_{suite}_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "suite": suite,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "search_solver_summary": {
                "correct": sum(1 for r in search_results if r["correct"]),
                "total": len(search_results),
                "success_rate": sum(1 for r in search_results if r["correct"]) / len(search_results),
                "total_time_s": search_time,
            },
            "llm_solver_summary": {
                "correct": sum(1 for r in llm_results if r["correct"]),
                "total": len(llm_results),
                "success_rate": sum(1 for r in llm_results if r["correct"]) / len(llm_results),
                "total_time_s": llm_time,
                "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
            },
            "search_results": search_results,
            "llm_results": llm_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
