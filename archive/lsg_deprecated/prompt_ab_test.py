"""
Prompt Engineering A/B Test: Zero-shot vs Few-shot for code repair.

Tests whether adding few-shot examples improves Qwen2.5-0.5B's
code repair success rate on the code-20 benchmark.
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


FEW_SHOT_EXAMPLES = [
    {
        "buggy": "def add_one(x):\n    return x",
        "fixed": "def add_one(x):\n    return x + 1",
    },
    {
        "buggy": "def double_it(x):\n    return x * 3",
        "fixed": "def double_it(x):\n    return x * 2",
    },
    {
        "buggy": "def max_instead_of_min(nums):\n    return max(nums)",
        "fixed": "def max_instead_of_min(nums):\n    return min(nums)",
    },
    {
        "buggy": "def wrong_sign(x):\n    return -abs(x)",
        "fixed": "def wrong_sign(x):\n    return abs(x)",
    },
    {
        "buggy": "def length_of_list(nums):\n    return nums[0]",
        "fixed": "def length_of_list(nums):\n    return len(nums)",
    },
]


class FewShotLlamaCppSolver(LlamaCppSolver):
    def _build_prompt(self, task: BenchmarkTask) -> str:
        if task.family != "code":
            return super()._build_prompt(task)

        examples_text = ""
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            examples_text += f"\nExample {i+1}:\nBuggy:\n{ex['buggy']}\nFixed:\n{ex['fixed']}\n"

        return (
            "Fix the buggy Python function. "
            "The function name describes what it should do. "
            "Return ONLY the fixed function.\n\n"
            f"Here are some examples:{examples_text}\n"
            "Now fix this function:\n"
            f"{task.prompt}\n\nFixed function:\n```python\n"
        )

    def _build_revision_prompt(self, task: BenchmarkTask, previous: SolverAttempt, feedback: str) -> str:
        if task.family != "code":
            return super()._build_revision_prompt(task, previous, feedback)

        return (
            "Fix the buggy Python function. Your previous attempt was wrong.\n\n"
            f"Original buggy code:\n{task.metadata.get('buggy_code', task.prompt)}\n\n"
            f"Your previous attempt:\n{previous.answer}\n\n"
            f"Test result: {feedback}\n\n"
            "Think about what the function name means and fix the bug. "
            "Return ONLY the corrected function.\n\n"
            "Fixed function:\n```python\n"
        )


class ChainOfThoughtSolver(LlamaCppSolver):
    def _build_prompt(self, task: BenchmarkTask) -> str:
        if task.family != "code":
            return super()._build_prompt(task)

        return (
            "Fix the buggy Python function below.\n\n"
            f"Function: {task.prompt}\n\n"
            "Step 1: Read function name → understand purpose\n"
            "Step 2: Compare name vs implementation → find bug\n"
            "Step 3: Write fixed version\n\n"
            "Output ONLY the def statement, nothing else:\n```python\n"
        )


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


def run_solver(solver, solver_name: str, tasks: list[BenchmarkTask]) -> list[dict]:
    verifier = BenchmarkVerifier()
    results = []
    total_start = time.time()

    for i, task in enumerate(tasks):
        t0 = time.time()
        attempt = solver.solve(task)
        try:
            vr = verifier.verify(task, attempt)
        except Exception as e:
            vr = type('VR', (), {'passed': False, 'feedback': f'verify_error:{e}'})()
        elapsed = time.time() - t0

        result = {
            "task_id": task.task_id,
            "bug_type": task.metadata["bug_type"],
            "difficulty": task.metadata["difficulty"],
            "confidence": round(attempt.confidence, 3),
            "correct": vr.passed,
            "feedback": vr.feedback,
            "latency_s": round(elapsed, 2),
            "answer_preview": attempt.answer[:120],
            "solver": solver_name,
        }
        results.append(result)

        status = "OK" if vr.passed else "FAIL"
        print(f"  [{i+1:2d}/20] {task.task_id:12s} bug={task.metadata['bug_type']:28s} [{status}] conf={attempt.confidence:.2f}")

    total_elapsed = time.time() - total_start
    return results, total_elapsed


def print_summary(results: list[dict], solver_name: str, total_time: float):
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n  === {solver_name} ===")
    print(f"  Success: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  Time:    {total_time:.1f}s")

    for diff in ["trivial", "easy", "medium"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            c = sum(1 for r in subset if r["correct"])
            print(f"    {diff:8s}: {c}/{len(subset)} ({c/len(subset)*100:.0f}%)")

    print(f"\n  By Bug Type:")
    for bt in sorted(set(r["bug_type"] for r in results)):
        subset = [r for r in results if r["bug_type"] == bt]
        c = sum(1 for r in subset if r["correct"])
        s = "OK" if c == len(subset) else "PARTIAL" if c > 0 else "FAIL"
        print(f"    [{s:7s}] {bt:30s}: {c}/{len(subset)}")


def main():
    print("=" * 70)
    print("Prompt Engineering A/B Test: Zero-shot vs Few-shot vs CoT")
    print("Model: Qwen2.5-0.5B-Instruct via llama.cpp")
    print("=" * 70)

    print("\n[1/4] Loading code-20 tasks...")
    tasks = load_code20_tasks()
    print(f"  Loaded {len(tasks)} tasks")

    print("\n[2/4] Running Zero-shot (baseline)...")
    baseline_solver = LlamaCppSolver(base_url="http://127.0.0.1:8081")
    baseline_results, baseline_time = run_solver(baseline_solver, "Zero-shot", tasks)
    print_summary(baseline_results, "Zero-shot", baseline_time)

    print("\n[3/4] Running Few-shot...")
    fewshot_solver = FewShotLlamaCppSolver(base_url="http://127.0.0.1:8081")
    fewshot_results, fewshot_time = run_solver(fewshot_solver, "Few-shot", tasks)
    print_summary(fewshot_results, "Few-shot", fewshot_time)

    print("\n[4/4] Running Chain-of-Thought...")
    cot_solver = ChainOfThoughtSolver(base_url="http://127.0.0.1:8081")
    cot_results, cot_time = run_solver(cot_solver, "Chain-of-Thought", tasks)
    print_summary(cot_results, "Chain-of-Thought", cot_time)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    b_correct = sum(1 for r in baseline_results if r["correct"])
    f_correct = sum(1 for r in fewshot_results if r["correct"])
    c_correct = sum(1 for r in cot_results if r["correct"])

    print(f"\n  {'Method':20s} {'Correct':>8s} {'Rate':>8s} {'Time':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Zero-shot':20s} {b_correct:>8d} {b_correct/20*100:>7.0f}% {baseline_time:>7.1f}s")
    print(f"  {'Few-shot':20s} {f_correct:>8d} {f_correct/20*100:>7.0f}% {fewshot_time:>7.1f}s")
    print(f"  {'Chain-of-Thought':20s} {c_correct:>8d} {c_correct/20*100:>7.0f}% {cot_time:>7.1f}s")

    b_by_id = {r["task_id"]: r for r in baseline_results}
    f_by_id = {r["task_id"]: r for r in fewshot_results}
    c_by_id = {r["task_id"]: r for r in cot_results}

    print(f"\n  Per-task comparison:")
    print(f"  {'Task':12s} {'Bug Type':28s} {'Zero':6s} {'Few':6s} {'CoT':6s}")
    print(f"  {'-'*12} {'-'*28} {'-'*6} {'-'*6} {'-'*6}")
    for tid in sorted(b_by_id.keys()):
        b_ok = "OK" if b_by_id[tid]["correct"] else "X"
        f_ok = "OK" if f_by_id[tid]["correct"] else "X"
        c_ok = "OK" if c_by_id[tid]["correct"] else "X"
        print(f"  {tid:12s} {b_by_id[tid]['bug_type']:28s} {b_ok:6s} {f_ok:6s} {c_ok:6s}")

    output_path = PROJECT_ROOT / "results" / "capability_benchmark" / f"prompt_ab_test_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "prompt_ab_test",
            "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_results": baseline_results,
            "fewshot_results": fewshot_results,
            "cot_results": cot_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
