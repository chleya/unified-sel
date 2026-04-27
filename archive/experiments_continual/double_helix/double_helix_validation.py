"""
Double-Helix Validation Experiment

Core hypothesis: "planning chain + maintain chain > planning chain alone"
in a code-repair closed-loop environment.

Environment: code repair tasks (from capability_benchmark)
Planning chain: SearchLocalSolver (generates candidate fixes)
Maintain chain: test runner + rollback (stabilizes output)
Energy: max 3 attempts per task
Death: attempts exhausted without passing

Single chain (baseline): solver generates one fix, test once
Double chain (experiment): solver generates fix, test, if fail → retry with error feedback
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    BenchmarkVerifier,
    SearchLocalSolver,
    HeuristicLocalSolver,
    OracleSolver,
    _run_code_task,
    _task_verifier_tests,
    _task_search_tests,
    generate_code_tasks,
    generate_reasoning_tasks,
)


@dataclass
class AttemptRecord:
    attempt_number: int
    code: str
    passed_visible: bool
    passed_all: bool
    error_feedback: str
    strategy: str


@dataclass
class TaskResult:
    task_id: str
    family: str
    bug_type: str
    difficulty: str
    mode: str
    solved: bool
    attempts: int
    max_attempts: int
    attempt_history: List[Dict[str, Any]]
    final_code: Optional[str]
    energy_remaining: int


def _solve_one_attempt(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    error_feedback: str = "",
) -> tuple[str, bool, bool, str]:
    visible_tests = _task_search_tests(task)
    fn_name = task.metadata.get("function_name", "solve")

    if error_feedback and hasattr(solver, "revise"):
        first_attempt = solver.solve(task)
        attempt = solver.revise(task, first_attempt, error_feedback)
    else:
        attempt = solver.solve(task)

    code = attempt.answer

    if not code or not code.strip():
        return code, False, False, "empty_code"

    vis_pass, vis_msg = _run_code_task(fn_name, code, visible_tests)
    if not vis_pass:
        return code, False, False, f"visible_test_fail: {vis_msg}"

    all_tests = _task_verifier_tests(task)
    all_pass, all_msg = _run_code_task(fn_name, code, all_tests)
    return code, vis_pass, all_pass, "all_tests_passed" if all_pass else f"hidden_test_fail: {all_msg}"


def run_single_chain(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    max_attempts: int = 1,
) -> TaskResult:
    code, vis_pass, all_pass, feedback = _solve_one_attempt(task, solver)
    return TaskResult(
        task_id=task.task_id,
        family=task.family,
        bug_type=task.metadata.get("bug_type", ""),
        difficulty=task.metadata.get("difficulty", ""),
        mode="single_chain",
        solved=all_pass,
        attempts=1,
        max_attempts=max_attempts,
        attempt_history=[{
            "attempt": 1,
            "passed_visible": vis_pass,
            "passed_all": all_pass,
            "feedback": feedback,
        }],
        final_code=code if all_pass else None,
        energy_remaining=max_attempts - 1,
    )


def run_double_chain(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    max_attempts: int = 3,
) -> TaskResult:
    history = []
    best_code = None
    solved = False
    energy = max_attempts

    for attempt_num in range(1, max_attempts + 1):
        error_feedback = ""
        if attempt_num > 1 and history:
            last = history[-1]
            if not last["passed_all"]:
                error_feedback = last["feedback"]

        code, vis_pass, all_pass, feedback = _solve_one_attempt(
            task, solver, error_feedback=error_feedback
        )

        record = {
            "attempt": attempt_num,
            "passed_visible": vis_pass,
            "passed_all": all_pass,
            "feedback": feedback,
            "had_feedback": attempt_num > 1,
        }
        history.append(record)
        energy -= 1

        if all_pass:
            best_code = code
            solved = True
            break

        if vis_pass and best_code is None:
            best_code = code

    return TaskResult(
        task_id=task.task_id,
        family=task.family,
        bug_type=task.metadata.get("bug_type", ""),
        difficulty=task.metadata.get("difficulty", ""),
        mode="double_chain",
        solved=solved,
        attempts=len(history),
        max_attempts=max_attempts,
        attempt_history=history,
        final_code=best_code,
        energy_remaining=energy,
    )


def run_experiment(
    num_tasks: int = 20,
    seed: int = 42,
    max_attempts: int = 3,
    suite_variant: str = "standard",
) -> Dict[str, Any]:
    tasks = generate_code_tasks(num_tasks, seed, variant=suite_variant)
    solver = SearchLocalSolver()

    single_results = []
    double_results = []

    for task in tasks:
        sr = run_single_chain(task, solver, max_attempts=1)
        single_results.append(sr)

        dr = run_double_chain(task, solver, max_attempts=max_attempts)
        double_results.append(dr)

    single_solved = sum(1 for r in single_results if r.solved)
    double_solved = sum(1 for r in double_results if r.solved)
    total = len(tasks)

    single_by_difficulty = {}
    double_by_difficulty = {}
    for r in single_results:
        d = r.difficulty or "unknown"
        single_by_difficulty.setdefault(d, {"solved": 0, "total": 0})
        single_by_difficulty[d]["total"] += 1
        if r.solved:
            single_by_difficulty[d]["solved"] += 1
    for r in double_results:
        d = r.difficulty or "unknown"
        double_by_difficulty.setdefault(d, {"solved": 0, "total": 0})
        double_by_difficulty[d]["total"] += 1
        if r.solved:
            double_by_difficulty[d]["solved"] += 1

    double_attempts = [r.attempts for r in double_results]
    feedback_helped = sum(
        1 for r in double_results
        if r.solved and r.attempts > 1
        and any(h.get("had_feedback") for h in r.attempt_history)
    )

    return {
        "experiment": "double_helix_validation",
        "hypothesis": "planning_chain + maintain_chain > planning_chain alone",
        "num_tasks": total,
        "max_attempts": max_attempts,
        "suite_variant": suite_variant,
        "seed": seed,
        "single_chain": {
            "solved": single_solved,
            "total": total,
            "rate": round(single_solved / max(total, 1), 4),
            "by_difficulty": single_by_difficulty,
        },
        "double_chain": {
            "solved": double_solved,
            "total": total,
            "rate": round(double_solved / max(total, 1), 4),
            "avg_attempts": round(sum(double_attempts) / max(len(double_attempts), 1), 2),
            "feedback_helped_count": feedback_helped,
            "by_difficulty": double_by_difficulty,
        },
        "delta": {
            "solved_diff": double_solved - single_solved,
            "rate_diff": round((double_solved - single_solved) / max(total, 1), 4),
        },
        "per_task": [
            {
                "task_id": sr.task_id,
                "bug_type": sr.bug_type,
                "difficulty": sr.difficulty,
                "single_solved": sr.solved,
                "double_solved": dr.solved,
                "double_attempts": dr.attempts,
                "double_improved": dr.solved and not sr.solved,
            }
            for sr, dr in zip(single_results, double_results)
        ],
    }


def main():
    print("=" * 70)
    print("Double-Helix Validation Experiment")
    print("Hypothesis: planning + maintain > planning alone")
    print("=" * 70)

    for variant in ["standard", "paraphrase"]:
        print(f"\n--- Variant: {variant} ---")
        result = run_experiment(num_tasks=20, seed=42, max_attempts=3, suite_variant=variant)

        sc = result["single_chain"]
        dc = result["double_chain"]
        delta = result["delta"]

        print(f"\nSingle chain: {sc['solved']}/{sc['total']} = {sc['rate']:.1%}")
        print(f"Double chain: {dc['solved']}/{dc['total']} = {dc['rate']:.1%}")
        print(f"Delta: {delta['solved_diff']:+d} tasks ({delta['rate_diff']:+.1%})")
        print(f"Double chain avg attempts: {dc['avg_attempts']}")
        print(f"Feedback helped: {dc['feedback_helped_count']} tasks")

        improved = [t for t in result["per_task"] if t["double_improved"]]
        if improved:
            print(f"\nTasks improved by maintain chain ({len(improved)}):")
            for t in improved:
                print(f"  {t['task_id']} ({t['bug_type']}, {t['difficulty']}) "
                      f"attempts={t['double_attempts']}")
        else:
            print("\nNo tasks improved by maintain chain.")

        by_diff = dc["by_difficulty"]
        print(f"\nBy difficulty:")
        for diff in sorted(set(list(sc["by_difficulty"].keys()) + list(dc["by_difficulty"].keys()))):
            s = sc["by_difficulty"].get(diff, {"solved": 0, "total": 0})
            d = dc["by_difficulty"].get(diff, {"solved": 0, "total": 0})
            print(f"  {diff}: single={s['solved']}/{s['total']} double={d['solved']}/{d['total']}")

    print("\n" + "=" * 70)
    print("Experiment complete.")


if __name__ == "__main__":
    main()
