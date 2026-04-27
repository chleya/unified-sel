"""
Double-Helix Minimal Validation

Core question: does a maintain chain (test feedback + retry) significantly
improve solve rate over a single-chain baseline?

Environment: code-repair tasks with deterministic test feedback
Planning chain: SearchLocalSolver (generates candidate fixes)
Maintain chain: test runner + error feedback + retry
Energy: max_attempts per task
Death: attempts exhausted without passing

Statistical design: 5 seeds, bootstrap CI, Cohen's d
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    SearchLocalSolver,
    _run_code_task,
    _task_verifier_tests,
    _task_search_tests,
    generate_code_tasks,
    generate_reasoning_tasks,
)


@dataclass
class AttemptRecord:
    attempt: int
    passed_visible: bool
    passed_all: bool
    feedback: str
    used_feedback: bool


@dataclass
class TaskResult:
    task_id: str
    bug_type: str
    difficulty: str
    solved: bool
    attempts: int
    history: List[Dict[str, Any]]


def _try_once(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    error_feedback: str = "",
) -> tuple[str, bool, bool, str]:
    fn_name = task.metadata.get("function_name", "solve")
    visible_tests = _task_search_tests(task)

    if error_feedback and hasattr(solver, "revise"):
        first = solver.solve(task)
        attempt = solver.revise(task, first, error_feedback)
    else:
        attempt = solver.solve(task)

    code = attempt.answer
    if not code or not code.strip():
        return code, False, False, "empty_code"

    vis_pass, vis_msg = _run_code_task(fn_name, code, visible_tests)
    if not vis_pass:
        return code, False, False, f"visible_fail:{vis_msg}"

    all_tests = _task_verifier_tests(task)
    all_pass, all_msg = _run_code_task(fn_name, code, all_tests)
    return code, vis_pass, all_pass, "pass" if all_pass else f"hidden_fail:{all_msg}"


def run_single_chain(task: BenchmarkTask, solver: SearchLocalSolver) -> TaskResult:
    code, vis, all_p, fb = _try_once(task, solver)
    return TaskResult(
        task_id=task.task_id,
        bug_type=task.metadata.get("bug_type", ""),
        difficulty=task.metadata.get("difficulty", ""),
        solved=all_p,
        attempts=1,
        history=[{"attempt": 1, "passed_visible": vis, "passed_all": all_p,
                  "feedback": fb, "used_feedback": False}],
    )


def run_double_chain(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    max_attempts: int = 3,
) -> TaskResult:
    history = []
    solved = False

    for i in range(1, max_attempts + 1):
        fb = history[-1]["feedback"] if i > 1 and history and not history[-1]["passed_all"] else ""
        code, vis, all_p, feedback = _try_once(task, solver, error_feedback=fb)
        history.append({
            "attempt": i,
            "passed_visible": vis,
            "passed_all": all_p,
            "feedback": feedback,
            "used_feedback": i > 1 and fb != "",
        })
        if all_p:
            solved = True
            break

    return TaskResult(
        task_id=task.task_id,
        bug_type=task.metadata.get("bug_type", ""),
        difficulty=task.metadata.get("difficulty", ""),
        solved=solved,
        attempts=len(history),
        history=history,
    )


def run_seed(
    seed: int,
    num_tasks: int = 20,
    max_attempts: int = 3,
    variant: str = "standard",
) -> Dict[str, Any]:
    all_tasks = generate_code_tasks(100, seed=0, variant=variant)
    rng = np.random.default_rng(seed)
    if len(all_tasks) > num_tasks:
        indices = rng.choice(len(all_tasks), size=num_tasks, replace=False)
        tasks = [all_tasks[i] for i in sorted(indices)]
    else:
        tasks = all_tasks

    solver = SearchLocalSolver()

    single_results = [run_single_chain(t, solver) for t in tasks]
    double_results = [run_double_chain(t, solver, max_attempts) for t in tasks]

    s_solved = sum(r.solved for r in single_results)
    d_solved = sum(r.solved for r in double_results)

    feedback_helped = sum(
        1 for r in double_results
        if r.solved and r.attempts > 1
        and any(h.get("used_feedback") for h in r.history)
    )

    by_diff_s = {}
    by_diff_d = {}
    for r in single_results:
        by_diff_s.setdefault(r.difficulty, [0, 0])
        by_diff_s[r.difficulty][1] += 1
        if r.solved:
            by_diff_s[r.difficulty][0] += 1
    for r in double_results:
        by_diff_d.setdefault(r.difficulty, [0, 0])
        by_diff_d[r.difficulty][1] += 1
        if r.solved:
            by_diff_d[r.difficulty][0] += 1

    return {
        "seed": seed,
        "variant": variant,
        "num_tasks": num_tasks,
        "max_attempts": max_attempts,
        "single_solved": s_solved,
        "double_solved": d_solved,
        "single_rate": round(s_solved / max(num_tasks, 1), 4),
        "double_rate": round(d_solved / max(num_tasks, 1), 4),
        "delta": d_solved - s_solved,
        "delta_rate": round((d_solved - s_solved) / max(num_tasks, 1), 4),
        "feedback_helped": feedback_helped,
        "avg_attempts": round(np.mean([r.attempts for r in double_results]), 2),
        "by_difficulty_single": {k: {"solved": v[0], "total": v[1]} for k, v in by_diff_s.items()},
        "by_difficulty_double": {k: {"solved": v[0], "total": v[1]} for k, v in by_diff_d.items()},
        "per_task": [
            {
                "task_id": s.task_id,
                "bug_type": s.bug_type,
                "difficulty": s.difficulty,
                "single": s.solved,
                "double": d.solved,
                "attempts": d.attempts,
                "improved": d.solved and not s.solved,
            }
            for s, d in zip(single_results, double_results)
        ],
    }


def run_full_experiment(
    seeds: List[int] = [42, 123, 456, 789, 1024],
    num_tasks: int = 20,
    max_attempts: int = 3,
    variants: List[str] = ["standard"],
) -> Dict[str, Any]:
    all_results = {}
    for variant in variants:
        seed_results = []
        for seed in seeds:
            r = run_seed(seed, num_tasks, max_attempts, variant)
            seed_results.append(r)
        all_results[variant] = seed_results

    summary = {}
    for variant, results in all_results.items():
        s_rates = [r["single_rate"] for r in results]
        d_rates = [r["double_rate"] for r in results]
        deltas = [r["delta_rate"] for r in results]
        helped = [r["feedback_helped"] for r in results]
        attempts = [r["avg_attempts"] for r in results]

        s_arr = np.array(s_rates)
        d_arr = np.array(d_rates)
        diff = d_arr - s_arr

        n_boot = 10000
        rng = np.random.default_rng(0)
        boot_means = []
        for _ in range(n_boot):
            sample = rng.choice(diff, size=len(diff), replace=True)
            boot_means.append(float(np.mean(sample)))
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        pooled_std = float(np.sqrt((s_arr.std() ** 2 + d_arr.std() ** 2) / 2))
        cohens_d = float(diff.mean() / pooled_std) if pooled_std > 0 else 0.0

        from scipy import stats as sp_stats
        t_stat, p_value = sp_stats.ttest_rel(d_arr, s_arr)

        summary[variant] = {
            "seeds": seeds,
            "num_tasks_per_seed": num_tasks,
            "single_rate_mean": round(float(s_arr.mean()), 4),
            "single_rate_std": round(float(s_arr.std()), 4),
            "double_rate_mean": round(float(d_arr.mean()), 4),
            "double_rate_std": round(float(d_arr.std()), 4),
            "delta_mean": round(float(diff.mean()), 4),
            "delta_std": round(float(diff.std()), 4),
            "bootstrap_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "cohens_d": round(cohens_d, 3),
            "paired_t": round(float(t_stat), 3),
            "p_value": round(float(p_value), 6),
            "significant_005": p_value < 0.05,
            "feedback_helped_mean": round(float(np.mean(helped)), 2),
            "avg_attempts_mean": round(float(np.mean(attempts)), 2),
        }

    return {"experiment": "double_helix_minimal_validation", "variants": summary, "raw": all_results}


def main():
    print("=" * 70)
    print("Double-Helix Minimal Validation")
    print("Hypothesis: maintain chain (feedback+retry) > single chain")
    print("=" * 70)

    result = run_full_experiment(
        seeds=[42, 123, 456, 789, 1024],
        num_tasks=20,
        max_attempts=3,
        variants=["standard", "paraphrase"],
    )

    for variant, s in result["variants"].items():
        print(f"\n{'='*50}")
        print(f"Variant: {variant}")
        print(f"{'='*50}")
        print(f"  Single chain: {s['single_rate_mean']:.1%} ± {s['single_rate_std']:.1%}")
        print(f"  Double chain: {s['double_rate_mean']:.1%} ± {s['double_rate_std']:.1%}")
        print(f"  Delta:        {s['delta_mean']:+.1%} ± {s['delta_std']:.1%}")
        print(f"  95% CI:       [{s['bootstrap_95ci'][0]:+.1%}, {s['bootstrap_95ci'][1]:+.1%}]")
        print(f"  Cohen's d:    {s['cohens_d']:.3f}")
        print(f"  Paired t:     {s['paired_t']:.3f}, p={s['p_value']:.6f}")
        print(f"  Significant:  {'YES' if s['significant_005'] else 'NO'} (alpha=0.05)")
        print(f"  Feedback helped: {s['feedback_helped_mean']:.1f} tasks/seed")
        print(f"  Avg attempts:    {s['avg_attempts_mean']:.1f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "double_helix" / "results" / f"validation_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("PRELIMINARY RESULTS SUMMARY")
    print("=" * 70)

    # Add notes on limitations
    print("\n⚠️  IMPORTANT LIMITATIONS:")
    print("  - Sample size: n=5 seeds, results may not be robust")
    print("  - No multiple comparison correction for multiple variants tested")
    print("  - These are preliminary findings, not definitive conclusions\n")

    significant_count = sum(1 for s in result["variants"].values() if s["significant_005"])
    total_variants = len(result["variants"])

    print(f"  {significant_count}/{total_variants} variants showed statistically significant benefit (p < 0.05)")
    print()

    for variant, s in result["variants"].items():
        sig_status = "✓ Significant benefit" if s["significant_005"] else "✗ No significant benefit"
        print(f"  {variant:<15s}: {sig_status} (delta={s['delta_mean']:+.1%}, p={s['p_value']:.4f})")

    print()
    print("  Preliminary evidence suggests feedback+retry may provide benefit for some task variants,")
    print("  but larger sample sizes and more rigorous testing are needed for definitive conclusions.")


if __name__ == "__main__":
    main()
