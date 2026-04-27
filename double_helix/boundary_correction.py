"""
Boundary Correction Experiment — 4-Group Controlled

Core question: does feedback-maintain repair near-boundary failures
more effectively than blind retry?

4 conditions (same total budget = 3 attempts):
  1. single-shot: 1 attempt, no retry
  2. blind_retry: 3 attempts, no feedback between attempts
  3. feedback_retry: 3 attempts, error feedback between attempts
  4. feedback_retry_escalate: 3 attempts, feedback + escalate to API on final attempt

Solver tiers:
  - SearchLocalSolver (has capability, tests correction mechanism)
  - Qwen2.5-0.5B-Instruct (below boundary, tests inverted-U prediction)
  - MiniMax-2.7 API (near boundary, tests where correction should peak)
"""
from __future__ import annotations

import os
import sys
import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

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
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("boundary_correction")


@dataclass
class AttemptRecord:
    attempt: int
    passed_visible: bool
    passed_all: bool
    feedback: str
    used_feedback: bool
    solver_name: str


@dataclass
class TaskResult:
    task_id: str
    bug_type: str
    difficulty: str
    condition: str
    solved: bool
    attempts: int
    history: List[Dict[str, Any]]


def _try_search(
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


def _call_minimax(prompt: str, temperature: float = 0.3, max_tokens: int = 384) -> str:
    API_KEY = os.environ.get("MINIMAX_API_KEY")
    if not API_KEY:
        raise ValueError("MINIMAX_API_KEY environment variable not set")
    API_URL = "https://api.minimaxi.com/anthropic/v1/messages"
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": "MiniMax-2.7",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    for attempt in range(3):
        try:
            import urllib.request, ssl
            ctx = ssl.create_default_context()
            req = urllib.request.Request(
                API_URL,
                data=json.dumps(payload).encode('utf-8'),
                headers=HEADERS,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                content = data.get('content', [])
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        return block.get('text', '')
            return ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


def _extract_code(text: str) -> str:
    for pat in [r"```python\n(.*?)```", r"```\n(.*?)```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "def solve" in code:
                return code
    lines = text.strip().split("\n")
    code_lines = []
    in_func = False
    for line in lines:
        if "def solve" in line:
            in_func = True
        if in_func:
            code_lines.append(line)
    return "\n".join(code_lines) if code_lines else ""


def _try_api(
    task: BenchmarkTask,
    error_feedback: str = "",
    temperature: float = 0.3,
) -> tuple[str, bool, bool, str]:
    fn_name = task.metadata.get("function_name", "solve")
    buggy_code = task.metadata.get("buggy_code", "")
    visible_tests = _task_search_tests(task)

    test_desc = ""
    for t in visible_tests:
        args_str = ", ".join(str(a) for a in t["args"])
        test_desc += f"  solve({args_str}) == {t['expected']}\n"

    prompt = (
        f"Fix the bug in this Python function.\n"
        f"Function: {fn_name}\n"
        f"Tests:\n{test_desc}\n"
        f"Buggy code:\n```python\n{buggy_code}```\n"
    )
    if error_feedback:
        prompt += f"\nPrevious attempt failed: {error_feedback}\nFix differently. Output ONLY the corrected function.\n"
    else:
        prompt += "Output ONLY the corrected function.\n"

    response = _call_minimax(prompt, temperature=temperature)
    code = _extract_code(response)
    if not code or "def solve" not in code:
        return code or "", False, False, "no_valid_function"

    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return code, False, False, f"syntax_error:{e}"

    vis_pass, vis_msg = _run_code_task(fn_name, code, visible_tests)
    if not vis_pass:
        return code, False, False, f"visible_fail:{vis_msg}"

    all_tests = _task_verifier_tests(task)
    all_pass, all_msg = _run_code_task(fn_name, code, all_tests)
    return code, vis_pass, all_pass, "pass" if all_pass else f"hidden_fail:{all_msg}"


def run_condition(
    task: BenchmarkTask,
    condition: str,
    solver_type: str = "search",
    max_attempts: int = 3,
) -> TaskResult:
    history = []
    solved = False

    if condition == "single_shot":
        if solver_type == "search":
            code, vis, all_p, fb = _try_search(task, SearchLocalSolver())
        else:
            code, vis, all_p, fb = _try_api(task)
        history.append({"attempt": 1, "passed_visible": vis, "passed_all": all_p,
                        "feedback": fb, "used_feedback": False, "solver": solver_type})
        solved = all_p

    elif condition == "blind_retry":
        for i in range(1, max_attempts + 1):
            if solver_type == "search":
                code, vis, all_p, fb = _try_search(task, SearchLocalSolver())
            else:
                code, vis, all_p, fb = _try_api(task, temperature=0.3 + 0.2 * (i - 1))
            history.append({"attempt": i, "passed_visible": vis, "passed_all": all_p,
                            "feedback": fb, "used_feedback": False, "solver": solver_type})
            if all_p:
                solved = True
                break

    elif condition == "feedback_retry":
        for i in range(1, max_attempts + 1):
            fb_in = ""
            if i > 1 and history and not history[-1]["passed_all"]:
                fb_in = history[-1]["feedback"]
            if solver_type == "search":
                code, vis, all_p, fb = _try_search(task, SearchLocalSolver(), error_feedback=fb_in)
            else:
                code, vis, all_p, fb = _try_api(task, error_feedback=fb_in,
                                                 temperature=0.3 + 0.2 * (i - 1))
            history.append({"attempt": i, "passed_visible": vis, "passed_all": all_p,
                            "feedback": fb, "used_feedback": i > 1 and fb_in != "",
                            "solver": solver_type})
            if all_p:
                solved = True
                break

    elif condition == "feedback_retry_escalate":
        for i in range(1, max_attempts + 1):
            fb_in = ""
            if i > 1 and history and not history[-1]["passed_all"]:
                fb_in = history[-1]["feedback"]

            use_escalation = (i == max_attempts) and not solved

            if use_escalation and solver_type == "search":
                code, vis, all_p, fb = _try_api(task, error_feedback=fb_in, temperature=0.5)
                actual_solver = "api_escalated"
            elif solver_type == "search":
                code, vis, all_p, fb = _try_search(task, SearchLocalSolver(), error_feedback=fb_in)
                actual_solver = solver_type
            else:
                code, vis, all_p, fb = _try_api(task, error_feedback=fb_in,
                                                 temperature=0.3 + 0.2 * (i - 1))
                actual_solver = solver_type

            history.append({"attempt": i, "passed_visible": vis, "passed_all": all_p,
                            "feedback": fb, "used_feedback": i > 1 and fb_in != "",
                            "solver": actual_solver, "escalated": use_escalation})
            if all_p:
                solved = True
                break

    return TaskResult(
        task_id=task.task_id,
        bug_type=task.metadata.get("bug_type", ""),
        difficulty=task.metadata.get("difficulty", ""),
        condition=condition,
        solved=solved,
        attempts=len(history),
        history=history,
    )


def run_experiment(
    solver_type: str = "search",
    num_tasks: int = 20,
    seed: int = 42,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    all_tasks = generate_code_tasks(100, seed=0)
    rng = np.random.default_rng(seed)
    if len(all_tasks) > num_tasks:
        indices = rng.choice(len(all_tasks), size=num_tasks, replace=False)
        tasks = [all_tasks[i] for i in sorted(indices)]
    else:
        tasks = all_tasks

    conditions = ["single_shot", "blind_retry", "feedback_retry", "feedback_retry_escalate"]
    results_by_condition = {}

    for cond in conditions:
        logger.info(f"Running condition: {cond} (solver={solver_type})")
        task_results = []
        for idx, task in enumerate(tasks):
            tr = run_condition(task, cond, solver_type=solver_type, max_attempts=max_attempts)
            task_results.append(tr)
            if (idx + 1) % 5 == 0:
                logger.info(f"  {cond}: {idx+1}/{len(tasks)} done, "
                            f"solved={sum(r.solved for r in task_results)}")
        results_by_condition[cond] = task_results

    summary = {}
    for cond, results in results_by_condition.items():
        solved = sum(r.solved for r in results)
        summary[cond] = {
            "solved": solved,
            "total": len(results),
            "rate": round(solved / max(len(results), 1), 4),
            "avg_attempts": round(np.mean([r.attempts for r in results]), 2),
        }

    by_difficulty = {}
    for cond, results in results_by_condition.items():
        for r in results:
            key = (r.difficulty, cond)
            by_difficulty.setdefault(key, [0, 0])
            by_difficulty[key][1] += 1
            if r.solved:
                by_difficulty[key][0] += 1

    feedback_vs_blind = {}
    for idx, task in enumerate(tasks):
        blind = results_by_condition["blind_retry"][idx]
        feedback = results_by_condition["feedback_retry"][idx]
        feedback_vs_blind[task.task_id] = {
            "blind_solved": blind.solved,
            "feedback_solved": feedback.solved,
            "feedback_better": feedback.solved and not blind.solved,
            "blind_better": blind.solved and not feedback.solved,
        }

    fb_better_count = sum(1 for v in feedback_vs_blind.values() if v["feedback_better"])
    blind_better_count = sum(1 for v in feedback_vs_blind.values() if v["blind_better"])

    return {
        "experiment": "boundary_correction_4group",
        "solver_type": solver_type,
        "num_tasks": num_tasks,
        "seed": seed,
        "max_attempts": max_attempts,
        "conditions": conditions,
        "summary": summary,
        "by_difficulty": {f"{k[0]}_{k[1]}": {"solved": v[0], "total": v[1]}
                         for k, v in by_difficulty.items()},
        "feedback_vs_blind": {
            "feedback_better": fb_better_count,
            "blind_better": blind_better_count,
            "equal": num_tasks - fb_better_count - blind_better_count,
        },
        "per_task": [
            {
                "task_id": task.task_id,
                "bug_type": task.metadata.get("bug_type", ""),
                "difficulty": task.metadata.get("difficulty", ""),
                **{cond: results_by_condition[cond][idx].solved for cond in conditions},
                **{f"{cond}_attempts": results_by_condition[cond][idx].attempts for cond in conditions},
            }
            for idx, task in enumerate(tasks)
        ],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["search", "api"], default="search")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--tasks", type=int, default=20, help="Tasks per seed")
    args = parser.parse_args()

    print("=" * 70)
    print("Boundary Correction Experiment — 4-Group Controlled")
    print("Question: feedback-maintain > blind retry near capability boundary?")
    print("=" * 70)

    all_results = []
    seed_list = [42, 123, 456][:args.seeds]

    for seed in seed_list:
        print(f"\n{'='*50}")
        print(f"Solver: {args.solver}, Seed: {seed}")
        print(f"{'='*50}")

        result = run_experiment(
            solver_type=args.solver, num_tasks=args.tasks, seed=seed, max_attempts=3
        )
        all_results.append(result)

        for cond in ["single_shot", "blind_retry", "feedback_retry", "feedback_retry_escalate"]:
            s = result["summary"][cond]
            print(f"  {cond:30s}: {s['solved']:2d}/{s['total']} = {s['rate']:.1%}")

        fvb = result["feedback_vs_blind"]
        print(f"  Feedback>Blind: {fvb['feedback_better']}  Blind>Feedback: {fvb['blind_better']}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "double_helix" / "results" / f"4group_{args.solver}_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved to {out_path}")

    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("Multi-seed summary")
        print(f"{'='*50}")
        for cond in ["single_shot", "blind_retry", "feedback_retry", "feedback_retry_escalate"]:
            rates = [r["summary"][cond]["rate"] for r in all_results]
            print(f"  {cond:30s}: {np.mean(rates):.1%} +/- {np.std(rates):.1%}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("Check: feedback_retry > blind_retry?")
    print("If YES: maintain chain has value beyond 'try more times'")
    print("If NO:  maintain chain is just retry, drop the narrative")


if __name__ == "__main__":
    main()
