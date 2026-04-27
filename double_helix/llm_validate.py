"""
Double-Helix LLM Validation

Core question: does a maintain chain (test feedback + retry) significantly
improve LLM solve rate over a single-call LLM baseline?

Planning chain: Qwen2.5-0.5B-Instruct (generates code)
Maintain chain: test runner + error feedback + retry
Environment: code-repair tasks with deterministic test feedback

This is the critical upgrade from validate.py:
  validate.py proved: search + feedback > search
  THIS experiment proves: LLM + feedback > LLM
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HF_CACHE = str(PROJECT_ROOT / "topomem" / "data" / "models" / "hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

from core.capability_benchmark import (
    BenchmarkTask,
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
logger = logging.getLogger("llm_validate")


class QwenSolver:
    """Qwen2.5-0.5B-Instruct based code solver."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device = torch.device("cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True,
            local_files_only=True
        )
        self._model = self._model.to(device)
        self._model.eval()
        logger.info("Model loaded")

    def solve(
        self,
        task: BenchmarkTask,
        error_feedback: str = "",
        temperature: float = 0.3,
        max_tokens: int = 384,
    ) -> str:
        import torch

        fn_name = task.metadata.get("function_name", "solve")
        buggy_code = task.metadata.get("buggy_code", "")
        visible_tests = _task_search_tests(task)

        test_desc = ""
        for t in visible_tests:
            args_str = ", ".join(str(a) for a in t["args"])
            test_desc += f"  solve({args_str}) == {t['expected']}\n"

        prompt = (
            f"You are a Python code fixer. Fix the bug in the function below.\n"
            f"Function name: {fn_name}\n"
            f"Visible tests:\n{test_desc}\n"
            f"Buggy code:\n```python\n{buggy_code}```\n"
        )

        if error_feedback:
            prompt += (
                f"\nYour previous attempt failed with this error:\n{error_feedback}\n"
                f"Please fix the bug differently. Output ONLY the corrected function.\n"
            )
        else:
            prompt += "Output ONLY the corrected function, no explanation.\n"

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        code = self._extract_code(response)
        if not code:
            code = buggy_code
        return code

    def _extract_code(self, text: str) -> str:
        patterns = [
            r"```python\n(.*?)```",
            r"```\n(.*?)```",
            r"def solve\(.+",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                code = m.group(1).strip() if m.lastindex else m.group(0).strip()
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
        if code_lines:
            return "\n".join(code_lines)
        return ""


@dataclass
class TaskResult:
    task_id: str
    bug_type: str
    difficulty: str
    solved: bool
    attempts: int
    history: List[Dict[str, Any]]


def _try_llm(
    task: BenchmarkTask,
    solver: QwenSolver,
    error_feedback: str = "",
    temperature: float = 0.3,
) -> tuple[str, bool, bool, str]:
    fn_name = task.metadata.get("function_name", "solve")
    visible_tests = _task_search_tests(task)

    code = solver.solve(task, error_feedback=error_feedback, temperature=temperature)

    if not code or not code.strip() or "def solve" not in code:
        return code, False, False, "no_valid_function"

    vis_pass, vis_msg = _run_code_task(fn_name, code, visible_tests)
    if not vis_pass:
        return code, False, False, f"visible_fail:{vis_msg}"

    all_tests = _task_verifier_tests(task)
    all_pass, all_msg = _run_code_task(fn_name, code, all_tests)
    return code, vis_pass, all_pass, "pass" if all_pass else f"hidden_fail:{all_msg}"


def run_single_chain(task: BenchmarkTask, solver: QwenSolver) -> TaskResult:
    code, vis, all_p, fb = _try_llm(task, solver)
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
    solver: QwenSolver,
    max_attempts: int = 3,
) -> TaskResult:
    history = []
    solved = False

    for i in range(1, max_attempts + 1):
        fb = ""
        if i > 1 and history and not history[-1]["passed_all"]:
            fb = history[-1]["feedback"]

        temp = 0.3 if i == 1 else 0.5 + 0.2 * (i - 2)
        code, vis, all_p, feedback = _try_llm(task, solver, error_feedback=fb, temperature=temp)
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


def run_experiment(
    num_tasks: int = 20,
    seed: int = 42,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    tasks = generate_code_tasks(num_tasks, seed)
    solver = QwenSolver()

    single_results = []
    double_results = []

    for idx, task in enumerate(tasks):
        logger.info(f"Task {idx+1}/{len(tasks)}: {task.task_id} ({task.metadata.get('bug_type','')})")

        sr = run_single_chain(task, solver)
        single_results.append(sr)
        logger.info(f"  Single: {'PASS' if sr.solved else 'FAIL'}")

        dr = run_double_chain(task, solver, max_attempts)
        double_results.append(dr)
        logger.info(f"  Double: {'PASS' if dr.solved else 'FAIL'} (attempts={dr.attempts})")

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
        "experiment": "double_helix_llm_validation",
        "solver": "Qwen2.5-0.5B-Instruct",
        "num_tasks": num_tasks,
        "max_attempts": max_attempts,
        "seed": seed,
        "single_chain": {
            "solved": s_solved,
            "total": num_tasks,
            "rate": round(s_solved / max(num_tasks, 1), 4),
            "by_difficulty": {k: {"solved": v[0], "total": v[1]} for k, v in by_diff_s.items()},
        },
        "double_chain": {
            "solved": d_solved,
            "total": num_tasks,
            "rate": round(d_solved / max(num_tasks, 1), 4),
            "avg_attempts": round(np.mean([r.attempts for r in double_results]), 2),
            "feedback_helped": feedback_helped,
            "by_difficulty": {k: {"solved": v[0], "total": v[1]} for k, v in by_diff_d.items()},
        },
        "delta": {
            "solved_diff": d_solved - s_solved,
            "rate_diff": round((d_solved - s_solved) / max(num_tasks, 1), 4),
        },
        "per_task": [
            {
                "task_id": s.task_id,
                "bug_type": s.bug_type,
                "difficulty": s.difficulty,
                "single_solved": s.solved,
                "double_solved": d.solved,
                "double_attempts": d.attempts,
                "double_improved": d.solved and not s.solved,
            }
            for s, d in zip(single_results, double_results)
        ],
    }


def main():
    print("=" * 70)
    print("Double-Helix LLM Validation")
    print("Hypothesis: LLM + maintain chain > LLM alone")
    print("Solver: Qwen2.5-0.5B-Instruct")
    print("=" * 70)

    result = run_experiment(num_tasks=20, seed=42, max_attempts=3)

    sc = result["single_chain"]
    dc = result["double_chain"]
    delta = result["delta"]

    print(f"\nSingle chain (LLM only): {sc['solved']}/{sc['total']} = {sc['rate']:.1%}")
    print(f"Double chain (LLM+feedback): {dc['solved']}/{dc['total']} = {dc['rate']:.1%}")
    print(f"Delta: {delta['solved_diff']:+d} tasks ({delta['rate_diff']:+.1%})")
    print(f"Feedback helped: {dc['feedback_helped']} tasks")
    print(f"Avg attempts: {dc['avg_attempts']}")

    improved = [t for t in result["per_task"] if t["double_improved"]]
    if improved:
        print(f"\nTasks improved by maintain chain ({len(improved)}):")
        for t in improved:
            print(f"  {t['task_id']} ({t['bug_type']}, {t['difficulty']}) "
                  f"attempts={t['double_attempts']}")
    else:
        print("\nNo tasks improved by maintain chain.")

    print(f"\nBy difficulty:")
    all_diffs = sorted(set(
        list(sc["by_difficulty"].keys()) + list(dc["by_difficulty"].keys())
    ))
    for diff in all_diffs:
        s = sc["by_difficulty"].get(diff, {"solved": 0, "total": 0})
        d = dc["by_difficulty"].get(diff, {"solved": 0, "total": 0})
        print(f"  {diff}: single={s['solved']}/{s['total']} double={d['solved']}/{d['total']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "double_helix" / "results" / f"llm_validation_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if delta["rate_diff"] > 0:
        print(f"MAINTAIN CHAIN IMPROVES LLM SOLVE RATE (+{delta['rate_diff']:.1%})")
        print("Double-helix hypothesis supported for LLM: feedback+retry > single call")
    elif delta["rate_diff"] == 0:
        print("NO IMPROVEMENT FROM MAINTAIN CHAIN")
        print("LLM may already be at ceiling, or feedback not useful for this solver")
    else:
        print(f"MAINTAIN CHAIN HURTS LLM ({delta['rate_diff']:+.1%})")
        print("Retry with different temperature/feedback strategy may be needed")


if __name__ == "__main__":
    main()
