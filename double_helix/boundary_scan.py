"""
Boundary Scan Experiment

Instead of varying model size (which we can't run), vary task difficulty
to construct the capability boundary.

Matrix:
  planner: search_solver, qwen_0.5b
  domain:  standard, paraphrase, stronger_paraphrase, naturalized
  condition: single, blind_retry, feedback_retry
  budget: max_attempts = 1, 2, 3

This gives us a boundary map from task-model combinations,
not just model size.
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

if os.name == "nt":
    try:
        import winreg
        _k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
        _mp, _ = winreg.QueryValueEx(_k, "Path")
        _k2 = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        _up, _ = winreg.QueryValueEx(_k2, "Path")
        os.environ["PATH"] = _mp + ";" + _up
    except Exception:
        pass

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
logger = logging.getLogger("boundary_scan")


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


def _try_qwen(
    task: BenchmarkTask,
    model,
    tokenizer,
    error_feedback: str = "",
    temperature: float = 0.3,
) -> tuple[str, bool, bool, str]:
    import torch

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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=384,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    code = _extract_code(response)
    if not code or "def solve" not in code:
        return code or "", False, False, "no_valid_function"

    code = _strip_imports(code)

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


def _try_gguf(
    task: BenchmarkTask,
    model_path: str,
    error_feedback: str = "",
    temperature: float = 0.3,
    n_ctx: int = 2048,
    n_predict: int = 384,
) -> tuple[str, bool, bool, str]:
    import subprocess

    fn_name = task.metadata.get("function_name", "solve")
    buggy_code = task.metadata.get("buggy_code", "")
    visible_tests = _task_search_tests(task)

    test_desc = ""
    for t in visible_tests:
        args_str = ", ".join(str(a) for a in t["args"])
        test_desc += f"  solve({args_str}) == {t['expected']}\n"

    user_msg = (
        f"Fix the bug in this Python function.\n"
        f"Function: {fn_name}\n"
        f"Tests:\n{test_desc}\n"
        f"Buggy code:\n```python\n{buggy_code}```\n"
    )
    if error_feedback:
        user_msg += f"\nPrevious attempt failed: {error_feedback}\nFix differently. Output ONLY the corrected function.\n"
    else:
        user_msg += "Output ONLY the corrected function.\n"

    model_name_lower = model_path.lower()
    if "phi" in model_name_lower:
        prompt = f"<|system|>You are a Python code fixer. Output ONLY the corrected function.<|end|>\n<|user|>\n{user_msg}<|end|>\n<|assistant| >\n"
    elif "gemma" in model_name_lower:
        prompt = f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"
    else:
        prompt = user_msg

    cmd = [
        "llama-cli",
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "-c", str(n_ctx),
        "--temp", str(temperature),
        "-ngl", "0",
        "--no-warmup",
        "--log-disable",
        "-no-cnv",
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
    ]

    try:
        env = os.environ.copy()
        result = subprocess.run(
            cmd, capture_output=True, timeout=120,
            env=env,
        )
        response = result.stdout.decode("utf-8", errors="replace").strip()
        for tok in ["<|end|>", "<|assistant| >", "<|assistant| >", "<|assistant| >", "<start_of_turn>", "<end_of_turn>"]:
            response = response.replace(tok, "")
    except subprocess.TimeoutExpired:
        return "", False, False, "timeout"
    except Exception as e:
        return "", False, False, f"subprocess_error:{e}"

    code = _extract_code(response)
    if not code or "def solve" not in code:
        return code or "", False, False, "no_valid_function"

    code = _strip_imports(code)

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


def _extract_code(text: str) -> str:
    text = text.replace("\r\n", "\n")
    for pat in [r"```python\n(.*?)```", r"```\n(.*?)```"]:
        matches = re.findall(pat, text, re.DOTALL)
        for m in reversed(matches):
            code = m.strip()
            if "def solve" in code:
                return code
    lines = text.strip().split("\n")
    code_lines = []
    in_func = False
    for line in lines:
        if "def solve" in line:
            in_func = True
            code_lines = [line]
            continue
        if in_func:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                break
            code_lines.append(line)
    return "\n".join(code_lines) if code_lines else ""


def _strip_imports(code: str) -> str:
    lines = code.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        kept.append(line)
    return "\n".join(kept)


def _build_belief_feedback(history: List[Dict]) -> str:
    if not history:
        return ""
    parts = []
    for h in history:
        if h["passed_all"]:
            continue
        attempt = h["attempt"]
        snippet = h.get("code_snippet", "")
        fb = h.get("feedback", "")
        parts.append(f"Attempt {attempt}: code was `{snippet}` → {fb}")
    if not parts:
        return ""
    return "Previous attempts ALL FAILED:\n" + "\n".join(parts) + "\nDo NOT repeat any of these approaches. Try a completely different fix."


def run_condition(
    tasks: List[BenchmarkTask],
    condition: str,
    solver_type: str,
    max_attempts: int = 3,
    model=None,
    tokenizer=None,
    gguf_path: str = "",
) -> List[Dict]:
    results = []
    solver = SearchLocalSolver() if solver_type == "search" else None

    for task in tasks:
        solved = False
        attempts = 0
        history = []

        for i in range(1, max_attempts + 1):
            fb_in = ""
            if condition == "feedback_retry" and i > 1 and history and not history[-1]["passed_all"]:
                fb_in = history[-1]["feedback"]
            elif condition == "belief_feedback" and i > 1:
                fb_in = _build_belief_feedback(history)

            if solver_type == "search":
                code, vis, all_p, fb = _try_search(task, solver, error_feedback=fb_in)
            elif gguf_path:
                temp = 0.3 if i == 1 else 0.5 + 0.2 * (i - 2)
                code, vis, all_p, fb = _try_gguf(
                    task, gguf_path,
                    error_feedback=fb_in, temperature=temp
                )
            else:
                temp = 0.3 if i == 1 else 0.5 + 0.2 * (i - 2)
                code, vis, all_p, fb = _try_qwen(
                    task, model, tokenizer,
                    error_feedback=fb_in, temperature=temp
                )

            attempts = i
            history.append({
                "attempt": i,
                "code_hash": hash(code) if code else 0,
                "code_snippet": code[:120] if code else "",
                "passed_visible": vis,
                "passed_all": all_p,
                "feedback": fb,
                "used_feedback": i > 1 and fb_in != "",
            })
            if all_p:
                solved = True
                break

            if condition == "single_shot":
                break

        results.append({
            "task_id": task.task_id,
            "bug_type": task.metadata.get("bug_type", ""),
            "difficulty": task.metadata.get("difficulty", ""),
            "solved": solved,
            "attempts": attempts,
            "code_changed_across_attempts": len(set(h["code_hash"] for h in history)) > 1,
            "history": history,
        })

    return results


def run_scan(
    solver_type: str = "search",
    variants: List[str] = ["standard", "paraphrase", "stronger_paraphrase", "naturalized"],
    conditions: List[str] = ["single_shot", "blind_retry", "feedback_retry"],
    budgets: List[int] = [1, 2, 3],
    num_tasks: int = 20,
    seed: int = 42,
    model=None,
    tokenizer=None,
    gguf_path: str = "",
    difficulty: str = "",
) -> Dict[str, Any]:
    cells = []

    for variant in variants:
        all_tasks = generate_code_tasks(100, seed=0, variant=variant, difficulty=difficulty)
        rng = np.random.default_rng(seed)
        if len(all_tasks) > num_tasks:
            indices = rng.choice(len(all_tasks), size=num_tasks, replace=False)
            tasks = [all_tasks[i] for i in sorted(indices)]
        else:
            tasks = all_tasks

        # Run all conditions for each budget
        for budget in budgets:
            for condition in conditions:
                if condition == "single_shot":
                    max_att = 1
                else:
                    max_att = budget

                logger.info(f"  {variant}/{condition}/budget={max_att} ...")
                results = run_condition(
                    tasks, condition, solver_type,
                    max_attempts=max_att, model=model, tokenizer=tokenizer,
                    gguf_path=gguf_path,
                )
                solved = sum(r["solved"] for r in results)
                rate = round(solved / max(len(results), 1), 4)

                cells.append({
                    "solver": solver_type,
                    "variant": variant,
                    "condition": condition,
                    "budget": max_att,
                    "solved": solved,
                    "total": len(results),
                    "rate": rate,
                    "per_task_results": results,
                })

    return {
        "experiment": "boundary_scan",
        "solver_type": solver_type,
        "variants": variants,
        "conditions": conditions,
        "budgets": budgets,
        "num_tasks": num_tasks,
        "seed": seed,
        "cells": cells,
    }


def print_boundary_map(result: Dict) -> None:
    cells = result["cells"]
    solver = result["solver_type"]
    budgets_available = sorted(set(c["budget"] for c in cells if c["condition"] != "single_shot"))
    max_budget = max(budgets_available) if budgets_available else 1

    print(f"\n{'='*70}")
    print(f"Boundary Map -- {solver}")
    print(f"{'='*70}")

    variants = sorted(set(c["variant"] for c in cells))
    conditions = ["single_shot", "blind_retry", "feedback_retry", "belief_feedback"]

    for variant in variants:
        print(f"\n  Variant: {variant}")
        header = f"  {'Condition':20s} |"
        for b in [1] + budgets_available:
            if b == 1:
                continue
            header += f" B={b:>1} |"
        header += " Delta(F-B)"
        print(header)
        print(f"  {'-'*60}")

        for cond in conditions:
            if cond == "single_shot":
                cell = next(
                    (c for c in cells
                     if c["variant"] == variant and c["condition"] == cond),
                    None
                )
                rate_str = f"{cell['rate']:.1%}" if cell else "  -  "
                print(f"  {cond:20s} | {rate_str:>6}")
            else:
                row_parts = []
                for b in budgets_available:
                    cell = next(
                        (c for c in cells
                         if c["variant"] == variant and c["condition"] == cond
                         and c["budget"] == b),
                        None
                    )
                    if cell:
                        row_parts.append((b, f"{cell['rate']:.1%}"))
                    else:
                        row_parts.append((b, "  -  "))

                line = f"  {cond:20s} |"
                for b, r in row_parts:
                    line += f" {r:>6} |"

                if cond in ("feedback_retry", "belief_feedback"):
                    blind_cell = next(
                        (c for c in cells
                         if c["variant"] == variant and c["condition"] == "blind_retry"
                         and c["budget"] == max_budget),
                        None
                    )
                    fb_cell = next(
                        (c for c in cells
                         if c["variant"] == variant and c["condition"] == cond
                         and c["budget"] == max_budget),
                        None
                    )
                    if blind_cell and fb_cell:
                        delta = fb_cell["rate"] - blind_cell["rate"]
                        line += f" {delta:+.1%}"
                print(line)

    print(f"\n  Near-boundary = where single_shot is 10-50% and feedback > blind")
    print(f"  Below-boundary = where single_shot < 10% and feedback ~ blind")
    print(f"  Above-boundary = where single_shot > 80% and feedback ~ single")

    all_results = []
    for c in cells:
        all_results.extend(c.get("per_task_results", []))
    if all_results:
        changed = sum(1 for r in all_results if r.get("code_changed_across_attempts", False))
        total = len(all_results)
        fb_tasks = [r for r in all_results if any(h.get("used_feedback") for h in r.get("history", []))]
        fb_changed = sum(1 for r in fb_tasks if r.get("code_changed_across_attempts", False))
        print(f"\n  Consistency: {changed}/{total} tasks changed code across attempts")
        if fb_tasks:
            print(f"  Feedback tasks: {fb_changed}/{len(fb_tasks)} changed code after feedback")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["search", "qwen", "qwen1.5", "gemma4b", "phi4mini"], default="search")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Task variants to scan (default: standard stronger_paraphrase naturalized)")
    parser.add_argument("--budget", nargs="+", type=int, default=[3],
                        help="Max attempts for retry conditions, can specify multiple values to scan budget dimension")
    parser.add_argument("--gguf-path", type=str, default=None,
                        help="Path to GGUF model file (overrides default paths)")
    parser.add_argument("--difficulty", choices=["trivial", "easy", "medium", "hard", ""], default="",
                        help="Filter tasks by difficulty (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("Boundary Scan Experiment")
    print("Constructing capability boundary via task difficulty, not model size")
    print("=" * 70)

    model = None
    tokenizer = None
    solver_tag = args.solver
    gguf_path = ""

    if args.solver in ("qwen", "qwen1.5"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.solver == "qwen1.5":
            model_name = "Qwen/Qwen2.5-1.5B"
        else:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
            trust_remote_code=True, local_files_only=True
        )
        model.eval()
        logger.info("Model loaded")

    elif args.solver in ("gemma4b", "phi4mini"):
        if args.gguf_path:
            gguf_path = args.gguf_path
        elif args.solver == "gemma4b":
            gguf_path = str(PROJECT_ROOT / "double_helix" / "google_gemma-3-4b-it-Q5_K_M (1).gguf")
        else:
            gguf_path = str(PROJECT_ROOT / "double_helix" / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf")

        if not Path(gguf_path).exists():
            logger.error(f"GGUF file not found: {gguf_path}")
            sys.exit(1)
        logger.info(f"Using GGUF model: {gguf_path}")

        import subprocess
        try:
            test_cmd = ["llama-cli", "--version"]
            subprocess.run(test_cmd, capture_output=True, timeout=10, check=True)
            logger.info("llama-cli found")
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.error("llama-cli not found. Install with: winget install llama.cpp")
            sys.exit(1)

    variants = args.variants or ["standard", "stronger_paraphrase", "naturalized"]
    conditions = ["single_shot", "blind_retry", "feedback_retry", "belief_feedback"]

    logger.info(f"Solver: {solver_tag}, Variants: {variants}, Tasks: {args.tasks}, Budgets: {args.budget}")

    result = run_scan(
        solver_type=solver_tag,
        variants=variants,
        conditions=conditions,
        budgets=args.budget,
        num_tasks=args.tasks,
        seed=args.seed,
        model=model,
        tokenizer=tokenizer,
        gguf_path=gguf_path,
        difficulty=args.difficulty,
    )

    print_boundary_map(result)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "double_helix" / "results" / f"boundary_scan_{solver_tag}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
