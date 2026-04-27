"""
Phase H (Simple): Cross-solver Validation

简化版本：直接复用 runtime_trace_boundary_experiment.py 的数据收集逻辑，
但是支持不同的 solver。

目标：验证 first-pass boundary classifier 在不同 solver 上的泛化能力
"""

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    SearchLocalSolver,
    HeuristicLocalSolver,
    OracleSolver,
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
logger = logging.getLogger("cross_solver")


@dataclass
class RuntimeTrace:
    """Runtime trace for a single task"""
    task_id: str = ""
    bug_type: str = ""
    difficulty: str = ""
    condition: str = ""
    solver_name: str = ""
    boundary_label: str = ""

    single_success: bool = False
    blind_success: bool = False
    feedback_success: bool = False

    first_attempt_parse_ok: bool = False
    first_attempt_syntax_ok: bool = False
    first_visible_pass: bool = False
    first_hidden_pass: bool = False
    first_error_type: str = ""
    first_error_message_len: int = 0
    has_expected_actual: bool = False
    expected_actual_distance: float = 0.0
    first_patch_size: int = 0
    first_changed_from_buggy: bool = False

    blind_changed_code: bool = False
    blind_parse_ok: bool = False
    blind_error_type: str = ""

    feedback_changed_code: bool = False
    feedback_parse_ok: bool = False
    feedback_error_type: str = ""
    feedback_uses_error_signal: bool = False
    feedback_patch_size_delta: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_error_type(error_msg: str) -> str:
    """从错误消息中提取错误类型"""
    if not error_msg or error_msg == "pass":
        return "pass"
    error_msg = str(error_msg).lower()
    if "nameerror" in error_msg or "name 'x' is not defined" in error_msg:
        return "NameError"
    if "assertionerror" in error_msg:
        return "AssertionError"
    if "typeerror" in error_msg:
        return "TypeError"
    if "syntaxerror" in error_msg:
        return "SyntaxError"
    if "indentationerror" in error_msg:
        return "IndentationError"
    if "attributeerror" in error_msg:
        return "AttributeError"
    if "indexerror" in error_msg:
        return "IndexError"
    if "keyerror" in error_msg:
        return "KeyError"
    if "valueerror" in error_msg:
        return "ValueError"
    if "zerodivisionerror" in error_msg:
        return "ZeroDivisionError"
    if "overflowerror" in error_msg:
        return "OverflowError"
    if "empty" in error_msg or "no code" in error_msg:
        return "empty"
    return "other"


def run_single_attempt(
    task: BenchmarkTask,
    solver,
    error_feedback: str = "",
) -> tuple[str, bool, bool, str, str, int]:
    """
    运行单次尝试

    Returns:
        code, parse_ok, syntax_ok, error_type, error_message, patch_size
    """
    fn_name = task.metadata.get("function_name", "solve")
    visible_tests = _task_search_tests(task)
    buggy_code = task.metadata.get("buggy_code", "")

    if error_feedback and hasattr(solver, "revise"):
        first = solver.solve(task)
        attempt = solver.revise(task, first, error_feedback)
    else:
        attempt = solver.solve(task)

    code = attempt.answer if attempt else ""

    parse_ok = bool(code and code.strip())
    syntax_ok = parse_ok

    if not parse_ok:
        return code, parse_ok, syntax_ok, "empty", "empty_code", 0

    vis_pass, vis_msg = _run_code_task(fn_name, code, visible_tests)
    if not vis_pass:
        error_type = parse_error_type(vis_msg)
        return code, parse_ok, syntax_ok, error_type, vis_msg, len(code)

    all_tests = _task_verifier_tests(task)
    all_pass, all_msg = _run_code_task(fn_name, code, all_tests)

    if all_pass:
        error_type = "pass"
        error_msg = "pass"
    else:
        error_type = parse_error_type(all_msg)
        error_msg = all_msg

    return code, parse_ok, syntax_ok, error_type, error_msg, len(code)


def collect_traces_for_solver(
    solver_name: str,
    num_seeds: int = 5,
    num_tasks_per_seed: int = 50,
) -> List[RuntimeTrace]:
    """Collect runtime traces for a specific solver"""
    logger.info("=" * 80)
    logger.info(f"Collecting traces for solver: {solver_name}")
    logger.info("=" * 80)

    traces = []

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx * 100
        logger.info(f"Running seed {seed} ({seed_idx+1}/{num_seeds})...")

        all_tasks = generate_code_tasks(100, seed=0, variant="standard")
        rng = np.random.default_rng(seed)
        if len(all_tasks) > num_tasks_per_seed:
            indices = rng.choice(len(all_tasks), size=num_tasks_per_seed, replace=False)
            tasks = [all_tasks[i] for i in sorted(indices)]
        else:
            tasks = all_tasks

        if solver_name == "SearchLocalSolver":
            solver = SearchLocalSolver()
        elif solver_name == "HeuristicLocalSolver":
            solver = HeuristicLocalSolver()
        elif solver_name == "OracleSolver":
            solver = OracleSolver()
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        for task_idx, task in enumerate(tasks):
            if (task_idx + 1) % 10 == 0:
                logger.info(f"  Seed {seed}: {task_idx+1}/{len(tasks)} tasks done")

            buggy_code = task.metadata.get("buggy_code", "")

            single_code, single_parse, single_syntax, single_etype, single_emsg, single_psize = run_single_attempt(
                task, solver, error_feedback=""
            )

            fn_name = task.metadata.get("function_name", "solve")
            single_vis, _ = _run_code_task(fn_name, single_code, _task_search_tests(task)) if single_parse else (False, "")
            single_all, single_all_msg = _run_code_task(fn_name, single_code, _task_verifier_tests(task)) if single_parse else (False, "")

            has_expected_actual_single = "expected" in single_all_msg.lower() and "actual" in single_all_msg.lower() if single_all_msg else False
            expected_actual_distance_single = 0.2 if has_expected_actual_single else 0.0

            blind_code, blind_parse, blind_syntax, blind_etype, blind_emsg, blind_psize = run_single_attempt(
                task, solver, error_feedback=""
            )
            blind_changed = blind_code != single_code

            fb_code, fb_parse, fb_syntax, fb_etype, fb_emsg, fb_psize = run_single_attempt(
                task, solver, error_feedback=single_all_msg if not single_all else ""
            )
            feedback_changed = fb_code != single_code
            feedback_uses_error = bool(single_all_msg and single_all_msg != "pass")
            feedback_patch_delta = len(fb_code) - len(single_code) if fb_code and single_code else 0

            fb_all, _ = _run_code_task(fn_name, fb_code, _task_verifier_tests(task)) if fb_parse else (False, "")

            if single_all:
                boundary_label = "above"
            elif fb_all:
                boundary_label = "near"
            else:
                boundary_label = "below"

            trace = RuntimeTrace(
                task_id=task.task_id,
                bug_type=task.metadata.get("bug_type", ""),
                difficulty=task.metadata.get("difficulty", ""),
                condition=f"seed_{seed}",
                solver_name=solver_name,
                boundary_label=boundary_label,

                single_success=single_all,
                blind_success=False,
                feedback_success=fb_all,

                first_attempt_parse_ok=single_parse,
                first_attempt_syntax_ok=single_syntax,
                first_visible_pass=single_vis,
                first_hidden_pass=single_all,
                first_error_type=single_etype,
                first_error_message_len=len(single_emsg) if single_emsg else 0,
                has_expected_actual=has_expected_actual_single,
                expected_actual_distance=expected_actual_distance_single,
                first_patch_size=single_psize,
                first_changed_from_buggy=single_code != buggy_code if buggy_code else False,

                blind_changed_code=blind_changed,
                blind_parse_ok=blind_parse,
                blind_error_type=blind_etype,

                feedback_changed_code=feedback_changed,
                feedback_parse_ok=fb_parse,
                feedback_error_type=fb_etype,
                feedback_uses_error_signal=feedback_uses_error,
                feedback_patch_size_delta=feedback_patch_delta,
            )

            traces.append(trace)

    logger.info(f"Collected {len(traces)} traces for {solver_name}")
    return traces


def main():
    print("=" * 80)
    print("Phase H (Simple): Cross-solver Validation")
    print("=" * 80)

    solvers = ["SearchLocalSolver", "HeuristicLocalSolver"]
    all_traces_data = {}

    for solver_name in solvers:
        traces = collect_traces_for_solver(solver_name, num_seeds=5, num_tasks_per_seed=50)

        results_dir = PROJECT_ROOT / "results" / "cross_solver_validation"
        results_dir.mkdir(parents=True, exist_ok=True)

        output_path = results_dir / f"traces_{solver_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        traces_data = {
            "solver": solver_name,
            "n_traces": len(traces),
            "traces": [t.to_dict() for t in traces],
        }

        with open(output_path, "w") as f:
            json.dump(traces_data, f, indent=2)

        logger.info(f"Traces saved to: {output_path}")
        all_traces_data[solver_name] = traces_data

    print("\n" + "=" * 80)
    print("Phase H (Simple) Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
