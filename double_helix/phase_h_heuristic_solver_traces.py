"""
Phase H: Collect HeuristicLocalSolver Traces

目标：收集 HeuristicLocalSolver 的 traces，用于跨 solver 验证
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
logger = logging.getLogger("phase_h")


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


def check_has_expected_actual(error_message: str) -> bool:
    """检查错误消息是否包含 expected/actual 信息"""
    if not error_message:
        return False
    error_lower = error_message.lower()
    return ("expected" in error_lower and "actual" in error_lower) or \
           ("expected" in error_lower and "got" in error_lower)


def extract_expected_actual_distance(error_message: str) -> float:
    """估算 expected 和 actual 之间的距离（归一化）"""
    if not error_message:
        return 1.0
    msg_len = len(error_message)
    if msg_len < 50:
        return 0.2
    elif msg_len < 100:
        return 0.4
    elif msg_len < 200:
        return 0.6
    else:
        return 0.8


def run_single_attempt(
    task: BenchmarkTask,
    solver,
    error_feedback: str = "",
) -> tuple[str, bool, bool, str, str, int]:
    """运行单次尝试"""
    fn_name = task.metadata.get("function_name", "solve")
    visible_tests = _task_search_tests(task)

    if error_feedback and hasattr(solver, "revise"):
        first = solver.solve(task)
        attempt = solver.revise(task, first, error_feedback)
    else:
        attempt = solver.solve(task)

    code = attempt.answer if attempt else ""

    parse_ok = bool(code and code.strip())
    syntax_ok = parse_ok

    if not parse_ok:
        return code, False, False, "empty", "", 0

    vis_pass, vis_msg = _run_code_task(fn_name, code, visible_tests)
    if not vis_pass:
        error_type = parse_error_type(vis_msg)
        return code, parse_ok, syntax_ok, error_type, vis_msg, len(code)

    all_tests = _task_verifier_tests(task)
    all_pass, all_msg = _run_code_task(fn_name, code, all_tests)

    if not all_pass:
        error_type = parse_error_type(all_msg)
        return code, parse_ok, syntax_ok, error_type, all_msg, len(code)

    return code, parse_ok, syntax_ok, "pass", "", len(code)


def collect_trace_for_task(
    task: BenchmarkTask,
    solver,
    solver_name: str,
    seed: int,
) -> RuntimeTrace:
    """为一个 task 收集完整的 runtime trace"""
    trace = RuntimeTrace()

    trace.task_id = task.metadata.get("task_id", f"task_{seed}")
    trace.bug_type = task.metadata.get("bug_type", "unknown")
    trace.difficulty = task.metadata.get("difficulty", "unknown")
    trace.condition = f"seed_{seed}"
    trace.solver_name = solver_name

    fn_name = task.metadata.get("function_name", "solve")
    buggy_code = task.metadata.get("buggy_code", "")

    code1, parse_ok1, syntax_ok1, err_type1, err_msg1, patch_size1 = run_single_attempt(task, solver, "")
    trace.first_attempt_parse_ok = parse_ok1
    trace.first_attempt_syntax_ok = syntax_ok1
    trace.first_error_type = err_type1
    trace.first_error_message_len = len(err_msg1) if err_msg1 else 0
    trace.has_expected_actual = check_has_expected_actual(err_msg1)
    trace.expected_actual_distance = extract_expected_actual_distance(err_msg1)
    trace.first_patch_size = patch_size1
    trace.first_changed_from_buggy = code1 != buggy_code if buggy_code else False

    if parse_ok1 and syntax_ok1:
        visible_tests = _task_search_tests(task)
        all_tests = _task_verifier_tests(task)
        vis_pass, _ = _run_code_task(fn_name, code1, visible_tests)
        all_pass, _ = _run_code_task(fn_name, code1, all_tests)
        trace.first_visible_pass = vis_pass
        trace.first_hidden_pass = all_pass
        trace.single_success = all_pass
    else:
        trace.first_visible_pass = False
        trace.first_hidden_pass = False
        trace.single_success = False

    if not trace.first_hidden_pass:
        code2, parse_ok2, _, err_type2, err_msg2, _ = run_single_attempt(task, solver, "")
        trace.blind_changed_code = (code2 != code1) if code1 and code2 else False
        trace.blind_parse_ok = parse_ok2
        trace.blind_error_type = err_type2

        if parse_ok2:
            all_tests = _task_verifier_tests(task)
            blind_pass, _ = _run_code_task(fn_name, code2, all_tests)
            trace.blind_success = blind_pass
        else:
            trace.blind_success = False
    else:
        trace.blind_changed_code = False
        trace.blind_parse_ok = True
        trace.blind_error_type = "pass"
        trace.blind_success = True

    feedback_for_retry = err_msg1 if not trace.single_success else ""
    code3, parse_ok3, _, err_type3, err_msg3, _ = run_single_attempt(task, solver, feedback_for_retry)
    trace.feedback_changed_code = (code3 != code1) if code1 and code3 else False
    trace.feedback_parse_ok = parse_ok3
    trace.feedback_error_type = err_type3
    trace.feedback_uses_error_signal = bool(feedback_for_retry and feedback_for_retry != "pass")
    trace.feedback_patch_size_delta = len(code3) - len(code1) if code3 and code1 else 0

    if parse_ok3:
        all_tests = _task_verifier_tests(task)
        fb_pass, _ = _run_code_task(fn_name, code3, all_tests)
        trace.feedback_success = fb_pass
    else:
        trace.feedback_success = False

    if trace.single_success:
        trace.boundary_label = "above"
    elif trace.feedback_success:
        trace.boundary_label = "near"
    else:
        trace.boundary_label = "below"

    return trace


def main():
    print("=" * 80)
    print("Phase H: Collect HeuristicLocalSolver Traces")
    print("=" * 80)

    solver_name = "HeuristicLocalSolver"
    solver = HeuristicLocalSolver()

    num_seeds = 5
    num_tasks_per_seed = 50
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

        for task_idx, task in enumerate(tasks):
            if (task_idx + 1) % 10 == 0:
                logger.info(f"  Seed {seed}: {task_idx+1}/{len(tasks)} tasks done")

            trace = collect_trace_for_task(task, solver, solver_name, seed)
            traces.append(trace)

    logger.info(f"Collected {len(traces)} traces for {solver_name}")

    results_dir = PROJECT_ROOT / "results" / "phase_h_heuristic_traces"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"traces_{solver_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_data = {
        "experiment": "phase_h_heuristic_solver_traces",
        "solver": solver_name,
        "num_seeds": num_seeds,
        "num_tasks_per_seed": num_tasks_per_seed,
        "n_traces": len(traces),
        "traces": [t.to_dict() for t in traces],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Traces saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Phase H (Heuristic Traces) Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
