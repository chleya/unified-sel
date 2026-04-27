"""
Runtime Trace Boundary Experiment - Phase D

目标：建立 runtime trace，为 deployable boundary scheduler 研究提供数据基础

Phase D 实验设计：
1. 修改采集协议，每个 task 记录丰富的运行时信号
2. 分离 Oracle 和 No-leak 两层分类器
3. 先做二分类（should_feedback），再升级三分类

每个 task 记录：
- 事后标注（不给 classifier）：task_id, bug_type, difficulty, boundary_label
- 第一轮运行时信号：parse_ok, syntax_ok, visible_pass, hidden_pass, error_type, message_len, has_expected_actual, distance, patch_size
- Blind retry 信号：changed_code, parse_ok, error_type
- Feedback retry 信号：changed_code, parse_ok, error_type, uses_error_signal, patch_size_delta
"""

import json
import logging
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

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
logger = logging.getLogger("runtime_trace")


@dataclass
class RuntimeTrace:
    """Runtime trace for a single task"""
    # 事后标注（不给 classifier）
    task_id: str = ""
    bug_type: str = ""
    difficulty: str = ""
    condition: str = ""
    boundary_label: str = ""  # above/near/below，事后由结果生成

    # 最终结果
    single_success: bool = False
    blind_success: bool = False
    feedback_success: bool = False

    # 第一轮运行时信号
    first_attempt_parse_ok: bool = False
    first_attempt_syntax_ok: bool = False
    first_visible_pass: bool = False
    first_hidden_pass: bool = False
    first_error_type: str = ""  # NameError/AssertionError/TypeError/SyntaxError/empty/pass
    first_error_message_len: int = 0
    has_expected_actual: bool = False
    expected_actual_distance: float = 0.0  # normalized distance
    first_patch_size: int = 0
    first_changed_from_buggy: bool = False

    # Blind retry 信号
    blind_changed_code: bool = False
    blind_parse_ok: bool = False
    blind_error_type: str = ""

    # Feedback retry 信号
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
    solver: SearchLocalSolver,
    error_feedback: str = "",
) -> tuple[str, bool, bool, str, str, int]:
    """
    运行单次尝试

    Returns:
        code, parse_ok, syntax_ok, error_type, error_message, patch_size
    """
    fn_name = task.metadata.get("function_name", "solve")
    visible_tests = _task_search_tests(task)

    # 获取代码
    if error_feedback and hasattr(solver, "revise"):
        first = solver.solve(task)
        attempt = solver.revise(task, first, error_feedback)
    else:
        attempt = solver.solve(task)

    code = attempt.answer if attempt else ""

    # 检查 parse 和 syntax
    parse_ok = bool(code and code.strip())
    syntax_ok = parse_ok  # SearchLocalSolver 保证语法正确

    # 运行测试
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
    # 简单启发式：看错误消息长度
    msg_len = len(error_message)
    if msg_len < 50:
        return 0.2
    elif msg_len < 100:
        return 0.4
    elif msg_len < 200:
        return 0.6
    else:
        return 0.8


def collect_runtime_trace(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    seed: int,
) -> RuntimeTrace:
    """为一个 task 收集完整的 runtime trace"""
    trace = RuntimeTrace()

    # 事后标注
    trace.task_id = task.metadata.get("task_id", f"task_{seed}")
    trace.bug_type = task.metadata.get("bug_type", "unknown")
    trace.difficulty = task.metadata.get("difficulty", "unknown")

    fn_name = task.metadata.get("function_name", "solve")

    # ============ 第一轮尝试 ============
    code1, parse_ok1, syntax_ok1, err_type1, err_msg1, patch_size1 = run_single_attempt(task, solver, "")
    trace.first_attempt_parse_ok = parse_ok1
    trace.first_attempt_syntax_ok = syntax_ok1
    trace.first_error_type = err_type1
    trace.first_error_message_len = len(err_msg1) if err_msg1 else 0
    trace.has_expected_actual = check_has_expected_actual(err_msg1)
    trace.expected_actual_distance = extract_expected_actual_distance(err_msg1)
    trace.first_patch_size = patch_size1

    # 检查 visible/hidden pass
    if parse_ok1 and syntax_ok1:
        visible_tests = _task_search_tests(task)
        all_tests = _task_verifier_tests(task)
        vis_pass, _ = _run_code_task(fn_name, code1, visible_tests)
        all_pass, _ = _run_code_task(fn_name, code1, all_tests)
        trace.first_visible_pass = vis_pass
        trace.first_hidden_pass = all_pass
    else:
        trace.first_visible_pass = False
        trace.first_hidden_pass = False

    # ============ Blind retry ============
    if not trace.first_hidden_pass:
        code2, parse_ok2, _, err_type2, err_msg2, _ = run_single_attempt(task, solver, "")
        trace.blind_changed_code = (code2 != code1) if code1 and code2 else False
        trace.blind_parse_ok = parse_ok2
        trace.blind_error_type = err_type2

        # 检查 blind 是否成功
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

    # ============ Feedback retry ============
    if not trace.blind_success:
        error_feedback = err_msg1 if err_msg1 else ""
        code3, parse_ok3, _, err_type3, err_msg3, patch_size3 = run_single_attempt(task, solver, error_feedback)
        trace.feedback_changed_code = (code3 != code1) if code1 and code3 else False
        trace.feedback_parse_ok = parse_ok3
        trace.feedback_error_type = err_type3
        trace.feedback_uses_error_signal = bool(error_feedback)
        trace.feedback_patch_size_delta = patch_size3 - patch_size1 if (patch_size3 and patch_size1) else 0

        # 检查 feedback 是否成功
        if parse_ok3:
            all_tests = _task_verifier_tests(task)
            fb_pass, _ = _run_code_task(fn_name, code3, all_tests)
            trace.feedback_success = fb_pass
        else:
            trace.feedback_success = False
    else:
        trace.feedback_changed_code = False
        trace.feedback_parse_ok = True
        trace.feedback_error_type = "pass"
        trace.feedback_uses_error_signal = False
        trace.feedback_patch_size_delta = 0
        trace.feedback_success = True

    # ============ Single shot success ============
    trace.single_success = trace.first_hidden_pass

    # ============ 事后生成 boundary_label ============
    if trace.single_success:
        trace.boundary_label = "above"
    elif trace.feedback_success:
        trace.boundary_label = "near"
    else:
        trace.boundary_label = "below"

    return trace


def run_experiment(
    num_tasks: int = 50,
    num_seeds: int = 5,
    difficulty: str = "mixed",
) -> List[RuntimeTrace]:
    """运行实验，收集 runtime traces"""
    all_traces: List[RuntimeTrace] = []

    seeds = [42, 123, 456, 789, 1024][:num_seeds]

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"Running seed {seed} ({seed_idx+1}/{len(seeds)})...")

        tasks = generate_code_tasks(num_tasks=num_tasks, seed=seed)
        solver = SearchLocalSolver()

        for task_idx, task in enumerate(tasks):
            trace = collect_runtime_trace(task, solver, seed * 1000 + task_idx)

            # 条件标签
            trace.condition = f"seed_{seed}"

            all_traces.append(trace)

            if (task_idx + 1) % 10 == 0:
                logger.info(f"  Seed {seed}: {task_idx+1}/{len(tasks)} tasks done")

    return all_traces


def analyze_traces(traces: List[RuntimeTrace]) -> Dict[str, Any]:
    """分析 runtime traces"""
    results = {
        "total_tasks": len(traces),
        "by_boundary": {},
        "oracle_predictor": {},
        "runtime_signals": {},
    }

    # 按 boundary label 分组
    for label in ["above", "near", "below"]:
        label_traces = [t for t in traces if t.boundary_label == label]
        if not label_traces:
            continue

        n = len(label_traces)
        results["by_boundary"][label] = {
            "count": n,
            "percentage": n / len(traces),
            "feedback_success_rate": sum(1 for t in label_traces if t.feedback_success) / n,
            "blind_success_rate": sum(1 for t in label_traces if t.blind_success) / n,
        }

    # 分析 runtime signals 的预测能力
    signals = [
        ("first_attempt_parse_ok", lambda t: t.first_attempt_parse_ok),
        ("first_visible_pass", lambda t: t.first_visible_pass),
        ("first_error_type", lambda t: t.first_error_type),
        ("has_expected_actual", lambda t: t.has_expected_actual),
        ("first_error_message_len", lambda t: t.first_error_message_len),
        ("first_patch_size", lambda t: t.first_patch_size),
        ("blind_changed_code", lambda t: t.blind_changed_code),
    ]

    for signal_name, signal_fn in signals:
        if signal_name == "first_error_type":
            # 分类信号
            error_types = {}
            for t in traces:
                et = signal_fn(t)
                if et not in error_types:
                    error_types[et] = {"total": 0, "feedback_success": 0}
                error_types[et]["total"] += 1
                if t.feedback_success:
                    error_types[et]["feedback_success"] += 1

            results["runtime_signals"][signal_name] = {
                error: {
                    "total": stats["total"],
                    "feedback_success_rate": stats["feedback_success"] / stats["total"] if stats["total"] > 0 else 0,
                }
                for error, stats in error_types.items()
                if stats["total"] >= 3
            }
        else:
            # 布尔/数值信号
            true_stats = {"total": 0, "feedback_success": 0}
            false_stats = {"total": 0, "feedback_success": 0}

            for t in traces:
                val = signal_fn(t)
                stats = true_stats if val else false_stats
                stats["total"] += 1
                if t.feedback_success:
                    stats["feedback_success"] += 1

            results["runtime_signals"][signal_name] = {
                "true_rate": true_stats["feedback_success"] / true_stats["total"] if true_stats["total"] > 0 else 0,
                "false_rate": false_stats["feedback_success"] / false_stats["total"] if false_stats["total"] > 0 else 0,
                "true_total": true_stats["total"],
                "false_total": false_stats["total"],
            }

    return results


def main():
    print("=" * 80)
    print("Runtime Trace Boundary Experiment - Phase D")
    print("=" * 80)
    print("\n目标：建立 runtime trace，为 deployable boundary scheduler 研究提供数据基础")
    print("\nPhase D 实验设计：")
    print("  1. 修改采集协议，每个 task 记录丰富的运行时信号")
    print("  2. 分离 Oracle 和 No-leak 两层分类器")
    print("  3. 先做二分类（should_feedback），再升级三分类")
    print("=" * 80)

    # 运行实验
    num_tasks = 50
    num_seeds = 5

    print(f"\n运行实验：{num_seeds} seeds × {num_tasks} tasks = {num_seeds * num_tasks} traces")
    traces = run_experiment(num_tasks=num_tasks, num_seeds=num_seeds, difficulty="mixed")

    print(f"\n收集了 {len(traces)} 个 runtime traces")

    # 分析
    print("\n分析 runtime traces...")
    analysis = analyze_traces(traces)

    # 打印结果
    print("\n" + "=" * 80)
    print("1. Boundary 分布")
    print("=" * 80)

    for label, stats in analysis["by_boundary"].items():
        print(f"\n  {label.upper()} (n={stats['count']}, {stats['percentage']:.1%}):")
        print(f"    Blind success rate: {stats['blind_success_rate']:.1%}")
        print(f"    Feedback success rate: {stats['feedback_success_rate']:.1%}")

    print("\n" + "=" * 80)
    print("2. Runtime Signals 预测能力")
    print("=" * 80)

    for signal_name, signal_stats in analysis["runtime_signals"].items():
        print(f"\n  {signal_name}:")
        if isinstance(signal_stats, dict) and "true_rate" in signal_stats:
            print(f"    True 时 feedback 成功率: {signal_stats['true_rate']:.1%}")
            print(f"    False 时 feedback 成功率: {signal_stats['false_rate']:.1%}")
        else:
            for error, stats in sorted(signal_stats.items(), key=lambda x: -x[1]["feedback_success_rate"]):
                print(f"    {error}: {stats['feedback_success_rate']:.1%} (n={stats['total']})")

    # 保存结果
    output = {
        "experiment": "runtime_trace_boundary_experiment",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_tasks": num_tasks,
            "num_seeds": num_seeds,
            "difficulty": "mixed",
        },
        "traces": [t.to_dict() for t in traces],
        "analysis": analysis,
    }

    results_dir = PROJECT_ROOT / "results" / "runtime_trace_boundary_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n结果保存到: {output_path}")

    # 打印数据示例
    print("\n" + "=" * 80)
    print("3. 数据示例（第一个 trace）")
    print("=" * 80)
    example = traces[0]
    for key, value in example.to_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
