"""
边界感知调度实验 - Capability Boundary Awareness Experiment

核心问题：
- 系统如何识别自己处于 below-boundary / near-boundary / above-boundary？
- TopoMem/结构信号能不能预测某个 solver-task 是否处于 near-boundary？

实验设计：
对每个 task-solver pair 记录：
1. single-shot 是否成功
2. blind retry 是否提升
3. feedback retry 是否提升
4. verifier error 类型
5. solver confidence / disagreement / repair-change rate

然后训练一个 boundary classifier，验证：
- near-boundary 区间里 feedback_retry 收益显著高于 blind_retry

数据来源：
- 使用 SearchLocalSolver（规则系统，无需 LLM）
- 使用已有的 benchmark tasks（code-20, mixed-40 等）
"""

from __future__ import annotations

import os
import sys
import json
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
logger = logging.getLogger("boundary_awareness")


@dataclass
class AttemptRecord:
    attempt: int
    passed_visible: bool
    passed_all: bool
    feedback: str
    used_feedback: bool


@dataclass
class TaskSignals:
    task_id: str
    bug_type: str
    
    single_shot_success: bool
    single_shot_confidence: float
    
    blind_retry_success: bool
    blind_retry_n_attempts: int
    blind_retry_improvement: bool
    
    feedback_retry_success: bool
    feedback_retry_n_attempts: int
    feedback_retry_improvement: bool
    
    verifier_error_type: str
    disagreement_score: float
    repair_change_rate: float
    
    boundary_label: str  # below / near / above (ground truth)


def _run_single_attempt(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    error_feedback: str = "",
) -> tuple[str, bool, bool, str, float]:
    """运行单次尝试，返回 (code, passed_visible, passed_all, error_type, confidence)"""
    fn_name = task.metadata.get("function_name", "solve")
    visible_tests = _task_search_tests(task)
    
    if error_feedback and hasattr(solver, "revise"):
        first = solver.solve(task)
        attempt = solver.revise(task, first, error_feedback)
    else:
        attempt = solver.solve(task)
    
    code = attempt.answer
    if not code or not code.strip():
        return code, False, False, "empty_code", 0.0
    
    vis_pass, vis_msg = _run_code_task(fn_name, code, visible_tests)
    if not vis_pass:
        return code, False, False, f"visible_fail:{vis_msg}", 0.0
    
    all_tests = _task_verifier_tests(task)
    all_pass, all_msg = _run_code_task(fn_name, code, all_tests)
    
    # 计算置信度：passed test ratio
    n_visible = len(visible_tests)
    n_all = len(all_tests)
    n_passed = sum(1 for t in all_tests if _run_code_task(fn_name, code, [t])[0])
    confidence = n_passed / n_all if n_all > 0 else 0.0
    
    error_type = "pass" if all_pass else f"hidden_fail:{all_msg}"
    return code, vis_pass, all_pass, error_type, confidence


def _blind_retry_loop(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    max_attempts: int = 3,
) -> tuple[bool, int, bool]:
    """Blind retry：不提供错误反馈，只重试"""
    success = False
    n_attempts = 0
    improvement = False
    
    prev_passed = False
    for attempt in range(max_attempts):
        code, vis_pass, all_pass, error_type, conf = _run_single_attempt(task, solver, "")
        n_attempts += 1
        
        if all_pass:
            success = True
            if prev_passed:
                improvement = True
            break
        
        if not prev_passed and vis_pass:
            improvement = True
        
        prev_passed = vis_pass
    
    return success, n_attempts, improvement


def _feedback_retry_loop(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    max_attempts: int = 3,
) -> tuple[bool, int, bool]:
    """Feedback retry：提供错误反馈，重试"""
    success = False
    n_attempts = 0
    improvement = False
    
    prev_error = None
    for attempt in range(max_attempts):
        error_feedback = prev_error if prev_error else ""
        code, vis_pass, all_pass, error_type, conf = _run_single_attempt(task, solver, error_feedback)
        n_attempts += 1
        
        if all_pass:
            success = True
            if prev_error is not None:
                improvement = True
            break
        
        if not vis_pass:
            prev_error = error_type
    
    return success, n_attempts, improvement


def _compute_disagreement(task: BenchmarkTask, solver: SearchLocalSolver) -> float:
    """计算多个尝试之间的分歧度"""
    attempts = []
    for _ in range(5):
        code1, _, _, _, _ = _run_single_attempt(task, solver, "")
        attempts.append(code1)
    
    if len(attempts) < 2:
        return 0.0
    
    # 计算分歧：code 不同的比例
    disagreements = 0
    for i in range(len(attempts)):
        for j in range(i + 1, len(attempts)):
            if attempts[i] != attempts[j]:
                disagreements += 1
    
    n_pairs = len(attempts) * (len(attempts) - 1) / 2
    return disagreements / n_pairs if n_pairs > 0 else 0.0


def _compute_repair_change_rate(task: BenchmarkTask, solver: SearchLocalSolver) -> float:
    """计算修复尝试之间的代码变化率"""
    codes = []
    prev_code = None
    prev_error = None
    
    for _ in range(3):
        error_feedback = prev_error if prev_error else ""
        code, vis_pass, all_pass, error_type, conf = _run_single_attempt(task, solver, error_feedback)
        codes.append(code)
        
        if all_pass:
            break
        
        if not vis_pass:
            prev_error = error_type
    
    if len(codes) < 2:
        return 0.0
    
    # 计算代码变化率
    changes = 0
    for i in range(1, len(codes)):
        if codes[i] != codes[i-1]:
            changes += 1
    
    return changes / (len(codes) - 1)


def extract_task_signals(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
    boundary_label: str,
) -> TaskSignals:
    """为一个 task 提取所有信号"""
    task_id = task.metadata.get("task_id", "unknown")
    bug_type = task.metadata.get("bug_type", "unknown")
    
    # 1. Single shot
    _, _, single_pass, _, single_conf = _run_single_attempt(task, solver, "")
    single_shot_success = single_pass
    
    # 2. Blind retry
    blind_success, blind_n, blind_imp = _blind_retry_loop(task, solver)
    blind_retry_success = blind_success
    blind_retry_n_attempts = blind_n
    blind_retry_improvement = blind_imp
    
    # 3. Feedback retry
    fb_success, fb_n, fb_imp = _feedback_retry_loop(task, solver)
    feedback_retry_success = fb_success
    feedback_retry_n_attempts = fb_n
    feedback_retry_improvement = fb_imp
    
    # 4. Verifier error type
    _, _, _, error_type, _ = _run_single_attempt(task, solver, "")
    verifier_error_type = error_type
    
    # 5. Disagreement score
    disagreement = _compute_disagreement(task, solver)
    
    # 6. Repair change rate
    repair_change = _compute_repair_change_rate(task, solver)
    
    return TaskSignals(
        task_id=task_id,
        bug_type=bug_type,
        single_shot_success=single_shot_success,
        single_shot_confidence=single_conf,
        blind_retry_success=blind_retry_success,
        blind_retry_n_attempts=blind_n,
        blind_retry_improvement=blind_retry_improvement,
        feedback_retry_success=feedback_retry_success,
        feedback_retry_n_attempts=fb_n,
        feedback_retry_improvement=feedback_retry_improvement,
        verifier_error_type=verifier_error_type,
        disagreement_score=disagreement,
        repair_change_rate=repair_change,
        boundary_label=boundary_label,
    )


def infer_boundary_label(
    single_shot: bool,
    blind_improvement: bool,
    feedback_improvement: bool,
) -> str:
    """从实验结果推断边界标签（用于没有 ground truth 的情况）"""
    if single_shot:
        return "above"
    elif feedback_improvement and not blind_improvement:
        return "near"
    elif blind_improvement:
        return "near"
    else:
        return "below"


def analyze_results(signals: List[TaskSignals]) -> Dict[str, Any]:
    """分析实验结果"""
    results = {
        "total_tasks": len(signals),
        "by_boundary": {},
        "feedback_vs_blind": {},
        "signal_stats": {},
    }
    
    # 按边界标签分组
    for label in ["above", "near", "below"]:
        tasks = [s for s in signals if s.boundary_label == label]
        if not tasks:
            continue
        
        blind_success = sum(1 for t in tasks if t.blind_retry_success)
        fb_success = sum(1 for t in tasks if t.feedback_retry_success)
        
        results["by_boundary"][label] = {
            "n_tasks": len(tasks),
            "blind_retry_success_rate": blind_success / len(tasks),
            "feedback_retry_success_rate": fb_success / len(tasks),
            "feedback_gain": (fb_success - blind_success) / len(tasks),
        }
    
    # 计算信号统计
    disagreement_values = [s.disagreement_score for s in signals]
    repair_change_values = [s.repair_change_rate for s in signals]
    
    results["signal_stats"] = {
        "mean_disagreement": float(np.mean(disagreement_values)),
        "std_disagreement": float(np.std(disagreement_values)),
        "mean_repair_change": float(np.mean(repair_change_values)),
        "std_repair_change": float(np.std(repair_change_values)),
    }
    
    # Near boundary 分析
    near_tasks = [s for s in signals if s.boundary_label == "near"]
    if near_tasks:
        near_blind = sum(1 for t in near_tasks if t.blind_retry_success)
        near_fb = sum(1 for t in near_tasks if t.feedback_retry_success)
        results["near_boundary"] = {
            "n_tasks": len(near_tasks),
            "blind_success_rate": near_blind / len(near_tasks),
            "feedback_success_rate": near_fb / len(near_tasks),
            "feedback_gain": (near_fb - near_blind) / len(near_tasks),
        }
    
    return results


def main():
    print("=" * 80)
    print("边界感知调度实验 - Capability Boundary Awareness Experiment")
    print("=" * 80)
    print("\n实验设计：")
    print("  1. 对每个 task-solver pair 记录多维度信号")
    print("  2. 分析 feedback_retry vs blind_retry 的效果差异")
    print("  3. 验证 near-boundary 区间里 feedback_retry 收益 > blind_retry")
    print("=" * 80)
    
    # 生成测试任务
    logger.info("生成测试任务...")
    tasks = generate_code_tasks(num_tasks=20, seed=42)
    logger.info(f"生成了 {len(tasks)} 个任务")
    
    # 创建 solver
    solver = SearchLocalSolver()
    
    # 收集信号
    all_signals: List[TaskSignals] = []
    
    for i, task in enumerate(tasks):
        task_id = task.metadata.get("task_id", f"task_{i}")
        logger.info(f"处理任务 {i+1}/{len(tasks)}: {task_id}")
        
        # 运行实验
        signals = extract_task_signals(task, solver, boundary_label="unknown")
        
        # 推断边界标签
        boundary_label = infer_boundary_label(
            signals.single_shot_success,
            signals.blind_retry_improvement,
            signals.feedback_retry_improvement,
        )
        signals.boundary_label = boundary_label
        
        logger.info(f"  Single shot: {signals.single_shot_success}, "
                    f"Blind: {signals.blind_retry_success}, "
                    f"Feedback: {signals.feedback_retry_success}, "
                    f"Boundary: {boundary_label}")
        
        all_signals.append(signals)
    
    # 分析结果
    logger.info("\n分析结果...")
    analysis = analyze_results(all_signals)
    
    print("\n" + "=" * 80)
    print("实验结果")
    print("=" * 80)
    
    print(f"\n总任务数: {analysis['total_tasks']}")
    
    print("\n按边界区域分组：")
    for label, stats in analysis["by_boundary"].items():
        print(f"\n  {label.upper()} (n={stats['n_tasks']}):")
        print(f"    Blind retry 成功率: {stats['blind_retry_success_rate']:.3f}")
        print(f"    Feedback retry 成功率: {stats['feedback_retry_success_rate']:.3f}")
        print(f"    Feedback 收益: {stats['feedback_gain']:+.3f}")
    
    if "near_boundary" in analysis:
        nb = analysis["near_boundary"]
        print(f"\n  NEAR BOUNDARY 详细分析 (n={nb['n_tasks']}):")
        print(f"    Blind retry 成功率: {nb['blind_success_rate']:.3f}")
        print(f"    Feedback retry 成功率: {nb['feedback_success_rate']:.3f}")
        print(f"    Feedback 收益: {nb['feedback_gain']:+.3f}")
        
        if nb['feedback_gain'] > nb['blind_success_rate'] - nb.get('feedback_success_rate', 0):
            print("\n  [验证成功] Feedback retry 在 near-boundary 区域确实比 blind retry 更有效！")
        else:
            print("\n  [需要更多数据] Feedback retry 在 near-boundary 区域的收益待验证")
    
    print("\n信号统计：")
    ss = analysis["signal_stats"]
    print(f"  平均 disagreement: {ss['mean_disagreement']:.3f} ± {ss['std_disagreement']:.3f}")
    print(f"  平均 repair change rate: {ss['mean_repair_change']:.3f} ± {ss['std_repair_change']:.3f}")
    
    # 保存结果
    output = {
        "experiment": "boundary_awareness",
        "timestamp": datetime.now().isoformat(),
        "n_tasks": len(all_signals),
        "signals": [
            {
                "task_id": s.task_id,
                "bug_type": s.bug_type,
                "single_shot_success": s.single_shot_success,
                "blind_retry_success": s.blind_retry_success,
                "feedback_retry_success": s.feedback_retry_success,
                "boundary_label": s.boundary_label,
                "disagreement_score": s.disagreement_score,
                "repair_change_rate": s.repair_change_rate,
            }
            for s in all_signals
        ],
        "analysis": analysis,
    }
    
    results_dir = PROJECT_ROOT / "results" / "boundary_awareness"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果保存到: {output_path}")
    
    return output


if __name__ == "__main__":
    main()
