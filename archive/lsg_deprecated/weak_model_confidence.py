"""
Alternative confidence estimation for weak LLMs (0.5B-3B).

Problem: llama.cpp API does not return logprobs, so we cannot use
traditional token-probability confidence. This module provides
heuristic-based confidence that correlates with actual correctness.

Design principles:
1. No reliance on logprobs or model internals
2. Use output structure, syntax, and semantic consistency
3. Calibrate on visible tests when available
4. Provide uncertainty estimate, not just point score
"""

from __future__ import annotations

import ast
import difflib
import re
from typing import Any, Dict, List, Tuple

from core.capability_benchmark import BenchmarkTask


def estimate_confidence_weak_model(
    code: str,
    task: BenchmarkTask,
    visible_test_results: List[bool] | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """Estimate confidence for weak model outputs.

    Returns:
        confidence: float in [0.05, 0.95]
        metadata: dict with component scores for debugging
    """
    if not code or len(code.strip()) == 0:
        return 0.05, {"error": "empty_output", "components": {}}

    components: Dict[str, float] = {}

    components["structure"] = _score_structure(code)
    components["syntax"] = _score_syntax(code)
    components["semantic"] = _score_semantic_consistency(code, task)
    components["test"] = _score_visible_tests(visible_test_results)

    if visible_test_results is not None:
        weights = {"structure": 0.15, "syntax": 0.15, "semantic": 0.20, "test": 0.50}
    else:
        weights = {"structure": 0.25, "syntax": 0.30, "semantic": 0.45, "test": 0.00}

    total = sum(components[k] * weights[k] for k in weights)

    calibrated = 0.05 + 0.9 * total

    confidence = float(min(max(calibrated, 0.05), 0.95))

    metadata = {
        "components": components,
        "weights": weights,
        "raw_score": total,
        "calibrated_score": calibrated,
        "visible_tests_available": visible_test_results is not None,
    }

    return confidence, metadata


def _score_structure(code: str) -> float:
    """Score code structure quality. Returns [0.0, 1.0]."""
    score = 0.0

    if "def " in code:
        score += 0.30

    if "return" in code:
        score += 0.30

    lines = code.strip().split("\n")
    n_lines = len(lines)
    if 3 <= n_lines <= 30:
        score += 0.20
    elif n_lines > 30:
        score += 0.10

    indent_levels = set()
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            indent_levels.add(indent)
    if len(indent_levels) > 1:
        score += 0.20

    return min(score, 1.0)


def _score_syntax(code: str) -> float:
    """Score syntax validity. Returns [0.0, 1.0]."""
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError as e:
        lines = code.strip().split("\n")
        error_line = getattr(e, "lineno", len(lines))
        ratio = error_line / max(len(lines), 1)
        return 0.5 * ratio
    except Exception:
        return 0.1


def _score_semantic_consistency(code: str, task: BenchmarkTask) -> float:
    """Score semantic consistency with task requirements. Returns [0.0, 1.0]."""
    score = 0.0

    expected_fn = task.metadata.get("function_name", "")
    if expected_fn:
        fn_match = re.search(r'def\s+(\w+)', code)
        if fn_match:
            if fn_match.group(1) == expected_fn:
                score += 0.35
            else:
                score += 0.05

    buggy_code = task.metadata.get("buggy_code", "")
    if buggy_code and code != buggy_code:
        similarity = difflib.SequenceMatcher(a=buggy_code, b=code).ratio()
        if similarity < 0.9:
            score += 0.30
        else:
            score += 0.10

    bug_type = task.metadata.get("bug_type", "")
    if bug_type:
        score += _check_bug_fix_pattern(code, bug_type)

    return min(score, 1.0)


def _check_bug_fix_pattern(code: str, bug_type: str) -> float:
    """Check if code appears to fix the specific bug type."""
    patterns = {
        "missing_return": lambda c: "return" in c,
        "wrong_comparison": lambda c: "zero" in c or c.count("return") >= 3,
        "reverse_words": lambda c: "reversed" in c or "[::-1]" in c,
        "running_max": lambda c: "> best" in c or ">=" in c,
        "normalize_spaces": lambda c: "split()" in c or "join" in c,
        "count_positive": lambda c: "> 0" in c or ">= 1" in c,
        "count_negative": lambda c: "< 0" in c or "<= -1" in c,
        "count_even": lambda c: "% 2 == 0" in c,
        "count_nonzero": lambda c: "!= 0" in c,
    }

    checker = patterns.get(bug_type)
    if checker and checker(code):
        return 0.35
    return 0.0


def _score_visible_tests(visible_test_results: List[bool] | None) -> float:
    """Score based on visible test results. Returns [0.0, 1.0]."""
    if visible_test_results is None or len(visible_test_results) == 0:
        return 0.0

    passed = sum(visible_test_results)
    total = len(visible_test_results)
    ratio = passed / total

    if ratio == 1.0:
        return 1.0
    elif ratio >= 0.5:
        return 0.6
    elif ratio > 0:
        return 0.3
    else:
        return 0.1


def estimate_uncertainty_band(confidence: float) -> str:
    """Map confidence to uncertainty band for routing decisions."""
    if confidence >= 0.75:
        return "LOW_UNCERTAINTY"
    elif confidence >= 0.50:
        return "MEDIUM_UNCERTAINTY"
    elif confidence >= 0.25:
        return "HIGH_UNCERTAINTY"
    else:
        return "VERY_HIGH_UNCERTAINTY"


def should_escalate_weak_model(
    confidence: float,
    uncertainty_band: str,
    task_difficulty: str = "medium",
) -> bool:
    """Decision helper: should weak model escalate to stronger model?"""
    if uncertainty_band == "VERY_HIGH_UNCERTAINTY":
        return True

    if uncertainty_band == "HIGH_UNCERTAINTY" and task_difficulty in ["hard", "expert"]:
        return True

    if uncertainty_band == "MEDIUM_UNCERTAINTY" and task_difficulty == "expert":
        return True

    return False
