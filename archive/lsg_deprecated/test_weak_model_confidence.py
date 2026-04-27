"""Tests for weak_model_confidence module."""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import BenchmarkTask
from core.weak_model_confidence import (
    estimate_confidence_weak_model,
    estimate_uncertainty_band,
    should_escalate_weak_model,
)


def make_task(bug_type="missing_return", function_name="solve", buggy_code="def solve(): pass"):
    return BenchmarkTask(
        task_id="test_001",
        family="code",
        prompt="Fix the code",
        expected_answer="def solve(): return 1",
        metadata={
            "bug_type": bug_type,
            "function_name": function_name,
            "buggy_code": buggy_code,
        }
    )


def test_empty_code():
    task = make_task()
    conf, meta = estimate_confidence_weak_model("", task)
    assert conf == 0.05
    assert meta["error"] == "empty_output"


def test_perfect_code():
    task = make_task("missing_return", "solve")
    code = "def solve():\n    return 42"
    conf_no_test, meta_no = estimate_confidence_weak_model(code, task, None)
    assert conf_no_test > 0.3
    assert meta_no["components"]["syntax"] == 1.0
    assert meta_no["components"]["structure"] > 0.5
    conf_with_test, meta_yes = estimate_confidence_weak_model(code, task, [True, True])
    assert conf_with_test > 0.7
    assert conf_with_test > conf_no_test


def test_syntax_error():
    task = make_task()
    code = "def solve(:\n    return 42"
    conf, meta = estimate_confidence_weak_model(code, task)
    assert meta["components"]["syntax"] < 0.5
    perfect_code = "def solve():\n    return 42"
    conf_perfect, _ = estimate_confidence_weak_model(perfect_code, task)
    assert conf < conf_perfect


def test_visible_tests_boost():
    task = make_task()
    code = "def solve():\n    return 42"
    conf_no_test, _ = estimate_confidence_weak_model(code, task, None)
    conf_with_test, _ = estimate_confidence_weak_model(code, task, [True, True])
    assert conf_with_test > conf_no_test


def test_uncertainty_bands():
    assert estimate_uncertainty_band(0.85) == "LOW_UNCERTAINTY"
    assert estimate_uncertainty_band(0.60) == "MEDIUM_UNCERTAINTY"
    assert estimate_uncertainty_band(0.35) == "HIGH_UNCERTAINTY"
    assert estimate_uncertainty_band(0.10) == "VERY_HIGH_UNCERTAINTY"


def test_escalation_rules():
    assert should_escalate_weak_model(0.1, "VERY_HIGH_UNCERTAINTY") is True
    assert should_escalate_weak_model(0.3, "HIGH_UNCERTAINTY", "hard") is True
    assert should_escalate_weak_model(0.3, "HIGH_UNCERTAINTY", "easy") is False
    assert should_escalate_weak_model(0.6, "MEDIUM_UNCERTAINTY", "expert") is True
    assert should_escalate_weak_model(0.6, "MEDIUM_UNCERTAINTY", "medium") is False
    assert should_escalate_weak_model(0.8, "LOW_UNCERTAINTY") is False
