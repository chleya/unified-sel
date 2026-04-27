from __future__ import annotations

import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.meta_controller.adapters import sel_lab_benchmark as adapter
from experiments.meta_controller.baselines import build_controllers
from experiments.meta_controller import run_experiment


def test_default_suites_validate_without_sel_lab_imports() -> None:
    adapter.validate_suites()
    source = inspect.getsource(adapter)
    assert "F:\\sel-lab" not in source
    assert "sel_lab" not in source.replace("sel_lab_benchmark", "")


def test_suite_expansion_is_deterministic() -> None:
    first = adapter.expand_suites()
    second = adapter.expand_suites()
    assert first == second
    assert len(first) == 19
    assert first[0].suite == "mainline_acceptance"
    assert first[0].profile == "v01"
    assert first[0].seed == 0
    assert first[0].cli_args()[:6] == ("--profile", "v01", "--mode", "train-eval", "--train-episodes", "240")


def test_controller_names_exist_in_local_registry() -> None:
    controllers = set(build_controllers(seed=0))
    for suite in adapter.default_suites():
        assert set(suite.controllers).issubset(controllers)


def test_masks_map_to_supported_names() -> None:
    suite = adapter.suite_by_name("b3_signal_masking")
    assert suite.masks == ("surprise", "memory", "conflict", "drift", "core_signals")
    adapter.validate_suite(suite)


def test_v03_long_horizon_drift_suite_is_declared() -> None:
    suite = adapter.suite_by_name("long_horizon_drift_v03")
    assert suite.profile == "v03"
    assert "planner_necessity_loose_controller" in suite.controllers
    assert "drift" in suite.masks
    adapter.validate_suite(suite)


def test_b7_cross_profile_suites_are_declared() -> None:
    v01 = adapter.suite_by_name("b7_cross_profile_v01")
    v02 = adapter.suite_by_name("b7_cross_profile_v02")
    assert v01.profile == "v01"
    assert v02.profile == "v02"
    assert "drift_aware_planner_necessity_controller" in v01.controllers
    assert "drift_aware_planner_necessity_controller" in v02.controllers
    adapter.validate_suite(v01)
    adapter.validate_suite(v02)


def test_acceptance_rules_evaluate_synthetic_results() -> None:
    suite = adapter.suite_by_name("mainline_acceptance")
    results = {
        "fixed_rule_controller": {
            "task_success": 1.0,
            "drift_under_horizon": 0.0,
            "planner_calls": 54.0,
            "memory_reads": 7.0,
        },
        "planner_necessity_loose_controller": {
            "task_success": 1.0,
            "drift_under_horizon": 0.0,
            "planner_calls": 49.9,
            "memory_reads": 7.2,
        },
    }
    checks = adapter.evaluate_acceptance(suite, results)
    assert checks
    assert all(checks.values())


def test_invalid_suite_is_rejected() -> None:
    suite = adapter.BenchmarkSuite(
        name="bad",
        profile="missing",
        seeds=(0,),
        train_episodes=1,
        eval_episodes=1,
        controllers=("fixed_rule_controller",),
    )
    try:
        adapter.validate_suite(suite)
    except ValueError as exc:
        assert "Unknown profile" in str(exc)
    else:
        raise AssertionError("invalid profile should fail validation")


def test_run_benchmark_suite_filters_results_and_checks_acceptance(monkeypatch) -> None:
    suite = adapter.BenchmarkSuite(
        name="tiny",
        profile="v01",
        seeds=(7,),
        train_episodes=1,
        eval_episodes=1,
        controllers=("fixed_rule_controller", "planner_necessity_loose_controller"),
        acceptance_rules=(
            adapter.AcceptanceRule(
                "planner_necessity_loose_controller",
                "planner_calls",
                "<",
                reference_controller="fixed_rule_controller",
            ),
        ),
    )

    def fake_run_train_eval_suite(**kwargs):
        assert kwargs["seed"] == 7
        assert kwargs["profile"] == "v01"
        return {
            "fixed_rule_controller": {"planner_calls": 54.0},
            "planner_necessity_loose_controller": {"planner_calls": 49.0},
            "unrelated_controller": {"planner_calls": 0.0},
        }

    monkeypatch.setattr(run_experiment, "run_train_eval_suite", fake_run_train_eval_suite)
    payload = run_experiment.run_benchmark_suite(suite)
    run_payload = payload["tiny:seed=7"]
    assert set(run_payload["results"]) == {"fixed_rule_controller", "planner_necessity_loose_controller"}
    assert all(run_payload["acceptance"].values())
