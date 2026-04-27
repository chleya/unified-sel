from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_proposal_replay import (
    classify_failure,
    explicit_state_from_json,
    replay_dataset,
    replay_proposal_case,
)


DATASET = PROJECT_ROOT / "data" / "lsg" / "proposal_replay_v0.json"


def test_explicit_state_from_json_requires_system_owned_fields() -> None:
    try:
        explicit_state_from_json({"a1_institutional_level": 0.1})
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("missing explicit fields should fail")
    print("[OK] explicit state requires system-owned fields")


def test_explicit_state_rejects_gate_string_bypass() -> None:
    payload = {
        "a1_institutional_level": 0.1,
        "p1_dependency_fanout": 0.1,
        "evidence_open": "false",
        "constitution_open": True,
        "log_ready": True,
    }
    try:
        explicit_state_from_json(payload)
    except ValueError as exc:
        assert "evidence_open" in str(exc)
        assert "boolean" in str(exc)
    else:
        raise AssertionError("string gate values should fail")
    print("[OK] explicit state rejects gate string bypass")


def test_explicit_state_rejects_bool_numeric_bypass() -> None:
    payload = {
        "a1_institutional_level": True,
        "p1_dependency_fanout": 0.1,
        "evidence_open": False,
        "constitution_open": True,
        "log_ready": True,
    }
    try:
        explicit_state_from_json(payload)
    except ValueError as exc:
        assert "a1_institutional_level" in str(exc)
        assert "numeric" in str(exc)
    else:
        raise AssertionError("boolean numeric fields should fail")
    print("[OK] explicit state rejects bool numeric bypass")


def test_classify_failure_priority() -> None:
    assert classify_failure(
        schema_error=True,
        authority_request=True,
        proxy_disagreement=True,
        committed=True,
        expected_committed=False,
    ) == "schema_error"
    assert classify_failure(
        schema_error=False,
        authority_request=True,
        proxy_disagreement=True,
        committed=True,
        expected_committed=False,
    ) == "false_commit"
    assert classify_failure(
        schema_error=False,
        authority_request=True,
        proxy_disagreement=True,
        committed=False,
        expected_committed=True,
    ) == "missed_commit"
    assert classify_failure(
        schema_error=False,
        authority_request=True,
        proxy_disagreement=True,
        committed=False,
        expected_committed=False,
    ) == "authority_request"
    print("[OK] failure classification priority")


def test_replay_dataset_summary() -> None:
    result = replay_dataset(DATASET)
    summary = result["summary"]
    assert summary["passed"] is True
    assert summary["num_cases"] == 4
    assert summary["failed_count"] == 0
    assert summary["false_commit_count"] == 0
    assert summary["missed_commit_count"] == 0
    assert summary["schema_error_count"] == 1
    assert summary["expected_schema_interception_count"] == 1
    assert summary["unexpected_schema_failure_count"] == 0
    assert summary["authority_request_count"] == 1
    assert summary["proxy_disagreement_count"] == 1
    print("[OK] replay dataset summary")


def test_schema_error_case_is_expected_intercept() -> None:
    raw_case = {
        "case_id": "bad",
        "proposal_origin": "fixture",
        "model_json": {"proposal_id": "p"},
        "explicit": {},
        "expected_error": "schema",
    }
    row = replay_proposal_case(raw_case)
    assert row.status == "passed"
    assert row.error_class == "schema"
    assert row.expected_error == "schema"
    assert row.failure_class == "schema_error"
    assert row.committed is None
    print("[OK] schema error case is expected intercept")


def run_all() -> None:
    test_explicit_state_from_json_requires_system_owned_fields()
    test_explicit_state_rejects_gate_string_bypass()
    test_explicit_state_rejects_bool_numeric_bypass()
    test_classify_failure_priority()
    test_replay_dataset_summary()
    test_schema_error_case_is_expected_intercept()
    print("All rewrite proposal replay tests passed")


if __name__ == "__main__":
    run_all()
