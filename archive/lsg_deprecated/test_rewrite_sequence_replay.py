from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_sequence_replay import (
    check_sequence_identity_invariants,
    replay_sequence_case,
    summarize_sequence_results,
)


DATASET = PROJECT_ROOT / "data" / "lsg" / "proposal_sequence_replay_v0.json"


def load_cases() -> list[dict[str, object]]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_sequence_replay_cases_pass() -> None:
    rows = [replay_sequence_case(case) for case in load_cases()]
    summary = summarize_sequence_results(rows)
    assert summary["passed"] is True
    assert summary["num_cases"] == 6
    assert summary["failed_count"] == 0
    assert summary["total_commit_log_count"] == 12
    assert summary["total_revision_log_count"] == 2
    assert summary["total_approved_revision_count"] == 1
    assert summary["total_executed_revision_count"] == 0
    assert summary["total_revision_execution_draft_count"] == 1
    assert summary["total_executed_revision_execution_count"] == 0
    assert summary["invariant_failed_count"] == 0
    assert summary["revision_invariant_failed_count"] == 0
    assert summary["revision_execution_invariant_failed_count"] == 0
    assert summary["identity_failed_count"] == 0
    print("[OK] sequence replay cases pass")


def test_late_gate_commits_once() -> None:
    row = replay_sequence_case(load_cases()[0])
    assert row["case_id"] == "late_gate_commit_once"
    assert row["committed"] is True
    assert row["commit_log_count"] == 1
    assert row["final_phase"] == "acknowledged"
    assert row["timeline"][-1]["commit_events"] == 1
    print("[OK] late gate commits once")


def test_acknowledged_absorbs_later_observation() -> None:
    row = replay_sequence_case(load_cases()[1])
    phases = [
        step["candidates"]["c_absorb"]["phase"]
        for step in row["timeline"]
    ]
    assert phases == ["acknowledged", "acknowledged"]
    first = row["timeline"][0]["candidates"]["c_absorb"]
    last = row["timeline"][-1]["candidates"]["c_absorb"]
    assert last["disturbance"] == first["disturbance"]
    assert last["stability"] == first["stability"]
    assert row["commit_log_count"] == 1
    print("[OK] acknowledged absorbs later observation")


def test_multi_candidate_bandwidth_commits_top3_only() -> None:
    row = replay_sequence_case(load_cases()[2])
    assert row["case_id"] == "multi_candidate_bandwidth_top3"
    assert row["commit_log_count"] == 3
    assert row["invariants"]["passed"] is True
    assert row["identity_invariants"]["passed"] is True
    candidates = row["timeline"][-1]["candidates"]
    assert candidates["c_bw_0"]["committed"] is True
    assert candidates["c_bw_1"]["committed"] is True
    assert candidates["c_bw_2"]["committed"] is True
    assert candidates["c_bw_3"]["committed"] is False
    assert candidates["c_bw_4"]["committed"] is False
    assert not set(row["timeline"][-1]["active_candidate_ids"]) & {"c_bw_0", "c_bw_1", "c_bw_2"}
    print("[OK] multi-candidate bandwidth commits top3 only")


def test_multi_step_refill_after_top3_commit() -> None:
    row = replay_sequence_case(load_cases()[3])
    assert row["case_id"] == "multi_step_refill_after_top3_commit"
    assert row["commit_log_count"] == 5
    assert row["invariants"]["passed"] is True
    assert row["identity_invariants"]["passed"] is True
    first_step = row["timeline"][0]["candidates"]
    assert first_step["c_refill_0"]["committed"] is True
    assert first_step["c_refill_1"]["committed"] is True
    assert first_step["c_refill_2"]["committed"] is True
    assert first_step["c_refill_3"]["committed"] is False
    assert first_step["c_refill_4"]["committed"] is False
    final_step = row["timeline"][-1]["candidates"]
    assert final_step["c_refill_3"]["committed"] is True
    assert final_step["c_refill_4"]["committed"] is True
    assert final_step["c_refill_3"]["phase"] == "acknowledged"
    assert final_step["c_refill_4"]["phase"] == "acknowledged"
    assert row["timeline"][0]["commit_events"] == 3
    assert row["timeline"][-1]["commit_events"] == 5
    print("[OK] multi-step refill after top3 commit")


def test_revision_request_after_acknowledgement_is_audit_only() -> None:
    row = replay_sequence_case(load_cases()[4])
    assert row["case_id"] == "revision_request_after_acknowledgement_audit_only"
    assert row["committed"] is True
    assert row["commit_log_count"] == 1
    assert row["revision_log_count"] == 1
    assert row["invariants"]["passed"] is True
    assert row["revision_invariants"]["passed"] is True
    assert row["identity_invariants"]["passed"] is True

    first_step = row["timeline"][0]["candidates"]["c_revision_anchor"]
    final_step = row["timeline"][-1]["candidates"]["c_revision_anchor"]
    assert first_step["phase"] == "acknowledged"
    assert final_step["phase"] == "acknowledged"
    assert final_step["disturbance"] == first_step["disturbance"]
    assert final_step["stability"] == first_step["stability"]
    assert final_step["version"] == first_step["version"] == 1
    assert row["timeline"][0]["revision_events"] == 1
    assert row["timeline"][-1]["revision_events"] == 1
    assert row["commit_log"][0]["candidate_version"] == 1
    assert row["revision_log"][0]["target_version"] == 1
    assert row["revision_log"][0]["revision_executed"] is False
    print("[OK] revision request after acknowledgement is audit-only")


def test_approved_revision_request_still_does_not_execute() -> None:
    row = replay_sequence_case(load_cases()[5])
    assert row["case_id"] == "approved_revision_request_remains_audit_only"
    assert row["committed"] is True
    assert row["commit_log_count"] == 1
    assert row["revision_log_count"] == 1
    assert row["invariants"]["passed"] is True
    assert row["revision_invariants"]["passed"] is True
    assert row["revision_execution_invariants"]["passed"] is True
    assert row["revision_invariants"]["approved_revision_count"] == 1
    assert row["revision_invariants"]["executed_revision_count"] == 0

    first_step = row["timeline"][0]["candidates"]["c_revision_approved"]
    final_step = row["timeline"][-1]["candidates"]["c_revision_approved"]
    assert final_step["phase"] == "acknowledged"
    assert final_step["version"] == first_step["version"] == 1
    assert final_step["disturbance"] == first_step["disturbance"]
    assert final_step["stability"] == first_step["stability"]
    assert row["revision_log"][0]["approval_open"] is True
    assert row["revision_log"][0]["revision_executed"] is False
    assert row["revision_log"][0]["target_version"] == 1
    assert len(row["revision_execution_log"]) == 1
    assert row["revision_execution_log"][0]["candidate_id"] == "c_revision_approved"
    assert row["revision_execution_log"][0]["from_version"] == 1
    assert row["revision_execution_log"][0]["to_version"] == 2
    assert row["revision_execution_log"][0]["execution_executed"] is False
    print("[OK] approved revision request still does not execute")


def sequence_observation(proposal_id: str, candidate_id: str) -> dict[str, object]:
    return {
        "proposal_origin": "identity_fixture",
        "model_json": {
            "proposal_id": proposal_id,
            "candidate_id": candidate_id,
            "candidate_summary": "identity boundary candidate",
            "proxy": {
                "u1_conflict": 0.9,
                "u2_mismatch": 0.8,
                "n1_goal_loss_if_ignored": 0.9,
                "n2_commitment_carry_cost": 0.8,
                "a1_institutional_level": 0.1,
                "a2_current_anchor_strength": 0.1,
                "p1_dependency_fanout": 0.1,
                "p2_rollback_cost": 0.1,
            },
        },
        "explicit": {
            "a1_institutional_level": 0.1,
            "a2_current_anchor_strength": 0.1,
            "p1_dependency_fanout": 0.1,
            "p2_rollback_cost": 0.1,
            "evidence_open": False,
            "constitution_open": True,
            "log_ready": True,
        },
    }


def test_sequence_identity_detects_duplicate_proposal_id() -> None:
    case = {
        "case_id": "duplicate_proposal_id",
        "candidate_id": "c_identity_a",
        "expected_committed": False,
        "expected_commit_count": 0,
        "steps": [
            {
                "observations": [
                    sequence_observation("p_duplicate", "c_identity_a"),
                    sequence_observation("p_duplicate", "c_identity_b"),
                ]
            }
        ],
    }
    identity = check_sequence_identity_invariants(case)
    assert identity["passed"] is False
    assert identity["duplicate_proposal_ids"] == ["p_duplicate"]
    assert identity["cross_candidate_reuse"] == ["p_duplicate"]
    row = replay_sequence_case(case)
    assert row["passed"] is False
    assert row["identity_invariants"]["passed"] is False
    assert row["commit_log_count"] == 0
    print("[OK] sequence identity detects duplicate proposal id")


def test_revision_request_rejects_unacknowledged_target() -> None:
    case = {
        "case_id": "revision_before_ack",
        "candidate_id": "c_revision_pending",
        "expected_committed": False,
        "expected_commit_count": 0,
        "steps": [
            {
                "observations": [
                    sequence_observation("p_revision_pending_1", "c_revision_pending"),
                ],
                "revision_requests": [
                    {
                        "candidate_id": "c_revision_pending",
                        "reason": "invalid revision before acknowledgement",
                        "disturbance_observed": 0.1,
                        "stability_observed": 0.9,
                        "evidence_open": True,
                        "constitution_open": True,
                        "log_ready": True,
                    }
                ],
            }
        ],
    }
    try:
        replay_sequence_case(case)
    except ValueError as exc:
        assert "already be acknowledged" in str(exc)
    else:
        raise AssertionError("revision request should reject unacknowledged target")
    print("[OK] revision request rejects unacknowledged target")


def test_revision_request_rejects_missing_target() -> None:
    case = {
        "case_id": "revision_missing_target",
        "candidate_id": "c_revision_observed",
        "expected_committed": False,
        "expected_commit_count": 0,
        "steps": [
            {
                "observations": [
                    sequence_observation("p_revision_observed_1", "c_revision_observed"),
                ],
                "revision_requests": [
                    {
                        "candidate_id": "c_missing_revision_target",
                        "reason": "invalid missing target",
                        "disturbance_observed": 0.1,
                        "stability_observed": 0.9,
                        "evidence_open": True,
                        "constitution_open": True,
                        "log_ready": True,
                    }
                ],
            }
        ],
    }
    try:
        replay_sequence_case(case)
    except ValueError as exc:
        assert "target candidate does not exist" in str(exc)
    else:
        raise AssertionError("revision request should reject missing target")
    print("[OK] revision request rejects missing target")


def test_revision_request_rejects_stale_target_version() -> None:
    case = {
        "case_id": "revision_stale_target_version",
        "candidate_id": "c_revision_versioned",
        "expected_committed": True,
        "expected_commit_count": 1,
        "steps": [
            {
                "observations": [
                    {
                        **sequence_observation("p_revision_versioned_1", "c_revision_versioned"),
                        "explicit": {
                            "a1_institutional_level": 0.1,
                            "a2_current_anchor_strength": 0.1,
                            "p1_dependency_fanout": 0.1,
                            "p2_rollback_cost": 0.1,
                            "evidence_open": True,
                            "constitution_open": True,
                            "log_ready": True,
                        },
                    },
                ],
                "revision_requests": [
                    {
                        "candidate_id": "c_revision_versioned",
                        "reason": "invalid stale target version",
                        "disturbance_observed": 0.1,
                        "stability_observed": 0.9,
                        "evidence_open": True,
                        "constitution_open": True,
                        "log_ready": True,
                        "target_version": 2,
                    }
                ],
            }
        ],
    }
    try:
        replay_sequence_case(case)
    except ValueError as exc:
        assert "target_version" in str(exc)
    else:
        raise AssertionError("revision request should reject stale target version")
    print("[OK] revision request rejects stale target version")


def run_all() -> None:
    test_sequence_replay_cases_pass()
    test_late_gate_commits_once()
    test_acknowledged_absorbs_later_observation()
    test_multi_candidate_bandwidth_commits_top3_only()
    test_multi_step_refill_after_top3_commit()
    test_revision_request_after_acknowledgement_is_audit_only()
    test_approved_revision_request_still_does_not_execute()
    test_sequence_identity_detects_duplicate_proposal_id()
    test_revision_request_rejects_unacknowledged_target()
    test_revision_request_rejects_missing_target()
    test_revision_request_rejects_stale_target_version()
    print("All rewrite sequence replay tests passed")


if __name__ == "__main__":
    run_all()
