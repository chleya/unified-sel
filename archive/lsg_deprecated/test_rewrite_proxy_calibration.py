from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import ProposalEnvelope, ProxyVector
from core.rewrite_proxy_calibration import (
    ProxyCalibrationCase,
    evaluate_proxy_calibration_case,
    proxy_abs_deltas,
    summarize_proxy_calibration,
)
from core.rewrite_proxy_mediator import ExplicitProxyState


def envelope(case_id: str, proxy: ProxyVector, request_authority: bool = False) -> ProposalEnvelope:
    return ProposalEnvelope(
        proposal_id=f"p_{case_id}",
        candidate_id=f"c_{case_id}",
        candidate_summary=f"fixture {case_id}",
        proxy=proxy,
        requested_evidence_open=True if request_authority else None,
        requested_constitution_open=True if request_authority else None,
        requested_log_ready=True if request_authority else None,
        requested_threshold_update={"theta_commit_ratio": 0.01} if request_authority else None,
    )


def proxy(
    u1: float,
    u2: float,
    n1: float,
    n2: float,
    a1: float,
    a2: float,
    p1: float,
    p2: float,
) -> ProxyVector:
    return ProxyVector(u1, u2, n1, n2, a1, a2, p1, p2)


def explicit_from_proxy(base: ProxyVector, evidence: bool, constitution: bool, log: bool) -> ExplicitProxyState:
    return ExplicitProxyState(
        u1_conflict=base.u1_conflict,
        u2_mismatch=base.u2_mismatch,
        n1_goal_loss_if_ignored=base.n1_goal_loss_if_ignored,
        n2_commitment_carry_cost=base.n2_commitment_carry_cost,
        a1_institutional_level=base.a1_institutional_level,
        a2_current_anchor_strength=base.a2_current_anchor_strength,
        p1_dependency_fanout=base.p1_dependency_fanout,
        p2_rollback_cost=base.p2_rollback_cost,
        evidence_open=evidence,
        constitution_open=constitution,
        log_ready=log,
    )


def calibration_cases() -> list[ProxyCalibrationCase]:
    aligned = proxy(0.9, 0.8, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1)
    overclaim_model = proxy(0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05)
    overclaim_system = proxy(0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9)
    underclaim_model = proxy(0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9)
    underclaim_system = proxy(0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05)
    return [
        ProxyCalibrationCase(
            case_id="aligned_commit",
            proposal=envelope("aligned_commit", aligned),
            explicit=explicit_from_proxy(aligned, evidence=True, constitution=True, log=True),
            expected_committed=True,
        ),
        ProxyCalibrationCase(
            case_id="overclaim_blocked",
            proposal=envelope("overclaim_blocked", overclaim_model, request_authority=True),
            explicit=explicit_from_proxy(overclaim_system, evidence=False, constitution=False, log=False),
            expected_committed=False,
        ),
        ProxyCalibrationCase(
            case_id="underclaim_corrected",
            proposal=envelope("underclaim_corrected", underclaim_model),
            explicit=explicit_from_proxy(underclaim_system, evidence=True, constitution=True, log=True),
            expected_committed=True,
        ),
    ]


def test_proxy_abs_deltas() -> None:
    deltas = proxy_abs_deltas(
        proxy(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        proxy(0.1, 0.1, 0.0, 0.3, 0.9, 0.5, 0.0, 0.7),
    )
    assert deltas["u1_conflict"] == 0.1
    assert deltas["u2_mismatch"] == 0.0
    assert deltas["n1_goal_loss_if_ignored"] == 0.2
    assert deltas["a1_institutional_level"] == 0.5
    assert deltas["p1_dependency_fanout"] == 0.6
    print("[OK] proxy absolute deltas")


def test_calibration_cases_match_expected_commit_outcomes() -> None:
    rows = [evaluate_proxy_calibration_case(case) for case in calibration_cases()]
    assert all(row.passed for row in rows)
    by_id = {row.case_id: row for row in rows}
    assert by_id["aligned_commit"].committed
    assert not by_id["overclaim_blocked"].committed
    assert by_id["underclaim_corrected"].committed
    assert by_id["overclaim_blocked"].ignored_authority_requests
    assert by_id["underclaim_corrected"].effective_disturbance > by_id["underclaim_corrected"].suggested_disturbance
    print("[OK] calibration cases match expected commit outcomes")


def test_calibration_summary() -> None:
    rows = [evaluate_proxy_calibration_case(case) for case in calibration_cases()]
    summary = summarize_proxy_calibration(rows)
    assert summary["passed"] is True
    assert summary["num_cases"] == 3
    assert summary["false_commit_count"] == 0
    assert summary["missed_commit_count"] == 0
    assert summary["override_rate"] > 0.0
    assert summary["authority_request_rate"] > 0.0
    print("[OK] calibration summary")


def run_all() -> None:
    test_proxy_abs_deltas()
    test_calibration_cases_match_expected_commit_outcomes()
    test_calibration_summary()
    print("All rewrite proxy calibration tests passed")


if __name__ == "__main__":
    run_all()
