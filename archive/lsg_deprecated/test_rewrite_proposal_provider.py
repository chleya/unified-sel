from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    RewriteDynamicsConfig,
    observation_from_proposal,
    proposal_audit_record,
    simulate_case,
)
from core.rewrite_proposal_provider import (
    MiniMaxProposalProvider,
    MockProposalProvider,
    ProposalRequest,
    proposal_from_model_json,
    proposal_from_provider,
)


def test_mock_provider_returns_proposal_envelope() -> None:
    provider = MockProposalProvider()
    request = ProposalRequest(
        request_id="r1",
        observation_summary="tool result contradicts current belief",
        current_order_summary="current belief is stable",
    )
    proposal = proposal_from_provider(provider, request)
    assert proposal.proposal_id == "proposal_r1"
    assert proposal.candidate_id == "candidate_r1"
    assert proposal.proposal_origin == "mock"
    assert proposal.requested_evidence_open is None
    print("[OK] mock provider returns proposal envelope")


def test_authority_requesting_provider_still_cannot_commit() -> None:
    provider = MockProposalProvider(request_authority=True)
    request = ProposalRequest(
        request_id="r2",
        observation_summary="model asks for direct formal commit",
        current_order_summary="protected state",
    )
    proposal = proposal_from_provider(provider, request)
    audit = proposal_audit_record(proposal)
    assert audit["authority_requests_ignored"] is True
    assert proposal.requested_threshold_update == {"theta_commit_ratio": 0.01}

    observation = observation_from_proposal(
        proposal,
        evidence_open=False,
        constitution_open=False,
        log_ready=False,
    )
    state, _ = simulate_case([[observation]], RewriteDynamicsConfig(alpha=1.0))
    assert not state.candidates[proposal.candidate_id].committed
    assert len(state.commit_log) == 0
    print("[OK] authority-requesting provider cannot commit")


def test_provider_proposal_commits_only_with_external_gates() -> None:
    provider = MockProposalProvider()
    request = ProposalRequest(
        request_id="r3",
        observation_summary="externally verified sustained drift",
        current_order_summary="old order is weak",
    )
    proposal = proposal_from_provider(provider, request)
    observation = observation_from_proposal(
        proposal,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
    )
    state, _ = simulate_case([[observation]], RewriteDynamicsConfig(alpha=1.0))
    assert state.candidates[proposal.candidate_id].committed
    assert len(state.commit_log) == 1
    print("[OK] provider proposal commits only with external gates")


def test_minimax_provider_is_skeleton_only() -> None:
    provider = MiniMaxProposalProvider(model="placeholder")
    request = ProposalRequest(
        request_id="r4",
        observation_summary="anything",
        current_order_summary="anything",
    )
    try:
        provider.propose(request)
        raise AssertionError("MiniMax skeleton must not make proposals yet")
    except NotImplementedError:
        pass
    print("[OK] MiniMax provider is skeleton only")


def valid_payload():
    return {
        "proposal_id": "json_1",
        "candidate_id": "json_candidate",
        "candidate_summary": "candidate from model JSON",
        "proxy": {
            "u1_conflict": 0.9,
            "u2_mismatch": 0.8,
            "n1_goal_loss_if_ignored": 0.7,
            "n2_commitment_carry_cost": 0.6,
            "a1_institutional_level": 0.2,
            "a2_current_anchor_strength": 0.2,
            "p1_dependency_fanout": 0.3,
            "p2_rollback_cost": 0.2,
        },
        "requested_evidence_open": True,
        "requested_constitution_open": True,
        "requested_log_ready": True,
        "requested_threshold_update": {"theta_commit_ratio": 0.01},
    }


def test_model_json_validates_to_proposal_envelope() -> None:
    proposal = proposal_from_model_json(valid_payload(), proposal_origin="minimax")
    assert proposal.proposal_id == "json_1"
    assert proposal.proposal_origin == "minimax"
    assert proposal.requested_evidence_open is True
    assert proposal.requested_threshold_update == {"theta_commit_ratio": 0.01}
    print("[OK] model JSON validates to proposal envelope")


def test_model_json_rejects_missing_proxy_field() -> None:
    payload = valid_payload()
    del payload["proxy"]["u1_conflict"]
    try:
        proposal_from_model_json(payload, proposal_origin="minimax")
        raise AssertionError("missing proxy field should fail")
    except ValueError as exc:
        assert "missing proxy fields" in str(exc)
    print("[OK] model JSON rejects missing proxy field")


def test_model_json_rejects_out_of_range_score() -> None:
    payload = valid_payload()
    payload["proxy"]["u1_conflict"] = 1.5
    try:
        proposal_from_model_json(payload, proposal_origin="minimax")
        raise AssertionError("out-of-range score should fail")
    except ValueError as exc:
        assert "must be in [0, 1]" in str(exc)
    print("[OK] model JSON rejects out-of-range score")


def test_model_json_rejects_unexpected_proxy_field() -> None:
    payload = valid_payload()
    payload["proxy"]["gate_override"] = 1.0
    try:
        proposal_from_model_json(payload, proposal_origin="minimax")
        raise AssertionError("unexpected proxy field should fail")
    except ValueError as exc:
        assert "unexpected proxy fields" in str(exc)
    print("[OK] model JSON rejects unexpected proxy field")


def test_model_json_rejects_bad_gate_type() -> None:
    payload = valid_payload()
    payload["requested_evidence_open"] = "yes"
    try:
        proposal_from_model_json(payload, proposal_origin="minimax")
        raise AssertionError("bad gate type should fail")
    except ValueError as exc:
        assert "requested_evidence_open" in str(exc)
    print("[OK] model JSON rejects bad gate type")


def test_model_json_rejects_proposal_candidate_id_collision() -> None:
    payload = valid_payload()
    payload["candidate_id"] = payload["proposal_id"]
    try:
        proposal_from_model_json(payload, proposal_origin="minimax")
        raise AssertionError("proposal_id must not equal candidate_id")
    except ValueError as exc:
        assert "proposal_id" in str(exc)
        assert "candidate_id" in str(exc)
    print("[OK] model JSON rejects proposal/candidate identity collision")


def run_all() -> None:
    test_mock_provider_returns_proposal_envelope()
    test_authority_requesting_provider_still_cannot_commit()
    test_provider_proposal_commits_only_with_external_gates()
    test_minimax_provider_is_skeleton_only()
    test_model_json_validates_to_proposal_envelope()
    test_model_json_rejects_missing_proxy_field()
    test_model_json_rejects_out_of_range_score()
    test_model_json_rejects_unexpected_proxy_field()
    test_model_json_rejects_bad_gate_type()
    test_model_json_rejects_proposal_candidate_id_collision()
    print("All rewrite proposal provider tests passed")


if __name__ == "__main__":
    run_all()
