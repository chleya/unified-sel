from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    ProposalEnvelope,
    ProxyVector,
    RewriteDynamicsConfig,
    simulate_case,
)
from core.rewrite_proxy_mediator import (
    ExplicitProxyState,
    mediate_proposal,
    mediated_audit_record,
    observation_from_mediated_proposal,
)


def model_proxy() -> ProxyVector:
    return ProxyVector(
        u1_conflict=0.95,
        u2_mismatch=0.95,
        n1_goal_loss_if_ignored=0.95,
        n2_commitment_carry_cost=0.95,
        a1_institutional_level=0.05,
        a2_current_anchor_strength=0.05,
        p1_dependency_fanout=0.05,
        p2_rollback_cost=0.05,
    )


def proposal() -> ProposalEnvelope:
    return ProposalEnvelope(
        proposal_id="p1",
        candidate_id="c1",
        candidate_summary="model wants low stability and open gates",
        proxy=model_proxy(),
        requested_evidence_open=True,
        requested_constitution_open=True,
        requested_log_ready=True,
        requested_threshold_update={"theta_commit_ratio": 0.01},
    )


def test_system_owned_a1_p1_override_model_proxy() -> None:
    mediated = mediate_proposal(
        proposal(),
        ExplicitProxyState(
            a1_institutional_level=0.9,
            p1_dependency_fanout=0.9,
            evidence_open=False,
            constitution_open=False,
            log_ready=False,
        ),
    )
    assert mediated.effective_proxy.a1_institutional_level == 0.9
    assert mediated.effective_proxy.p1_dependency_fanout == 0.9
    assert "a1_institutional_level" in mediated.overridden_fields
    assert "p1_dependency_fanout" in mediated.overridden_fields
    print("[OK] system-owned a1/p1 override model proxy")


def test_authority_requests_are_ignored_by_mediator() -> None:
    mediated = mediate_proposal(
        proposal(),
        ExplicitProxyState(
            a1_institutional_level=0.9,
            p1_dependency_fanout=0.9,
            evidence_open=False,
            constitution_open=False,
            log_ready=False,
        ),
    )
    audit = mediated_audit_record(mediated)
    assert set(audit["ignored_authority_requests"]) == {
        "requested_evidence_open",
        "requested_constitution_open",
        "requested_log_ready",
        "requested_threshold_update",
    }
    assert audit["evidence_open"] is False
    assert audit["constitution_open"] is False
    assert audit["log_ready"] is False
    print("[OK] authority requests are ignored by mediator")


def test_explicit_proxy_state_rejects_invalid_direct_values() -> None:
    try:
        ExplicitProxyState(
            a1_institutional_level=True,
            p1_dependency_fanout=0.1,
            evidence_open=False,
            constitution_open=False,
            log_ready=False,
        )
    except ValueError as exc:
        assert "a1_institutional_level" in str(exc)
    else:
        raise AssertionError("boolean proxy value should fail")

    try:
        ExplicitProxyState(
            a1_institutional_level=0.1,
            p1_dependency_fanout=0.1,
            evidence_open="false",
            constitution_open=False,
            log_ready=False,
        )
    except ValueError as exc:
        assert "evidence_open" in str(exc)
    else:
        raise AssertionError("string gate value should fail")
    print("[OK] explicit proxy state rejects invalid direct values")


def test_mediated_high_pressure_cannot_commit_when_system_gates_closed() -> None:
    mediated = mediate_proposal(
        proposal(),
        ExplicitProxyState(
            a1_institutional_level=0.9,
            p1_dependency_fanout=0.9,
            evidence_open=False,
            constitution_open=False,
            log_ready=False,
        ),
    )
    observation = observation_from_mediated_proposal(mediated)
    state, _ = simulate_case([[observation]], RewriteDynamicsConfig(alpha=1.0))
    assert not state.candidates["c1"].committed
    assert len(state.commit_log) == 0
    print("[OK] mediated high pressure cannot commit when gates closed")


def test_mediated_proposal_can_commit_when_system_gates_open_and_stability_low() -> None:
    mediated = mediate_proposal(
        proposal(),
        ExplicitProxyState(
            a1_institutional_level=0.1,
            p1_dependency_fanout=0.1,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
        ),
    )
    observation = observation_from_mediated_proposal(mediated)
    state, _ = simulate_case([[observation]], RewriteDynamicsConfig(alpha=1.0))
    assert state.candidates["c1"].committed
    assert len(state.commit_log) == 1
    print("[OK] mediated proposal can commit when system gates open and stability low")


def run_all() -> None:
    test_system_owned_a1_p1_override_model_proxy()
    test_authority_requests_are_ignored_by_mediator()
    test_explicit_proxy_state_rejects_invalid_direct_values()
    test_mediated_high_pressure_cannot_commit_when_system_gates_closed()
    test_mediated_proposal_can_commit_when_system_gates_open_and_stability_low()
    print("All rewrite proxy mediator tests passed")


if __name__ == "__main__":
    run_all()
