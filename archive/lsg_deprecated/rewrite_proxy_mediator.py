"""Proxy mediation for LSG model proposals.

The mediator separates model-suggested proxy values from system-owned explicit
statistics.  It prevents a valid JSON proposal from becoming authority over
institutional level, fanout, evidence, constitutional, or log gates.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rewrite_dynamics import CandidateObservation, ProposalEnvelope, ProxyVector, observation_from_proxy


@dataclass(frozen=True)
class ExplicitProxyState:
    a1_institutional_level: float
    p1_dependency_fanout: float
    evidence_open: bool
    constitution_open: bool
    log_ready: bool
    u1_conflict: float | None = None
    u2_mismatch: float | None = None
    n1_goal_loss_if_ignored: float | None = None
    n2_commitment_carry_cost: float | None = None
    a2_current_anchor_strength: float | None = None
    p2_rollback_cost: float | None = None

    def __post_init__(self) -> None:
        for field in (
            "a1_institutional_level",
            "p1_dependency_fanout",
            "u1_conflict",
            "u2_mismatch",
            "n1_goal_loss_if_ignored",
            "n2_commitment_carry_cost",
            "a2_current_anchor_strength",
            "p2_rollback_cost",
        ):
            value = getattr(self, field)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"ExplicitProxyState.{field} must be numeric")
            if float(value) < 0.0 or float(value) > 1.0:
                raise ValueError(f"ExplicitProxyState.{field} must be in [0, 1]")

        for field in ("evidence_open", "constitution_open", "log_ready"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"ExplicitProxyState.{field} must be boolean")


@dataclass(frozen=True)
class MediatedProposal:
    proposal: ProposalEnvelope
    suggested_proxy: ProxyVector
    effective_proxy: ProxyVector
    evidence_open: bool
    constitution_open: bool
    log_ready: bool
    overridden_fields: tuple[str, ...]
    ignored_authority_requests: tuple[str, ...]


def _choose(system_value: float | None, model_value: float, field: str, overrides: list[str]) -> float:
    if system_value is None:
        return model_value
    if abs(system_value - model_value) > 1e-12:
        overrides.append(field)
    return system_value


def mediate_proposal(
    proposal: ProposalEnvelope,
    explicit: ExplicitProxyState,
) -> MediatedProposal:
    suggested = proposal.proxy
    overridden: list[str] = []

    effective = ProxyVector(
        u1_conflict=_choose(explicit.u1_conflict, suggested.u1_conflict, "u1_conflict", overridden),
        u2_mismatch=_choose(explicit.u2_mismatch, suggested.u2_mismatch, "u2_mismatch", overridden),
        n1_goal_loss_if_ignored=_choose(
            explicit.n1_goal_loss_if_ignored,
            suggested.n1_goal_loss_if_ignored,
            "n1_goal_loss_if_ignored",
            overridden,
        ),
        n2_commitment_carry_cost=_choose(
            explicit.n2_commitment_carry_cost,
            suggested.n2_commitment_carry_cost,
            "n2_commitment_carry_cost",
            overridden,
        ),
        a1_institutional_level=explicit.a1_institutional_level,
        a2_current_anchor_strength=_choose(
            explicit.a2_current_anchor_strength,
            suggested.a2_current_anchor_strength,
            "a2_current_anchor_strength",
            overridden,
        ),
        p1_dependency_fanout=explicit.p1_dependency_fanout,
        p2_rollback_cost=_choose(
            explicit.p2_rollback_cost,
            suggested.p2_rollback_cost,
            "p2_rollback_cost",
            overridden,
        ),
    )

    if abs(effective.a1_institutional_level - suggested.a1_institutional_level) > 1e-12:
        overridden.append("a1_institutional_level")
    if abs(effective.p1_dependency_fanout - suggested.p1_dependency_fanout) > 1e-12:
        overridden.append("p1_dependency_fanout")

    ignored_authority: list[str] = []
    if proposal.requested_evidence_open is not None:
        ignored_authority.append("requested_evidence_open")
    if proposal.requested_constitution_open is not None:
        ignored_authority.append("requested_constitution_open")
    if proposal.requested_log_ready is not None:
        ignored_authority.append("requested_log_ready")
    if proposal.requested_threshold_update:
        ignored_authority.append("requested_threshold_update")

    return MediatedProposal(
        proposal=proposal,
        suggested_proxy=suggested,
        effective_proxy=effective,
        evidence_open=explicit.evidence_open,
        constitution_open=explicit.constitution_open,
        log_ready=explicit.log_ready,
        overridden_fields=tuple(dict.fromkeys(overridden)),
        ignored_authority_requests=tuple(ignored_authority),
    )


def mediated_audit_record(mediated: MediatedProposal) -> dict[str, object]:
    return {
        "proposal_id": mediated.proposal.proposal_id,
        "candidate_id": mediated.proposal.candidate_id,
        "proposal_origin": mediated.proposal.proposal_origin,
        "overridden_fields": list(mediated.overridden_fields),
        "ignored_authority_requests": list(mediated.ignored_authority_requests),
        "evidence_open": mediated.evidence_open,
        "constitution_open": mediated.constitution_open,
        "log_ready": mediated.log_ready,
    }


def observation_from_mediated_proposal(mediated: MediatedProposal) -> CandidateObservation:
    return observation_from_proxy(
        mediated.proposal.candidate_id,
        mediated.effective_proxy,
        evidence_open=mediated.evidence_open,
        constitution_open=mediated.constitution_open,
        log_ready=mediated.log_ready,
    )
