"""Deterministic rewrite-qualification dynamics for LSG Phase 0.

This module intentionally has no LLM calls and no dependency on CEE.  It tests
the smallest useful abstraction: a candidate's disturbance pressure D against
the current order's self-stability S.  Durable acknowledgement is only allowed
from commit_review with evidence, constitutional, and log gates open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable
from uuid import uuid4


class RewritePhase(str, Enum):
    SUPPRESSED = "suppressed"
    BACKGROUND = "background"
    FOREGROUND = "foreground"
    VERIFY = "verify"
    COMMIT_REVIEW = "commit_review"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"


ACTIVE_PHASES = {
    RewritePhase.FOREGROUND,
    RewritePhase.VERIFY,
    RewritePhase.COMMIT_REVIEW,
}


@dataclass(frozen=True)
class RewriteDynamicsConfig:
    alpha: float = 0.6
    epsilon: float = 1e-6
    delta_max: float | None = None
    bandwidth_limit: int = 3
    theta_bg_enter: float = 0.20
    theta_fg_enter: float = 0.15
    theta_fg_exit: float = 0.05
    theta_verify_ratio: float = 1.25
    theta_commit_ratio: float = 2.00
    theta_protected_stability: float = 0.70

    def __post_init__(self) -> None:
        _require_unit_interval(self.alpha, "RewriteDynamicsConfig.alpha")
        _require_positive(self.epsilon, "RewriteDynamicsConfig.epsilon")
        if self.delta_max is not None:
            _require_positive(self.delta_max, "RewriteDynamicsConfig.delta_max")
        if not isinstance(self.bandwidth_limit, int) or isinstance(self.bandwidth_limit, bool):
            raise ValueError("RewriteDynamicsConfig.bandwidth_limit must be an integer")
        if self.bandwidth_limit < 1:
            raise ValueError("RewriteDynamicsConfig.bandwidth_limit must be >= 1")
        for field_name in (
            "theta_bg_enter",
            "theta_fg_enter",
            "theta_fg_exit",
            "theta_verify_ratio",
            "theta_commit_ratio",
            "theta_protected_stability",
        ):
            _require_nonnegative(getattr(self, field_name), f"RewriteDynamicsConfig.{field_name}")
        if self.theta_verify_ratio > self.theta_commit_ratio:
            raise ValueError("RewriteDynamicsConfig.theta_verify_ratio must be <= theta_commit_ratio")
        _require_unit_interval(
            self.theta_protected_stability,
            "RewriteDynamicsConfig.theta_protected_stability",
        )


@dataclass
class CandidateState:
    candidate_id: str
    disturbance: float = 0.0
    stability: float = 1.0
    disturbance_velocity: float = 0.0
    stability_velocity: float = 0.0
    phase: RewritePhase = RewritePhase.SUPPRESSED
    evidence_open: bool = False
    constitution_open: bool = False
    log_ready: bool = False
    committed: bool = False
    initialized: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        _require_nonempty_string(self.candidate_id, "CandidateState.candidate_id")
        _require_positive_integer(self.version, "CandidateState.version")

    @property
    def ratio(self) -> float:
        return self.disturbance / (self.stability + 1e-6)

    @property
    def margin(self) -> float:
        return self.disturbance - self.stability


@dataclass(frozen=True)
class CandidateObservation:
    candidate_id: str
    disturbance_observed: float
    stability_observed: float
    evidence_open: bool
    constitution_open: bool
    log_ready: bool

    def __post_init__(self) -> None:
        _require_nonempty_string(self.candidate_id, "CandidateObservation.candidate_id")
        _require_unit_interval(
            self.disturbance_observed,
            "CandidateObservation.disturbance_observed",
        )
        _require_unit_interval(
            self.stability_observed,
            "CandidateObservation.stability_observed",
        )
        _require_bool(self.evidence_open, "CandidateObservation.evidence_open")
        _require_bool(self.constitution_open, "CandidateObservation.constitution_open")
        _require_bool(self.log_ready, "CandidateObservation.log_ready")


@dataclass(frozen=True)
class ProxyVector:
    """Engineering proxy interface for LSG's U/N/A/P/R layer.

    Candidate evidence is deliberately not included here.  Candidate evidence
    belongs to the evidence gate, while current-state acknowledgement depth
    contributes to A.
    """

    u1_conflict: float
    u2_mismatch: float
    n1_goal_loss_if_ignored: float
    n2_commitment_carry_cost: float
    a1_institutional_level: float
    a2_current_anchor_strength: float
    p1_dependency_fanout: float
    p2_rollback_cost: float

    def __post_init__(self) -> None:
        for field_name in (
            "u1_conflict",
            "u2_mismatch",
            "n1_goal_loss_if_ignored",
            "n2_commitment_carry_cost",
            "a1_institutional_level",
            "a2_current_anchor_strength",
            "p1_dependency_fanout",
            "p2_rollback_cost",
        ):
            _require_unit_interval(getattr(self, field_name), f"ProxyVector.{field_name}")


@dataclass(frozen=True)
class GovernanceScalars:
    U: float
    N: float
    A: float
    P: float
    R_raw: float
    disturbance: float
    stability: float


@dataclass(frozen=True)
class ProposalEnvelope:
    """Untrusted model proposal for a rewrite candidate.

    This is the only shape a future LLM adapter may produce.  Gate fields and
    threshold requests are recorded for audit but ignored by the simulator.
    """

    proposal_id: str
    candidate_id: str
    candidate_summary: str
    proxy: ProxyVector
    requested_evidence_open: bool | None = None
    requested_constitution_open: bool | None = None
    requested_log_ready: bool | None = None
    requested_threshold_update: dict[str, float] | None = None
    proposal_origin: str = "model"

    def __post_init__(self) -> None:
        _require_nonempty_string(self.proposal_id, "ProposalEnvelope.proposal_id")
        _require_nonempty_string(self.candidate_id, "ProposalEnvelope.candidate_id")
        _require_nonempty_string(self.candidate_summary, "ProposalEnvelope.candidate_summary")
        _require_nonempty_string(self.proposal_origin, "ProposalEnvelope.proposal_origin")
        if self.proposal_id == self.candidate_id:
            raise ValueError("ProposalEnvelope.proposal_id must be distinct from candidate_id")
        if not isinstance(self.proxy, ProxyVector):
            raise ValueError("ProposalEnvelope.proxy must be a ProxyVector")
        for field_name in (
            "requested_evidence_open",
            "requested_constitution_open",
            "requested_log_ready",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_bool(value, f"ProposalEnvelope.{field_name}")
        if self.requested_threshold_update is not None:
            if not isinstance(self.requested_threshold_update, dict):
                raise ValueError("ProposalEnvelope.requested_threshold_update must be a dict")
            for key, value in self.requested_threshold_update.items():
                _require_nonempty_string(key, "ProposalEnvelope.requested_threshold_update key")
                _require_nonnegative(value, f"ProposalEnvelope.requested_threshold_update[{key}]")


@dataclass(frozen=True)
class CommitEvent:
    event_id: str
    step_index: int
    candidate_id: str
    from_phase: str
    to_phase: str
    disturbance: float
    stability: float
    ratio: float
    evidence_open: bool
    constitution_open: bool
    log_ready: bool
    commit_executed: bool
    candidate_version: int = 1

    def __post_init__(self) -> None:
        _require_nonempty_string(self.event_id, "CommitEvent.event_id")
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool):
            raise ValueError("CommitEvent.step_index must be an integer")
        if self.step_index < 0:
            raise ValueError("CommitEvent.step_index must be nonnegative")
        _require_nonempty_string(self.candidate_id, "CommitEvent.candidate_id")
        _require_positive_integer(self.candidate_version, "CommitEvent.candidate_version")
        _require_nonempty_string(self.from_phase, "CommitEvent.from_phase")
        _require_nonempty_string(self.to_phase, "CommitEvent.to_phase")
        _require_unit_interval(self.disturbance, "CommitEvent.disturbance")
        _require_unit_interval(self.stability, "CommitEvent.stability")
        _require_nonnegative(self.ratio, "CommitEvent.ratio")
        _require_bool(self.evidence_open, "CommitEvent.evidence_open")
        _require_bool(self.constitution_open, "CommitEvent.constitution_open")
        _require_bool(self.log_ready, "CommitEvent.log_ready")
        _require_bool(self.commit_executed, "CommitEvent.commit_executed")
        if self.commit_executed and self.to_phase == RewritePhase.ACKNOWLEDGED.value:
            if not (self.evidence_open and self.constitution_open and self.log_ready):
                raise ValueError("Executed acknowledgement CommitEvent requires all gates open")


@dataclass(frozen=True)
class RevisionProposalEvent:
    """Audit-only proposal to revise an already acknowledged candidate.

    Acknowledged candidates are absorbing for ordinary observations.  Later
    corrections must enter through this explicit audit channel rather than
    silently mutating the acknowledged candidate state.
    """

    event_id: str
    step_index: int
    candidate_id: str
    reason: str
    disturbance: float
    stability: float
    evidence_open: bool
    constitution_open: bool
    log_ready: bool
    approval_open: bool
    revision_executed: bool
    target_version: int = 1

    def __post_init__(self) -> None:
        _require_nonempty_string(self.event_id, "RevisionProposalEvent.event_id")
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool):
            raise ValueError("RevisionProposalEvent.step_index must be an integer")
        if self.step_index < 0:
            raise ValueError("RevisionProposalEvent.step_index must be nonnegative")
        _require_nonempty_string(self.candidate_id, "RevisionProposalEvent.candidate_id")
        _require_positive_integer(self.target_version, "RevisionProposalEvent.target_version")
        _require_nonempty_string(self.reason, "RevisionProposalEvent.reason")
        _require_unit_interval(self.disturbance, "RevisionProposalEvent.disturbance")
        _require_unit_interval(self.stability, "RevisionProposalEvent.stability")
        _require_bool(self.evidence_open, "RevisionProposalEvent.evidence_open")
        _require_bool(self.constitution_open, "RevisionProposalEvent.constitution_open")
        _require_bool(self.log_ready, "RevisionProposalEvent.log_ready")
        _require_bool(self.approval_open, "RevisionProposalEvent.approval_open")
        _require_bool(self.revision_executed, "RevisionProposalEvent.revision_executed")
        if self.revision_executed:
            if not (
                self.evidence_open
                and self.constitution_open
                and self.log_ready
                and self.approval_open
            ):
                raise ValueError("Executed revision requires evidence, constitution, log, and approval gates open")


@dataclass(frozen=True)
class RevisionExecutionEvent:
    """Schema for a future state-changing revision transition.

    Phase 24 only permits drafting this event.  It does not mutate
    `RewriteSystemState`, does not increment candidate versions, and is
    deliberately separate from rollback.
    """

    event_id: str
    step_index: int
    proposal_event_id: str
    candidate_id: str
    from_version: int
    to_version: int
    disturbance: float
    stability: float
    evidence_open: bool
    constitution_open: bool
    log_ready: bool
    approval_open: bool
    execution_executed: bool

    def __post_init__(self) -> None:
        _require_nonempty_string(self.event_id, "RevisionExecutionEvent.event_id")
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool):
            raise ValueError("RevisionExecutionEvent.step_index must be an integer")
        if self.step_index < 0:
            raise ValueError("RevisionExecutionEvent.step_index must be nonnegative")
        _require_nonempty_string(
            self.proposal_event_id,
            "RevisionExecutionEvent.proposal_event_id",
        )
        _require_nonempty_string(self.candidate_id, "RevisionExecutionEvent.candidate_id")
        _require_positive_integer(self.from_version, "RevisionExecutionEvent.from_version")
        _require_positive_integer(self.to_version, "RevisionExecutionEvent.to_version")
        if self.to_version != self.from_version + 1:
            raise ValueError("RevisionExecutionEvent.to_version must equal from_version + 1")
        _require_unit_interval(self.disturbance, "RevisionExecutionEvent.disturbance")
        _require_unit_interval(self.stability, "RevisionExecutionEvent.stability")
        _require_bool(self.evidence_open, "RevisionExecutionEvent.evidence_open")
        _require_bool(self.constitution_open, "RevisionExecutionEvent.constitution_open")
        _require_bool(self.log_ready, "RevisionExecutionEvent.log_ready")
        _require_bool(self.approval_open, "RevisionExecutionEvent.approval_open")
        _require_bool(self.execution_executed, "RevisionExecutionEvent.execution_executed")
        if not self.approval_open:
            raise ValueError("RevisionExecutionEvent requires approval_open")


@dataclass(frozen=True)
class RollbackEvent:
    """Rollback an acknowledged candidate to its previous version.

    Rollback is a separate event family from revision execution:
    - Revision execution: version increments (from_version + 1)
    - Rollback: version decrements (from_version - 1, minimum 1)

    Rollback restores the candidate to its previous disturbance/stability
    state, reversing the effect of the most recent revision execution.
    """

    event_id: str
    step_index: int
    proposal_event_id: str
    candidate_id: str
    from_version: int
    to_version: int
    disturbance: float
    stability: float
    evidence_open: bool
    constitution_open: bool
    log_ready: bool
    approval_open: bool
    rollback_executed: bool

    def __post_init__(self) -> None:
        _require_nonempty_string(self.event_id, "RollbackEvent.event_id")
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool):
            raise ValueError("RollbackEvent.step_index must be an integer")
        if self.step_index < 0:
            raise ValueError("RollbackEvent.step_index must be nonnegative")
        _require_nonempty_string(self.proposal_event_id, "RollbackEvent.proposal_event_id")
        _require_nonempty_string(self.candidate_id, "RollbackEvent.candidate_id")
        _require_positive_integer(self.from_version, "RollbackEvent.from_version")
        _require_positive_integer(self.to_version, "RollbackEvent.to_version")
        if self.to_version != self.from_version - 1:
            raise ValueError("RollbackEvent.to_version must equal from_version - 1")
        if self.from_version < 2:
            raise ValueError("RollbackEvent.from_version must be >= 2 (nothing to roll back from version 1)")
        _require_unit_interval(self.disturbance, "RollbackEvent.disturbance")
        _require_unit_interval(self.stability, "RollbackEvent.stability")
        _require_bool(self.evidence_open, "RollbackEvent.evidence_open")
        _require_bool(self.constitution_open, "RollbackEvent.constitution_open")
        _require_bool(self.log_ready, "RollbackEvent.log_ready")
        _require_bool(self.approval_open, "RollbackEvent.approval_open")
        _require_bool(self.rollback_executed, "RollbackEvent.rollback_executed")
        if not self.approval_open:
            raise ValueError("RollbackEvent requires approval_open")


@dataclass(frozen=True)
class CEEProjection:
    """CEE-style projection of an LSG acknowledgement.

    This is a schema-alignment adapter, not a runtime integration.  It mirrors
    CEE's CommitmentEvent -> ModelRevisionEvent boundary so Phase 3 can test
    whether LSG acknowledgement can be handed to a CEE-like commit layer.
    """

    commitment_event: dict[str, object]
    revision_event: dict[str, object]


@dataclass
class RewriteSystemState:
    candidates: dict[str, CandidateState] = field(default_factory=dict)
    active_candidate_ids: list[str] = field(default_factory=list)
    bandwidth_limit: int = 3
    commit_log: list[CommitEvent] = field(default_factory=list)
    revision_log: list[RevisionProposalEvent] = field(default_factory=list)
    revision_execution_log: list[RevisionExecutionEvent] = field(default_factory=list)
    rollback_log: list[RollbackEvent] = field(default_factory=list)
    step_index: int = 0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _require_bool(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_number(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _require_unit_interval(value: object, field_name: str) -> None:
    number = _require_number(value, field_name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")


def _require_nonnegative(value: object, field_name: str) -> None:
    number = _require_number(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be nonnegative")


def _require_positive(value: object, field_name: str) -> None:
    number = _require_number(value, field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be positive")


def _require_positive_integer(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")


def noisy_or(a: float, b: float) -> float:
    a = clamp01(a)
    b = clamp01(b)
    return 1.0 - (1.0 - a) * (1.0 - b)


def compute_governance_scalars(
    proxy: ProxyVector,
    *,
    epsilon: float = 1e-6,
) -> GovernanceScalars:
    """Map the eight engineering proxies to U/N/A/P/R and D/S values."""

    U = noisy_or(proxy.u1_conflict, proxy.u2_mismatch)
    N = noisy_or(proxy.n1_goal_loss_if_ignored, proxy.n2_commitment_carry_cost)
    A = clamp01(proxy.a1_institutional_level) * clamp01(
        proxy.a2_current_anchor_strength
    )
    P = noisy_or(proxy.p1_dependency_fanout, proxy.p2_rollback_cost)
    R_raw = (U * N) / (A * P + epsilon)

    # D/S is the root dynamic representation.  R is a boundary statistic.
    disturbance = clamp01(U * N)
    stability = clamp01(A * P)

    return GovernanceScalars(
        U=U,
        N=N,
        A=A,
        P=P,
        R_raw=R_raw,
        disturbance=disturbance,
        stability=stability,
    )


def observation_from_proxy(
    candidate_id: str,
    proxy: ProxyVector,
    *,
    evidence_open: bool,
    constitution_open: bool,
    log_ready: bool,
    epsilon: float = 1e-6,
) -> CandidateObservation:
    scalars = compute_governance_scalars(proxy, epsilon=epsilon)
    return CandidateObservation(
        candidate_id=candidate_id,
        disturbance_observed=scalars.disturbance,
        stability_observed=scalars.stability,
        evidence_open=evidence_open,
        constitution_open=constitution_open,
        log_ready=log_ready,
    )


def observation_from_proposal(
    proposal: ProposalEnvelope,
    *,
    evidence_open: bool,
    constitution_open: bool,
    log_ready: bool,
    epsilon: float = 1e-6,
) -> CandidateObservation:
    """Convert an untrusted proposal into a trusted observation.

    A model may propose proxy values, but gate values are supplied by the caller
    from explicit evidence/constitutional/log checks.  Requested gate overrides
    and threshold updates are intentionally ignored.
    """

    return observation_from_proxy(
        proposal.candidate_id,
        proposal.proxy,
        evidence_open=evidence_open,
        constitution_open=constitution_open,
        log_ready=log_ready,
        epsilon=epsilon,
    )


def proposal_audit_record(proposal: ProposalEnvelope) -> dict[str, object]:
    """Return an audit record showing ignored authority requests."""

    return {
        "proposal_id": proposal.proposal_id,
        "candidate_id": proposal.candidate_id,
        "proposal_origin": proposal.proposal_origin,
        "candidate_summary": proposal.candidate_summary,
        "requested_evidence_open": proposal.requested_evidence_open,
        "requested_constitution_open": proposal.requested_constitution_open,
        "requested_log_ready": proposal.requested_log_ready,
        "requested_threshold_update": proposal.requested_threshold_update or {},
        "authority_requests_ignored": True,
    }


def _smooth_value(
    previous: float,
    observed: float,
    *,
    alpha: float,
    delta_max: float | None,
) -> float:
    observed = clamp01(observed)
    raw = (1.0 - alpha) * previous + alpha * observed
    if delta_max is None:
        return clamp01(raw)
    lower = previous - delta_max
    upper = previous + delta_max
    return clamp01(max(lower, min(upper, raw)))


def _select_phase(
    candidate: CandidateState,
    config: RewriteDynamicsConfig,
) -> RewritePhase:
    d = candidate.disturbance
    s = candidate.stability
    margin = d - s
    ratio = d / (s + config.epsilon)

    if candidate.phase == RewritePhase.ACKNOWLEDGED:
        return RewritePhase.ACKNOWLEDGED
    if candidate.phase == RewritePhase.REJECTED:
        return RewritePhase.REJECTED

    if d < config.theta_bg_enter and s >= d:
        return RewritePhase.SUPPRESSED

    if candidate.phase in ACTIVE_PHASES and margin > config.theta_fg_exit:
        if ratio >= config.theta_commit_ratio and s < config.theta_protected_stability:
            return RewritePhase.COMMIT_REVIEW
        if ratio >= config.theta_verify_ratio:
            return RewritePhase.VERIFY
        return RewritePhase.FOREGROUND

    if margin < config.theta_fg_enter:
        return RewritePhase.BACKGROUND

    if ratio >= config.theta_commit_ratio and s < config.theta_protected_stability:
        return RewritePhase.COMMIT_REVIEW

    if ratio >= config.theta_verify_ratio:
        return RewritePhase.VERIFY

    return RewritePhase.FOREGROUND


def candidate_priority(candidate: CandidateState) -> tuple[float, float, float, str]:
    return (
        candidate.ratio,
        candidate.margin,
        candidate.disturbance,
        candidate.candidate_id,
    )


def update_candidate(
    candidate: CandidateState,
    observation: CandidateObservation,
    config: RewriteDynamicsConfig,
) -> CandidateState:
    if candidate.committed or candidate.phase == RewritePhase.ACKNOWLEDGED:
        return candidate

    prev_d = candidate.disturbance
    prev_s = candidate.stability

    if not candidate.initialized:
        next_d = clamp01(observation.disturbance_observed)
        next_s = clamp01(observation.stability_observed)
    else:
        next_d = _smooth_value(
            prev_d,
            observation.disturbance_observed,
            alpha=config.alpha,
            delta_max=config.delta_max,
        )
        next_s = _smooth_value(
            prev_s,
            observation.stability_observed,
            alpha=config.alpha,
            delta_max=config.delta_max,
        )

    updated = CandidateState(
        candidate_id=candidate.candidate_id,
        disturbance=next_d,
        stability=next_s,
        disturbance_velocity=next_d - prev_d,
        stability_velocity=next_s - prev_s,
        phase=candidate.phase,
        evidence_open=observation.evidence_open,
        constitution_open=observation.constitution_open,
        log_ready=observation.log_ready,
        committed=candidate.committed,
        initialized=True,
        version=candidate.version,
    )
    updated.phase = _select_phase(updated, config)
    return updated


def _apply_bandwidth_limit(
    state: RewriteSystemState,
    config: RewriteDynamicsConfig,
) -> None:
    active = [
        c for c in state.candidates.values()
        if c.phase in ACTIVE_PHASES and not c.committed
    ]
    active.sort(key=candidate_priority, reverse=True)
    retained = {c.candidate_id for c in active[:config.bandwidth_limit]}

    for candidate in active[config.bandwidth_limit:]:
        candidate.phase = RewritePhase.BACKGROUND

    state.active_candidate_ids = [
        c.candidate_id for c in active[:config.bandwidth_limit]
        if c.candidate_id in retained
    ]


def _acknowledge_ready_candidates(
    state: RewriteSystemState,
    config: RewriteDynamicsConfig,
) -> None:
    active = set(state.active_candidate_ids)
    for candidate in state.candidates.values():
        if candidate.candidate_id not in active:
            continue
        if candidate.phase != RewritePhase.COMMIT_REVIEW:
            continue
        if not (
            candidate.evidence_open
            and candidate.constitution_open
            and candidate.log_ready
        ):
            continue

        prior_phase = candidate.phase
        candidate.phase = RewritePhase.ACKNOWLEDGED
        candidate.committed = True
        state.commit_log.append(CommitEvent(
            event_id=f"ce_{uuid4().hex}",
            step_index=state.step_index,
            candidate_id=candidate.candidate_id,
            from_phase=prior_phase.value,
            to_phase=RewritePhase.ACKNOWLEDGED.value,
            disturbance=candidate.disturbance,
            stability=candidate.stability,
            ratio=candidate.disturbance / (candidate.stability + config.epsilon),
            evidence_open=candidate.evidence_open,
            constitution_open=candidate.constitution_open,
            log_ready=candidate.log_ready,
            commit_executed=True,
            candidate_version=candidate.version,
        ))

    state.active_candidate_ids = [
        cid for cid in state.active_candidate_ids
        if not state.candidates[cid].committed
    ]


def step_system(
    state: RewriteSystemState,
    observations: Iterable[CandidateObservation],
    config: RewriteDynamicsConfig | None = None,
) -> RewriteSystemState:
    cfg = config or RewriteDynamicsConfig(bandwidth_limit=state.bandwidth_limit)
    state.bandwidth_limit = cfg.bandwidth_limit
    state.step_index += 1

    for observation in observations:
        candidate = state.candidates.get(observation.candidate_id)
        if candidate is None:
            candidate = CandidateState(candidate_id=observation.candidate_id)
        state.candidates[observation.candidate_id] = update_candidate(
            candidate,
            observation,
            cfg,
        )

    _apply_bandwidth_limit(state, cfg)
    _acknowledge_ready_candidates(state, cfg)
    return state


def propose_revision_for_acknowledged_candidate(
    state: RewriteSystemState,
    *,
    candidate_id: str,
    reason: str,
    disturbance_observed: float,
    stability_observed: float,
    evidence_open: bool,
    constitution_open: bool,
    log_ready: bool,
    approval_open: bool = False,
    target_version: int | None = None,
) -> RevisionProposalEvent:
    """Record a revision proposal without mutating acknowledged state."""

    _require_nonempty_string(candidate_id, "candidate_id")
    _require_nonempty_string(reason, "reason")
    _require_unit_interval(disturbance_observed, "disturbance_observed")
    _require_unit_interval(stability_observed, "stability_observed")
    _require_bool(evidence_open, "evidence_open")
    _require_bool(constitution_open, "constitution_open")
    _require_bool(log_ready, "log_ready")
    _require_bool(approval_open, "approval_open")
    if target_version is not None:
        _require_positive_integer(target_version, "target_version")

    candidate = state.candidates.get(candidate_id)
    if candidate is None:
        raise ValueError("Revision proposal target candidate does not exist")
    if not candidate.committed or candidate.phase != RewritePhase.ACKNOWLEDGED:
        raise ValueError("Revision proposal target must already be acknowledged")
    if target_version is not None and target_version != candidate.version:
        raise ValueError("Revision proposal target_version must match current candidate version")

    event = RevisionProposalEvent(
        event_id=f"rpe_{uuid4().hex}",
        step_index=state.step_index,
        candidate_id=candidate_id,
        reason=reason,
        disturbance=disturbance_observed,
        stability=stability_observed,
        evidence_open=evidence_open,
        constitution_open=constitution_open,
        log_ready=log_ready,
        approval_open=approval_open,
        revision_executed=False,
        target_version=candidate.version,
    )
    state.revision_log.append(event)
    return event


def check_revision_log_invariants(state: RewriteSystemState) -> dict[str, object]:
    """Validate Phase 19 revision semantics.

    Phase 19 only permits audit-only revision proposals.  Execution requires a
    future, separate state-transition protocol.
    """

    issues: list[str] = []
    seen_event_ids: set[str] = set()
    for event in state.revision_log:
        if event.event_id in seen_event_ids:
            issues.append(f"duplicate_revision_event_id:{event.event_id}")
        seen_event_ids.add(event.event_id)

        candidate = state.candidates.get(event.candidate_id)
        if candidate is None:
            issues.append(f"missing_revision_target:{event.candidate_id}")
            continue
        if not candidate.committed or candidate.phase != RewritePhase.ACKNOWLEDGED:
            issues.append(f"revision_target_not_acknowledged:{event.candidate_id}")
        if event.target_version != candidate.version:
            issues.append(
                f"revision_target_version_mismatch:{event.candidate_id}:{event.target_version}!={candidate.version}"
            )
        if event.revision_executed:
            issues.append(f"executed_revision_not_supported:{event.event_id}")

    return {
        "passed": not issues,
        "issues": issues,
        "num_revision_events": len(state.revision_log),
        "approved_revision_count": sum(1 for event in state.revision_log if event.approval_open),
        "executed_revision_count": sum(1 for event in state.revision_log if event.revision_executed),
    }


def _find_revision_proposal(state: RewriteSystemState, proposal_event_id: str) -> RevisionProposalEvent:
    _require_nonempty_string(proposal_event_id, "proposal_event_id")
    matches = [
        event for event in state.revision_log
        if event.event_id == proposal_event_id
    ]
    if not matches:
        raise ValueError("Revision execution proposal_event_id was not found")
    if len(matches) > 1:
        raise ValueError("Revision execution proposal_event_id is duplicated")
    return matches[0]


def draft_revision_execution_event(
    state: RewriteSystemState,
    *,
    proposal_event_id: str,
) -> RevisionExecutionEvent:
    """Draft, but do not execute, a version-incrementing revision event."""

    proposal = _find_revision_proposal(state, proposal_event_id)
    if not proposal.approval_open:
        raise ValueError("Revision execution draft requires an approved revision proposal")
    if proposal.revision_executed:
        raise ValueError("Revision proposal is already marked executed")

    candidate = state.candidates.get(proposal.candidate_id)
    if candidate is None:
        raise ValueError("Revision execution target candidate does not exist")
    if not candidate.committed or candidate.phase != RewritePhase.ACKNOWLEDGED:
        raise ValueError("Revision execution target must already be acknowledged")
    if proposal.target_version != candidate.version:
        raise ValueError("Revision execution target_version must match current candidate version")

    return RevisionExecutionEvent(
        event_id=f"ree_{uuid4().hex}",
        step_index=state.step_index,
        proposal_event_id=proposal.event_id,
        candidate_id=proposal.candidate_id,
        from_version=proposal.target_version,
        to_version=proposal.target_version + 1,
        disturbance=proposal.disturbance,
        stability=proposal.stability,
        evidence_open=proposal.evidence_open,
        constitution_open=proposal.constitution_open,
        log_ready=proposal.log_ready,
        approval_open=proposal.approval_open,
        execution_executed=False,
    )


def draft_revision_execution_for_candidate(
    state: RewriteSystemState,
    *,
    candidate_id: str,
    target_version: int | None = None,
) -> RevisionExecutionEvent:
    """Draft an execution event from the latest approved proposal for a candidate."""

    _require_nonempty_string(candidate_id, "candidate_id")
    if target_version is not None:
        _require_positive_integer(target_version, "target_version")
    candidates = [
        event for event in state.revision_log
        if event.candidate_id == candidate_id
        and event.approval_open
        and not event.revision_executed
        and (target_version is None or event.target_version == target_version)
    ]
    if not candidates:
        raise ValueError("No approved revision proposal found for candidate")
    proposal = candidates[-1]
    return draft_revision_execution_event(
        state,
        proposal_event_id=proposal.event_id,
    )


def record_revision_execution_draft(
    state: RewriteSystemState,
    *,
    proposal_event_id: str | None = None,
    candidate_id: str | None = None,
    target_version: int | None = None,
) -> RevisionExecutionEvent:
    """Append a non-executed revision execution draft to the execution log."""

    if proposal_event_id is not None and candidate_id is not None:
        raise ValueError("Provide either proposal_event_id or candidate_id, not both")
    if proposal_event_id is None and candidate_id is None:
        raise ValueError("Revision execution draft requires proposal_event_id or candidate_id")
    if proposal_event_id is not None:
        event = draft_revision_execution_event(
            state,
            proposal_event_id=proposal_event_id,
        )
    else:
        assert candidate_id is not None
        event = draft_revision_execution_for_candidate(
            state,
            candidate_id=candidate_id,
            target_version=target_version,
        )
    state.revision_execution_log.append(event)
    return event


def check_revision_execution_event_against_state(
    state: RewriteSystemState,
    event: RevisionExecutionEvent,
) -> dict[str, object]:
    """Validate a drafted execution event without applying it.

    For already-executed events, checks that the post-execution state
    is consistent (version should be to_version, not from_version).
    """

    issues: list[str] = []

    if event.execution_executed:
        candidate = state.candidates.get(event.candidate_id)
        if candidate is None:
            issues.append(f"revision_execution_missing_candidate:{event.candidate_id}")
        else:
            if candidate.version != event.to_version:
                has_rollback = any(
                    rb.candidate_id == event.candidate_id
                    and rb.from_version == event.to_version
                    and rb.rollback_executed
                    for rb in state.rollback_log
                )
                if not has_rollback:
                    issues.append(
                        f"revision_execution_post_version_mismatch:{event.candidate_id}:{candidate.version}!={event.to_version}"
                    )
        return {
            "passed": not issues,
            "issues": issues,
            "proposal_event_id": event.proposal_event_id,
            "candidate_id": event.candidate_id,
            "from_version": event.from_version,
            "to_version": event.to_version,
            "execution_executed": event.execution_executed,
        }

    # Pre-execution validation
    try:
        proposal = _find_revision_proposal(state, event.proposal_event_id)
    except ValueError as exc:
        proposal = None
        issues.append(str(exc))

    candidate = state.candidates.get(event.candidate_id)
    if candidate is None:
        issues.append(f"revision_execution_missing_candidate:{event.candidate_id}")
    else:
        if not candidate.committed or candidate.phase != RewritePhase.ACKNOWLEDGED:
            issues.append(f"revision_execution_target_not_acknowledged:{event.candidate_id}")
        if event.from_version != candidate.version:
            issues.append(
                f"revision_execution_from_version_mismatch:{event.candidate_id}:{event.from_version}!={candidate.version}"
            )

    if proposal is not None:
        if proposal.candidate_id != event.candidate_id:
            issues.append("revision_execution_candidate_mismatch")
        if proposal.target_version != event.from_version:
            issues.append("revision_execution_proposal_version_mismatch")
        if not proposal.approval_open:
            issues.append("revision_execution_proposal_not_approved")
        if proposal.revision_executed:
            issues.append("revision_execution_proposal_already_executed")

    return {
        "passed": not issues,
        "issues": issues,
        "proposal_event_id": event.proposal_event_id,
        "candidate_id": event.candidate_id,
        "from_version": event.from_version,
        "to_version": event.to_version,
        "execution_executed": event.execution_executed,
    }


def execute_revision_event(
    state: RewriteSystemState,
    *,
    execution_event_id: str,
) -> RevisionExecutionEvent:
    """Execute an approved revision event, mutating candidate state.

    Phase 25 implements the controlled state transition that Phase 24
    deliberately left unimplemented.  Execution:

    1. Validates the execution event is ready (not already executed)
    2. Validates the target candidate is acknowledged and version matches
    3. Validates the source proposal is approved and not already executed
    4. Mutates candidate disturbance/stability/version
    5. Marks both proposal and execution event as executed
    6. Records the execution in the execution log

    This is the only function in LSG that mutates an acknowledged candidate.
    """

    # Find the execution event in the log
    matches = [e for e in state.revision_execution_log if e.event_id == execution_event_id]
    if not matches:
        raise ValueError(f"Revision execution event not found: {execution_event_id}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate revision execution event: {execution_event_id}")
    event = matches[0]

    if event.execution_executed:
        raise ValueError(f"Revision execution already executed: {execution_event_id}")

    # Validate against current state
    check = check_revision_execution_event_against_state(state, event)
    if not check["passed"]:
        raise ValueError(
            f"Revision execution validation failed: {', '.join(check['issues'])}"
        )

    # Find source proposal
    proposal = _find_revision_proposal(state, event.proposal_event_id)

    # Find target candidate
    candidate = state.candidates.get(event.candidate_id)
    if candidate is None:
        raise ValueError(f"Candidate not found: {event.candidate_id}")

    # Mutate candidate state (this is the only place acknowledged candidates are mutated)
    candidate.disturbance = event.disturbance
    candidate.stability = event.stability
    candidate.version = event.to_version

    # Mark proposal as executed (requires mutable field)
    object.__setattr__(proposal, "revision_executed", True)

    # Mark execution event as executed (requires mutable field)
    object.__setattr__(event, "execution_executed", True)

    return event


def check_revision_execution_log_invariants(state: RewriteSystemState) -> dict[str, object]:
    """Validate execution events, including executed ones."""

    issues: list[str] = []
    seen_event_ids: set[str] = set()
    for event in state.revision_execution_log:
        if event.event_id in seen_event_ids:
            issues.append(f"duplicate_revision_execution_event_id:{event.event_id}")
        seen_event_ids.add(event.event_id)
        event_check = check_revision_execution_event_against_state(state, event)
        if event_check["passed"] is not True:
            issues.extend(str(issue) for issue in event_check["issues"])

    return {
        "passed": not issues,
        "issues": issues,
        "num_revision_execution_events": len(state.revision_execution_log),
        "executed_revision_execution_count": sum(
            1 for event in state.revision_execution_log
            if event.execution_executed
        ),
    }


def _find_previous_state(
    state: RewriteSystemState,
    candidate_id: str,
    current_version: int,
) -> tuple[float, float] | None:
    """Find the disturbance/stability of a candidate at a previous version.

    Searches commit_log and revision_execution_log for the state that
    was recorded when the candidate was at (current_version - 1).
    """
    target_version = current_version - 1

    for event in reversed(state.revision_execution_log):
        if (
            event.candidate_id == candidate_id
            and event.to_version == target_version
            and event.execution_executed
        ):
            return (event.disturbance, event.stability)

    for event in reversed(state.commit_log):
        if event.candidate_id == candidate_id:
            if getattr(event, "candidate_version", 1) == target_version:
                return (event.disturbance, event.stability)
            if target_version == 1 and not hasattr(event, "candidate_version"):
                return (event.disturbance, event.stability)

    return None


def draft_rollback_event(
    state: RewriteSystemState,
    *,
    proposal_event_id: str,
) -> RollbackEvent:
    """Draft a rollback event from an approved revision proposal.

    Unlike revision execution (version + 1), rollback decrements version.
    The disturbance/stability values are restored from the previous state.
    """
    proposal = _find_revision_proposal(state, proposal_event_id)

    if not proposal.approval_open:
        raise ValueError(f"Rollback requires approved proposal: {proposal_event_id}")

    candidate = state.candidates.get(proposal.candidate_id)
    if candidate is None:
        raise ValueError(f"Candidate not found: {proposal.candidate_id}")

    if candidate.version < 2:
        raise ValueError(f"Cannot roll back candidate at version 1: {proposal.candidate_id}")

    prev_state = _find_previous_state(state, candidate.candidate_id, candidate.version)
    if prev_state is None:
        raise ValueError(
            f"Cannot find previous state for candidate {candidate.candidate_id} at version {candidate.version - 1}"
        )

    prev_disturbance, prev_stability = prev_state

    event = RollbackEvent(
        event_id=f"rb_{proposal_event_id}_{state.step_index}",
        step_index=state.step_index,
        proposal_event_id=proposal_event_id,
        candidate_id=proposal.candidate_id,
        from_version=candidate.version,
        to_version=candidate.version - 1,
        disturbance=prev_disturbance,
        stability=prev_stability,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
        approval_open=True,
        rollback_executed=False,
    )

    state.rollback_log.append(event)
    return event


def check_rollback_event_against_state(
    state: RewriteSystemState,
    event: RollbackEvent,
) -> dict[str, object]:
    """Validate a rollback event against current state."""
    issues: list[str] = []

    if event.rollback_executed:
        candidate = state.candidates.get(event.candidate_id)
        if candidate and candidate.version != event.to_version:
            has_subsequent_rollback = any(
                rb.candidate_id == event.candidate_id
                and rb.from_version == event.to_version
                and rb.rollback_executed
                for rb in state.rollback_log
                if rb.event_id != event.event_id
            )
            if not has_subsequent_rollback:
                issues.append(
                    f"rollback_post_version_mismatch: candidate version {candidate.version} != event.to_version {event.to_version}"
                )
        return {"passed": not issues, "issues": issues}

    candidate = state.candidates.get(event.candidate_id)
    if candidate is None:
        issues.append(f"candidate_not_found: {event.candidate_id}")
        return {"passed": False, "issues": issues}

    if candidate.version != event.from_version:
        issues.append(
            f"rollback_version_mismatch: candidate version {candidate.version} != event.from_version {event.from_version}"
        )

    if candidate.phase != RewritePhase.ACKNOWLEDGED:
        issues.append(f"rollback_target_not_acknowledged: {candidate.phase}")

    proposal_matches = [
        p for p in state.revision_log if p.event_id == event.proposal_event_id
    ]
    if not proposal_matches:
        issues.append(f"rollback_proposal_not_found: {event.proposal_event_id}")
    else:
        proposal = proposal_matches[0]
        if not proposal.approval_open:
            issues.append(f"rollback_proposal_not_approved: {event.proposal_event_id}")

    return {"passed": not issues, "issues": issues}


def execute_rollback_event(
    state: RewriteSystemState,
    *,
    rollback_event_id: str,
) -> RollbackEvent:
    """Execute a rollback event, restoring candidate to previous version.

    Rollback is the second mechanism (after revision execution) that can
    mutate an acknowledged candidate. Key differences:
    - Version decrements instead of incrementing
    - Disturbance/stability are restored from previous state
    - Rollback from version 1 is impossible
    """
    matches = [e for e in state.rollback_log if e.event_id == rollback_event_id]
    if not matches:
        raise ValueError(f"Rollback event not found: {rollback_event_id}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate rollback event: {rollback_event_id}")
    event = matches[0]

    if event.rollback_executed:
        raise ValueError(f"Rollback already executed: {rollback_event_id}")

    check = check_rollback_event_against_state(state, event)
    if not check["passed"]:
        raise ValueError(
            f"Rollback validation failed: {', '.join(check['issues'])}"
        )

    candidate = state.candidates.get(event.candidate_id)
    if candidate is None:
        raise ValueError(f"Candidate not found: {event.candidate_id}")

    proposal = _find_revision_proposal(state, event.proposal_event_id)

    candidate.disturbance = event.disturbance
    candidate.stability = event.stability
    candidate.version = event.to_version

    object.__setattr__(proposal, "revision_executed", True)
    object.__setattr__(event, "rollback_executed", True)

    return event


def check_rollback_log_invariants(state: RewriteSystemState) -> dict[str, object]:
    """Validate rollback events in the log."""
    issues: list[str] = []
    seen_event_ids: set[str] = set()

    for event in state.rollback_log:
        if event.event_id in seen_event_ids:
            issues.append(f"duplicate_rollback_event_id:{event.event_id}")
        seen_event_ids.add(event.event_id)

        event_check = check_rollback_event_against_state(state, event)
        if event_check["passed"] is not True:
            issues.extend(str(issue) for issue in event_check["issues"])

    return {
        "passed": not issues,
        "issues": issues,
        "num_rollback_events": len(state.rollback_log),
        "executed_rollback_count": sum(
            1 for event in state.rollback_log if event.rollback_executed
        ),
    }


def simulate_case(
    observations_by_step: list[list[CandidateObservation]],
    config: RewriteDynamicsConfig | None = None,
) -> tuple[RewriteSystemState, list[dict[str, object]]]:
    cfg = config or RewriteDynamicsConfig()
    state = RewriteSystemState(bandwidth_limit=cfg.bandwidth_limit)
    timeline: list[dict[str, object]] = []

    for observations in observations_by_step:
        step_system(state, observations, cfg)
        timeline.append({
            "step_index": state.step_index,
            "active_candidate_ids": list(state.active_candidate_ids),
            "candidates": {
                cid: {
                    "disturbance": c.disturbance,
                    "stability": c.stability,
                    "ratio": c.disturbance / (c.stability + cfg.epsilon),
                    "margin": c.margin,
                    "phase": c.phase.value,
                    "committed": c.committed,
                    "version": c.version,
                }
                for cid, c in sorted(state.candidates.items())
            },
            "commit_events": len(state.commit_log),
            "revision_events": len(state.revision_log),
        })

    return state, timeline


def count_phase_flips(timeline: list[dict[str, object]], candidate_id: str) -> int:
    phases: list[str] = []
    for row in timeline:
        candidates = row["candidates"]
        if isinstance(candidates, dict) and candidate_id in candidates:
            phases.append(candidates[candidate_id]["phase"])  # type: ignore[index]
    return sum(1 for a, b in zip(phases, phases[1:]) if a != b)


def project_commit_event_to_cee(
    event: CommitEvent,
    *,
    source_state_id: str,
    resulting_state_id: str,
    target_ref_prefix: str = "lsg.acknowledged",
) -> CEEProjection:
    """Project an LSG CommitEvent into CEE-style event dictionaries.

    The projection intentionally returns plain dictionaries to avoid coupling
    unified-sel to the CEE runtime.  The field names follow the CEE
    CommitmentEvent and ModelRevisionEvent shape closely enough for schema
    alignment tests.
    """

    if not event.commit_executed:
        raise ValueError("Only executed LSG commit events can be projected")
    if not (event.evidence_open and event.constitution_open and event.log_ready):
        raise ValueError("Projected commit must have evidence, constitution, and log gates open")

    commitment_id = f"lsg-ce-{event.event_id}"
    revision_id = f"lsg-rev-{event.event_id}"
    target_ref = f"{target_ref_prefix}.{event.candidate_id}"
    anchor_summary = f"LSG acknowledged candidate {event.candidate_id}"

    commitment = {
        "schema_version": "cee.commitment.v1",
        "event_type": "commitment",
        "event_id": commitment_id,
        "source_state_id": source_state_id,
        "commitment_kind": "internal_commit",
        "intent_summary": f"acknowledge LSG candidate {event.candidate_id}",
        "expected_world_change": [],
        "expected_self_change": [target_ref],
        "affected_entity_ids": [event.candidate_id],
        "affected_relation_ids": [],
        "action_summary": "formal acknowledgement through LSG rewrite dynamics",
        "external_result_summary": "acknowledged",
        "observation_summaries": [
            f"D={event.disturbance:.6f}",
            f"S={event.stability:.6f}",
            f"ratio={event.ratio:.6f}",
        ],
        "success": True,
        "reversibility": "reversible",
        "requires_approval": False,
        "cost": 0.0,
        "risk_realized": 0.0,
    }

    revision = {
        "schema_version": "cee.revision.v1",
        "event_type": "revision",
        "revision_id": revision_id,
        "prior_state_id": source_state_id,
        "caused_by_event_id": commitment_id,
        "revision_kind": "confirmation",
        "deltas": [
            {
                "delta_id": f"delta-{event.candidate_id}",
                "target_kind": "anchor_add",
                "target_ref": target_ref,
                "before_summary": "candidate not formally acknowledged",
                "after_summary": anchor_summary,
                "justification": (
                    "LSG commit event passed evidence, constitutional, and log gates"
                ),
                "raw_value": {
                    "candidate_id": event.candidate_id,
                    "disturbance": event.disturbance,
                    "stability": event.stability,
                    "ratio": event.ratio,
                    "candidate_version": event.candidate_version,
                },
            }
        ],
        "discarded_hypothesis_ids": [],
        "strengthened_hypothesis_ids": [],
        "new_anchor_fact_summaries": [anchor_summary],
        "resulting_state_id": resulting_state_id,
        "revision_summary": f"LSG acknowledgement projection for {event.candidate_id}",
    }

    return CEEProjection(
        commitment_event=commitment,
        revision_event=revision,
    )
