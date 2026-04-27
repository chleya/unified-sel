"""Calibration utilities for LSG proxy mediation.

The calibration layer measures how far model-suggested proxy values differ
from system-owned explicit values.  It does not grant authority to the model;
it only produces diagnostics for future proxy-head training or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rewrite_dynamics import ProposalEnvelope, ProxyVector, RewriteDynamicsConfig, compute_governance_scalars, simulate_case
from .rewrite_proxy_mediator import ExplicitProxyState, mediate_proposal, observation_from_mediated_proposal


PROXY_FIELD_NAMES: tuple[str, ...] = (
    "u1_conflict",
    "u2_mismatch",
    "n1_goal_loss_if_ignored",
    "n2_commitment_carry_cost",
    "a1_institutional_level",
    "a2_current_anchor_strength",
    "p1_dependency_fanout",
    "p2_rollback_cost",
)


@dataclass(frozen=True)
class ProxyCalibrationCase:
    case_id: str
    proposal: ProposalEnvelope
    explicit: ExplicitProxyState
    expected_committed: bool


@dataclass(frozen=True)
class ProxyCalibrationRow:
    case_id: str
    committed: bool
    expected_committed: bool
    passed: bool
    mean_abs_delta: float
    max_abs_delta: float
    overridden_fields: tuple[str, ...]
    ignored_authority_requests: tuple[str, ...]
    suggested_disturbance: float
    suggested_stability: float
    effective_disturbance: float
    effective_stability: float


def proxy_abs_deltas(suggested: ProxyVector, effective: ProxyVector) -> dict[str, float]:
    return {
        field: abs(float(getattr(suggested, field)) - float(getattr(effective, field)))
        for field in PROXY_FIELD_NAMES
    }


def evaluate_proxy_calibration_case(
    case: ProxyCalibrationCase,
    config: RewriteDynamicsConfig | None = None,
) -> ProxyCalibrationRow:
    cfg = config or RewriteDynamicsConfig(alpha=1.0)
    mediated = mediate_proposal(case.proposal, case.explicit)
    observation = observation_from_mediated_proposal(mediated)
    state, _ = simulate_case([[observation]], cfg)
    committed = state.candidates[case.proposal.candidate_id].committed

    deltas = proxy_abs_deltas(mediated.suggested_proxy, mediated.effective_proxy)
    suggested_scalars = compute_governance_scalars(mediated.suggested_proxy)
    effective_scalars = compute_governance_scalars(mediated.effective_proxy)
    return ProxyCalibrationRow(
        case_id=case.case_id,
        committed=committed,
        expected_committed=case.expected_committed,
        passed=committed == case.expected_committed,
        mean_abs_delta=sum(deltas.values()) / len(deltas),
        max_abs_delta=max(deltas.values()),
        overridden_fields=mediated.overridden_fields,
        ignored_authority_requests=mediated.ignored_authority_requests,
        suggested_disturbance=suggested_scalars.disturbance,
        suggested_stability=suggested_scalars.stability,
        effective_disturbance=effective_scalars.disturbance,
        effective_stability=effective_scalars.stability,
    )


def summarize_proxy_calibration(rows: list[ProxyCalibrationRow]) -> dict[str, object]:
    if not rows:
        return {
            "num_cases": 0,
            "passed": False,
            "pass_rate": 0.0,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
            "override_rate": 0.0,
            "authority_request_rate": 0.0,
        }

    num_cases = len(rows)
    passed_count = sum(1 for row in rows if row.passed)
    override_count = sum(1 for row in rows if row.overridden_fields)
    authority_count = sum(1 for row in rows if row.ignored_authority_requests)
    return {
        "num_cases": num_cases,
        "passed": passed_count == num_cases,
        "pass_rate": passed_count / num_cases,
        "mean_abs_delta": sum(row.mean_abs_delta for row in rows) / num_cases,
        "max_abs_delta": max(row.max_abs_delta for row in rows),
        "override_rate": override_count / num_cases,
        "authority_request_rate": authority_count / num_cases,
        "false_commit_count": sum(1 for row in rows if row.committed and not row.expected_committed),
        "missed_commit_count": sum(1 for row in rows if not row.committed and row.expected_committed),
    }
