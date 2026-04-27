"""Proposal-provider boundary for LSG rewrite dynamics.

Providers may suggest candidate summaries and proxy values.  They may not open
evidence, constitutional, or log gates, and they may not change thresholds.
MiniMax support starts here as a contract skeleton; no network calls are made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .rewrite_dynamics import ProposalEnvelope, ProxyVector


PROXY_FIELDS = (
    "u1_conflict",
    "u2_mismatch",
    "n1_goal_loss_if_ignored",
    "n2_commitment_carry_cost",
    "a1_institutional_level",
    "a2_current_anchor_strength",
    "p1_dependency_fanout",
    "p2_rollback_cost",
)

IGNORED_AUTHORITY_FIELDS = (
    "requested_evidence_open",
    "requested_constitution_open",
    "requested_log_ready",
    "requested_threshold_update",
)


@dataclass(frozen=True)
class ProposalRequest:
    request_id: str
    observation_summary: str
    current_order_summary: str
    goal_summary: str = ""
    source: str = "system"


class ProposalProvider(Protocol):
    name: str

    def propose(self, request: ProposalRequest) -> ProposalEnvelope: ...


@dataclass(frozen=True)
class MockProposalProvider:
    """Deterministic provider for tests and Phase 0-3 experiments."""

    name: str = "mock"
    proxy: ProxyVector = ProxyVector(
        u1_conflict=0.9,
        u2_mismatch=0.8,
        n1_goal_loss_if_ignored=0.8,
        n2_commitment_carry_cost=0.7,
        a1_institutional_level=0.2,
        a2_current_anchor_strength=0.2,
        p1_dependency_fanout=0.2,
        p2_rollback_cost=0.2,
    )
    request_authority: bool = False

    def propose(self, request: ProposalRequest) -> ProposalEnvelope:
        return ProposalEnvelope(
            proposal_id=f"proposal_{request.request_id}",
            candidate_id=f"candidate_{request.request_id}",
            candidate_summary=(
                f"{request.observation_summary} | against: "
                f"{request.current_order_summary}"
            ),
            proxy=self.proxy,
            requested_evidence_open=True if self.request_authority else None,
            requested_constitution_open=True if self.request_authority else None,
            requested_log_ready=True if self.request_authority else None,
            requested_threshold_update=(
                {"theta_commit_ratio": 0.01} if self.request_authority else None
            ),
            proposal_origin=self.name,
        )


@dataclass(frozen=True)
class MiniMaxProposalProvider:
    """Skeleton for a future MiniMax-backed proposal provider.

    This class deliberately raises until a later phase defines prompt format,
    response schema validation, timeout policy, and key handling.  It exists so
    the project has a stable integration point without silently adding network
    behavior.
    """

    name: str = "minimax"
    model: str = "unset"

    def propose(self, request: ProposalRequest) -> ProposalEnvelope:
        raise NotImplementedError(
            "MiniMaxProposalProvider is a contract skeleton only. "
            "Use MockProposalProvider until network-backed proposal generation "
            "is explicitly added."
        )


def proposal_from_provider(
    provider: ProposalProvider,
    request: ProposalRequest,
) -> ProposalEnvelope:
    proposal = provider.propose(request)
    if not isinstance(proposal, ProposalEnvelope):
        raise TypeError("ProposalProvider must return ProposalEnvelope")
    return proposal


def _require_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field '{field}' must be a non-empty string")
    return value


def _require_score(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"proxy field '{field}' must be numeric")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"proxy field '{field}' must be in [0, 1]")
    return score


def proposal_from_model_json(
    payload: dict[str, Any],
    *,
    proposal_origin: str,
) -> ProposalEnvelope:
    """Validate untrusted model JSON into a ProposalEnvelope.

    Authority requests are copied into the envelope for audit only.  The
    simulator ignores them when creating observations.
    """

    if not isinstance(payload, dict):
        raise ValueError("model proposal payload must be an object")

    proposal_id = _require_str(payload, "proposal_id")
    candidate_id = _require_str(payload, "candidate_id")
    candidate_summary = _require_str(payload, "candidate_summary")

    proxy_payload = payload.get("proxy")
    if not isinstance(proxy_payload, dict):
        raise ValueError("field 'proxy' must be an object")

    unexpected = set(proxy_payload) - set(PROXY_FIELDS)
    if unexpected:
        raise ValueError(f"unexpected proxy fields: {sorted(unexpected)}")

    missing = [field for field in PROXY_FIELDS if field not in proxy_payload]
    if missing:
        raise ValueError(f"missing proxy fields: {missing}")

    proxy_values = {
        field: _require_score(proxy_payload, field)
        for field in PROXY_FIELDS
    }

    threshold_update = payload.get("requested_threshold_update")
    if threshold_update is not None and not isinstance(threshold_update, dict):
        raise ValueError("requested_threshold_update must be an object when present")

    for field in IGNORED_AUTHORITY_FIELDS[:3]:
        value = payload.get(field)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{field} must be boolean when present")

    return ProposalEnvelope(
        proposal_id=proposal_id,
        candidate_id=candidate_id,
        candidate_summary=candidate_summary,
        proxy=ProxyVector(**proxy_values),
        requested_evidence_open=payload.get("requested_evidence_open"),
        requested_constitution_open=payload.get("requested_constitution_open"),
        requested_log_ready=payload.get("requested_log_ready"),
        requested_threshold_update=threshold_update,
        proposal_origin=proposal_origin,
    )
