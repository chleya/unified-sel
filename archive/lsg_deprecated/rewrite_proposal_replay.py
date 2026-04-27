"""Replay local LSG proposal JSON fixtures through the trusted boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rewrite_proposal_provider import proposal_from_model_json
from .rewrite_proxy_calibration import (
    ProxyCalibrationCase,
    evaluate_proxy_calibration_case,
    proxy_abs_deltas,
)
from .rewrite_proxy_mediator import ExplicitProxyState, mediate_proposal


@dataclass(frozen=True)
class ProposalReplayRow:
    case_id: str
    status: str
    committed: bool | None
    expected_committed: bool | None
    error_class: str | None
    expected_error: str | None
    failure_class: str | None
    mean_abs_delta: float | None
    max_abs_delta: float | None
    overridden_fields: tuple[str, ...]
    ignored_authority_requests: tuple[str, ...]


def load_replay_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("proposal replay dataset must be a list")
    return payload


def explicit_state_from_json(payload: dict[str, Any]) -> ExplicitProxyState:
    if not isinstance(payload, dict):
        raise ValueError("explicit state must be an object")
    required = {
        "a1_institutional_level",
        "p1_dependency_fanout",
        "evidence_open",
        "constitution_open",
        "log_ready",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"explicit state missing fields: {missing}")
    return ExplicitProxyState(
        a1_institutional_level=_required_float(payload, "a1_institutional_level"),
        p1_dependency_fanout=_required_float(payload, "p1_dependency_fanout"),
        evidence_open=_required_bool(payload, "evidence_open"),
        constitution_open=_required_bool(payload, "constitution_open"),
        log_ready=_required_bool(payload, "log_ready"),
        u1_conflict=_optional_float(payload, "u1_conflict"),
        u2_mismatch=_optional_float(payload, "u2_mismatch"),
        n1_goal_loss_if_ignored=_optional_float(payload, "n1_goal_loss_if_ignored"),
        n2_commitment_carry_cost=_optional_float(payload, "n2_commitment_carry_cost"),
        a2_current_anchor_strength=_optional_float(payload, "a2_current_anchor_strength"),
        p2_rollback_cost=_optional_float(payload, "p2_rollback_cost"),
    )


def _required_bool(payload: dict[str, Any], field: str) -> bool:
    value = payload[field]
    if not isinstance(value, bool):
        raise ValueError(f"explicit field '{field}' must be boolean")
    return value


def _required_float(payload: dict[str, Any], field: str) -> float:
    value = payload[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"explicit field '{field}' must be numeric")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"explicit field '{field}' must be in [0, 1]")
    return score


def _optional_float(payload: dict[str, Any], field: str) -> float | None:
    if field not in payload or payload[field] is None:
        return None
    value = payload[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"explicit field '{field}' must be numeric when present")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"explicit field '{field}' must be in [0, 1]")
    return score


def classify_failure(
    *,
    schema_error: bool,
    authority_request: bool,
    proxy_disagreement: bool,
    committed: bool | None,
    expected_committed: bool | None,
) -> str | None:
    if schema_error:
        return "schema_error"
    if committed is True and expected_committed is False:
        return "false_commit"
    if committed is False and expected_committed is True:
        return "missed_commit"
    if authority_request:
        return "authority_request"
    if proxy_disagreement:
        return "proxy_disagreement"
    return None


def replay_proposal_case(raw_case: dict[str, Any], *, disagreement_threshold: float = 0.25) -> ProposalReplayRow:
    case_id = str(raw_case.get("case_id", ""))
    if not case_id:
        raise ValueError("replay case missing case_id")
    expected_error = raw_case.get("expected_error")
    expected_committed = raw_case.get("expected_committed")
    if expected_committed is not None and not isinstance(expected_committed, bool):
        raise ValueError(f"case {case_id}: expected_committed must be boolean when present")

    try:
        proposal = proposal_from_model_json(
            raw_case.get("model_json"),
            proposal_origin=str(raw_case.get("proposal_origin", "fixture")),
        )
    except ValueError as exc:
        status = "passed" if expected_error == "schema" else "failed"
        return ProposalReplayRow(
            case_id=case_id,
            status=status,
            committed=None,
            expected_committed=expected_committed,
            error_class="schema",
            expected_error=expected_error if isinstance(expected_error, str) else None,
            failure_class=classify_failure(
                schema_error=True,
                authority_request=False,
                proxy_disagreement=False,
                committed=None,
                expected_committed=expected_committed,
            ),
            mean_abs_delta=None,
            max_abs_delta=None,
            overridden_fields=(),
            ignored_authority_requests=(),
        )

    explicit = explicit_state_from_json(raw_case.get("explicit", {}))
    mediated = mediate_proposal(proposal, explicit)
    deltas = proxy_abs_deltas(mediated.suggested_proxy, mediated.effective_proxy)
    mean_abs_delta = sum(deltas.values()) / len(deltas)
    max_abs_delta = max(deltas.values())
    proxy_disagreement = max_abs_delta >= disagreement_threshold
    authority_request = bool(mediated.ignored_authority_requests)
    if expected_committed is None:
        raise ValueError(f"case {case_id}: valid proposal requires expected_committed")

    calibration = evaluate_proxy_calibration_case(
        ProxyCalibrationCase(
            case_id=case_id,
            proposal=proposal,
            explicit=explicit,
            expected_committed=expected_committed,
        )
    )
    failure_class = classify_failure(
        schema_error=False,
        authority_request=authority_request,
        proxy_disagreement=proxy_disagreement,
        committed=calibration.committed,
        expected_committed=expected_committed,
    )
    outcome_ok = calibration.committed == expected_committed
    status = "passed" if outcome_ok else "failed"
    return ProposalReplayRow(
        case_id=case_id,
        status=status,
        committed=calibration.committed,
        expected_committed=expected_committed,
        error_class=None,
        expected_error=None,
        failure_class=failure_class,
        mean_abs_delta=mean_abs_delta,
        max_abs_delta=max_abs_delta,
        overridden_fields=mediated.overridden_fields,
        ignored_authority_requests=mediated.ignored_authority_requests,
    )


def summarize_replay(rows: list[ProposalReplayRow]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.failure_class or "none"
        counts[key] = counts.get(key, 0) + 1
    failed_count = sum(1 for row in rows if row.status != "passed")
    return {
        "num_cases": len(rows),
        "passed": failed_count == 0,
        "failed_count": failed_count,
        "failure_class_counts": counts,
        "false_commit_count": sum(1 for row in rows if row.failure_class == "false_commit"),
        "missed_commit_count": sum(1 for row in rows if row.failure_class == "missed_commit"),
        "schema_error_count": sum(1 for row in rows if row.failure_class == "schema_error"),
        "expected_schema_interception_count": sum(
            1 for row in rows
            if row.failure_class == "schema_error" and row.expected_error == "schema"
        ),
        "unexpected_schema_failure_count": sum(
            1 for row in rows
            if row.failure_class == "schema_error" and row.expected_error != "schema"
        ),
        "authority_request_count": sum(1 for row in rows if row.failure_class == "authority_request"),
        "proxy_disagreement_count": sum(1 for row in rows if row.failure_class == "proxy_disagreement"),
    }


def replay_dataset(path: Path, *, disagreement_threshold: float = 0.25) -> dict[str, object]:
    rows = [
        replay_proposal_case(raw_case, disagreement_threshold=disagreement_threshold)
        for raw_case in load_replay_dataset(path)
    ]
    return {
        "dataset": str(path),
        "summary": summarize_replay(rows),
        "cases": [row.__dict__ for row in rows],
    }
