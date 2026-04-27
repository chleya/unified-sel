"""Multi-step replay for LSG proposal observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rewrite_dynamics import (
    CandidateObservation,
    RewriteDynamicsConfig,
    RewriteSystemState,
    check_revision_execution_log_invariants,
    check_revision_log_invariants,
    propose_revision_for_acknowledged_candidate,
    record_revision_execution_draft,
    step_system,
)
from .rewrite_proposal_provider import proposal_from_model_json
from .rewrite_proposal_replay import explicit_state_from_json
from .rewrite_proxy_mediator import mediate_proposal, observation_from_mediated_proposal


@dataclass(frozen=True)
class SequenceReplayResult:
    case_id: str
    committed: bool
    expected_committed: bool
    passed: bool
    commit_log_count: int
    expected_commit_count: int | None
    final_phase: str | None
    expected_final_phase: str | None
    revision_log_count: int
    expected_revision_count: int | None


def observation_from_sequence_step(step: dict[str, Any]) -> CandidateObservation:
    proposal = proposal_from_model_json(
        step.get("model_json"),
        proposal_origin=str(step.get("proposal_origin", "sequence")),
    )
    explicit = explicit_state_from_json(step.get("explicit", {}))
    mediated = mediate_proposal(proposal, explicit)
    return observation_from_mediated_proposal(mediated)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _require_unit_interval(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return number


def _timeline_row(state: RewriteSystemState, config: RewriteDynamicsConfig) -> dict[str, object]:
    return {
        "step_index": state.step_index,
        "active_candidate_ids": list(state.active_candidate_ids),
        "candidates": {
            cid: {
                "disturbance": c.disturbance,
                "stability": c.stability,
                "ratio": c.disturbance / (c.stability + config.epsilon),
                "margin": c.margin,
                "phase": c.phase.value,
                "committed": c.committed,
                "version": c.version,
            }
            for cid, c in sorted(state.candidates.items())
        },
        "commit_events": len(state.commit_log),
        "revision_events": len(state.revision_log),
        "revision_execution_events": len(state.revision_execution_log),
    }


def apply_revision_request(state: RewriteSystemState, request: dict[str, Any], *, case_id: str, step_index: int) -> None:
    candidate_id = request.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError(f"sequence case {case_id}: step {step_index} revision candidate_id must be non-empty")
    reason = request.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"sequence case {case_id}: step {step_index} revision reason must be non-empty")
    disturbance = _require_unit_interval(
        request.get("disturbance_observed"),
        f"sequence case {case_id}: step {step_index} revision disturbance_observed",
    )
    stability = _require_unit_interval(
        request.get("stability_observed"),
        f"sequence case {case_id}: step {step_index} revision stability_observed",
    )
    evidence_open = _require_bool(
        request.get("evidence_open"),
        f"sequence case {case_id}: step {step_index} revision evidence_open",
    )
    constitution_open = _require_bool(
        request.get("constitution_open"),
        f"sequence case {case_id}: step {step_index} revision constitution_open",
    )
    log_ready = _require_bool(
        request.get("log_ready"),
        f"sequence case {case_id}: step {step_index} revision log_ready",
    )
    approval_open = request.get("approval_open", False)
    approval_open = _require_bool(
        approval_open,
        f"sequence case {case_id}: step {step_index} revision approval_open",
    )
    target_version = request.get("target_version")
    if target_version is not None:
        if not isinstance(target_version, int) or isinstance(target_version, bool):
            raise ValueError(f"sequence case {case_id}: step {step_index} revision target_version must be an integer")
        if target_version < 1:
            raise ValueError(f"sequence case {case_id}: step {step_index} revision target_version must be >= 1")

    propose_revision_for_acknowledged_candidate(
        state,
        candidate_id=candidate_id,
        reason=reason,
        disturbance_observed=disturbance,
        stability_observed=stability,
        evidence_open=evidence_open,
        constitution_open=constitution_open,
        log_ready=log_ready,
        approval_open=approval_open,
        target_version=target_version,
    )


def apply_revision_execution_draft(
    state: RewriteSystemState,
    request: dict[str, Any],
    *,
    case_id: str,
    step_index: int,
) -> None:
    proposal_event_id = request.get("proposal_event_id")
    candidate_id = request.get("candidate_id")
    if proposal_event_id is not None and not isinstance(proposal_event_id, str):
        raise ValueError(
            f"sequence case {case_id}: step {step_index} execution proposal_event_id must be a string"
        )
    if candidate_id is not None and not isinstance(candidate_id, str):
        raise ValueError(
            f"sequence case {case_id}: step {step_index} execution candidate_id must be a string"
        )
    target_version = request.get("target_version")
    if target_version is not None:
        if not isinstance(target_version, int) or isinstance(target_version, bool):
            raise ValueError(f"sequence case {case_id}: step {step_index} execution target_version must be an integer")
        if target_version < 1:
            raise ValueError(f"sequence case {case_id}: step {step_index} execution target_version must be >= 1")
    record_revision_execution_draft(
        state,
        proposal_event_id=proposal_event_id,
        candidate_id=candidate_id,
        target_version=target_version,
    )


def check_sequence_identity_invariants(raw_case: dict[str, Any]) -> dict[str, object]:
    """Check proposal-event identity separately from candidate-state identity.

    A proposal_id identifies one proposal event.  A candidate_id identifies the
    durable candidate state updated by observations.  They must not collapse,
    and proposal IDs must not be reused within a sequence.
    """

    errors: list[str] = []
    proposal_to_candidate: dict[str, str] = {}
    duplicate_proposal_ids: list[str] = []
    cross_candidate_reuse: list[str] = []
    proposal_candidate_collisions: list[str] = []
    steps_payload = raw_case.get("steps")
    if not isinstance(steps_payload, list):
        return {
            "passed": False,
            "errors": ["steps must be a list"],
            "proposal_count": 0,
            "duplicate_proposal_ids": [],
            "cross_candidate_reuse": [],
            "proposal_candidate_collisions": [],
        }

    proposal_count = 0
    for step_index, step in enumerate(steps_payload):
        if not isinstance(step, dict):
            errors.append(f"step {step_index} must be an object")
            continue
        observations_payload = step.get("observations")
        if observations_payload is None:
            observations_payload = [step]
        if not isinstance(observations_payload, list):
            errors.append(f"step {step_index} observations must be a list")
            continue
        for obs_index, observation in enumerate(observations_payload):
            if not isinstance(observation, dict):
                errors.append(f"step {step_index} observation {obs_index} must be an object")
                continue
            model_json = observation.get("model_json")
            if not isinstance(model_json, dict):
                errors.append(f"step {step_index} observation {obs_index} model_json must be an object")
                continue
            proposal_id = model_json.get("proposal_id")
            candidate_id = model_json.get("candidate_id")
            if not isinstance(proposal_id, str) or not proposal_id.strip():
                errors.append(f"step {step_index} observation {obs_index} proposal_id must be non-empty")
                continue
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                errors.append(f"step {step_index} observation {obs_index} candidate_id must be non-empty")
                continue
            proposal_count += 1
            if proposal_id == candidate_id:
                proposal_candidate_collisions.append(proposal_id)
            previous_candidate = proposal_to_candidate.get(proposal_id)
            if previous_candidate is not None:
                duplicate_proposal_ids.append(proposal_id)
                if previous_candidate != candidate_id:
                    cross_candidate_reuse.append(proposal_id)
            else:
                proposal_to_candidate[proposal_id] = candidate_id

    for proposal_id in sorted(set(proposal_candidate_collisions)):
        errors.append(f"proposal_id equals candidate_id: {proposal_id}")
    for proposal_id in sorted(set(duplicate_proposal_ids)):
        errors.append(f"duplicate proposal_id: {proposal_id}")
    for proposal_id in sorted(set(cross_candidate_reuse)):
        errors.append(f"proposal_id reused across candidates: {proposal_id}")

    return {
        "passed": not errors,
        "errors": errors,
        "proposal_count": proposal_count,
        "duplicate_proposal_ids": sorted(set(duplicate_proposal_ids)),
        "cross_candidate_reuse": sorted(set(cross_candidate_reuse)),
        "proposal_candidate_collisions": sorted(set(proposal_candidate_collisions)),
    }


def replay_sequence_case(
    raw_case: dict[str, Any],
    *,
    config: RewriteDynamicsConfig | None = None,
) -> dict[str, object]:
    case_id = raw_case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("sequence case missing case_id")
    steps_payload = raw_case.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError(f"sequence case {case_id}: steps must be a non-empty list")

    identity_invariants = check_sequence_identity_invariants(raw_case)
    cfg = config or RewriteDynamicsConfig()
    state = RewriteSystemState(bandwidth_limit=cfg.bandwidth_limit)
    timeline: list[dict[str, object]] = []
    for index, step in enumerate(steps_payload):
        if not isinstance(step, dict):
            raise ValueError(f"sequence case {case_id}: step {index} must be an object")
        observations_payload = step.get("observations")
        if observations_payload is None:
            observations_payload = [step]
        if not isinstance(observations_payload, list) or not observations_payload:
            raise ValueError(f"sequence case {case_id}: step {index} observations must be non-empty")
        observations = [
            observation_from_sequence_step(observation)
            for observation in observations_payload
        ]
        step_system(state, observations, cfg)

        revision_requests = step.get("revision_requests", [])
        if revision_requests is None:
            revision_requests = []
        if not isinstance(revision_requests, list):
            raise ValueError(f"sequence case {case_id}: step {index} revision_requests must be a list")
        for request_index, request in enumerate(revision_requests):
            if not isinstance(request, dict):
                raise ValueError(
                    f"sequence case {case_id}: step {index} revision request {request_index} must be an object"
                )
            apply_revision_request(state, request, case_id=case_id, step_index=index)

        execution_drafts = step.get("revision_execution_drafts", [])
        if execution_drafts is None:
            execution_drafts = []
        if not isinstance(execution_drafts, list):
            raise ValueError(f"sequence case {case_id}: step {index} revision_execution_drafts must be a list")
        for request_index, request in enumerate(execution_drafts):
            if not isinstance(request, dict):
                raise ValueError(
                    f"sequence case {case_id}: step {index} revision execution draft {request_index} must be an object"
                )
            apply_revision_execution_draft(state, request, case_id=case_id, step_index=index)

        timeline.append(_timeline_row(state, cfg))

    invariants = check_commit_log_invariants(state)
    revision_invariants = check_revision_log_invariants(state)
    revision_execution_invariants = check_revision_execution_log_invariants(state)
    candidate_id = raw_case.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        if len(state.candidates) != 1:
            raise ValueError(f"sequence case {case_id}: candidate_id required for multi-candidate case")
        candidate_id = next(iter(state.candidates))
    candidate = state.candidates.get(candidate_id)
    if candidate is None:
        raise ValueError(f"sequence case {case_id}: candidate {candidate_id} was not observed")

    expected_committed = raw_case.get("expected_committed")
    if not isinstance(expected_committed, bool):
        raise ValueError(f"sequence case {case_id}: expected_committed must be boolean")
    expected_commit_count = raw_case.get("expected_commit_count")
    if expected_commit_count is not None:
        if not isinstance(expected_commit_count, int) or isinstance(expected_commit_count, bool):
            raise ValueError(f"sequence case {case_id}: expected_commit_count must be integer")
    expected_final_phase = raw_case.get("expected_final_phase")
    if expected_final_phase is not None and not isinstance(expected_final_phase, str):
        raise ValueError(f"sequence case {case_id}: expected_final_phase must be string")
    expected_revision_count = raw_case.get("expected_revision_count")
    if expected_revision_count is not None:
        if not isinstance(expected_revision_count, int) or isinstance(expected_revision_count, bool):
            raise ValueError(f"sequence case {case_id}: expected_revision_count must be integer")

    result = SequenceReplayResult(
        case_id=case_id,
        committed=candidate.committed,
        expected_committed=expected_committed,
        passed=(
            candidate.committed == expected_committed
            and invariants["passed"] is True
            and revision_invariants["passed"] is True
            and revision_execution_invariants["passed"] is True
            and identity_invariants["passed"] is True
            and (
                expected_commit_count is None
                or len(state.commit_log) == expected_commit_count
            )
            and (
                expected_revision_count is None
                or len(state.revision_log) == expected_revision_count
            )
            and (
                expected_final_phase is None
                or candidate.phase.value == expected_final_phase
            )
        ),
        commit_log_count=len(state.commit_log),
        expected_commit_count=expected_commit_count,
        final_phase=candidate.phase.value,
        expected_final_phase=expected_final_phase,
        revision_log_count=len(state.revision_log),
        expected_revision_count=expected_revision_count,
    )
    return {
        **result.__dict__,
        "invariants": invariants,
        "revision_invariants": revision_invariants,
        "revision_execution_invariants": revision_execution_invariants,
        "identity_invariants": identity_invariants,
        "timeline": timeline,
        "commit_log": [event.__dict__ for event in state.commit_log],
        "revision_log": [event.__dict__ for event in state.revision_log],
        "revision_execution_log": [event.__dict__ for event in state.revision_execution_log],
    }


def check_commit_log_invariants(state: RewriteSystemState) -> dict[str, object]:
    errors: list[str] = []
    seen_candidate_ids: set[str] = set()
    for event in state.commit_log:
        if event.candidate_id in seen_candidate_ids:
            errors.append(f"duplicate commit for candidate {event.candidate_id}")
        seen_candidate_ids.add(event.candidate_id)
        candidate = state.candidates.get(event.candidate_id)
        if candidate is None:
            errors.append(f"commit references missing candidate {event.candidate_id}")
            continue
        if not candidate.committed:
            errors.append(f"commit references uncommitted candidate {event.candidate_id}")
        if candidate.phase.value != "acknowledged":
            errors.append(f"commit candidate {event.candidate_id} final phase is {candidate.phase.value}")
        if event.candidate_version != candidate.version:
            errors.append(
                f"commit candidate {event.candidate_id} version mismatch "
                f"{event.candidate_version}!={candidate.version}"
            )
        if not (event.evidence_open and event.constitution_open and event.log_ready):
            errors.append(f"commit event {event.event_id} does not have all gates open")
        if not event.commit_executed:
            errors.append(f"commit event {event.event_id} was not executed")
    committed_candidate_ids = {
        candidate.candidate_id
        for candidate in state.candidates.values()
        if candidate.committed
    }
    logged_candidate_ids = {event.candidate_id for event in state.commit_log}
    for candidate_id in sorted(committed_candidate_ids - logged_candidate_ids):
        errors.append(f"committed candidate {candidate_id} missing commit log event")
    return {
        "passed": not errors,
        "errors": errors,
        "commit_log_count": len(state.commit_log),
        "committed_candidate_count": len(committed_candidate_ids),
    }


def summarize_sequence_results(rows: list[dict[str, object]]) -> dict[str, object]:
    revision_logs = [
        event
        for row in rows
        if isinstance(row.get("revision_log"), list)
        for event in row["revision_log"]
        if isinstance(event, dict)
    ]
    revision_execution_logs = [
        event
        for row in rows
        if isinstance(row.get("revision_execution_log"), list)
        for event in row["revision_execution_log"]
        if isinstance(event, dict)
    ]
    return {
        "num_cases": len(rows),
        "passed": all(row["passed"] is True for row in rows),
        "failed_count": sum(1 for row in rows if row["passed"] is not True),
        "total_commit_log_count": sum(int(row["commit_log_count"]) for row in rows),
        "total_revision_log_count": sum(int(row["revision_log_count"]) for row in rows),
        "total_approved_revision_count": sum(
            1 for event in revision_logs
            if event.get("approval_open") is True
        ),
        "total_executed_revision_count": sum(
            1 for event in revision_logs
            if event.get("revision_executed") is True
        ),
        "total_revision_execution_draft_count": len(revision_execution_logs),
        "total_executed_revision_execution_count": sum(
            1 for event in revision_execution_logs
            if event.get("execution_executed") is True
        ),
        "invariant_failed_count": sum(
            1 for row in rows
            if not isinstance(row.get("invariants"), dict)
            or row["invariants"].get("passed") is not True
        ),
        "revision_invariant_failed_count": sum(
            1 for row in rows
            if not isinstance(row.get("revision_invariants"), dict)
            or row["revision_invariants"].get("passed") is not True
        ),
        "revision_execution_invariant_failed_count": sum(
            1 for row in rows
            if not isinstance(row.get("revision_execution_invariants"), dict)
            or row["revision_execution_invariants"].get("passed") is not True
        ),
        "identity_failed_count": sum(
            1 for row in rows
            if not isinstance(row.get("identity_invariants"), dict)
            or row["identity_invariants"].get("passed") is not True
        ),
    }
