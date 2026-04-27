"""Offline capture protocol for LSG proposal-provider outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rewrite_proposal_provider import PROXY_FIELDS, ProposalRequest


@dataclass(frozen=True)
class ProviderCaptureRequest:
    request: ProposalRequest
    explicit: dict[str, Any]
    expected_committed: bool | None = None
    expected_error: str | None = None


@dataclass(frozen=True)
class ProviderCaptureRecord:
    capture_id: str
    provider_name: str
    request: ProposalRequest
    prompt: str
    raw_model_json: dict[str, Any]
    explicit: dict[str, Any]
    expected_committed: bool | None = None
    expected_error: str | None = None


def build_provider_prompt(request: ProposalRequest) -> str:
    proxy_lines = "\n".join(f'- "{field}": number in [0, 1]' for field in PROXY_FIELDS)
    return (
        "You are generating an untrusted LSG rewrite proposal.\n"
        "Return exactly one JSON object. Do not include markdown.\n"
        "The JSON object must contain:\n"
        '- "proposal_id": non-empty string\n'
        '- "candidate_id": non-empty string\n'
        '- "candidate_summary": non-empty string\n'
        '- "proxy": object with exactly these fields:\n'
        f"{proxy_lines}\n"
        "You may include requested_evidence_open, requested_constitution_open, "
        "requested_log_ready, or requested_threshold_update, but these are audit-only "
        "and will not grant authority.\n\n"
        f"request_id: {request.request_id}\n"
        f"source: {request.source}\n"
        f"observation_summary: {request.observation_summary}\n"
        f"current_order_summary: {request.current_order_summary}\n"
        f"goal_summary: {request.goal_summary}\n"
    )


def capture_provider_output(
    *,
    capture_id: str,
    provider_name: str,
    capture_request: ProviderCaptureRequest,
    raw_model_json: dict[str, Any],
) -> ProviderCaptureRecord:
    if not capture_id:
        raise ValueError("capture_id must be non-empty")
    if not provider_name:
        raise ValueError("provider_name must be non-empty")
    if not isinstance(raw_model_json, dict):
        raise ValueError("raw_model_json must be an object")
    return ProviderCaptureRecord(
        capture_id=capture_id,
        provider_name=provider_name,
        request=capture_request.request,
        prompt=build_provider_prompt(capture_request.request),
        raw_model_json=raw_model_json,
        explicit=capture_request.explicit,
        expected_committed=capture_request.expected_committed,
        expected_error=capture_request.expected_error,
    )


def capture_record_to_json(record: ProviderCaptureRecord) -> dict[str, Any]:
    return {
        "capture_id": record.capture_id,
        "provider_name": record.provider_name,
        "request": {
            "request_id": record.request.request_id,
            "observation_summary": record.request.observation_summary,
            "current_order_summary": record.request.current_order_summary,
            "goal_summary": record.request.goal_summary,
            "source": record.request.source,
        },
        "prompt": record.prompt,
        "raw_model_json": record.raw_model_json,
        "explicit": record.explicit,
        "expected_committed": record.expected_committed,
        "expected_error": record.expected_error,
    }


def write_capture_jsonl(records: list[ProviderCaptureRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(capture_record_to_json(record), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_capture_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"capture JSONL line {index} must be an object")
        records.append(value)
    return records


def capture_json_to_replay_case(record: dict[str, Any]) -> dict[str, Any]:
    case_id = record.get("capture_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("capture record missing capture_id")
    model_json = record.get("raw_model_json")
    if not isinstance(model_json, dict):
        raise ValueError(f"capture {case_id}: raw_model_json must be an object")
    explicit = record.get("explicit")
    if not isinstance(explicit, dict):
        raise ValueError(f"capture {case_id}: explicit must be an object")
    replay_case = {
        "case_id": case_id,
        "proposal_origin": str(record.get("provider_name", "provider")),
        "model_json": model_json,
        "explicit": explicit,
    }
    if record.get("expected_error") is not None:
        replay_case["expected_error"] = record["expected_error"]
    else:
        expected_committed = record.get("expected_committed")
        if not isinstance(expected_committed, bool):
            raise ValueError(f"capture {case_id}: expected_committed must be boolean unless expected_error is set")
        replay_case["expected_committed"] = expected_committed
    return replay_case


def capture_jsonl_to_replay_dataset(path: Path) -> list[dict[str, Any]]:
    return [capture_json_to_replay_case(record) for record in load_capture_jsonl(path)]
