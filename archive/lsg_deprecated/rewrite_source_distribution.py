"""Compare LSG proposal-source replay distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .rewrite_proposal_replay import ProposalReplayRow, load_replay_dataset, replay_proposal_case
from .rewrite_provider_capture import capture_jsonl_to_replay_dataset


SourceKind = Literal["replay_json", "capture_jsonl"]


@dataclass(frozen=True)
class ProposalSourceSpec:
    name: str
    kind: SourceKind
    path: Path


@dataclass(frozen=True)
class ProposalSourceSummary:
    name: str
    kind: SourceKind
    path: str
    num_cases: int
    passed: bool
    schema_error_rate: float
    expected_schema_interception_rate: float
    unexpected_schema_failure_rate: float
    authority_request_rate: float
    proxy_disagreement_rate: float
    false_commit_count: int
    missed_commit_count: int
    mean_proxy_delta: float
    max_proxy_delta: float
    failure_class_counts: dict[str, int]


def rows_from_source(spec: ProposalSourceSpec, *, disagreement_threshold: float = 0.25) -> list[ProposalReplayRow]:
    if spec.kind == "replay_json":
        cases = load_replay_dataset(spec.path)
    elif spec.kind == "capture_jsonl":
        cases = capture_jsonl_to_replay_dataset(spec.path)
    else:
        raise ValueError(f"unknown source kind: {spec.kind}")
    return [
        replay_proposal_case(case, disagreement_threshold=disagreement_threshold)
        for case in cases
    ]


def summarize_source(
    spec: ProposalSourceSpec,
    rows: list[ProposalReplayRow],
) -> ProposalSourceSummary:
    num_cases = len(rows)
    if num_cases == 0:
        return ProposalSourceSummary(
            name=spec.name,
            kind=spec.kind,
            path=str(spec.path),
            num_cases=0,
            passed=False,
            schema_error_rate=0.0,
            expected_schema_interception_rate=0.0,
            unexpected_schema_failure_rate=0.0,
            authority_request_rate=0.0,
            proxy_disagreement_rate=0.0,
            false_commit_count=0,
            missed_commit_count=0,
            mean_proxy_delta=0.0,
            max_proxy_delta=0.0,
            failure_class_counts={},
        )

    counts: dict[str, int] = {}
    for row in rows:
        key = row.failure_class or "none"
        counts[key] = counts.get(key, 0) + 1

    valid_delta_rows = [row for row in rows if row.mean_abs_delta is not None]
    mean_proxy_delta = (
        sum(float(row.mean_abs_delta) for row in valid_delta_rows) / len(valid_delta_rows)
        if valid_delta_rows
        else 0.0
    )
    max_proxy_delta = (
        max(float(row.max_abs_delta) for row in valid_delta_rows if row.max_abs_delta is not None)
        if valid_delta_rows
        else 0.0
    )
    return ProposalSourceSummary(
        name=spec.name,
        kind=spec.kind,
        path=str(spec.path),
        num_cases=num_cases,
        passed=all(row.status == "passed" for row in rows),
        schema_error_rate=counts.get("schema_error", 0) / num_cases,
        expected_schema_interception_rate=sum(
            1 for row in rows
            if row.failure_class == "schema_error" and row.expected_error == "schema"
        ) / num_cases,
        unexpected_schema_failure_rate=sum(
            1 for row in rows
            if row.failure_class == "schema_error" and row.expected_error != "schema"
        ) / num_cases,
        authority_request_rate=counts.get("authority_request", 0) / num_cases,
        proxy_disagreement_rate=counts.get("proxy_disagreement", 0) / num_cases,
        false_commit_count=counts.get("false_commit", 0),
        missed_commit_count=counts.get("missed_commit", 0),
        mean_proxy_delta=mean_proxy_delta,
        max_proxy_delta=max_proxy_delta,
        failure_class_counts=counts,
    )


def compare_sources(
    specs: list[ProposalSourceSpec],
    *,
    disagreement_threshold: float = 0.25,
) -> dict[str, object]:
    summaries = [
        summarize_source(
            spec,
            rows_from_source(spec, disagreement_threshold=disagreement_threshold),
        )
        for spec in specs
    ]
    return {
        "passed": all(summary.passed for summary in summaries),
        "num_sources": len(summaries),
        "sources": [summary.__dict__ for summary in summaries],
    }
