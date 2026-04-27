from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_source_distribution import (
    ProposalSourceSpec,
    compare_sources,
    rows_from_source,
    summarize_source,
)


REPLAY = PROJECT_ROOT / "data" / "lsg" / "proposal_replay_v0.json"
CAPTURE = PROJECT_ROOT / "data" / "lsg" / "provider_capture_v0.jsonl"


def test_rows_from_replay_source() -> None:
    spec = ProposalSourceSpec("hand_authored", "replay_json", REPLAY)
    rows = rows_from_source(spec)
    assert len(rows) == 4
    assert any(row.failure_class == "authority_request" for row in rows)
    assert any(row.failure_class == "proxy_disagreement" for row in rows)
    print("[OK] rows from replay source")


def test_rows_from_capture_source() -> None:
    spec = ProposalSourceSpec("capture_fixture", "capture_jsonl", CAPTURE)
    rows = rows_from_source(spec)
    assert len(rows) == 2
    assert any(row.failure_class == "schema_error" for row in rows)
    assert all(row.failure_class != "false_commit" for row in rows)
    print("[OK] rows from capture source")


def test_summarize_source_rates() -> None:
    replay_spec = ProposalSourceSpec("hand_authored", "replay_json", REPLAY)
    replay_summary = summarize_source(replay_spec, rows_from_source(replay_spec))
    assert replay_summary.num_cases == 4
    assert replay_summary.passed is True
    assert replay_summary.schema_error_rate == 0.25
    assert replay_summary.expected_schema_interception_rate == 0.25
    assert replay_summary.unexpected_schema_failure_rate == 0.0
    assert replay_summary.authority_request_rate == 0.25
    assert replay_summary.proxy_disagreement_rate == 0.25
    assert replay_summary.false_commit_count == 0
    assert replay_summary.missed_commit_count == 0

    capture_spec = ProposalSourceSpec("capture_fixture", "capture_jsonl", CAPTURE)
    capture_summary = summarize_source(capture_spec, rows_from_source(capture_spec))
    assert capture_summary.num_cases == 2
    assert capture_summary.schema_error_rate == 0.5
    assert capture_summary.expected_schema_interception_rate == 0.5
    assert capture_summary.unexpected_schema_failure_rate == 0.0
    assert capture_summary.authority_request_rate == 0.0
    print("[OK] summarize source rates")


def test_compare_sources() -> None:
    result = compare_sources([
        ProposalSourceSpec("hand_authored", "replay_json", REPLAY),
        ProposalSourceSpec("capture_fixture", "capture_jsonl", CAPTURE),
    ])
    assert result["passed"] is True
    assert result["num_sources"] == 2
    assert len(result["sources"]) == 2
    by_name = {row["name"]: row for row in result["sources"]}
    assert by_name["hand_authored"]["authority_request_rate"] == 0.25
    assert by_name["capture_fixture"]["schema_error_rate"] == 0.5
    assert by_name["capture_fixture"]["unexpected_schema_failure_rate"] == 0.0
    print("[OK] compare sources")


def run_all() -> None:
    test_rows_from_replay_source()
    test_rows_from_capture_source()
    test_summarize_source_rates()
    test_compare_sources()
    print("All rewrite source distribution tests passed")


if __name__ == "__main__":
    run_all()
