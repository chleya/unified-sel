from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_proposal_provider import PROXY_FIELDS, ProposalRequest
from core.rewrite_proposal_replay import replay_proposal_case, summarize_replay
from core.rewrite_provider_capture import (
    ProviderCaptureRequest,
    build_provider_prompt,
    capture_jsonl_to_replay_dataset,
    capture_provider_output,
    capture_record_to_json,
)


CAPTURE_FIXTURE = PROJECT_ROOT / "data" / "lsg" / "provider_capture_v0.jsonl"


def test_provider_prompt_names_required_proxy_fields() -> None:
    prompt = build_provider_prompt(
        ProposalRequest(
            request_id="r1",
            observation_summary="obs",
            current_order_summary="order",
            goal_summary="goal",
        )
    )
    for field in PROXY_FIELDS:
        assert field in prompt
    assert "audit-only" in prompt
    print("[OK] provider prompt names required proxy fields")


def test_capture_record_to_replay_case() -> None:
    request = ProposalRequest(
        request_id="capture_test",
        observation_summary="obs",
        current_order_summary="order",
    )
    raw_json = {
        "proposal_id": "p",
        "candidate_id": "c",
        "candidate_summary": "summary",
        "proxy": {field: 0.2 for field in PROXY_FIELDS},
    }
    record = capture_provider_output(
        capture_id="capture_test",
        provider_name="fixture_provider",
        capture_request=ProviderCaptureRequest(
            request=request,
            explicit={
                "a1_institutional_level": 0.2,
                "p1_dependency_fanout": 0.2,
                "evidence_open": False,
                "constitution_open": True,
                "log_ready": True,
            },
            expected_committed=False,
        ),
        raw_model_json=raw_json,
    )
    payload = capture_record_to_json(record)
    assert payload["capture_id"] == "capture_test"
    assert payload["raw_model_json"]["proposal_id"] == "p"
    assert payload["expected_committed"] is False
    print("[OK] capture record serializes")


def test_capture_jsonl_replays_through_boundary() -> None:
    replay_cases = capture_jsonl_to_replay_dataset(CAPTURE_FIXTURE)
    rows = [replay_proposal_case(case) for case in replay_cases]
    summary = summarize_replay(rows)
    assert summary["passed"] is True
    assert summary["num_cases"] == 2
    assert summary["false_commit_count"] == 0
    assert summary["missed_commit_count"] == 0
    assert summary["schema_error_count"] == 1
    print("[OK] capture JSONL replays through boundary")


def run_all() -> None:
    test_provider_prompt_names_required_proxy_fields()
    test_capture_record_to_replay_case()
    test_capture_jsonl_replays_through_boundary()
    print("All rewrite provider capture tests passed")


if __name__ == "__main__":
    run_all()
