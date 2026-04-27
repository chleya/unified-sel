from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_live_provider_capture import (
    LiveProviderConfig,
    build_live_provider_payload,
    capture_live_provider_output,
    extract_json_object_from_text,
    extract_model_json_from_response,
)
from core.rewrite_proposal_provider import PROXY_FIELDS, ProposalRequest
from core.rewrite_provider_capture import ProviderCaptureRequest


class FakeTransport:
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append({
            "url": url,
            "headers": headers,
            "payload": payload,
            "timeout_seconds": timeout_seconds,
        })
        return self.response


def request() -> ProviderCaptureRequest:
    return ProviderCaptureRequest(
        request=ProposalRequest(
            request_id="live_test",
            observation_summary="obs",
            current_order_summary="order",
            goal_summary="goal",
        ),
        explicit={
            "a1_institutional_level": 0.2,
            "p1_dependency_fanout": 0.2,
            "evidence_open": False,
            "constitution_open": True,
            "log_ready": True,
        },
        expected_committed=False,
    )


def proposal_json() -> dict[str, Any]:
    return {
        "proposal_id": "p_live",
        "candidate_id": "c_live",
        "candidate_summary": "live proposal",
        "proxy": {field: 0.2 for field in PROXY_FIELDS},
    }


def test_live_config_requires_env(monkeypatch=None) -> None:
    old = {name: os.environ.get(name) for name in ("MINIMAX_API_KEY", "MINIMAX_API_URL", "MINIMAX_MODEL")}
    try:
        for name in old:
            os.environ.pop(name, None)
        try:
            LiveProviderConfig.from_env(provider_name="minimax")
        except RuntimeError as exc:
            assert "MINIMAX_API_KEY" in str(exc)
            assert "MINIMAX_API_URL" in str(exc)
            assert "MINIMAX_MODEL" in str(exc)
        else:
            raise AssertionError("missing env should fail")
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    print("[OK] live config requires env")


def test_extract_json_object_from_text() -> None:
    assert extract_json_object_from_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object_from_text('prefix {"a": 2} suffix') == {"a": 2}
    try:
        extract_json_object_from_text("no json here")
    except ValueError:
        pass
    else:
        raise AssertionError("non-json text should fail")
    print("[OK] extract JSON object from text")


def test_extract_model_json_from_response_shapes() -> None:
    direct = proposal_json()
    assert extract_model_json_from_response(direct)["proposal_id"] == "p_live"
    chat = {"choices": [{"message": {"content": '{"proposal_id":"p2"}'}}]}
    assert extract_model_json_from_response(chat)["proposal_id"] == "p2"
    text = {"choices": [{"text": '{"proposal_id":"p3"}'}]}
    assert extract_model_json_from_response(text)["proposal_id"] == "p3"
    print("[OK] extract model JSON from response shapes")


def test_capture_live_provider_output_with_fake_transport() -> None:
    config = LiveProviderConfig(
        provider_name="minimax",
        api_url="https://example.invalid/chat",
        api_key="secret",
        model="fake-model",
    )
    fake = FakeTransport({"choices": [{"message": {"content": '{"proposal_id":"p_live","candidate_id":"c_live","candidate_summary":"summary","proxy":{"u1_conflict":0.2,"u2_mismatch":0.2,"n1_goal_loss_if_ignored":0.2,"n2_commitment_carry_cost":0.2,"a1_institutional_level":0.2,"a2_current_anchor_strength":0.2,"p1_dependency_fanout":0.2,"p2_rollback_cost":0.2}}'}}]})
    record = capture_live_provider_output(
        capture_id="capture_live_test",
        config=config,
        capture_request=request(),
        transport=fake,
    )
    assert record.capture_id == "capture_live_test"
    assert record.provider_name == "minimax"
    assert record.raw_model_json["proposal_id"] == "p_live"
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert fake.calls[0]["payload"]["model"] == "fake-model"
    print("[OK] capture live provider output with fake transport")


def test_build_live_provider_payload() -> None:
    config = LiveProviderConfig(
        provider_name="minimax",
        api_url="https://example.invalid/chat",
        api_key="secret",
        model="fake-model",
        temperature=0.3,
    )
    payload = build_live_provider_payload(config, request())
    assert payload["model"] == "fake-model"
    assert payload["temperature"] == 0.3
    assert payload["response_format"] == {"type": "json_object"}
    assert "messages" in payload
    print("[OK] build live provider payload")


def run_all() -> None:
    test_live_config_requires_env()
    test_extract_json_object_from_text()
    test_extract_model_json_from_response_shapes()
    test_capture_live_provider_output_with_fake_transport()
    test_build_live_provider_payload()
    print("All rewrite live provider capture tests passed")


if __name__ == "__main__":
    run_all()
