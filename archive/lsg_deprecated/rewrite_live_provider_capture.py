from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_live_provider_capture import LiveProviderConfig, capture_live_provider_output
from core.rewrite_proposal_provider import ProposalRequest
from core.rewrite_provider_capture import (
    ProviderCaptureRequest,
    build_provider_prompt,
    write_capture_jsonl,
)
from experiments.capability.rewrite_provider_capture_replay import run_capture_replay


def build_capture_request(args: argparse.Namespace) -> ProviderCaptureRequest:
    return ProviderCaptureRequest(
        request=ProposalRequest(
            request_id=args.capture_id,
            observation_summary=args.observation_summary,
            current_order_summary=args.current_order_summary,
            goal_summary=args.goal_summary,
            source=args.source,
        ),
        explicit={
            "a1_institutional_level": args.a1,
            "p1_dependency_fanout": args.p1,
            "evidence_open": args.evidence_open,
            "constitution_open": args.constitution_open,
            "log_ready": args.log_ready,
        },
        expected_committed=args.expected_committed,
        expected_error=args.expected_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Actually call the configured provider")
    parser.add_argument("--provider-name", default="minimax")
    parser.add_argument("--capture-id", default="manual_capture")
    parser.add_argument("--label", default="manual")
    parser.add_argument("--observation-summary", default="candidate observation")
    parser.add_argument("--current-order-summary", default="current formal state")
    parser.add_argument("--goal-summary", default="")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--a1", type=float, default=0.2)
    parser.add_argument("--p1", type=float, default=0.2)
    parser.add_argument("--evidence-open", action="store_true")
    parser.add_argument("--constitution-open", action="store_true")
    parser.add_argument("--log-ready", action="store_true")
    parser.add_argument("--expected-committed", action="store_true")
    parser.add_argument("--expected-error", default=None)
    args = parser.parse_args()

    capture_request = build_capture_request(args)
    if not args.live:
        prompt = build_provider_prompt(capture_request.request)
        print(json.dumps({
            "mode": "dry_run",
            "provider_name": args.provider_name,
            "required_env": [
                f"{args.provider_name.upper()}_API_KEY",
                f"{args.provider_name.upper()}_API_URL",
                f"{args.provider_name.upper()}_MODEL",
            ],
            "prompt": prompt,
        }, indent=2, sort_keys=True))
        return

    config = LiveProviderConfig.from_env(provider_name=args.provider_name)
    record = capture_live_provider_output(
        capture_id=args.capture_id,
        config=config,
        capture_request=capture_request,
    )
    capture_dir = PROJECT_ROOT / "data" / "lsg" / "provider_captures"
    capture_path = capture_dir / f"{args.provider_name}_{args.label}.jsonl"
    write_capture_jsonl([record], capture_path)

    replay = run_capture_replay(capture_path)
    out_dir = PROJECT_ROOT / "results" / "capability_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rewrite_live_provider_capture_{args.provider_name}_{args.label}.json"
    out_path.write_text(json.dumps(replay, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(replay["summary"], indent=2, sort_keys=True))
    print(f"[OK] wrote capture {capture_path}")
    print(f"[OK] wrote replay {out_path}")
    if not replay["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
