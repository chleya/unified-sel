from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_proposal_provider import proposal_from_model_json


def valid_payload():
    return {
        "proposal_id": "schema_valid",
        "candidate_id": "schema_candidate",
        "candidate_summary": "candidate from future MiniMax JSON",
        "proxy": {
            "u1_conflict": 0.9,
            "u2_mismatch": 0.8,
            "n1_goal_loss_if_ignored": 0.7,
            "n2_commitment_carry_cost": 0.6,
            "a1_institutional_level": 0.2,
            "a2_current_anchor_strength": 0.2,
            "p1_dependency_fanout": 0.3,
            "p2_rollback_cost": 0.2,
        },
        "requested_evidence_open": True,
        "requested_constitution_open": True,
        "requested_log_ready": True,
        "requested_threshold_update": {"theta_commit_ratio": 0.01},
    }


def validate_case(name: str, payload: dict, should_pass: bool) -> dict[str, object]:
    try:
        proposal = proposal_from_model_json(payload, proposal_origin="minimax")
        return {
            "case": name,
            "passed_validation": True,
            "expected": should_pass,
            "accepted": should_pass,
            "proposal_id": proposal.proposal_id,
        }
    except ValueError as exc:
        return {
            "case": name,
            "passed_validation": False,
            "expected": should_pass,
            "accepted": not should_pass,
            "error": str(exc),
        }


def main() -> None:
    cases = []

    payload = valid_payload()
    cases.append(validate_case("valid", payload, True))

    payload = valid_payload()
    del payload["proxy"]["u1_conflict"]
    cases.append(validate_case("missing_proxy", payload, False))

    payload = valid_payload()
    payload["proxy"]["u1_conflict"] = -0.1
    cases.append(validate_case("score_below_range", payload, False))

    payload = valid_payload()
    payload["proxy"]["unexpected"] = 0.5
    cases.append(validate_case("unexpected_proxy", payload, False))

    payload = valid_payload()
    payload["requested_log_ready"] = "true"
    cases.append(validate_case("bad_gate_type", payload, False))

    result = {
        "passed": all(row["accepted"] for row in cases),
        "cases": cases,
    }

    out_dir = PROJECT_ROOT / "results" / "rewrite_proposal_schema_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"rewrite_proposal_schema_validation_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "passed": result["passed"],
        "num_cases": len(cases),
        "result_file": str(out_path),
    }, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

