from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import ProposalEnvelope, ProxyVector
from core.rewrite_proxy_calibration import (
    ProxyCalibrationCase,
    evaluate_proxy_calibration_case,
    proxy_abs_deltas,
    summarize_proxy_calibration,
)
from core.rewrite_proxy_mediator import ExplicitProxyState, mediate_proposal


def envelope(case_id: str, proxy: ProxyVector, request_authority: bool = False) -> ProposalEnvelope:
    return ProposalEnvelope(
        proposal_id=f"p_{case_id}",
        candidate_id=f"c_{case_id}",
        candidate_summary=f"calibration fixture {case_id}",
        proxy=proxy,
        requested_evidence_open=True if request_authority else None,
        requested_constitution_open=True if request_authority else None,
        requested_log_ready=True if request_authority else None,
        requested_threshold_update={"theta_commit_ratio": 0.01} if request_authority else None,
    )


def proxy(
    u1: float,
    u2: float,
    n1: float,
    n2: float,
    a1: float,
    a2: float,
    p1: float,
    p2: float,
) -> ProxyVector:
    return ProxyVector(u1, u2, n1, n2, a1, a2, p1, p2)


def explicit_from_proxy(base: ProxyVector, evidence: bool, constitution: bool, log: bool) -> ExplicitProxyState:
    return ExplicitProxyState(
        u1_conflict=base.u1_conflict,
        u2_mismatch=base.u2_mismatch,
        n1_goal_loss_if_ignored=base.n1_goal_loss_if_ignored,
        n2_commitment_carry_cost=base.n2_commitment_carry_cost,
        a1_institutional_level=base.a1_institutional_level,
        a2_current_anchor_strength=base.a2_current_anchor_strength,
        p1_dependency_fanout=base.p1_dependency_fanout,
        p2_rollback_cost=base.p2_rollback_cost,
        evidence_open=evidence,
        constitution_open=constitution,
        log_ready=log,
    )


def calibration_cases() -> list[ProxyCalibrationCase]:
    aligned = proxy(0.9, 0.8, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1)
    overclaim_model = proxy(0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05)
    overclaim_system = proxy(0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9)
    underclaim_model = proxy(0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9)
    underclaim_system = proxy(0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05)
    return [
        ProxyCalibrationCase(
            case_id="aligned_commit",
            proposal=envelope("aligned_commit", aligned),
            explicit=explicit_from_proxy(aligned, evidence=True, constitution=True, log=True),
            expected_committed=True,
        ),
        ProxyCalibrationCase(
            case_id="overclaim_blocked",
            proposal=envelope("overclaim_blocked", overclaim_model, request_authority=True),
            explicit=explicit_from_proxy(overclaim_system, evidence=False, constitution=False, log=False),
            expected_committed=False,
        ),
        ProxyCalibrationCase(
            case_id="underclaim_corrected",
            proposal=envelope("underclaim_corrected", underclaim_model),
            explicit=explicit_from_proxy(underclaim_system, evidence=True, constitution=True, log=True),
            expected_committed=True,
        ),
    ]


def run_experiment() -> dict[str, object]:
    cases = calibration_cases()
    rows = [evaluate_proxy_calibration_case(case) for case in cases]
    case_records = []
    for case, row in zip(cases, rows):
        mediated = mediate_proposal(case.proposal, case.explicit)
        case_records.append({
            **asdict(row),
            "proxy_abs_deltas": proxy_abs_deltas(mediated.suggested_proxy, mediated.effective_proxy),
        })
    return {
        "experiment": "rewrite_proxy_calibration_fixture",
        "summary": summarize_proxy_calibration(rows),
        "cases": case_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()
    result = run_experiment()
    out_dir = PROJECT_ROOT / "results" / "capability_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rewrite_proxy_calibration_fixture_{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"[OK] wrote {out_path}")
    if not result["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
