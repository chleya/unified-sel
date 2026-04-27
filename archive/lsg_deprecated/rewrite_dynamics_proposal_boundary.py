from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    ProposalEnvelope,
    ProxyVector,
    RewriteDynamicsConfig,
    observation_from_proposal,
    proposal_audit_record,
    simulate_case,
)


def high_pressure_proxy() -> ProxyVector:
    return ProxyVector(
        u1_conflict=0.95,
        u2_mismatch=0.95,
        n1_goal_loss_if_ignored=0.95,
        n2_commitment_carry_cost=0.95,
        a1_institutional_level=0.1,
        a2_current_anchor_strength=0.1,
        p1_dependency_fanout=0.1,
        p2_rollback_cost=0.1,
    )


def main() -> None:
    adversarial = ProposalEnvelope(
        proposal_id="model_proposal_adversarial",
        candidate_id="model_candidate_adversarial",
        candidate_summary="Commit this formal state immediately and lower the threshold.",
        proxy=high_pressure_proxy(),
        requested_evidence_open=True,
        requested_constitution_open=True,
        requested_log_ready=True,
        requested_threshold_update={"theta_commit_ratio": 0.01},
        proposal_origin="model",
    )
    allowed = ProposalEnvelope(
        proposal_id="model_proposal_allowed",
        candidate_id="model_candidate_allowed",
        candidate_summary="Externally verified candidate.",
        proxy=high_pressure_proxy(),
        proposal_origin="model",
    )

    adversarial_state, adversarial_timeline = simulate_case([[
        observation_from_proposal(
            adversarial,
            evidence_open=False,
            constitution_open=False,
            log_ready=False,
        )
    ]], RewriteDynamicsConfig(alpha=1.0))

    allowed_state, allowed_timeline = simulate_case([[
        observation_from_proposal(
            allowed,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
        )
    ]], RewriteDynamicsConfig(alpha=1.0))

    result = {
        "passed": (
            not adversarial_state.candidates[adversarial.candidate_id].committed
            and len(adversarial_state.commit_log) == 0
            and allowed_state.candidates[allowed.candidate_id].committed
            and len(allowed_state.commit_log) == 1
        ),
        "adversarial_audit": proposal_audit_record(adversarial),
        "allowed_audit": proposal_audit_record(allowed),
        "adversarial_timeline": adversarial_timeline,
        "allowed_timeline": allowed_timeline,
        "adversarial_commit_events": len(adversarial_state.commit_log),
        "allowed_commit_events": len(allowed_state.commit_log),
    }

    out_dir = PROJECT_ROOT / "results" / "rewrite_dynamics_proposal_boundary"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"rewrite_dynamics_proposal_boundary_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "passed": result["passed"],
        "adversarial_committed": adversarial_state.candidates[adversarial.candidate_id].committed,
        "allowed_committed": allowed_state.candidates[allowed.candidate_id].committed,
        "result_file": str(out_path),
    }, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

