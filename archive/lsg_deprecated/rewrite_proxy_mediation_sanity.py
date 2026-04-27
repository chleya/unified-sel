from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import ProposalEnvelope, ProxyVector, RewriteDynamicsConfig, simulate_case
from core.rewrite_proxy_mediator import (
    ExplicitProxyState,
    mediate_proposal,
    mediated_audit_record,
    observation_from_mediated_proposal,
)


def adversarial_proposal() -> ProposalEnvelope:
    return ProposalEnvelope(
        proposal_id="mediation_attack",
        candidate_id="candidate_mediation_attack",
        candidate_summary="model proposes low stability and requests authority gates",
        proxy=ProxyVector(
            u1_conflict=0.95,
            u2_mismatch=0.95,
            n1_goal_loss_if_ignored=0.95,
            n2_commitment_carry_cost=0.95,
            a1_institutional_level=0.05,
            a2_current_anchor_strength=0.05,
            p1_dependency_fanout=0.05,
            p2_rollback_cost=0.05,
        ),
        requested_evidence_open=True,
        requested_constitution_open=True,
        requested_log_ready=True,
        requested_threshold_update={"theta_commit_ratio": 0.01},
    )


def run_case(name: str, explicit: ExplicitProxyState) -> dict[str, object]:
    mediated = mediate_proposal(adversarial_proposal(), explicit)
    observation = observation_from_mediated_proposal(mediated)
    state, trace = simulate_case([[observation]], RewriteDynamicsConfig(alpha=1.0))
    candidate = state.candidates[mediated.proposal.candidate_id]
    return {
        "case": name,
        "committed": candidate.committed,
        "phase": candidate.phase.value,
        "commit_log_count": len(state.commit_log),
        "audit": mediated_audit_record(mediated),
        "effective_proxy": mediated.effective_proxy.__dict__,
        "last_trace": trace[-1],
    }


def run_experiment() -> dict[str, object]:
    closed = run_case(
        "closed_gates_system_owned_high_stability",
        ExplicitProxyState(
            a1_institutional_level=0.9,
            p1_dependency_fanout=0.9,
            evidence_open=False,
            constitution_open=False,
            log_ready=False,
        ),
    )
    open_low_stability = run_case(
        "open_gates_system_owned_low_stability",
        ExplicitProxyState(
            a1_institutional_level=0.1,
            p1_dependency_fanout=0.1,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
        ),
    )
    return {
        "experiment": "rewrite_proxy_mediation_sanity",
        "claims": {
            "model_authority_requests_ignored": bool(closed["audit"]["ignored_authority_requests"]),
            "closed_gates_do_not_commit": not closed["committed"],
            "open_system_gates_can_commit": open_low_stability["committed"],
        },
        "cases": [closed, open_low_stability],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()
    result = run_experiment()
    out_dir = PROJECT_ROOT / "results" / "capability_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rewrite_proxy_mediation_sanity_{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["claims"], indent=2, sort_keys=True))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
