from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    CandidateObservation,
    RewriteDynamicsConfig,
    project_commit_event_to_cee,
    simulate_case,
)


def obs(candidate_id, d, s, e=False, k=True, log=True):
    return CandidateObservation(
        candidate_id=candidate_id,
        disturbance_observed=d,
        stability_observed=s,
        evidence_open=e,
        constitution_open=k,
        log_ready=log,
    )


def build_projection():
    state, timeline = simulate_case([
        [obs("cee_roundtrip", 0.2, 0.8, e=False)],
        [obs("cee_roundtrip", 0.75, 0.45, e=True)],
        [obs("cee_roundtrip", 0.9, 0.25, e=True)],
    ], RewriteDynamicsConfig(alpha=1.0))

    if len(state.commit_log) != 1:
        raise RuntimeError(f"expected 1 commit event, got {len(state.commit_log)}")

    projection = project_commit_event_to_cee(
        state.commit_log[0],
        source_state_id="ws_0",
        resulting_state_id="ws_1",
    )
    return state, timeline, projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cee-src",
        default="F:\\cognitive-execution-engine\\src",
        help="Path to CEE src directory",
    )
    args = parser.parse_args()

    cee_src = Path(args.cee_src)
    if not cee_src.exists():
        raise SystemExit(f"CEE src path does not exist: {cee_src}")
    if str(cee_src) not in sys.path:
        sys.path.insert(0, str(cee_src))

    from cee_core.commitment import CommitmentEvent
    from cee_core.event_log import EventLog
    from cee_core.revision import ModelRevisionEvent

    state, timeline, projection = build_projection()

    cee_commitment = CommitmentEvent.from_dict(projection.commitment_event)
    cee_revision = ModelRevisionEvent.from_dict(projection.revision_event)

    log = EventLog()
    log.append(cee_commitment)
    log.append(cee_revision)
    replayed = log.replay_world_state()

    expected_anchor = projection.revision_event["new_anchor_fact_summaries"][0]
    passed = (
        cee_revision.caused_by_event_id == cee_commitment.event_id
        and replayed.state_id == "ws_1"
        and expected_anchor in replayed.anchored_fact_summaries
        and len(log.commitment_events()) == 1
        and len(log.revision_events()) == 1
    )

    result = {
        "passed": passed,
        "cee_src": str(cee_src),
        "lsg_commit_events": len(state.commit_log),
        "cee_commitment_event_id": cee_commitment.event_id,
        "cee_revision_id": cee_revision.revision_id,
        "causal_link_ok": cee_revision.caused_by_event_id == cee_commitment.event_id,
        "replayed_state_id": replayed.state_id,
        "expected_anchor": expected_anchor,
        "anchor_replayed": expected_anchor in replayed.anchored_fact_summaries,
        "timeline": timeline,
    }

    out_dir = PROJECT_ROOT / "results" / "rewrite_dynamics_cee_roundtrip"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"rewrite_dynamics_cee_roundtrip_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "passed": passed,
        "causal_link_ok": result["causal_link_ok"],
        "anchor_replayed": result["anchor_replayed"],
        "result_file": str(out_path),
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

