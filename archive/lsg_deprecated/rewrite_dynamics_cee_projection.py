from __future__ import annotations

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


def main() -> None:
    state, timeline = simulate_case([
        [obs("cee_candidate", 0.2, 0.8, e=False)],
        [obs("cee_candidate", 0.75, 0.45, e=True)],
        [obs("cee_candidate", 0.9, 0.25, e=True)],
    ], RewriteDynamicsConfig(alpha=1.0))

    if len(state.commit_log) != 1:
        raise SystemExit(f"expected 1 commit event, got {len(state.commit_log)}")

    projection = project_commit_event_to_cee(
        state.commit_log[0],
        source_state_id="ws_lsg_0",
        resulting_state_id="ws_lsg_1",
    )
    result = {
        "passed": True,
        "timeline": timeline,
        "lsg_commit_event": state.commit_log[0].__dict__,
        "cee_projection": {
            "commitment_event": projection.commitment_event,
            "revision_event": projection.revision_event,
        },
    }

    out_dir = PROJECT_ROOT / "results" / "rewrite_dynamics_cee_projection"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"rewrite_dynamics_cee_projection_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "passed": True,
        "commitment_event_id": projection.commitment_event["event_id"],
        "revision_id": projection.revision_event["revision_id"],
        "result_file": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()

