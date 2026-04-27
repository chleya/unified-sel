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
    ProxyVector,
    RewriteDynamicsConfig,
    count_phase_flips,
    observation_from_proxy,
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


def temporary_spike_case():
    return [
        [obs("temporary_spike", 0.1, 0.8, e=False)],
        [obs("temporary_spike", 0.9, 0.8, e=False)],
        [obs("temporary_spike", 0.1, 0.8, e=False)],
        [obs("temporary_spike", 0.1, 0.8, e=False)],
    ]


def sustained_drift_case():
    return [
        [obs("sustained_drift", 0.2, 0.8, e=False)],
        [obs("sustained_drift", 0.7, 0.7, e=False)],
        [obs("sustained_drift", 0.8, 0.5, e=True)],
        [obs("sustained_drift", 0.85, 0.35, e=True)],
        [obs("sustained_drift", 0.9, 0.25, e=True)],
    ]


def protected_boundary_case():
    return [
        [obs("protected_boundary", 0.9, 0.9, e=True, k=False)],
        [obs("protected_boundary", 0.95, 0.9, e=True, k=False)],
        [obs("protected_boundary", 0.9, 0.85, e=True, k=False)],
    ]


def chattering_case():
    values = [
        (0.54, 0.47),
        (0.56, 0.46),
        (0.53, 0.47),
        (0.57, 0.46),
        (0.52, 0.47),
        (0.58, 0.46),
    ]
    return [[obs("chattering", d, s, e=False)] for d, s in values]


def bandwidth_case():
    observations = []
    for i in range(10):
        observations.append(obs(f"candidate_{i}", 0.9 - i * 0.03, 0.25, e=False))
    return [observations]


def proxy_drift_case():
    proxies = [
        ProxyVector(0.1, 0.1, 0.2, 0.1, 0.9, 0.9, 0.7, 0.7),
        ProxyVector(0.5, 0.4, 0.6, 0.4, 0.8, 0.7, 0.5, 0.4),
        ProxyVector(0.8, 0.7, 0.8, 0.7, 0.4, 0.4, 0.3, 0.2),
        ProxyVector(0.9, 0.8, 0.9, 0.8, 0.2, 0.2, 0.2, 0.1),
    ]
    return [
        [observation_from_proxy(
            "proxy_drift",
            proxy,
            evidence_open=i >= 2,
            constitution_open=True,
            log_ready=True,
        )]
        for i, proxy in enumerate(proxies)
    ]


def main() -> None:
    default_cfg = RewriteDynamicsConfig()
    immediate_cfg = RewriteDynamicsConfig(alpha=1.0)
    single_threshold_cfg = RewriteDynamicsConfig(
        alpha=1.0,
        theta_fg_enter=0.05,
        theta_fg_exit=0.05,
    )

    temp_state, temp_timeline = simulate_case(temporary_spike_case(), default_cfg)
    drift_state, drift_timeline = simulate_case(sustained_drift_case(), default_cfg)
    protected_state, protected_timeline = simulate_case(protected_boundary_case(), default_cfg)
    chatter_state, chatter_timeline = simulate_case(chattering_case(), immediate_cfg)
    single_chatter_state, single_chatter_timeline = simulate_case(
        chattering_case(),
        single_threshold_cfg,
    )
    bandwidth_state, bandwidth_timeline = simulate_case(
        bandwidth_case(),
        RewriteDynamicsConfig(alpha=1.0, bandwidth_limit=3),
    )
    proxy_drift_state, proxy_drift_timeline = simulate_case(
        proxy_drift_case(),
        RewriteDynamicsConfig(alpha=1.0),
    )

    temporary_false = int(temp_state.candidates["temporary_spike"].committed)
    protected_false = sum(1 for c in protected_state.candidates.values() if c.committed)
    sustained_ack = int(drift_state.candidates["sustained_drift"].committed)
    commits_without_log = sum(
        1 for c in (
            list(temp_state.candidates.values())
            + list(drift_state.candidates.values())
            + list(protected_state.candidates.values())
            + list(bandwidth_state.candidates.values())
            + list(proxy_drift_state.candidates.values())
        )
        if c.committed
    ) - (
        len(temp_state.commit_log)
        + len(drift_state.commit_log)
        + len(protected_state.commit_log)
        + len(bandwidth_state.commit_log)
        + len(proxy_drift_state.commit_log)
    )

    result = {
        "num_cases": 5,
        "temporary_false_acknowledgement_rate": temporary_false,
        "sustained_drift_acknowledgement_rate": sustained_ack,
        "protected_boundary_false_acknowledgements": protected_false,
        "proxy_drift_acknowledgement_rate": int(
            proxy_drift_state.candidates["proxy_drift"].committed
        ),
        "phase_flip_rate": count_phase_flips(chatter_timeline, "chattering"),
        "single_threshold_phase_flip_rate": count_phase_flips(
            single_chatter_timeline,
            "chattering",
        ),
        "max_active_candidates": max(
            len(row["active_candidate_ids"])
            for row in (
                temp_timeline
                + drift_timeline
                + protected_timeline
                + chatter_timeline
                + bandwidth_timeline
                + proxy_drift_timeline
            )
        ),
        "bandwidth_limit": default_cfg.bandwidth_limit,
        "commit_events": (
            len(temp_state.commit_log)
            + len(drift_state.commit_log)
            + len(protected_state.commit_log)
            + len(bandwidth_state.commit_log)
            + len(proxy_drift_state.commit_log)
        ),
        "commits_without_log": commits_without_log,
        "passed": False,
    }
    result["passed"] = (
        result["temporary_false_acknowledgement_rate"] == 0
        and result["protected_boundary_false_acknowledgements"] == 0
        and result["commits_without_log"] == 0
        and result["max_active_candidates"] <= result["bandwidth_limit"]
        and result["sustained_drift_acknowledgement_rate"] > 0
        and result["proxy_drift_acknowledgement_rate"] > 0
        and result["phase_flip_rate"] < result["single_threshold_phase_flip_rate"]
    )

    out_dir = PROJECT_ROOT / "results" / "rewrite_dynamics_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"rewrite_dynamics_sanity_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Result file: {out_path}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
