from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    CandidateObservation,
    RewriteDynamicsConfig,
    simulate_case,
)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def obs(candidate_id: str, d: float, s: float, e: bool, k: bool = True, log: bool = True):
    return CandidateObservation(
        candidate_id=candidate_id,
        disturbance_observed=_clip(d),
        stability_observed=_clip(s),
        evidence_open=e,
        constitution_open=k,
        log_ready=log,
    )


def jitter(rng: random.Random, scale: float = 0.04) -> float:
    return rng.uniform(-scale, scale)


def temporary_spike(seed: int):
    rng = random.Random(seed)
    cid = f"temp_{seed}"
    return [
        [obs(cid, 0.10 + jitter(rng), 0.82 + jitter(rng), False)],
        [obs(cid, 0.88 + jitter(rng), 0.82 + jitter(rng), False)],
        [obs(cid, 0.12 + jitter(rng), 0.82 + jitter(rng), False)],
        [obs(cid, 0.10 + jitter(rng), 0.82 + jitter(rng), False)],
    ], cid


def small_disturbance_stream(seed: int, steps: int = 8):
    rng = random.Random(seed)
    cid = f"small_{seed}"
    case = []
    for _ in range(steps):
        case.append([obs(
            cid,
            0.18 + jitter(rng, 0.03),
            0.70 + jitter(rng, 0.05),
            False,
        )])
    return case, cid


def sustained_drift(seed: int):
    rng = random.Random(seed)
    cid = f"drift_{seed}"
    d_values = [0.20, 0.62, 0.75, 0.84, 0.90]
    s_values = [0.80, 0.68, 0.48, 0.34, 0.24]
    case = []
    for i, (d, s) in enumerate(zip(d_values, s_values)):
        case.append([obs(
            cid,
            d + jitter(rng),
            s + jitter(rng),
            i >= 2,
        )])
    return case, cid


def protected_boundary(seed: int):
    rng = random.Random(seed)
    cid = f"protected_{seed}"
    case = []
    for _ in range(4):
        case.append([obs(
            cid,
            0.90 + jitter(rng, 0.03),
            0.86 + jitter(rng, 0.03),
            True,
            k=False,
        )])
    return case, cid


def run_family(
    name: str,
    builder,
    seeds: list[int],
    config: RewriteDynamicsConfig,
) -> dict[str, object]:
    committed = 0
    commit_events = 0
    max_active = 0
    per_seed = []

    for seed in seeds:
        case, cid = builder(seed)
        state, timeline = simulate_case(case, config)
        did_commit = state.candidates[cid].committed
        committed += int(did_commit)
        commit_events += len(state.commit_log)
        max_active = max(max_active, *(len(row["active_candidate_ids"]) for row in timeline))
        per_seed.append({
            "seed": seed,
            "candidate_id": cid,
            "committed": did_commit,
            "final_phase": state.candidates[cid].phase.value,
            "commit_events": len(state.commit_log),
            "final_disturbance": state.candidates[cid].disturbance,
            "final_stability": state.candidates[cid].stability,
            "final_ratio": state.candidates[cid].ratio,
        })

    return {
        "family": name,
        "num_seeds": len(seeds),
        "commit_rate": committed / len(seeds),
        "commit_events": commit_events,
        "max_active_candidates": max_active,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="7,42,123,256,999,1337,2024,3141,4096,65537")
    parser.add_argument("--label", default="phase2")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    config = RewriteDynamicsConfig()

    families = [
        run_family("temporary_spike", temporary_spike, seeds, config),
        run_family("small_disturbance", small_disturbance_stream, seeds, config),
        run_family("sustained_drift", sustained_drift, seeds, config),
        run_family("protected_boundary", protected_boundary, seeds, config),
    ]
    by_name = {row["family"]: row for row in families}

    passed = (
        by_name["temporary_spike"]["commit_rate"] == 0.0
        and by_name["small_disturbance"]["commit_rate"] == 0.0
        and by_name["protected_boundary"]["commit_rate"] == 0.0
        and by_name["sustained_drift"]["commit_rate"] >= 0.8
        and max(row["max_active_candidates"] for row in families) <= config.bandwidth_limit
    )

    result = {
        "label": args.label,
        "seeds": seeds,
        "config": {
            "alpha": config.alpha,
            "bandwidth_limit": config.bandwidth_limit,
            "theta_verify_ratio": config.theta_verify_ratio,
            "theta_commit_ratio": config.theta_commit_ratio,
            "theta_protected_stability": config.theta_protected_stability,
        },
        "families": families,
        "passed": passed,
    }

    out_dir = PROJECT_ROOT / "results" / "rewrite_dynamics_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"rewrite_dynamics_sweep_{args.label}_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({
        "label": result["label"],
        "passed": result["passed"],
        "family_commit_rates": {
            row["family"]: row["commit_rate"] for row in families
        },
        "result_file": str(out_path),
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

