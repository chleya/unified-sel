from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    CandidateObservation,
    RewriteDynamicsConfig,
    RewriteSystemState,
    check_revision_log_invariants,
    propose_revision_for_acknowledged_candidate,
    step_system,
)
from core.rewrite_sequence_replay import check_commit_log_invariants


def random_observation(rng: random.Random, candidate_id: str) -> CandidateObservation:
    high_pressure = rng.random() < 0.45
    if high_pressure:
        disturbance = rng.uniform(0.65, 1.0)
        stability = rng.uniform(0.02, 0.45)
    else:
        disturbance = rng.uniform(0.0, 0.65)
        stability = rng.uniform(0.35, 1.0)
    gates_open = rng.random() < 0.35
    return CandidateObservation(
        candidate_id=candidate_id,
        disturbance_observed=disturbance,
        stability_observed=stability,
        evidence_open=gates_open or rng.random() < 0.15,
        constitution_open=gates_open or rng.random() < 0.25,
        log_ready=gates_open or rng.random() < 0.25,
    )


def generate_sequence(
    *,
    seed: int,
    num_steps: int,
    num_candidates: int,
    max_observations_per_step: int,
) -> list[list[CandidateObservation]]:
    rng = random.Random(seed)
    candidate_ids = [f"fuzz_{index}" for index in range(num_candidates)]
    sequence: list[list[CandidateObservation]] = []
    for _ in range(num_steps):
        width = rng.randint(1, max_observations_per_step)
        observations = [
            random_observation(rng, rng.choice(candidate_ids))
            for _ in range(width)
        ]
        sequence.append(observations)
    return sequence


def run_fuzz_seed(
    *,
    seed: int,
    num_steps: int,
    num_candidates: int,
    max_observations_per_step: int,
    config: RewriteDynamicsConfig,
) -> dict[str, object]:
    sequence = generate_sequence(
        seed=seed,
        num_steps=num_steps,
        num_candidates=num_candidates,
        max_observations_per_step=max_observations_per_step,
    )
    rng = random.Random(seed + 10_000)
    state = RewriteSystemState(bandwidth_limit=config.bandwidth_limit)
    timeline: list[dict[str, object]] = []
    revision_mutation_errors: list[str] = []
    for observations in sequence:
        step_system(state, observations, config)
        acknowledged_before = {
            candidate_id: (candidate.disturbance, candidate.stability)
            for candidate_id, candidate in state.candidates.items()
            if candidate.committed
        }
        for candidate_id in sorted(acknowledged_before):
            if rng.random() >= 0.35:
                continue
            propose_revision_for_acknowledged_candidate(
                state,
                candidate_id=candidate_id,
                reason="fuzz audit-only revision request",
                disturbance_observed=rng.uniform(0.0, 0.25),
                stability_observed=rng.uniform(0.75, 1.0),
                evidence_open=True,
                constitution_open=True,
                log_ready=True,
            )
        acknowledged_after = {
            candidate_id: (candidate.disturbance, candidate.stability)
            for candidate_id, candidate in state.candidates.items()
            if candidate.committed
        }
        for candidate_id, snapshot in acknowledged_before.items():
            if acknowledged_after.get(candidate_id) != snapshot:
                revision_mutation_errors.append(candidate_id)
        timeline.append({
            "step_index": state.step_index,
            "active_candidate_ids": list(state.active_candidate_ids),
            "candidates": {
                cid: {
                    "disturbance": c.disturbance,
                    "stability": c.stability,
                    "phase": c.phase.value,
                    "committed": c.committed,
                    "version": c.version,
                }
                for cid, c in sorted(state.candidates.items())
            },
            "commit_events": len(state.commit_log),
            "revision_events": len(state.revision_log),
        })

    if not state.revision_log:
        committed_ids = sorted(
            candidate_id
            for candidate_id, candidate in state.candidates.items()
            if candidate.committed
        )
        if committed_ids:
            candidate_id = committed_ids[0]
            before = (
                state.candidates[candidate_id].disturbance,
                state.candidates[candidate_id].stability,
            )
            propose_revision_for_acknowledged_candidate(
                state,
                candidate_id=candidate_id,
                reason="fuzz fallback audit-only revision request",
                disturbance_observed=0.0,
                stability_observed=1.0,
                evidence_open=True,
                constitution_open=True,
                log_ready=True,
            )
            after = (
                state.candidates[candidate_id].disturbance,
                state.candidates[candidate_id].stability,
            )
            if after != before:
                revision_mutation_errors.append(candidate_id)

    invariants = check_commit_log_invariants(state)
    revision_invariants = check_revision_log_invariants(state)
    max_active = max((len(row["active_candidate_ids"]) for row in timeline), default=0)
    acknowledged_mutation_errors = []
    first_ack_snapshot: dict[str, tuple[float, float]] = {}
    for row in timeline:
        candidates = row["candidates"]
        if not isinstance(candidates, dict):
            continue
        for candidate_id, snapshot in candidates.items():
            if snapshot["phase"] != "acknowledged":
                continue
            current = (float(snapshot["disturbance"]), float(snapshot["stability"]))
            if candidate_id not in first_ack_snapshot:
                first_ack_snapshot[candidate_id] = current
            elif first_ack_snapshot[candidate_id] != current:
                acknowledged_mutation_errors.append(candidate_id)
    passed = (
        invariants["passed"] is True
        and revision_invariants["passed"] is True
        and max_active <= config.bandwidth_limit
        and not acknowledged_mutation_errors
        and not revision_mutation_errors
    )
    return {
        "seed": seed,
        "passed": passed,
        "num_steps": num_steps,
        "num_candidates": num_candidates,
        "max_active": max_active,
        "bandwidth_limit": config.bandwidth_limit,
        "commit_log_count": len(state.commit_log),
        "revision_log_count": len(state.revision_log),
        "committed_candidate_count": sum(1 for candidate in state.candidates.values() if candidate.committed),
        "invariants": invariants,
        "revision_invariants": revision_invariants,
        "acknowledged_mutation_errors": sorted(set(acknowledged_mutation_errors)),
        "revision_mutation_errors": sorted(set(revision_mutation_errors)),
    }


def run_fuzz(
    *,
    seeds: list[int],
    num_steps: int,
    num_candidates: int,
    max_observations_per_step: int,
    bandwidth_limit: int,
) -> dict[str, object]:
    config = RewriteDynamicsConfig(alpha=1.0, bandwidth_limit=bandwidth_limit)
    rows = [
        run_fuzz_seed(
            seed=seed,
            num_steps=num_steps,
            num_candidates=num_candidates,
            max_observations_per_step=max_observations_per_step,
            config=config,
        )
        for seed in seeds
    ]
    return {
        "passed": all(row["passed"] is True for row in rows),
        "num_seeds": len(seeds),
        "failed_seeds": [row["seed"] for row in rows if row["passed"] is not True],
        "total_commit_log_count": sum(int(row["commit_log_count"]) for row in rows),
        "total_revision_log_count": sum(int(row["revision_log_count"]) for row in rows),
        "max_active_observed": max((int(row["max_active"]) for row in rows), default=0),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-observations-per-step", type=int, default=5)
    parser.add_argument("--bandwidth-limit", type=int, default=3)
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()

    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    result = run_fuzz(
        seeds=seeds,
        num_steps=args.num_steps,
        num_candidates=args.num_candidates,
        max_observations_per_step=args.max_observations_per_step,
        bandwidth_limit=args.bandwidth_limit,
    )
    out_dir = PROJECT_ROOT / "results" / "capability_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rewrite_sequence_fuzz_{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "num_seeds": result["num_seeds"],
        "failed_seeds": result["failed_seeds"],
        "total_commit_log_count": result["total_commit_log_count"],
        "total_revision_log_count": result["total_revision_log_count"],
        "max_active_observed": result["max_active_observed"],
    }, indent=2, sort_keys=True))
    print(f"[OK] wrote {out_path}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
