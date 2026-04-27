from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.capability.rewrite_sequence_fuzz import run_fuzz


def test_sequence_fuzz_fixed_seeds() -> None:
    result = run_fuzz(
        seeds=[0, 1, 2],
        num_steps=8,
        num_candidates=6,
        max_observations_per_step=4,
        bandwidth_limit=3,
    )
    assert result["passed"] is True
    assert result["failed_seeds"] == []
    assert result["max_active_observed"] <= 3
    assert result["total_revision_log_count"] > 0
    for row in result["rows"]:
        assert row["invariants"]["passed"] is True
        assert row["revision_invariants"]["passed"] is True
        assert row["acknowledged_mutation_errors"] == []
        assert row["revision_mutation_errors"] == []
    print("[OK] sequence fuzz fixed seeds")


def run_all() -> None:
    test_sequence_fuzz_fixed_seeds()
    print("All rewrite sequence fuzz tests passed")


if __name__ == "__main__":
    run_all()
