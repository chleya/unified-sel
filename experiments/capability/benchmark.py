"""
Capability benchmark runner for the next project line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import run_capability_benchmark
from core.runtime import get_results_path, save_json, timestamp


SEMANTIC_FAMILIES = [
    "threshold",
    "parity",
    "zero_role",
    "prime",
    "divisibility",
    "word_symmetry",
    "word_repeat",
    "word_vowel",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="mixed", choices=["reasoning", "code", "mixed"])
    parser.add_argument("--suite-variant", type=str, default="standard", choices=["standard", "paraphrase"])
    parser.add_argument(
        "--protocol",
        type=str,
        default="local_only",
        choices=[
            "local_only",
            "local_verify",
            "local_escalate",
            "confidence_threshold",
            "surprise_gate",
            "monitor_gate",
            "monitor_triage",
            "monitor_repair_triage",
            "verifier_first",
            "escalation_first",
        ],
    )
    parser.add_argument(
        "--local-solver",
        type=str,
        default="search",
        choices=["heuristic", "search"],
    )
    parser.add_argument("--num-tasks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--routing-signal-threshold", type=float, default=0.5)
    parser.add_argument("--escalation-signal-threshold", type=float, default=0.9)
    parser.add_argument("--low-signal-guard-band", type=float, default=0.15)
    parser.add_argument(
        "--routing-monitor",
        type=str,
        default="diagnostic",
        choices=["confidence", "diagnostic", "external", "counterfactual", "behavioral", "surface", "semantic", "hybrid", "topo_surprise", "topo_semantic_fusion"],
    )
    parser.add_argument(
        "--semantic-disable-family",
        action="append",
        default=[],
        choices=SEMANTIC_FAMILIES,
        help="Disable one or more semantic ambiguity families for held-out evaluations.",
    )
    args = parser.parse_args()

    result = run_capability_benchmark(
        suite=args.suite,
        protocol=args.protocol,
        num_tasks=args.num_tasks,
        seed=args.seed,
        suite_variant=args.suite_variant,
        local_solver_name=args.local_solver,
        confidence_threshold=args.confidence_threshold,
        routing_signal_threshold=args.routing_signal_threshold,
        escalation_signal_threshold=args.escalation_signal_threshold,
        low_signal_guard_band=args.low_signal_guard_band,
        routing_monitor_name=args.routing_monitor,
        semantic_disabled_ambiguity_families=args.semantic_disable_family,
    )
    output_dir = get_results_path("capability_benchmark")
    output_path = output_dir / f"{timestamp()}.json"
    save_json(result, output_path)

    result["saved_to"] = str(output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
