"""
Capability generalization runner for held-out family and paraphrase checks.
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
    parser.add_argument("--suite", type=str, default="code", choices=["code", "mixed"])
    parser.add_argument("--protocol", type=str, default="monitor_repair_triage", choices=["monitor_gate", "monitor_triage", "monitor_repair_triage"])
    parser.add_argument("--variants", nargs="+", default=["standard", "paraphrase"], choices=["standard", "paraphrase"])
    parser.add_argument("--routing-monitors", nargs="+", default=["semantic", "counterfactual", "diagnostic"], choices=["confidence", "diagnostic", "external", "counterfactual", "behavioral", "surface", "semantic", "hybrid"])
    parser.add_argument("--semantic-holdout-family", action="append", default=[], choices=SEMANTIC_FAMILIES)
    parser.add_argument("--local-solver", type=str, default="search", choices=["heuristic", "search"])
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--routing-signal-threshold", type=float, default=0.5)
    parser.add_argument("--escalation-signal-threshold", type=float, default=0.9)
    parser.add_argument("--low-signal-guard-band", type=float, default=0.15)
    args = parser.parse_args()

    runs = []
    for variant in args.variants:
        for monitor in args.routing_monitors:
            result = run_capability_benchmark(
                suite=args.suite,
                protocol=args.protocol,
                num_tasks=args.num_tasks,
                seed=args.seed,
                suite_variant=variant,
                local_solver_name=args.local_solver,
                routing_signal_threshold=args.routing_signal_threshold,
                escalation_signal_threshold=args.escalation_signal_threshold,
                low_signal_guard_band=args.low_signal_guard_band,
                routing_monitor_name=monitor,
            )
            runs.append(
                {
                    "label": f"{monitor}:{variant}",
                    "monitor": monitor,
                    "suite_variant": variant,
                    "semantic_disabled_ambiguity_families": [],
                    "summary": result["summary"],
                    "result": result,
                }
            )

        for family in args.semantic_holdout_family:
            result = run_capability_benchmark(
                suite=args.suite,
                protocol=args.protocol,
                num_tasks=args.num_tasks,
                seed=args.seed,
                suite_variant=variant,
                local_solver_name=args.local_solver,
                routing_signal_threshold=args.routing_signal_threshold,
                escalation_signal_threshold=args.escalation_signal_threshold,
                low_signal_guard_band=args.low_signal_guard_band,
                routing_monitor_name="semantic",
                semantic_disabled_ambiguity_families=[family],
            )
            runs.append(
                {
                    "label": f"semantic_holdout_{family}:{variant}",
                    "monitor": "semantic",
                    "suite_variant": variant,
                    "semantic_disabled_ambiguity_families": [family],
                    "summary": result["summary"],
                    "result": result,
                }
            )

    payload = {
        "suite": args.suite,
        "protocol": args.protocol,
        "num_tasks": args.num_tasks,
        "seed": args.seed,
        "variants": args.variants,
        "routing_monitors": args.routing_monitors,
        "semantic_holdout_family": args.semantic_holdout_family,
        "low_signal_guard_band": args.low_signal_guard_band,
        "runs": runs,
    }

    output_dir = get_results_path("capability_generalization")
    output_path = output_dir / f"{timestamp()}.json"
    save_json(payload, output_path)
    payload["saved_to"] = str(output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
