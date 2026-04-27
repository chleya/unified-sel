"""
Capability benchmark CLI: capbench

Usage:
    python -m experiments.capability.capbench run [options]
    python -m experiments.capability.capbench list-monitors
    python -m experiments.capability.capbench list-policies
    python -m experiments.capability.capbench compare <baseline.json> <experiment.json>
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


MONITORS = {
    "confidence": "Solver confidence only. No code analysis.",
    "diagnostic": "Solver search diagnostics. Requires solver-internal metadata.",
    "external": "Surface-level heuristics (difficulty, edit distance, confidence). No code execution.",
    "counterfactual": "Repair candidate enumeration. Strongest reference monitor.",
    "behavioral": "Synthesized challenge tests. Answer-only, no bug_type labels.",
    "surface": "Output string patterns. Same blind spots as behavioral.",
    "semantic": "Surface analysis + ambiguity detection + probe tests. Current best.",
    "hybrid": "Confidence + diagnostic blend.",
    "topo_surprise": "TopoMem embedding novelty. REJECTED as routing monitor (novelty != correctness).",
    "topo_semantic_fusion": "30% topo + 70% semantic. No advantage over semantic-only.",
}

POLICIES = {
    "local_only": "Always accept local answer. No verification.",
    "local_verify": "Always verify then revise. No routing.",
    "local_escalate": "Verify + escalate if low confidence. No routing signal.",
    "confidence_threshold": "Uses confidence only, not routing signal.",
    "surprise_gate": "Signal < threshold => accept; else verify -> revise -> escalate.",
    "monitor_gate": "Same as surprise_gate but with monitor signal.",
    "monitor_triage": "3-tier: low signal => accept; high => escalate; medium => verify/revise.",
    "monitor_repair_triage": "3-tier with repairability check. Best current policy.",
    "monitor_no_revision_triage": "3-tier for weak revisers: low accept, high direct escalate, medium verify/escalate.",
    "hybrid_gate": "3-tier with fixed thresholds (0.40/0.90).",
    "verifier_first": "Always verify first, then decide.",
    "escalation_first": "Escalate immediately for high-signal tasks.",
}

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


def cmd_run(args: argparse.Namespace) -> None:
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
    s = result.get("summary", {})
    print(f"Suite: {result['suite']} | Protocol: {result['protocol']} | Monitor: {result['routing_monitor_name']}")
    print(f"Success: {s.get('success_rate', 'N/A'):.4f} | Cost: {s.get('mean_cost_units', 'N/A'):.4f} | Saved: {output_path}")


def cmd_list_monitors(_args: argparse.Namespace) -> None:
    print("Available routing monitors:\n")
    for name, desc in MONITORS.items():
        print(f"  {name:25s} {desc}")
    print()


def cmd_list_policies(_args: argparse.Namespace) -> None:
    print("Available routing policies:\n")
    for name, desc in POLICIES.items():
        print(f"  {name:25s} {desc}")
    print()


def cmd_compare(args: argparse.Namespace) -> None:
    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.experiment, encoding="utf-8") as f:
        experiment = json.load(f)

    b_meta = baseline.get("metadata", {})
    e_meta = experiment.get("metadata", {})
    b_sum = baseline.get("summary", {})
    e_sum = experiment.get("summary", {})

    print(f"Baseline:    {args.baseline}")
    print(f"  Suite: {baseline.get('suite')} | Protocol: {baseline.get('protocol')} | Monitor: {baseline.get('routing_monitor_name')}")
    print(f"  Schema: {baseline.get('schema_version', 'N/A')} | Data: {b_meta.get('data_source', 'N/A')} | Cost: {b_meta.get('cost_model', 'N/A')}")
    if b_meta.get("benchmark_suite"):
        print(f"  Suite: {b_meta['benchmark_suite']} | Policy: {b_meta.get('verifier_policy', 'N/A')} | Seeds: {b_meta.get('seeds', [])}")
    print()
    print(f"Experiment:  {args.experiment}")
    print(f"  Suite: {experiment.get('suite')} | Protocol: {experiment.get('protocol')} | Monitor: {experiment.get('routing_monitor_name')}")
    print(f"  Schema: {experiment.get('schema_version', 'N/A')} | Data: {e_meta.get('data_source', 'N/A')} | Cost: {e_meta.get('cost_model', 'N/A')}")
    if e_meta.get("benchmark_suite"):
        print(f"  Suite: {e_meta['benchmark_suite']} | Policy: {e_meta.get('verifier_policy', 'N/A')} | Seeds: {e_meta.get('seeds', [])}")
    print()

    metrics = ["success_rate", "mean_cost_units", "escalation_rate", "revision_rate", "verifier_rate", "direct_escalation_rate", "accepted_without_verifier_rate"]
    print(f"{'Metric':35s} {'Baseline':>10s} {'Experiment':>10s} {'Delta':>10s}")
    print("-" * 70)
    for m in metrics:
        bv = b_sum.get(m)
        ev = e_sum.get(m)
        if bv is not None and ev is not None:
            delta = ev - bv
            print(f"  {m:33s} {bv:10.4f} {ev:10.4f} {delta:+10.4f}")
        elif bv is not None or ev is not None:
            print(f"  {m:33s} {bv!s:>10s} {ev!s:>10s}")


def cmd_report(args: argparse.Namespace) -> None:
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    summary = data.get("summary", {})
    results = data.get("results", [])
    tasks = data.get("tasks", [])

    lines: list[str] = []

    lines.append("# Capability Benchmark Report")
    lines.append("")
    lines.append(f"**Suite**: {data.get('suite', 'N/A')} | **Protocol**: {data.get('protocol', 'N/A')} | **Monitor**: {data.get('routing_monitor_name', 'N/A')}")
    lines.append(f"**Seed**: {data.get('seed', 'N/A')} | **Tasks**: {data.get('num_tasks', 'N/A')} | **Solver**: {data.get('local_solver_name', 'N/A')}")
    lines.append(f"**Schema**: {data.get('schema_version', 'N/A')} | **Data source**: {meta.get('data_source', 'N/A')} | **Cost model**: {meta.get('cost_model', 'N/A')}")
    if meta.get("benchmark_suite"):
        lines.append(f"**Benchmark**: {meta['benchmark_suite']} | **Verifier policy**: {meta.get('verifier_policy', 'N/A')} | **Seeds**: {meta.get('seeds', [])}")
    if meta.get("oracle_assumption"):
        lines.append("**WARNING: Oracle assumption**: Escalation path uses OracleSolver (100% success by assumption)")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    for key, label in [
        ("success_rate", "Success Rate"),
        ("mean_cost_units", "Mean Cost (assumed units)"),
        ("escalation_rate", "Escalation Rate"),
        ("revision_rate", "Revision Rate"),
        ("verifier_rate", "Verifier Rate"),
        ("direct_escalation_rate", "Direct Escalation Rate"),
        ("accepted_without_verifier_rate", "Accepted w/o Verifier"),
        ("mean_routing_signal", "Mean Routing Signal"),
    ]:
        val = summary.get(key)
        if val is not None:
            lines.append(f"| {label} | {val:.4f} |")
    lines.append("")

    batch_health = data.get("batch_health", {})
    if batch_health:
        lines.append("## Batch Health (OBD)")
        lines.append("")
        lines.append(f"| Signal | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Status | {batch_health.get('status', 'N/A')} |")
        lines.append(f"| Tasks observed | {batch_health.get('n_observed', 'N/A')} |")
        half_drift = batch_health.get("half_split_drift")
        if half_drift is not None:
            lines.append(f"| Half-split drift | {half_drift:.4f} |")
        mean_sim = batch_health.get("mean_pairwise_similarity")
        if mean_sim is not None:
            lines.append(f"| Mean pairwise sim | {mean_sim:.4f} |")
        lines.append("")

    family_summary = summary.get("family_summary", {})
    if family_summary:
        lines.append("## Per-Family Breakdown")
        lines.append("")
        lines.append("| Family | Count | Success | Cost | Escalation | Revision |")
        lines.append("|--------|-------|---------|------|------------|----------|")
        for fam, fs in sorted(family_summary.items()):
            lines.append(f"| {fam} | {fs.get('count', 0)} | {fs.get('success_rate', 0):.3f} | {fs.get('mean_cost_units', 0):.3f} | {fs.get('escalation_rate', 0):.3f} | {fs.get('revision_rate', 0):.3f} |")
        lines.append("")

    route_counts: dict[str, int] = {}
    for r in results:
        path = r.get("decision_path", "unknown")
        route_counts[path] = route_counts.get(path, 0) + 1

    if route_counts:
        lines.append("## Route Trace")
        lines.append("")
        lines.append("| Decision Path | Count |")
        lines.append("|---------------|-------|")
        for path, count in sorted(route_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {path} | {count} |")
        lines.append("")

    failures = [r for r in results if not r.get("success", True)]
    if failures:
        lines.append("## Failure Examples")
        lines.append("")
        for i, r in enumerate(failures[:5]):
            task_id = r.get("task_id", "?")
            family = r.get("family", "?")
            signal = r.get("routing_signal", 0)
            path = r.get("decision_path", "?")
            lines.append(f"### Failure {i+1}: `{task_id}` ({family})")
            lines.append(f"- Routing signal: {signal:.4f}")
            lines.append(f"- Decision path: {path}")
            lines.append(f"- Escalated: {r.get('escalated', False)} | Revised: {r.get('revised', False)}")
            lines.append("")
    else:
        lines.append("## Failures")
        lines.append("")
        lines.append("No failures — all tasks resolved successfully.")
        lines.append("")

    report = "\n".join(lines)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)


def main() -> None:
    parser = argparse.ArgumentParser(prog="capbench", description="Capability benchmark toolkit")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a capability benchmark experiment")
    run_p.add_argument("--suite", default="mixed", choices=["reasoning", "code", "mixed"])
    run_p.add_argument("--suite-variant", default="standard", choices=["standard", "paraphrase"])
    run_p.add_argument("--protocol", default="local_only", choices=list(POLICIES.keys()))
    run_p.add_argument("--local-solver", default="search", choices=["heuristic", "search"])
    run_p.add_argument("--num-tasks", type=int, default=12)
    run_p.add_argument("--seed", type=int, default=7)
    run_p.add_argument("--confidence-threshold", type=float, default=0.95)
    run_p.add_argument("--routing-signal-threshold", type=float, default=0.5)
    run_p.add_argument("--escalation-signal-threshold", type=float, default=0.9)
    run_p.add_argument("--low-signal-guard-band", type=float, default=0.15)
    run_p.add_argument("--routing-monitor", default="semantic", choices=list(MONITORS.keys()))
    run_p.add_argument("--semantic-disable-family", action="append", default=[], choices=SEMANTIC_FAMILIES)

    sub.add_parser("list-monitors", help="List available routing monitors")
    sub.add_parser("list-policies", help="List available routing policies")

    cmp_p = sub.add_parser("compare", help="Compare two result JSON files")
    cmp_p.add_argument("baseline", type=str, help="Path to baseline result JSON")
    cmp_p.add_argument("experiment", type=str, help="Path to experiment result JSON")

    rpt_p = sub.add_parser("report", help="Generate Markdown report from result JSON")
    rpt_p.add_argument("input", type=str, help="Path to result JSON file")
    rpt_p.add_argument("--output", type=str, default="", help="Output Markdown file (default: stdout)")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list-monitors":
        cmd_list_monitors(args)
    elif args.command == "list-policies":
        cmd_list_policies(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
