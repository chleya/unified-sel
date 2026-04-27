"""
A4: Transfer Matrix Runner and Summarizer

Produces a success-rate × mean-cost matrix over:
  - Domains: standard, paraphrase, stronger_paraphrase, naturalized
  - Monitors: semantic, semantic+guard, counterfactual, confidence
  - Protocols: monitor_repair_triage (primary), monitor_gate (secondary)
  - Seeds: multiple (frozen: 7, 11, 17)

Usage:
    # Phase 1 sanity: single seed
    python experiments/transfer_matrix.py --protocol monitor_repair_triage --seeds 7

    # Phase 1 full: 3-seed matrix
    python experiments/transfer_matrix.py --protocol monitor_repair_triage --seeds 7,11,17

    # Phase 1 invariance check: monitor_gate single seed
    python experiments/transfer_matrix.py --protocol monitor_gate --seeds 7

Output:
    results/capability_generalization/transfer_matrix_<label>_<timestamp>.json
    results/capability_generalization/transfer_matrix_<label>_<timestamp>.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.capability_benchmark import (
    run_capability_benchmark,
    build_routing_monitor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "capability_generalization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen canonical config
DOMAINS = ["standard", "paraphrase", "stronger_paraphrase", "naturalized"]
MONITORS = ["semantic", "semantic+guard", "counterfactual", "confidence"]
PROTOCOL_EPOCH = "post_A1"

# Routing signal threshold override per protocol (for hybrid_gate: lower threshold for accept)
ROUTING_SIGNAL_THRESHOLDS = {
    "hybrid_gate": 0.40,   # hybrid uses lower accept threshold (calibrated on stronger_paraphrase)
    "monitor_gate": 0.50,
    "monitor_repair_triage": 0.50,
}


def run_transfer_cell(
    protocol: str,
    domain: str,
    monitor: str,
    num_tasks: int,
    seed: int,
) -> Dict[str, Any]:
    """Run a single cell of the transfer matrix."""
    semantic_disabled = (
        ["threshold", "parity", "zero_role", "prime", "divisibility",
         "word_symmetry", "word_repeat", "word_vowel"]
        if monitor == "semantic+guard"
        else []
    )
    routing_monitor_name = (
        "semantic" if monitor in ("semantic", "semantic+guard") else monitor
    )
    rs_thresh = ROUTING_SIGNAL_THRESHOLDS.get(protocol, 0.5)

    result = run_capability_benchmark(
        suite="code",
        protocol=protocol,
        num_tasks=num_tasks,
        seed=seed,
        suite_variant=domain,
        local_solver_name="search",
        confidence_threshold=0.95,
        routing_signal_threshold=rs_thresh,
        escalation_signal_threshold=0.9,
        routing_monitor_name=routing_monitor_name,
        semantic_disabled_ambiguity_families=semantic_disabled,
        low_signal_guard_band=0.15,
    )
    return result


def extract_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    s = result.get("summary", {})
    return {
        "success_rate": s.get("success_rate", 0.0),
        "mean_latency_units": s.get("mean_latency_units", 0.0),
        "mean_cost_units": s.get("mean_cost_units", 0.0),
        "mean_routing_signal": s.get("mean_routing_signal", 0.0),
        "escalation_rate": s.get("escalation_rate", 0.0),
        "revision_rate": s.get("revision_rate", 0.0),
        "verifier_rate": s.get("verifier_rate", 0.0),
        "accepted_without_verifier_rate": s.get("accepted_without_verifier_rate", 0.0),
        "direct_escalation_rate": s.get("direct_escalation_rate", 0.0),
    }


def run_matrix(
    protocol: str,
    domains: List[str],
    monitors: List[str],
    num_tasks: int,
    seeds: List[int],
    label: str = "default",
) -> Dict[str, Any]:
    """Run full transfer matrix across domains, monitors, and seeds."""
    per_seed: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {s: {} for s in seeds}
    runs_log = []
    total_cells = len(domains) * len(monitors) * len(seeds)
    cell_idx = 0

    for seed in seeds:
        for domain in domains:
            per_seed[seed][domain] = {}
            for monitor in monitors:
                cell_idx += 1
                tag = f"{protocol}/{domain}/{monitor}/seed={seed}"
                print(f"[{cell_idx}/{total_cells}] {tag}")

                t0 = time.time()
                try:
                    result = run_transfer_cell(
                        protocol=protocol,
                        domain=domain,
                        monitor=monitor,
                        num_tasks=num_tasks,
                        seed=seed,
                    )
                    elapsed = time.time() - t0
                    cell = extract_summary(result)
                    per_seed[seed][domain][monitor] = cell

                    log_entry = {
                        "protocol": protocol,
                        "domain": domain,
                        "monitor": monitor,
                        "seed": seed,
                        "elapsed_s": round(elapsed, 1),
                        "protocol_epoch": PROTOCOL_EPOCH,
                        **cell,
                    }
                    runs_log.append(log_entry)
                    print(f"  → sr={cell['success_rate']:.2f}  cost={cell['mean_cost_units']:.2f}  "
                          f"ver={cell['verifier_rate']:.2f}  acc_no_ver={cell['accepted_without_verifier_rate']:.2f}")

                except Exception as ex:
                    print(f"  → ERROR: {ex}")
                    per_seed[seed][domain][monitor] = {
                        "success_rate": None, "mean_cost_units": None,
                        "verifier_rate": None, "accepted_without_verifier_rate": None,
                        "error": str(ex),
                    }
                    runs_log.append({
                        "protocol": protocol, "domain": domain, "monitor": monitor,
                        "seed": seed, "elapsed_s": round(time.time() - t0, 1),
                        "protocol_epoch": PROTOCOL_EPOCH,
                        "success_rate": None, "mean_cost_units": None,
                        "verifier_rate": None, "accepted_without_verifier_rate": None,
                        "error": str(ex),
                    })

    # Aggregate across seeds
    aggregated = _aggregate(per_seed, domains, monitors, seeds)

    output = {
        "protocol": protocol,
        "protocol_epoch": PROTOCOL_EPOCH,
        "num_tasks": num_tasks,
        "seeds": seeds,
        "domains": domains,
        "monitors": monitors,
        "label": label,
        "per_seed": per_seed,
        "aggregated": aggregated,
        "runs_log": runs_log,
    }
    return output


def _aggregate(
    per_seed: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    domains: List[str],
    monitors: List[str],
    seeds: List[int],
) -> Dict[str, Any]:
    """Compute mean/std across seeds per cell."""
    agg = {}
    for domain in domains:
        agg[domain] = {}
        for monitor in monitors:
            vals = {
                "success_rate": [],
                "mean_cost_units": [],
                "mean_latency_units": [],
                "mean_routing_signal": [],
                "escalation_rate": [],
                "revision_rate": [],
                "verifier_rate": [],
                "accepted_without_verifier_rate": [],
                "direct_escalation_rate": [],
            }
            for seed in seeds:
                cell = per_seed[seed][domain][monitor]
                if cell.get("success_rate") is not None:
                    for k in vals:
                        vals[k].append(cell[k])

            if vals["success_rate"]:
                agg[domain][monitor] = {
                    k: {"mean": float(np.mean(v)), "std": float(np.std(v)) if len(v) > 1 else 0.0}
                    for k, v in vals.items()
                }
            else:
                agg[domain][monitor] = None
    return agg


def render_markdown(output: Dict[str, Any]) -> str:
    domains = output["domains"]
    monitors = output["monitors"]
    seeds = output["seeds"]
    label = output["label"]
    per_seed = output["per_seed"]
    agg = output["aggregated"]
    is_multi_seed = len(seeds) > 1

    lines = [f"# Transfer Matrix\n"]
    lines.append(f"**Protocol**: `{output['protocol']}`  |  **Epoch**: `{output['protocol_epoch']}`  |  "
                 f"**Tasks**: {output['num_tasks']}  |  **Seeds**: {seeds}  |  **Label**: {label}\n")

    metrics = [
        ("Success Rate", "success_rate", ".2f"),
        ("Mean Cost", "mean_cost_units", ".2f"),
        ("Verifier Rate", "verifier_rate", ".2f"),
        ("Accepted Without Verifier Rate", "accepted_without_verifier_rate", ".2f"),
        ("Direct Escalation Rate", "direct_escalation_rate", ".2f"),
        ("Mean Routing Signal", "mean_routing_signal", ".2f"),
    ]

    for metric_name, metric_key, fmt in metrics:
        if is_multi_seed:
            lines.append(f"## {metric_name} (aggregated, mean±std)\n")
        else:
            lines.append(f"## {metric_name}\n")

        lines.append(f"| Domain | {' | '.join(monitors)} |")
        lines.append(f"|---|---|{'|'.join(['---'] * len(monitors))}|")

        for domain in domains:
            row = [domain]
            for monitor in monitors:
                cell = agg[domain][monitor]
                if cell is None:
                    row.append("—")
                else:
                    stat = cell.get(metric_key)
                    if stat is None:
                        row.append("—")
                    elif is_multi_seed:
                        row.append(f"{stat['mean']:{fmt}}±{stat['std']:{fmt}}")
                    else:
                        row.append(f"{stat['mean']:{fmt}}")
            lines.append(f"| {' | '.join(row)} |")
        lines.append("")

    if is_multi_seed:
        lines.append("## Per-Seed Breakdown (Success Rate)\n")
        for seed in seeds:
            lines.append(f"### Seed {seed}\n")
            lines.append(f"| Domain | {' | '.join(monitors)} |")
            lines.append(f"|---|---|{'|'.join(['---'] * len(monitors))}|")
            for domain in domains:
                row = [domain]
                for monitor in monitors:
                    v = per_seed[seed][domain][monitor].get("success_rate")
                    row.append("—" if v is None else f"{v:.2f}")
                lines.append(f"| {' | '.join(row)} |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="A4: Transfer Matrix Runner")
    parser.add_argument("--protocol", type=str, required=True,
                        choices=["monitor_repair_triage", "monitor_gate", "monitor_triage", "hybrid_gate"])
    parser.add_argument("--seeds", type=str, default="7",
                        help="Comma-separated seed list (default: 7)")
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated domains (default: all 4)")
    parser.add_argument("--monitors", type=str, default=None,
                        help="Comma-separated monitors (default: all 4)")
    parser.add_argument("--label", type=str, default="",
                        help="Label for output files (default: protocol name)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    domains = args.domains.split(",") if args.domains else DOMAINS
    monitors = args.monitors.split(",") if args.monitors else MONITORS
    label = args.label or args.protocol

    print(f"=== Transfer Matrix ===")
    print(f"Protocol: {args.protocol}  epoch: {PROTOCOL_EPOCH}")
    print(f"Tasks: {args.num_tasks}  Seeds: {seeds}")
    print(f"Domains: {domains}")
    print(f"Monitors: {monitors}")
    print()

    t0 = time.time()
    result = run_matrix(
        protocol=args.protocol,
        domains=domains,
        monitors=monitors,
        num_tasks=args.num_tasks,
        seeds=seeds,
        label=label,
    )
    total_elapsed = time.time() - t0

    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"transfer_matrix_{label}_{ts}"
    json_path = RESULTS_DIR / f"{stem}.json"
    md_path = RESULTS_DIR / f"{stem}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"\nJSON: {json_path}")

    md_text = render_markdown(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"MD:   {md_path}")
    print(f"Total time: {total_elapsed:.1f}s")

    # Always update latest pointer
    latest_json = RESULTS_DIR / "transfer_matrix_latest.json"
    latest_md = RESULTS_DIR / "transfer_matrix_latest.md"
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_json_default)
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Latest: {latest_json}")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
