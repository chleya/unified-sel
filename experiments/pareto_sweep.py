"""
A5 / Phase 2: Guard-Band Pareto Sweep

Sweeps low_signal_guard_band from 0.00 to 0.30 (step 0.02) and compares:
  - Swept: semantic signal across guard bands
  - Fixed: counterfactual, confidence
  - Fixed: semantic+guard at canonical band 0.15

Protocol: monitor_gate (Week 1 finding: guard protection works here)
Domains: stronger_paraphrase + naturalized (shifted domains, primary)
         standard (baseline)
Seeds: 7, 11, 17

Usage:
    python experiments/pareto_sweep.py --num-tasks 20

Output:
    results/capability_pareto/pareto_sweep_<timestamp>.json
    results/capability_pareto/pareto_frontier_<timestamp>.json
    results/capability_pareto/pareto_latest.json / .md
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
RESULTS_DIR = PROJECT_ROOT / "results" / "capability_pareto"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOL_EPOCH = "post_A1"
GUARD_BANDS = [round(x * 0.02, 2) for x in range(0, 16)]  # 0.00 to 0.30 step 0.02
DOMAINS_PARETO = ["stronger_paraphrase", "naturalized"]
DOMAINS_BASELINE = ["standard"]
ALL_DOMAINS = ["standard", "stronger_paraphrase", "naturalized"]
FIXED_BANDS = {
    "semantic+guard@0.15": 0.15,
}


def run_cell(
    protocol: str,
    domain: str,
    monitor: str,
    low_signal_guard_band: float,
    num_tasks: int,
    seed: int,
) -> Dict[str, Any]:
    """Run a single benchmark cell.

    monitor: "semantic" | "semantic+guard" | "counterfactual" | "confidence"
      - "semantic":           routing_monitor_name="semantic",  no family disabling
      - "semantic+guard":     routing_monitor_name="semantic",  8 families disabled (= guard effect)
      - "counterfactual":      routing_monitor_name="counterfactual"
      - "confidence":         routing_monitor_name="confidence"
    """
    # Guard effect = disabling 8 ambiguity families in SemanticRoutingMonitor.
    # low_signal_guard_band is only effective in monitor_triage/monitor_repair_triage.
    # For monitor_gate, it is inert and kept only for API compatibility.
    semantic_disabled = (
        ["threshold", "parity", "zero_role", "prime", "divisibility",
         "word_symmetry", "word_repeat", "word_vowel"]
        if monitor == "semantic+guard"
        else []
    )
    routing_monitor_name = (
        "semantic" if monitor in ("semantic", "semantic+guard") else monitor
    )

    result = run_capability_benchmark(
        suite="code",
        protocol=protocol,
        num_tasks=num_tasks,
        seed=seed,
        suite_variant=domain,
        local_solver_name="search",
        confidence_threshold=0.95,
        routing_signal_threshold=0.5,
        escalation_signal_threshold=0.9,
        routing_monitor_name=routing_monitor_name,
        semantic_disabled_ambiguity_families=semantic_disabled,
        low_signal_guard_band=low_signal_guard_band,
    )
    return result


def extract_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    s = result.get("summary", {})
    return {
        "success_rate": s.get("success_rate", 0.0),
        "mean_cost_units": s.get("mean_cost_units", 0.0),
        "verifier_rate": s.get("verifier_rate", 0.0),
        "accepted_without_verifier_rate": s.get("accepted_without_verifier_rate", 0.0),
    }


def run_sweep(protocol: str, num_tasks: int, seeds: List[int], domains: List[str]) -> Dict[str, Any]:
    """Run the full Pareto sweep."""
    all_runs = []

    # --- Part 1: Guard band sweep (semantic only, varying band) ---
    print(f"\n=== Part 1: Guard-band sweep (semantic) ===")
    for seed in seeds:
        for band in GUARD_BANDS:
            for domain in domains:
                label = f"semantic@{band}"
                tag = f"{protocol}/{domain}/{label}/seed={seed}"
                print(f"  {tag}")
                t0 = time.time()
                try:
                    result = run_cell(protocol, domain, "semantic", band, num_tasks, seed)
                    elapsed = time.time() - t0
                    cell = extract_summary(result)
                    all_runs.append({
                        "protocol": protocol, "domain": domain, "monitor_label": label,
                        "low_signal_guard_band": band,
                        "semantic_disabled": [],
                        "seed": seed, "protocol_epoch": PROTOCOL_EPOCH,
                        "elapsed_s": round(elapsed, 1),
                        **cell,
                    })
                    print(f"    → sr={cell['success_rate']:.2f}  cost={cell['mean_cost_units']:.2f}")
                except Exception as ex:
                    print(f"    → ERROR: {ex}")
                    all_runs.append({
                        "protocol": protocol, "domain": domain, "monitor_label": label,
                        "low_signal_guard_band": band,
                        "semantic_disabled": [],
                        "seed": seed, "protocol_epoch": PROTOCOL_EPOCH,
                        "elapsed_s": round(time.time() - t0, 1),
                        "success_rate": None, "mean_cost_units": None,
                        "verifier_rate": None, "accepted_without_verifier_rate": None,
                        "error": str(ex),
                    })

    # --- Part 2: Fixed comparators ---
    print(f"\n=== Part 2: Fixed comparators ===")
    comparators = [
        ("semantic+guard@0.15", "semantic+guard", 0.15),
        ("counterfactual", "counterfactual", 0.15),
        ("confidence", "confidence", 0.15),
    ]
    for seed in seeds:
        for domain in domains:
            for label, monitor, band in comparators:
                tag = f"{protocol}/{domain}/{label}/seed={seed}"
                print(f"  {tag}")
                t0 = time.time()
                try:
                    result = run_cell(protocol, domain, monitor, band, num_tasks, seed)
                    elapsed = time.time() - t0
                    cell = extract_summary(result)
                    disabled = (
                        ["threshold", "parity", "zero_role", "prime", "divisibility",
                         "word_symmetry", "word_repeat", "word_vowel"]
                        if monitor == "semantic+guard"
                        else []
                    )
                    all_runs.append({
                        "protocol": protocol, "domain": domain, "monitor_label": label,
                        "low_signal_guard_band": band,
                        "semantic_disabled": disabled,
                        "seed": seed, "protocol_epoch": PROTOCOL_EPOCH,
                        "elapsed_s": round(elapsed, 1),
                        **cell,
                    })
                    print(f"    → sr={cell['success_rate']:.2f}  cost={cell['mean_cost_units']:.2f}")
                except Exception as ex:
                    print(f"    → ERROR: {ex}")
                    all_runs.append({
                        "protocol": protocol, "domain": domain, "monitor_label": label,
                        "low_signal_guard_band": band,
                        "semantic_disabled": [],
                        "seed": seed, "protocol_epoch": PROTOCOL_EPOCH,
                        "elapsed_s": round(time.time() - t0, 1),
                        "success_rate": None, "mean_cost_units": None,
                        "verifier_rate": None, "accepted_without_verifier_rate": None,
                        "error": str(ex),
                    })

    return {"protocol": protocol, "protocol_epoch": PROTOCOL_EPOCH,
            "num_tasks": num_tasks, "seeds": seeds, "domains": domains,
            "guard_bands_swept": GUARD_BANDS, "all_runs": all_runs}


def build_frontier(sweep_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build Pareto frontier across guard bands, per domain."""
    all_runs = sweep_result["all_runs"]
    domains = sweep_result["domains"]

    frontier = {}
    for domain in domains:
        domain_runs = [r for r in all_runs if r["domain"] == domain and r["success_rate"] is not None]

        # Separate sweep runs from fixed comparators
        sweep_runs = [r for r in domain_runs if r["monitor_label"].startswith("semantic@")]
        comp_runs = {r["monitor_label"]: r for r in domain_runs
                     if not r["monitor_label"].startswith("semantic@")}

        # Build cost vs success tradeoff curve for swept semantic
        # Group by guard_band and average across seeds
        by_band: Dict[float, List[Dict]] = {}
        for r in sweep_runs:
            by_band.setdefault(r["low_signal_guard_band"], []).append(r)

        tradeoff_curve = []
        for band, runs in sorted(by_band.items()):
            sr_vals = [r["success_rate"] for r in runs]
            cost_vals = [r["mean_cost_units"] for r in runs]
            tradeoff_curve.append({
                "low_signal_guard_band": band,
                "monitor_label": f"semantic@{band}",
                "success_rate_mean": float(np.mean(sr_vals)),
                "success_rate_std": float(np.std(sr_vals)) if len(sr_vals) > 1 else 0.0,
                "mean_cost_units_mean": float(np.mean(cost_vals)),
                "mean_cost_units_std": float(np.std(cost_vals)) if len(cost_vals) > 1 else 0.0,
            })

        frontier[domain] = {
            "tradeoff_curve": tradeoff_curve,
            "comparators": {k: v for k, v in comp_runs.items()},
        }

    return frontier


def render_pareto_markdown(sweep_result: Dict, frontier: Dict) -> str:
    domains = sweep_result["domains"]
    seeds = sweep_result["seeds"]

    lines = [f"# Guard-Band Pareto Sweep\n"]
    lines.append(f"**Protocol**: `{sweep_result['protocol']}`  |  **Epoch**: `{sweep_result['protocol_epoch']}`  |  "
                 f"**Tasks**: {sweep_result['num_tasks']}  |  **Seeds**: {seeds}\n")

    for domain in domains:
        lines.append(f"## Domain: {domain}\n")

        fd = frontier[domain]
        curve = fd["tradeoff_curve"]
        comps = fd["comparators"]

        # Table: guard_band sweep + comparators
        lines.append("### Success Rate vs Guard Band\n")
        all_rows = []
        for row in curve:
            all_rows.append((row["low_signal_guard_band"], row["monitor_label"],
                            row["success_rate_mean"], row["success_rate_std"], None))
        for label, r in comps.items():
            all_rows.append((r["low_signal_guard_band"], label,
                            r["success_rate"], 0.0, None))

        all_rows.sort(key=lambda x: (x[2] if x[2] is not None else -1, x[0]))
        lines.append(f"| Guard Band | Monitor | Success Rate | Mean Cost |")
        lines.append("|---|---|---|---|")
        for band, label, sr, sr_std, _ in all_rows:
            if sr is None:
                continue
            cost = None
            for row in curve:
                if row["low_signal_guard_band"] == band:
                    cost = row["mean_cost_units_mean"]
            if label in comps:
                cost = comps[label].get("mean_cost_units")
            cost_str = f"{cost:.2f}" if cost is not None else "—"
            sr_str = f"{sr:.2f}"
            sr_extra = f"±{sr_std:.2f}" if sr_std else ""
            lines.append(f"| {band:.2f} | {label} | {sr_str}{sr_extra} | {cost_str} |")

        # Pareto frontier: maximal success at minimal cost
        lines.append("\n### Pareto Frontier (max success, min cost)\n")
        pareto = []
        for row in curve:
            sr = row["success_rate_mean"]
            cost = row["mean_cost_units_mean"]
            dominated = any(
                other["success_rate_mean"] >= sr and other["mean_cost_units_mean"] <= cost
                and (other["success_rate_mean"] > sr or other["mean_cost_units_mean"] < cost)
                for other in curve
                if other != row
            )
            if not dominated:
                pareto.append(row)

        if pareto:
            pareto.sort(key=lambda x: x["low_signal_guard_band"])
            lines.append(f"| Guard Band | Success Rate | Mean Cost |")
            lines.append("|---|---|---|")
            for row in pareto:
                lines.append(f"| {row['low_signal_guard_band']:.2f} | {row['success_rate_mean']:.2f} | {row['mean_cost_units_mean']:.2f} |")

        # Comparator summary
        lines.append("\n### Comparator Baselines\n")
        lines.append(f"| Monitor | Success Rate | Mean Cost |")
        lines.append("|---|---|---|")
        for label, r in comps.items():
            sr = r.get("success_rate")
            cost = r.get("mean_cost_units")
            lines.append(f"| {label} | {(f'{sr:.2f}' if sr is not None else '—')} | {(f'{cost:.2f}' if cost is not None else '—')} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="A5/Phase 2: Guard-Band Pareto Sweep")
    parser.add_argument("--protocol", type=str, default="monitor_gate",
                        choices=["monitor_gate", "monitor_repair_triage"])
    parser.add_argument("--seeds", type=str, default="7,11,17")
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated domains (default: standard,stronger_paraphrase,naturalized)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    domains = args.domains.split(",") if args.domains else ALL_DOMAINS

    print(f"=== Guard-Band Pareto Sweep ===")
    print(f"Protocol: {args.protocol}  epoch: {PROTOCOL_EPOCH}")
    print(f"Tasks: {args.num_tasks}  Seeds: {seeds}")
    print(f"Domains: {domains}")
    print(f"Guard bands: {GUARD_BANDS[0]:.2f} to {GUARD_BANDS[-1]:.2f} step 0.02")
    print(f"Comparators: semantic+guard@0.15, counterfactual, confidence")

    t0 = time.time()
    sweep_result = run_sweep(args.protocol, args.num_tasks, seeds, domains)
    elapsed = time.time() - t0
    print(f"\nSweep done in {elapsed:.1f}s")

    frontier = build_frontier(sweep_result)

    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"pareto_{args.protocol}_{ts}"

    sweep_path = RESULTS_DIR / f"{stem}.json"
    frontier_path = RESULTS_DIR / f"pareto_frontier_{ts}.json"
    md_path = RESULTS_DIR / f"{stem}.md"

    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_result, f, indent=2, default=_json_default)
    print(f"Sweep JSON: {sweep_path}")

    with open(frontier_path, "w", encoding="utf-8") as f:
        json.dump({"protocol": args.protocol, "protocol_epoch": PROTOCOL_EPOCH,
                   "num_tasks": args.num_tasks, "seeds": seeds, "domains": domains,
                   "guard_bands": GUARD_BANDS, "frontier": frontier}, f, indent=2, default=_json_default)
    print(f"Frontier JSON: {frontier_path}")

    md_text = render_pareto_markdown(sweep_result, frontier)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"MD: {md_path}")

    # Update latest
    for name, data in [("pareto_latest", sweep_result), ("pareto_frontier_latest", frontier)]:
        p = RESULTS_DIR / f"{name}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)
    latest_md = RESULTS_DIR / "pareto_latest.md"
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Latest: {RESULTS_DIR / 'pareto_latest.json'}")


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
