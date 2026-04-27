"""
Diagnose high-variance seeds from cleaned-route no-boundary runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import get_results_path, load_json, save_json, timestamp


def latest_json(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON result files found in {directory}")
    return candidates[-1]


def summarize_windows(windows: List[Dict]) -> Dict:
    if not windows:
        return {
            "count": 0,
            "avg_loss": 0.0,
            "avg_surprise": 0.0,
            "avg_route_l2": 0.0,
            "avg_route_cosine": 0.0,
        }
    return {
        "count": len(windows),
        "avg_loss": float(np.mean([row["avg_loss"] for row in windows])),
        "avg_surprise": float(np.mean([row["avg_surprise"] for row in windows])),
        "avg_route_l2": float(np.mean([row["avg_route_l2"] for row in windows])),
        "avg_route_cosine": float(np.mean([row["avg_route_cosine"] for row in windows])),
    }


def phase_for_step(step: int, checkpoint_step: int, total_steps: int) -> str:
    if step <= checkpoint_step:
        return "early"
    if step <= ((checkpoint_step + total_steps) // 2):
        return "mid"
    return "late"


def event_counts_by_phase(run: Dict, checkpoint_step: int, total_steps: int) -> Dict[str, Dict[str, int]]:
    counts = {
        "early": {"reinforce": 0, "branch": 0, "create": 0, "boundary_stabilize": 0},
        "mid": {"reinforce": 0, "branch": 0, "create": 0, "boundary_stabilize": 0},
        "late": {"reinforce": 0, "branch": 0, "create": 0, "boundary_stabilize": 0},
    }
    for row in run.get("step_trace", []):
        phase = phase_for_step(int(row["step"]), checkpoint_step, total_steps)
        event = str(row.get("event", ""))
        if event in counts[phase]:
            counts[phase][event] += 1
    return counts


def diagnose_run(run: Dict, checkpoint_step: int, total_steps: int) -> Dict:
    windows = run.get("window_summaries", [])
    early_windows = [row for row in windows if int(row["step_end"]) <= checkpoint_step]
    mid_windows = [row for row in windows if checkpoint_step < int(row["step_end"]) <= ((checkpoint_step + total_steps) // 2)]
    late_windows = [row for row in windows if int(row["step_end"]) > ((checkpoint_step + total_steps) // 2)]
    phase_windows = {
        "early": summarize_windows(early_windows),
        "mid": summarize_windows(mid_windows),
        "late": summarize_windows(late_windows),
    }

    phase_events = event_counts_by_phase(run, checkpoint_step, total_steps)
    final_stats = run.get("final_stats", {})
    avg_accuracy = 0.5 * (
        float(run.get("task_0_accuracy_final", 0.0))
        + float(run.get("task_1_accuracy_final", 0.0))
    )
    forgetting = float(run.get("forgetting_task_0", 0.0))
    task1 = float(run.get("task_1_accuracy_final", 0.0))

    diagnosis = "balanced"
    if forgetting > 0.0 and task1 < 0.35:
        if phase_events["late"]["boundary_stabilize"] >= 10:
            diagnosis = "retention_bias_under_late_stabilization"
        elif phase_events["mid"]["branch"] + phase_events["mid"]["create"] >= 4:
            diagnosis = "mid_phase_churn_without_recovery"
        else:
            diagnosis = "retention_bias_without_task1_adaptation"
    elif forgetting <= 0.0 and task1 < 0.25:
        diagnosis = "task0_locked_task1_sacrificed"
    elif forgetting <= 0.0 and task1 >= 0.45:
        diagnosis = "balanced_transfer"
    elif forgetting > 0.0 and task1 >= 0.45:
        diagnosis = "task1_adapts_but_task0_decays"

    return {
        "seed": int(run["seed"]),
        "avg_accuracy": avg_accuracy,
        "forgetting_task_0": forgetting,
        "task_0_accuracy_after_early_stream": float(run["task_0_accuracy_after_early_stream"]),
        "task_0_accuracy_final": float(run["task_0_accuracy_final"]),
        "task_1_accuracy_final": task1,
        "recent_mean_loss": float(run.get("recent_mean_loss", 0.0)),
        "recent_route_l2": float(final_stats.get("recent_route_l2", 0.0)),
        "recent_route_cosine": float(final_stats.get("recent_route_cosine", 1.0)),
        "phase_windows": phase_windows,
        "phase_events": phase_events,
        "diagnosis": diagnosis,
    }


def aggregate_seed_reports(seed_reports: List[Dict]) -> Dict:
    forgettings = np.asarray([row["forgetting_task_0"] for row in seed_reports], dtype=float)
    task1 = np.asarray([row["task_1_accuracy_final"] for row in seed_reports], dtype=float)
    late_route_l2 = np.asarray([row["phase_windows"]["late"]["avg_route_l2"] for row in seed_reports], dtype=float)
    late_route_cosine = np.asarray([row["phase_windows"]["late"]["avg_route_cosine"] for row in seed_reports], dtype=float)

    return {
        "num_runs": len(seed_reports),
        "means": {
            "forgetting_task_0": float(np.mean(forgettings)),
            "task_1_accuracy_final": float(np.mean(task1)),
            "late_avg_route_l2": float(np.mean(late_route_l2)),
            "late_avg_route_cosine": float(np.mean(late_route_cosine)),
        },
        "high_forgetting_seeds": [
            row["seed"] for row in seed_reports if row["forgetting_task_0"] > float(np.mean(forgettings))
        ],
        "low_task1_seeds": [
            row["seed"] for row in seed_reports if row["task_1_accuracy_final"] < float(np.mean(task1))
        ],
        "diagnosis_counts": {
            label: sum(1 for row in seed_reports if row["diagnosis"] == label)
            for label in sorted({row["diagnosis"] for row in seed_reports})
        },
    }


def compare_inputs(payloads: List[Dict]) -> Dict:
    if len(payloads) < 2:
        return {"seed_comparisons": []}

    seed_maps: List[Dict[int, Dict]] = []
    labels: List[str] = []
    for payload in payloads:
        labels.append(payload["label"])
        seed_maps.append({row["seed"]: row for row in payload["seed_reports"]})

    shared_seeds = sorted(set.intersection(*[set(seed_map.keys()) for seed_map in seed_maps]))
    seed_comparisons = []
    for seed in shared_seeds:
        per_label = {}
        for label, seed_map in zip(labels, seed_maps):
            row = seed_map[seed]
            per_label[label] = {
                "avg_accuracy": row["avg_accuracy"],
                "forgetting_task_0": row["forgetting_task_0"],
                "task_1_accuracy_final": row["task_1_accuracy_final"],
                "late_avg_route_l2": row["phase_windows"]["late"]["avg_route_l2"],
                "late_avg_route_cosine": row["phase_windows"]["late"]["avg_route_cosine"],
                "late_boundary_stabilize": row["phase_events"]["late"]["boundary_stabilize"],
                "late_churn": row["phase_events"]["late"]["branch"] + row["phase_events"]["late"]["create"],
                "diagnosis": row["diagnosis"],
            }

        baseline = per_label[labels[0]]
        current = per_label[labels[-1]]
        seed_comparisons.append(
            {
                "seed": seed,
                "per_input": per_label,
                "delta_last_minus_first": {
                    "avg_accuracy": current["avg_accuracy"] - baseline["avg_accuracy"],
                    "forgetting_task_0": current["forgetting_task_0"] - baseline["forgetting_task_0"],
                    "task_1_accuracy_final": current["task_1_accuracy_final"] - baseline["task_1_accuracy_final"],
                    "late_avg_route_l2": current["late_avg_route_l2"] - baseline["late_avg_route_l2"],
                    "late_avg_route_cosine": current["late_avg_route_cosine"] - baseline["late_avg_route_cosine"],
                    "late_boundary_stabilize": current["late_boundary_stabilize"] - baseline["late_boundary_stabilize"],
                    "late_churn": current["late_churn"] - baseline["late_churn"],
                },
                "lambda_insensitive": (
                    abs(current["forgetting_task_0"] - baseline["forgetting_task_0"]) < 0.01
                    and abs(current["task_1_accuracy_final"] - baseline["task_1_accuracy_final"]) < 0.02
                ),
            }
        )
    return {"seed_comparisons": seed_comparisons}


def load_input(path_str: str) -> Dict:
    path = Path(path_str)
    if path.is_dir():
        path = latest_json(path)
    payload = load_json(path)
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in {path}")

    checkpoint_step = int(payload.get("checkpoint_step", 200))
    total_steps = int(payload.get("steps", 600))
    label = f"lambda_{float(payload.get('ewc_lambda', 0.0)):g}"
    seed_reports = [diagnose_run(run, checkpoint_step, total_steps) for run in runs]
    return {
        "label": label,
        "source_result": str(path),
        "ewc_lambda": float(payload.get("ewc_lambda", 0.0)),
        "num_runs": len(seed_reports),
        "seed_reports": seed_reports,
        "summary": aggregate_seed_reports(seed_reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input no-boundary result files or directories",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional output path")
    args = parser.parse_args()

    reports = [load_input(item) for item in args.inputs]
    output = {
        "experiment": "variance_diagnostics",
        "inputs": reports,
        "comparison": compare_inputs(reports),
    }

    output_dir = get_results_path("analysis_variance")
    output_path = Path(args.output) if args.output else output_dir / f"{timestamp()}.json"
    save_json(output, output_path)

    print(json.dumps(output, indent=2))
    print(f"\nSaved variance diagnostics: {output_path}")


if __name__ == "__main__":
    main()
