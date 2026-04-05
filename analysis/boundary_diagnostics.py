"""
Diagnose endogenous boundary formation from no-boundary experiment traces.
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


def phase_name(progress: float) -> str:
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "mid"
    return "late"


def summarize_phase(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "count": 0,
            "avg_loss": 0.0,
            "avg_surprise": 0.0,
            "avg_tension": 0.0,
            "avg_utility": 0.0,
            "avg_structures": 0.0,
            "event_counts": {"reinforce": 0, "branch": 0, "create": 0},
        }

    event_counts = {"reinforce": 0, "branch": 0, "create": 0}
    for row in rows:
        event = row.get("event", "")
        if event in event_counts:
            event_counts[event] += 1

    return {
        "count": len(rows),
        "avg_loss": float(np.mean([row["loss"] for row in rows])),
        "avg_surprise": float(np.mean([row["surprise"] for row in rows])),
        "avg_tension": float(np.mean([row["avg_tension"] for row in rows])),
        "avg_utility": float(np.mean([row["avg_utility"] for row in rows])),
        "avg_structures": float(np.mean([row["n_structures"] for row in rows])),
        "event_counts": event_counts,
    }


def score_window(window: Dict, max_event_total: int) -> float:
    event_counts = window.get("event_counts", {})
    novelty_events = int(event_counts.get("branch", 0)) + int(event_counts.get("create", 0))
    pressure = novelty_events / max(max_event_total, 1)
    return float(
        0.5 * pressure
        + 0.3 * float(window.get("avg_surprise", 0.0))
        + 0.2 * float(window.get("avg_loss", 0.0))
    )


def summarize_boundary_metrics(scored_windows: List[Dict]) -> Dict:
    if not scored_windows:
        return {
            "activation_threshold": 0.0,
            "active_window_count": 0,
            "active_window_ratio": 0.0,
            "episode_count": 0,
            "first_emergence_window_index": None,
            "first_emergence_step_end": None,
            "first_emergence_phase": None,
            "longest_episode_length": 0,
            "longest_episode_phase_span": [],
            "collapse_count": 0,
            "reactivation_count": 0,
            "late_active_window_count": 0,
            "late_reactivation_count": 0,
            "stability_score": 0.0,
            "status": "no_boundary",
        }

    scores = np.asarray([float(window["pressure_score"]) for window in scored_windows], dtype=float)
    threshold = float(max(0.16, np.quantile(scores, 0.75)))

    active_flags: List[bool] = []
    for window in scored_windows:
        event_counts = window.get("event_counts", {})
        novelty_events = int(event_counts.get("branch", 0)) + int(event_counts.get("create", 0))
        is_active = (
            float(window["pressure_score"]) >= threshold
            and (
                novelty_events > 0
                or float(window.get("avg_surprise", 0.0)) >= 0.33
                or float(window.get("avg_loss", 0.0)) >= 0.30
            )
        )
        active_flags.append(bool(is_active))

    episodes: List[Dict] = []
    start_idx = None
    for idx, is_active in enumerate(active_flags):
        if is_active and start_idx is None:
            start_idx = idx
        if not is_active and start_idx is not None:
            episodes.append(_episode_summary(scored_windows, start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        episodes.append(_episode_summary(scored_windows, start_idx, len(scored_windows) - 1))

    active_window_count = int(sum(active_flags))
    late_active_window_count = int(
        sum(
            1
            for idx, is_active in enumerate(active_flags)
            if is_active and scored_windows[idx]["phase"] == "late"
        )
    )

    collapse_count = max(0, len(episodes) - 1)
    reactivation_count = max(0, len(episodes) - 1)
    late_reactivation_count = max(
        0,
        sum(1 for episode in episodes[1:] if "late" in episode["phase_span"]),
    )

    longest_episode = max(episodes, key=lambda row: row["length"]) if episodes else None
    first_episode = episodes[0] if episodes else None

    active_ratio = active_window_count / max(len(scored_windows), 1)
    longest_ratio = (longest_episode["length"] / max(len(scored_windows), 1)) if longest_episode else 0.0
    late_pressure_ratio = late_active_window_count / max(len(scored_windows), 1)
    stability_score = float(
        0.5 * longest_ratio
        + 0.3 * active_ratio
        - 0.2 * min(1.0, reactivation_count / max(len(scored_windows), 1))
        - 0.2 * late_pressure_ratio
    )

    status = "transient"
    if not episodes:
        status = "no_boundary"
    elif len(episodes) == 1 and late_active_window_count == 0 and longest_ratio >= 0.20:
        status = "stable"
    elif late_reactivation_count > 0 or late_active_window_count >= 2:
        status = "recurrent_pressure"

    return {
        "activation_threshold": threshold,
        "active_window_count": active_window_count,
        "active_window_ratio": float(active_ratio),
        "episode_count": len(episodes),
        "first_emergence_window_index": first_episode["start_window_index"] if first_episode else None,
        "first_emergence_step_end": first_episode["start_step_end"] if first_episode else None,
        "first_emergence_phase": first_episode["start_phase"] if first_episode else None,
        "longest_episode_length": int(longest_episode["length"]) if longest_episode else 0,
        "longest_episode_phase_span": longest_episode["phase_span"] if longest_episode else [],
        "collapse_count": collapse_count,
        "reactivation_count": reactivation_count,
        "late_active_window_count": late_active_window_count,
        "late_reactivation_count": late_reactivation_count,
        "stability_score": stability_score,
        "status": status,
        "episodes": episodes,
        "active_window_indices": [
            int(scored_windows[idx]["window_index"])
            for idx, is_active in enumerate(active_flags)
            if is_active
        ],
    }


def _episode_summary(scored_windows: List[Dict], start_idx: int, end_idx: int) -> Dict:
    episode_windows = scored_windows[start_idx : end_idx + 1]
    phase_span: List[str] = []
    for window in episode_windows:
        phase = str(window["phase"])
        if phase not in phase_span:
            phase_span.append(phase)
    return {
        "start_window_index": int(episode_windows[0]["window_index"]),
        "end_window_index": int(episode_windows[-1]["window_index"]),
        "start_step_end": int(episode_windows[0]["step_end"]),
        "end_step_end": int(episode_windows[-1]["step_end"]),
        "start_phase": str(episode_windows[0]["phase"]),
        "end_phase": str(episode_windows[-1]["phase"]),
        "phase_span": phase_span,
        "length": len(episode_windows),
        "mean_pressure_score": float(np.mean([window["pressure_score"] for window in episode_windows])),
    }


def candidate_steps(step_trace: List[Dict], top_k: int = 12) -> List[Dict]:
    if not step_trace:
        return []

    surprises = np.asarray([float(row["surprise"]) for row in step_trace], dtype=float)
    tensions = np.asarray([float(row.get("avg_tension", 0.0)) for row in step_trace], dtype=float)
    surprise_threshold = float(np.quantile(surprises, 0.9))
    tension_threshold = float(np.quantile(tensions, 0.9))

    candidates: List[Dict] = []
    prev_structures = None
    for row in step_trace:
        structures = int(row.get("n_structures", 0))
        event = row.get("event", "")
        structure_jump = prev_structures is not None and structures != prev_structures
        high_surprise = float(row.get("surprise", 0.0)) >= surprise_threshold
        high_tension = float(row.get("avg_tension", 0.0)) >= tension_threshold
        if event != "reinforce" or structure_jump or high_surprise or high_tension:
            candidates.append(
                {
                    "step": int(row["step"]),
                    "progress": float(row["progress"]),
                    "phase": phase_name(float(row["progress"])),
                    "event": event,
                    "surprise": float(row["surprise"]),
                    "avg_tension": float(row.get("avg_tension", 0.0)),
                    "avg_utility": float(row.get("avg_utility", 0.0)),
                    "n_structures": structures,
                    "structure_jump": bool(structure_jump),
                    "high_surprise": bool(high_surprise),
                    "high_tension": bool(high_tension),
                }
            )
        prev_structures = structures

    candidates.sort(
        key=lambda row: (
            row["structure_jump"],
            row["event"] != "reinforce",
            row["high_surprise"],
            row["high_tension"],
            row["surprise"] + row["avg_tension"],
        ),
        reverse=True,
    )
    return candidates[:top_k]


def analyze_run(run: Dict) -> Dict:
    step_trace = run.get("step_trace", [])
    window_summaries = run.get("window_summaries", [])
    if not step_trace:
        raise ValueError(f"Run for seed {run.get('seed')} does not contain step_trace")

    phases = {"early": [], "mid": [], "late": []}
    for row in step_trace:
        phases[phase_name(float(row["progress"]))].append(row)

    phase_summary = {name: summarize_phase(rows) for name, rows in phases.items()}

    max_window_events = 1
    for window in window_summaries:
        event_counts = window.get("event_counts", {})
        total = int(event_counts.get("reinforce", 0)) + int(event_counts.get("branch", 0)) + int(event_counts.get("create", 0))
        max_window_events = max(max_window_events, total)

    scored_windows = []
    for idx, window in enumerate(window_summaries):
        scored = dict(window)
        scored["window_index"] = int(window.get("window_index", idx))
        scored["phase"] = phase_name(float(scored["step_end"]) / max(int(run["steps"]), 1))
        scored["pressure_score"] = score_window(scored, max_window_events)
        scored_windows.append(scored)
    scored_windows.sort(key=lambda row: row["pressure_score"], reverse=True)
    temporal_windows = sorted(scored_windows, key=lambda row: row["window_index"])
    boundary_metrics = summarize_boundary_metrics(temporal_windows)

    early = phase_summary["early"]
    mid = phase_summary["mid"]
    late = phase_summary["late"]
    return {
        "seed": int(run["seed"]),
        "forgetting_task_0": float(run["forgetting_task_0"]),
        "task_0_accuracy_after_early_stream": float(run["task_0_accuracy_after_early_stream"]),
        "task_0_accuracy_final": float(run["task_0_accuracy_final"]),
        "task_1_accuracy_final": float(run["task_1_accuracy_final"]),
        "max_structures_seen": int(run["max_structures_seen"]),
        "phase_summary": phase_summary,
        "transition_deltas": {
            "mid_minus_early_surprise": mid["avg_surprise"] - early["avg_surprise"],
            "late_minus_mid_surprise": late["avg_surprise"] - mid["avg_surprise"],
            "mid_minus_early_tension": mid["avg_tension"] - early["avg_tension"],
            "late_minus_mid_tension": late["avg_tension"] - mid["avg_tension"],
            "late_minus_early_structures": late["avg_structures"] - early["avg_structures"],
        },
        "boundary_metrics": boundary_metrics,
        "top_boundary_windows": scored_windows[:5],
        "boundary_candidate_steps": candidate_steps(step_trace),
    }


def aggregate_report(run_reports: List[Dict]) -> Dict:
    forgettings = np.asarray([report["forgetting_task_0"] for report in run_reports], dtype=float)
    mid_surprise = np.asarray(
        [report["phase_summary"]["mid"]["avg_surprise"] for report in run_reports],
        dtype=float,
    )
    late_surprise = np.asarray(
        [report["phase_summary"]["late"]["avg_surprise"] for report in run_reports],
        dtype=float,
    )
    mid_tension = np.asarray(
        [report["phase_summary"]["mid"]["avg_tension"] for report in run_reports],
        dtype=float,
    )
    late_tension = np.asarray(
        [report["phase_summary"]["late"]["avg_tension"] for report in run_reports],
        dtype=float,
    )
    stability_scores = np.asarray(
        [report["boundary_metrics"]["stability_score"] for report in run_reports],
        dtype=float,
    )
    active_ratios = np.asarray(
        [report["boundary_metrics"]["active_window_ratio"] for report in run_reports],
        dtype=float,
    )
    collapse_counts = np.asarray(
        [report["boundary_metrics"]["collapse_count"] for report in run_reports],
        dtype=float,
    )

    return {
        "num_runs": len(run_reports),
        "forgetting": {
            "mean": float(np.mean(forgettings)),
            "std": float(np.std(forgettings)),
            "min": float(np.min(forgettings)),
            "max": float(np.max(forgettings)),
        },
        "phase_means": {
            "mid_avg_surprise": float(np.mean(mid_surprise)),
            "late_avg_surprise": float(np.mean(late_surprise)),
            "mid_avg_tension": float(np.mean(mid_tension)),
            "late_avg_tension": float(np.mean(late_tension)),
        },
        "boundary_summary": {
            "mean_stability_score": float(np.mean(stability_scores)),
            "mean_active_window_ratio": float(np.mean(active_ratios)),
            "mean_collapse_count": float(np.mean(collapse_counts)),
            "status_counts": {
                status: sum(1 for report in run_reports if report["boundary_metrics"]["status"] == status)
                for status in sorted({report["boundary_metrics"]["status"] for report in run_reports})
            },
        },
        "high_forgetting_seeds": [
            report["seed"]
            for report in run_reports
            if report["forgetting_task_0"] > float(np.mean(forgettings))
        ],
        "high_instability_seeds": [
            report["seed"]
            for report in run_reports
            if report["boundary_metrics"]["status"] != "stable"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Input no-boundary result file or directory")
    parser.add_argument("--output", type=str, default=None, help="Optional output path")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else latest_json(get_results_path("continual_no_boundary"))
    if input_path.is_dir():
        input_path = latest_json(input_path)

    payload = load_json(input_path)
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in {input_path}")

    run_reports = [analyze_run(run) for run in runs]
    report = {
        "source_result": str(input_path),
        "experiment": payload.get("experiment", "no_boundary"),
        "num_runs": len(run_reports),
        "summary": aggregate_report(run_reports),
        "runs": run_reports,
    }

    output_dir = get_results_path("analysis_boundary")
    output_path = Path(args.output) if args.output else output_dir / f"{timestamp()}.json"
    save_json(report, output_path)

    print(json.dumps(report, indent=2))
    print(f"\nSaved boundary diagnostics: {output_path}")


if __name__ == "__main__":
    main()
