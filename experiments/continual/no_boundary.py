"""
Unified-SEL no-boundary continual-learning experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_config import NoBoundaryConfig
from core.learner import UnifiedSELClassifier
from core.runtime import get_results_path, save_json, summarize_runs, timestamp


DEFAULT_CONFIG = NoBoundaryConfig()


def make_eval_task(
    task_id: int,
    n_samples: int,
    seed: int,
    in_size: int = DEFAULT_CONFIG.in_size,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, in_size))
    boundary = X[:, 0] + X[:, 1]
    if task_id == 0:
        y = (boundary > 0.0).astype(int)
    elif task_id == 1:
        y = (boundary < 0.0).astype(int)
    else:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return X, y


def stream_sample(
    progress: float,
    rng: np.random.Generator,
    in_size: int = DEFAULT_CONFIG.in_size,
) -> tuple[np.ndarray, int]:
    x = rng.normal(0.0, 1.0, size=in_size)
    boundary = float(x[0] + x[1])
    task_0_label = int(boundary > 0.0)
    task_1_label = int(boundary < 0.0)
    y = task_0_label if rng.random() > progress else task_1_label
    return x, y


def _window_summary(window_steps: List[Dict]) -> Dict:
    if not window_steps:
        return {
            "count": 0,
            "avg_loss": 0.0,
            "avg_surprise": 0.0,
            "event_counts": {"reinforce": 0, "branch": 0, "create": 0},
            "avg_structures": 0.0,
            "max_structures": 0,
        }

    event_counts = {"reinforce": 0, "branch": 0, "create": 0}
    for row in window_steps:
        event = row.get("event", "")
        if event in event_counts:
            event_counts[event] += 1

    return {
        "count": len(window_steps),
        "step_start": int(window_steps[0]["step"]),
        "step_end": int(window_steps[-1]["step"]),
        "avg_loss": float(np.mean([row["loss"] for row in window_steps])),
        "avg_surprise": float(np.mean([row["surprise"] for row in window_steps])),
        "avg_structures": float(np.mean([row["n_structures"] for row in window_steps])),
        "max_structures": int(max(row["n_structures"] for row in window_steps)),
        "event_counts": event_counts,
    }


def run_seed(
    seed: int,
    config: NoBoundaryConfig,
    window_size: int = 50,
) -> Dict:
    rng = np.random.default_rng(seed)
    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        pool_config=config.pool.to_pool_kwargs(),
        seed=seed,
    )

    X_task_0, y_task_0 = make_eval_task(
        task_id=0,
        n_samples=config.eval_samples_per_task,
        seed=seed + 1000,
        in_size=config.in_size,
    )
    X_task_1, y_task_1 = make_eval_task(
        task_id=1,
        n_samples=config.eval_samples_per_task,
        seed=seed + 2000,
        in_size=config.in_size,
    )

    structure_trace: List[int] = []
    loss_trace: List[float] = []
    step_trace: List[Dict] = []
    checkpoint_metrics: List[Dict] = []
    window_summaries: List[Dict] = []
    next_window_cutoff = window_size
    task_0_accuracy_after_early_stream = 0.0

    for step in range(config.steps):
        progress = step / max(config.steps - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        loss = clf.fit_one(x, y)
        stats = clf.get_stats()
        event = clf._history[-1]["event"] if clf._history else ""
        surprise = float(clf._history[-1]["surprise"]) if clf._history else 0.0
        loss_trace.append(loss)
        structure_trace.append(stats["n_structures"])
        step_trace.append(
            {
                "step": step + 1,
                "progress": float(progress),
                "loss": float(loss),
                "surprise": surprise,
                "event": event,
                "n_structures": stats["n_structures"],
                "avg_tension": float(stats.get("avg_tension", 0.0)),
                "avg_utility": float(stats.get("avg_utility", 0.0)),
                "total_clones": int(stats.get("total_clones", 0)),
                "total_prunes": int(stats.get("total_prunes", 0)),
            }
        )

        if step + 1 >= next_window_cutoff:
            window_summaries.append(_window_summary(step_trace[-window_size:]))
            next_window_cutoff += window_size

        if step + 1 == config.checkpoint_step:
            task_0_accuracy_after_early_stream = clf.accuracy(X_task_0, y_task_0)
            checkpoint_metrics.append(
                {
                    "step": step + 1,
                    "task_0_accuracy": task_0_accuracy_after_early_stream,
                    "task_1_accuracy": clf.accuracy(X_task_1, y_task_1),
                    "forgetting_task_0": None,
                    "event_counts": clf.get_event_counts(),
                    "stats": clf.get_stats(),
                }
            )

    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)
    event_counts = clf.get_event_counts()
    final_stats = clf.get_stats()

    if checkpoint_metrics:
        checkpoint_metrics[-1]["forgetting_task_0"] = (
            checkpoint_metrics[-1]["task_0_accuracy"] - final_task_0_accuracy
        )

    return {
        "seed": seed,
        "steps": config.steps,
        "checkpoint_step": config.checkpoint_step,
        "max_structures": config.pool.max_structures,
        "window_size": window_size,
        "task_0_accuracy_after_early_stream": task_0_accuracy_after_early_stream,
        "task_0_accuracy_final": final_task_0_accuracy,
        "task_1_accuracy_final": final_task_1_accuracy,
        "forgetting_task_0": task_0_accuracy_after_early_stream - final_task_0_accuracy,
        "event_counts": event_counts,
        "final_stats": final_stats,
        "max_structures_seen": max(structure_trace) if structure_trace else 0,
        "mean_structures_seen": float(np.mean(structure_trace)) if structure_trace else 0.0,
        "recent_mean_loss": float(np.mean(loss_trace[-50:])) if loss_trace else 0.0,
        "step_trace": step_trace,
        "window_summaries": [
            {
                "window_index": idx,
                "step_end": window["step_end"],
                "avg_loss": window["avg_loss"],
                "avg_surprise": window["avg_surprise"],
                "event_counts": window["event_counts"],
                "avg_structures": window["avg_structures"],
                "max_structures": window["max_structures"],
            }
            for idx, window in enumerate(window_summaries)
        ],
        "checkpoint_metrics": checkpoint_metrics,
    }


def build_summary(runs: List[Dict]) -> Dict:
    task_0_final = summarize_runs(runs, key="task_0_accuracy_final")
    task_1_final = summarize_runs(runs, key="task_1_accuracy_final")
    forgetting = summarize_runs(runs, key="forgetting_task_0")
    max_structures = summarize_runs(runs, key="max_structures_seen")

    aggregate_events = {"reinforce": 0, "branch": 0, "create": 0}
    total_clones = 0
    total_prunes = 0
    for run in runs:
        for key in aggregate_events:
            aggregate_events[key] += int(run["event_counts"].get(key, 0))
        total_clones += int(run["final_stats"].get("total_clones", 0))
        total_prunes += int(run["final_stats"].get("total_prunes", 0))

    return {
        "task_0_accuracy_final": task_0_final,
        "task_1_accuracy_final": task_1_final,
        "forgetting_task_0": forgetting,
        "max_structures_seen": max_structures,
        "aggregate_event_counts": aggregate_events,
        "aggregate_total_clones": total_clones,
        "aggregate_total_prunes": total_prunes,
    }


def run_experiment(
    num_seeds: int = len(DEFAULT_CONFIG.seeds),
    max_structures: int = DEFAULT_CONFIG.pool.max_structures,
) -> Dict:
    return run_experiment_with_window(
        num_seeds=num_seeds,
        max_structures=max_structures,
        window_size=50,
    )


def run_experiment_with_window(
    num_seeds: int = len(DEFAULT_CONFIG.seeds),
    max_structures: int = DEFAULT_CONFIG.pool.max_structures,
    window_size: int = 50,
    steps: int = DEFAULT_CONFIG.steps,
    checkpoint_step: int = DEFAULT_CONFIG.checkpoint_step,
) -> Dict:
    config = NoBoundaryConfig()
    config.steps = steps
    config.checkpoint_step = checkpoint_step
    config.seeds = [7 + i for i in range(num_seeds)]
    config.pool.max_structures = max_structures

    runs = [run_seed(seed=seed, config=config, window_size=window_size) for seed in config.seeds]
    summary = build_summary(runs)

    checkpoint_summary = []
    for run in runs:
        checkpoint_summary.extend(run.get("checkpoint_metrics", []))

    result = {
        "experiment": "no_boundary",
        "num_seeds": num_seeds,
        "max_structures": max_structures,
        "window_size": window_size,
        "steps": steps,
        "checkpoint_step": checkpoint_step,
        "seeds": config.seeds,
        "stream_spec": {
            "input_dim": config.in_size,
            "n_eval_samples": config.eval_samples_per_task,
            "drift_protocol": "probabilistic two-rule shift from task 0 to task 1",
            "task_descriptions": {
                "task_0": "boundary = x0 + x1 > 0",
                "task_1": "boundary = x0 + x1 < 0",
            },
        },
        "config": config.to_dict(),
        "summary": summary,
        "checkpoint_summary": checkpoint_summary,
        "runs": runs,
    }

    results_dir = get_results_path("continual_no_boundary")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(result, output_path)
    result["saved_to"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_CONFIG.seeds), help="Number of random seeds to run")
    parser.add_argument(
        "--max-structures",
        type=int,
        default=DEFAULT_CONFIG.pool.max_structures,
        help="Maximum number of structures",
    )
    parser.add_argument("--window-size", type=int, default=50, help="Window size for diagnostic summaries")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG.steps, help="Number of stream steps per seed")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=DEFAULT_CONFIG.checkpoint_step,
        help="Step at which to record checkpoint diagnostics",
    )
    args = parser.parse_args()

    result = run_experiment_with_window(
        num_seeds=args.seeds,
        max_structures=args.max_structures,
        window_size=args.window_size,
        steps=args.steps,
        checkpoint_step=args.checkpoint_step,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
