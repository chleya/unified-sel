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
            "avg_route_l2": 0.0,
            "avg_route_cosine": 1.0,
            "event_counts": {
                "reinforce": 0,
                "branch": 0,
                "create": 0,
                "boundary_stabilize": 0,
                "reinforce_suppressed": 0,
            },
            "avg_structures": 0.0,
            "max_structures": 0,
        }

    event_counts = {
        "reinforce": 0,
        "branch": 0,
        "create": 0,
        "boundary_stabilize": 0,
        "reinforce_suppressed": 0,
    }
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
        "avg_route_l2": float(np.mean([row.get("route_l2", 0.0) for row in window_steps])),
        "avg_route_cosine": float(np.mean([row.get("route_cosine", 1.0) for row in window_steps])),
        "avg_structures": float(np.mean([row["n_structures"] for row in window_steps])),
        "max_structures": int(max(row["n_structures"] for row in window_steps)),
        "event_counts": event_counts,
    }


def run_seed(
    seed: int,
    config: NoBoundaryConfig,
    window_size: int = 50,
    ewc_lambda: float = 0.0,
    anchor_lambda: float = 0.0,
    dual_path_alpha: float = 0.0,
    snapshot_expert: bool = False,
    snapshot_surprise_threshold: float = 0.5,
    best_snapshot: bool = False,
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
        ewc_lambda=ewc_lambda,
        readout_mode=config.readout_mode,
        shared_readout_scale=config.shared_readout_scale,
        shared_readout_post_checkpoint_scale=config.shared_readout_post_checkpoint_scale,
        local_readout_lr_scale=config.local_readout_lr_scale,
        local_readout_start_step=config.local_readout_start_step,
        local_readout_surprise_threshold=config.local_readout_surprise_threshold,
        local_readout_young_age_max=config.local_readout_young_age_max,
        local_readout_training_events=config.local_readout_training_events,
        local_readout_inference_surprise_threshold=config.local_readout_inference_surprise_threshold,
        local_readout_episode_events=config.local_readout_episode_events,
        local_readout_episode_window_steps=config.local_readout_episode_window_steps,
        local_readout_pressure_window_steps=config.local_readout_pressure_window_steps,
        anchor_lambda=anchor_lambda,
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
    best_task_0_acc = 0.0
    best_snapshot_state = None

    for step in range(config.steps):
        if step < config.checkpoint_step:
            progress = 0.0
        else:
            progress = (step - config.checkpoint_step) / max(config.steps - config.checkpoint_step - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        loss = clf.fit_one(x, y)
        stats = clf.get_stats()
        event = clf._history[-1]["event"] if clf._history else ""
        surprise = float(clf._history[-1]["surprise"]) if clf._history else 0.0
        route_l2 = float(clf._history[-1].get("route_l2", 0.0)) if clf._history else 0.0
        route_cosine = float(clf._history[-1].get("route_cosine", 1.0)) if clf._history else 1.0
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
                "pressure_active": bool(clf._history[-1].get("pressure_active", False)) if clf._history else False,
                "persistent_pressure": bool(clf._history[-1].get("persistent_pressure", False)) if clf._history else False,
                "high_pressure": bool(clf._history[-1].get("high_pressure", False)) if clf._history else False,
                "can_boundary_stabilize": bool(clf._history[-1].get("can_boundary_stabilize", False)) if clf._history else False,
                "route_l2": route_l2,
                "route_cosine": route_cosine,
                "avg_tension": float(stats.get("avg_tension", 0.0)),
                "avg_utility": float(stats.get("avg_utility", 0.0)),
                "total_clones": int(stats.get("total_clones", 0)),
                "total_prunes": int(stats.get("total_prunes", 0)),
            }
        )

        if step + 1 >= next_window_cutoff:
            window_summaries.append(_window_summary(step_trace[-window_size:]))
            next_window_cutoff += window_size

        if best_snapshot and step < config.checkpoint_step and (step + 1) % 25 == 0:
            current_t0_acc = clf.accuracy(X_task_0, y_task_0)
            if current_t0_acc > best_task_0_acc:
                best_task_0_acc = current_t0_acc
                import copy
                best_snapshot_state = {
                    "W_out": clf.W_out.copy(),
                    "structures": [],
                }
                for s in clf.pool.structures:
                    best_snapshot_state["structures"].append({
                        "id": s.id,
                        "weights": s.weights.copy(),
                        "feedback": s.feedback.copy(),
                        "local_readout": s.local_readout.copy() if s.local_readout is not None else None,
                    })

        if step + 1 == config.checkpoint_step:
            task_0_accuracy_after_early_stream = clf.accuracy(X_task_0, y_task_0)
            if ewc_lambda > 0 and not clf.fisher_estimated:
                clf.estimate_w_out_fisher(X_task_0, y_task_0)
            if anchor_lambda > 0:
                n_anchored = clf.pool.set_anchors(X_task_0, y_task_0, out_size=config.out_size, min_age=10)
                print(f"  [checkpoint] Set {n_anchored} anchors (pool size: {len(clf.pool.structures)})")
            if dual_path_alpha > 0:
                clf.activate_dual_path(alpha=dual_path_alpha)
                print(f"  [checkpoint] Activated dual-path readout (alpha={dual_path_alpha})")
            if snapshot_expert:
                if best_snapshot and best_snapshot_state is not None:
                    clf._snapshot_experts.append(best_snapshot_state)
                    clf._snapshot_confidence_ratio_threshold = snapshot_surprise_threshold
                    print(f"  [checkpoint] Created BEST snapshot expert (best_t0={best_task_0_acc:.4f}, current_t0={task_0_accuracy_after_early_stream:.4f}, total: {len(clf._snapshot_experts)})")
                else:
                    clf.snapshot_expert(confidence_ratio_threshold=snapshot_surprise_threshold)
                    print(f"  [checkpoint] Created snapshot expert (total: {len(clf._snapshot_experts)})")
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
    
    oracle_task_0_acc = 0.0
    oracle_task_1_acc = 0.0
    if snapshot_expert and clf._snapshot_experts:
        correct_t0 = 0
        for i in range(len(X_task_0)):
            x = X_task_0[i]
            y = int(y_task_0[i])
            boundary = x[0] + x[1]
            if boundary > 0.0:
                pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            else:
                pred = int(np.argmax(clf.predict_proba_single(x)))
            if pred == y:
                correct_t0 += 1
        oracle_task_0_acc = correct_t0 / len(X_task_0)
        
        correct_t1 = 0
        for i in range(len(X_task_1)):
            x = X_task_1[i]
            y = int(y_task_1[i])
            boundary = x[0] + x[1]
            if boundary > 0.0:
                pred = int(np.argmax(clf._predict_with_snapshot(x, clf._snapshot_experts[0])))
            else:
                pred = int(np.argmax(clf.predict_proba_single(x)))
            if pred == y:
                correct_t1 += 1
        oracle_task_1_acc = correct_t1 / len(X_task_1)
    
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
        "oracle_task_0_accuracy": oracle_task_0_acc,
        "oracle_task_1_accuracy": oracle_task_1_acc,
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
                "avg_route_l2": window["avg_route_l2"],
                "avg_route_cosine": window["avg_route_cosine"],
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

    aggregate_events = {
        "reinforce": 0,
        "branch": 0,
        "create": 0,
        "boundary_stabilize": 0,
        "reinforce_suppressed": 0,
    }
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


# ------------------------------------------------------------------
# TopoMem Fusion Variant
# ------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


def run_seed_with_topo(
    seed: int,
    config: NoBoundaryConfig,
    window_size: int = 50,
    use_topo_health: bool = True,
    topo_health_interval: int = 50,
) -> Dict:
    """Run no_boundary experiment with TopoMem ECU health integration.

    Key differences from run_seed():
    - Uses TopoMemUnifiedClassifier instead of UnifiedSELClassifier
    - Runs a minimal TopoMem system for health monitoring
    - Health signals modulate pool lifecycle decisions
    """
    from core.learner import TopoMemUnifiedClassifier

    rng = np.random.default_rng(seed)

    clf = TopoMemUnifiedClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        pool_config=config.pool.to_pool_kwargs(),
        seed=seed,
        use_topo_health=use_topo_health,
        topo_health_update_interval=topo_health_interval,
    )

    topo_system = None
    if use_topo_health:
        try:
            from topomem.config import TopoMemConfig, EmbeddingConfig, TopologyConfig, MemoryConfig
            from topomem.system import TopoMemSystem

            topo_config = TopoMemConfig(
                embedding=EmbeddingConfig(),
                topology=TopologyConfig(max_homology_dim=2),
                memory=MemoryConfig(
                    topo_recompute_interval=10,
                    max_nodes=100,
                ),
            )
            topo_system = TopoMemSystem(topo_config)
        except Exception as e:
            logger.warning(f"TopoMem not available, running without health integration: {e}")
            use_topo_health = False

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
        loss = clf.fit_one(x, y, topo_system=topo_system)
        stats = clf.get_stats()
        event = clf._history[-1]["event"] if clf._history else ""
        surprise = float(clf._history[-1]["surprise"]) if clf._history else 0.0
        route_l2 = float(clf._history[-1].get("route_l2", 0.0)) if clf._history else 0.0
        route_cosine = float(clf._history[-1].get("route_cosine", 1.0)) if clf._history else 1.0
        loss_trace.append(loss)
        structure_trace.append(stats["n_structures"])

        health_info = {}
        if use_topo_health and topo_system is not None:
            try:
                dash = topo_system.get_health_dashboard()
                current = dash.get("current", {})
                health_info = {
                    "health_score": current.get("health_score", 1.0),
                    "h1_health": current.get("h1_health", 1.0),
                    "h2_health": current.get("h2_health", 1.0),
                }
            except Exception:
                pass

        step_trace.append(
            {
                "step": step + 1,
                "progress": float(progress),
                "loss": float(loss),
                "surprise": surprise,
                "event": event,
                "n_structures": stats["n_structures"],
                "route_l2": route_l2,
                "route_cosine": route_cosine,
                "avg_tension": float(stats.get("avg_tension", 0.0)),
                "avg_utility": float(stats.get("avg_utility", 0.0)),
                "total_clones": int(stats.get("total_clones", 0)),
                "total_prunes": int(stats.get("total_prunes", 0)),
                **health_info,
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
                    **health_info,
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
        "use_topo_health": use_topo_health,
        "topo_health_interval": topo_health_interval,
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
                "avg_route_l2": window["avg_route_l2"],
                "avg_route_cosine": window["avg_route_cosine"],
                "event_counts": window["event_counts"],
                "avg_structures": window["avg_structures"],
                "max_structures": window["max_structures"],
            }
            for idx, window in enumerate(window_summaries)
        ],
        "checkpoint_metrics": checkpoint_metrics,
    }


def run_experiment_with_topo(
    num_seeds: int = 5,
    max_structures: int = 12,
    window_size: int = 50,
    steps: int = 600,
    checkpoint_step: int = 200,
    use_topo_health: bool = True,
    topo_health_interval: int = 50,
    compare_baseline: bool = True,
) -> Dict:
    """Run no_boundary experiment WITH TopoMem ECU integration.

    This is the primary experiment for Phase 4 fusion prototype.
    Set compare_baseline=True to compare against non-topo baseline.
    """
    config = NoBoundaryConfig()
    config.steps = steps
    config.checkpoint_step = checkpoint_step
    config.seeds = [7 + i for i in range(num_seeds)]
    config.pool.max_structures = max_structures

    runs = [
        run_seed_with_topo(
            seed=seed,
            config=config,
            window_size=window_size,
            use_topo_health=use_topo_health,
            topo_health_interval=topo_health_interval,
        )
        for seed in config.seeds
    ]
    summary = build_summary(runs)

    comparison = {}
    if compare_baseline:
        baseline_runs = [
            run_seed_with_topo(
                seed=seed,
                config=config,
                window_size=window_size,
                use_topo_health=False,
            )
            for seed in config.seeds
        ]
        baseline_summary = build_summary(baseline_runs)
        comparison = {
            "baseline_task_0_final": baseline_summary["task_0_accuracy_final"],
            "baseline_task_1_final": baseline_summary["task_1_final"],
            "baseline_forgetting": baseline_summary["forgetting_task_0"],
            "topo_task_0_final": summary["task_0_accuracy_final"],
            "topo_task_1_final": summary["task_1_final"],
            "topo_forgetting": summary["forgetting_task_0"],
            "improvement_task_0": (
                summary["task_0_accuracy_final"]["mean"] - baseline_summary["task_0_accuracy_final"]["mean"]
            ),
            "improvement_forgetting": (
                baseline_summary["forgetting_task_0"]["mean"] - summary["forgetting_task_0"]["mean"]
            ),
        }

    checkpoint_summary = []
    for run in runs:
        checkpoint_summary.extend(run.get("checkpoint_metrics", []))

    result = {
        "experiment": "no_boundary_topo_fusion",
        "num_seeds": num_seeds,
        "max_structures": max_structures,
        "window_size": window_size,
        "steps": steps,
        "checkpoint_step": checkpoint_step,
        "use_topo_health": use_topo_health,
        "topo_health_interval": topo_health_interval,
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
        "comparison": comparison,
        "checkpoint_summary": checkpoint_summary,
        "runs": runs,
    }

    results_dir = get_results_path("continual_no_boundary_topo")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(result, output_path)
    result["saved_to"] = str(output_path)
    return result


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
    ewc_lambda: float = 0.0,
    start_seed: int = DEFAULT_CONFIG.seeds[0],
    readout_mode: str = DEFAULT_CONFIG.readout_mode,
    shared_readout_scale: float = DEFAULT_CONFIG.shared_readout_scale,
    shared_readout_post_checkpoint_scale: float = DEFAULT_CONFIG.shared_readout_post_checkpoint_scale,
    local_readout_lr_scale: float = DEFAULT_CONFIG.local_readout_lr_scale,
    local_readout_start_step: int = DEFAULT_CONFIG.local_readout_start_step,
    local_readout_surprise_threshold: float | None = DEFAULT_CONFIG.local_readout_surprise_threshold,
    local_readout_young_age_max: int | None = DEFAULT_CONFIG.local_readout_young_age_max,
    local_readout_training_events: List[str] | None = DEFAULT_CONFIG.local_readout_training_events,
    local_readout_inference_surprise_threshold: float | None = DEFAULT_CONFIG.local_readout_inference_surprise_threshold,
    local_readout_episode_events: List[str] | None = DEFAULT_CONFIG.local_readout_episode_events,
    local_readout_episode_window_steps: int = DEFAULT_CONFIG.local_readout_episode_window_steps,
    local_readout_pressure_window_steps: int = DEFAULT_CONFIG.local_readout_pressure_window_steps,
) -> Dict:
    config = NoBoundaryConfig()
    config.steps = steps
    config.checkpoint_step = checkpoint_step
    config.seeds = [start_seed + i for i in range(num_seeds)]
    config.readout_mode = readout_mode
    config.shared_readout_scale = shared_readout_scale
    config.shared_readout_post_checkpoint_scale = shared_readout_post_checkpoint_scale
    config.local_readout_lr_scale = local_readout_lr_scale
    config.local_readout_start_step = local_readout_start_step
    config.local_readout_surprise_threshold = local_readout_surprise_threshold
    config.local_readout_young_age_max = local_readout_young_age_max
    config.local_readout_training_events = local_readout_training_events
    config.local_readout_inference_surprise_threshold = local_readout_inference_surprise_threshold
    config.local_readout_episode_events = local_readout_episode_events
    config.local_readout_episode_window_steps = local_readout_episode_window_steps
    config.local_readout_pressure_window_steps = local_readout_pressure_window_steps
    config.pool.max_structures = max_structures

    runs = [run_seed(seed=seed, config=config, window_size=window_size, ewc_lambda=ewc_lambda) for seed in config.seeds]
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
        "ewc_lambda": ewc_lambda,
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
        "--start-seed",
        type=int,
        default=DEFAULT_CONFIG.seeds[0],
        help="First random seed to run",
    )
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
    # TopoMem fusion args
    parser.add_argument(
        "--topo-mode",
        action="store_true",
        default=False,
        help="Run with TopoMem ECU health integration (Phase 4 fusion)",
    )
    parser.add_argument(
        "--topo-health-interval",
        type=int,
        default=50,
        help="Steps between TopoMem health updates",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        default=False,
        help="Skip baseline comparison in topo mode",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=0.0,
        help="EWC lambda for W_out Fisher protection (0=disabled)",
    )
    parser.add_argument(
        "--readout-mode",
        type=str,
        default=DEFAULT_CONFIG.readout_mode,
        choices=["shared", "hybrid_local", "exclusive_local"],
        help="Readout path: shared global head or shared+local residual head",
    )
    parser.add_argument(
        "--shared-readout-scale",
        type=float,
        default=DEFAULT_CONFIG.shared_readout_scale,
        help="Scale for the shared W_out contribution",
    )
    parser.add_argument(
        "--shared-readout-post-checkpoint-scale",
        type=float,
        default=DEFAULT_CONFIG.shared_readout_post_checkpoint_scale,
        help="Scale for shared W_out updates after local readout becomes active",
    )
    parser.add_argument(
        "--local-readout-lr-scale",
        type=float,
        default=DEFAULT_CONFIG.local_readout_lr_scale,
        help="Learning-rate scale for per-structure local readout heads",
    )
    parser.add_argument(
        "--local-readout-start-step",
        type=int,
        default=DEFAULT_CONFIG.local_readout_start_step,
        help="Step from which local readout heads become active",
    )
    parser.add_argument(
        "--local-readout-surprise-threshold",
        type=float,
        default=DEFAULT_CONFIG.local_readout_surprise_threshold,
        help="Only use local readout when best-match surprise exceeds this threshold",
    )
    parser.add_argument(
        "--local-readout-young-age-max",
        type=int,
        default=DEFAULT_CONFIG.local_readout_young_age_max,
        help="Only use local readout when the routed structure age is at most this value",
    )
    parser.add_argument(
        "--local-readout-training-events",
        nargs="*",
        default=DEFAULT_CONFIG.local_readout_training_events,
        help="Optional training events that allow local-readout learning",
    )
    parser.add_argument(
        "--local-readout-inference-surprise-threshold",
        type=float,
        default=DEFAULT_CONFIG.local_readout_inference_surprise_threshold,
        help="Optional stricter surprise threshold for local-readout use at inference",
    )
    parser.add_argument(
        "--local-readout-episode-events",
        nargs="*",
        default=DEFAULT_CONFIG.local_readout_episode_events,
        help="Optional events that open a temporary local-readout episode window",
    )
    parser.add_argument(
        "--local-readout-episode-window-steps",
        type=int,
        default=DEFAULT_CONFIG.local_readout_episode_window_steps,
        help="Number of steps a structure keeps local-readout eligibility after an episode event",
    )
    parser.add_argument(
        "--local-readout-pressure-window-steps",
        type=int,
        default=DEFAULT_CONFIG.local_readout_pressure_window_steps,
        help="Number of steps the current pressured route keeps local-readout eligibility",
    )
    args = parser.parse_args()

    if args.topo_mode:
        result = run_experiment_with_topo(
            num_seeds=args.seeds,
            max_structures=args.max_structures,
            window_size=args.window_size,
            steps=args.steps,
            checkpoint_step=args.checkpoint_step,
            use_topo_health=True,
            topo_health_interval=args.topo_health_interval,
            compare_baseline=not args.no_compare,
        )
    else:
        result = run_experiment_with_window(
            num_seeds=args.seeds,
            max_structures=args.max_structures,
            window_size=args.window_size,
            steps=args.steps,
            checkpoint_step=args.checkpoint_step,
            ewc_lambda=args.ewc_lambda,
            start_seed=args.start_seed,
            readout_mode=args.readout_mode,
            shared_readout_scale=args.shared_readout_scale,
            shared_readout_post_checkpoint_scale=args.shared_readout_post_checkpoint_scale,
            local_readout_lr_scale=args.local_readout_lr_scale,
            local_readout_start_step=args.local_readout_start_step,
            local_readout_surprise_threshold=args.local_readout_surprise_threshold,
            local_readout_young_age_max=args.local_readout_young_age_max,
            local_readout_training_events=args.local_readout_training_events,
            local_readout_inference_surprise_threshold=args.local_readout_inference_surprise_threshold,
            local_readout_episode_events=args.local_readout_episode_events,
            local_readout_episode_window_steps=args.local_readout_episode_window_steps,
            local_readout_pressure_window_steps=args.local_readout_pressure_window_steps,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
