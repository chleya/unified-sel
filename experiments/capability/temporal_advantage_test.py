"""
Temporal Advantage Test: PredictiveHealthMonitor vs BatchHealthMonitor

Runs both monitors on the same task stream (code -> reasoning domain shift)
and records which one detects the shift first.

Usage:
    python experiments/capability/temporal_advantage_test.py [--seeds 7 42 123]
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.capability_benchmark import build_task_suite, BatchHealthMonitor
from core.predictive_health import PredictiveHealthMonitor
from topomem.embedding import EmbeddingManager


def encode_task(embed_mgr, task):
    parts = [task.prompt]
    if task.family == "code":
        fn = task.metadata.get("function_name", "")
        if fn:
            parts.append(fn)
    return embed_mgr.encode(" ".join(parts))


def run_temporal_comparison(embed_mgr, seed, n_code=20, n_reasoning=20):
    code_tasks = build_task_suite("code", n_code, seed=seed)
    reasoning_tasks = build_task_suite("reasoning", n_reasoning, seed=seed)
    all_tasks = code_tasks + reasoning_tasks
    shift_point = n_code

    batch_mon = BatchHealthMonitor(window_size=10)
    pred_mon = PredictiveHealthMonitor(
        latent_dim=384,
        window_size=10,
        warmup_windows=2,
        learning_rate=0.05,
    )

    batch_first_alert = None
    pred_first_alert = None

    per_task_results = []

    for i, task in enumerate(all_tasks):
        z = encode_task(embed_mgr, task)

        batch_sig = batch_mon.observe(task)
        pred_sig = pred_mon.observe(z)

        batch_alerting = batch_sig.get("status", "") in (
            "gradual_drift",
            "domain_shift_detected",
        )
        pred_alerting = pred_sig.status in (
            "gradual_drift",
            "domain_shift",
            "anomaly",
        )

        if batch_first_alert is None and batch_alerting:
            batch_first_alert = i
        if pred_first_alert is None and pred_alerting:
            pred_first_alert = i

        per_task_results.append({
            "task_idx": i,
            "family": task.family,
            "is_post_shift": i >= shift_point,
            "batch_drift": batch_sig.get("drift_signal", 0.0),
            "batch_status": batch_sig.get("status", "unknown"),
            "pred_residual": pred_sig.residual_mean,
            "pred_status": pred_sig.status,
        })

    return {
        "seed": seed,
        "shift_point": shift_point,
        "n_tasks": len(all_tasks),
        "batch_first_alert": batch_first_alert,
        "pred_first_alert": pred_first_alert,
        "batch_lead": (
            (shift_point - batch_first_alert)
            if batch_first_alert is not None and batch_first_alert >= shift_point
            else (
                -(batch_first_alert - shift_point)
                if batch_first_alert is not None
                else None
            )
        ),
        "pred_lead": (
            (shift_point - pred_first_alert)
            if pred_first_alert is not None and pred_first_alert >= shift_point
            else (
                -(pred_first_alert - shift_point)
                if pred_first_alert is not None
                else None
            )
        ),
        "per_task": per_task_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123, 256, 999])
    args = parser.parse_args()

    print("=" * 60)
    print("Temporal Advantage Test")
    print("PredictiveHealthMonitor vs BatchHealthMonitor")
    print("=" * 60)
    print(f"Seeds: {args.seeds}")
    print()

    print("[1/2] Loading embedding model...")
    embed_mgr = EmbeddingManager()

    print("[2/2] Running temporal comparison...")
    results = []
    for seed in args.seeds:
        r = run_temporal_comparison(embed_mgr, seed)
        results.append(r)

        batch_alert = r["batch_first_alert"]
        pred_alert = r["pred_first_alert"]
        shift = r["shift_point"]

        batch_label = (
            f"task {batch_alert} ({batch_alert - shift:+d} from shift)"
            if batch_alert is not None
            else "never"
        )
        pred_label = (
            f"task {pred_alert} ({pred_alert - shift:+d} from shift)"
            if pred_alert is not None
            else "never"
        )

        print(f"\n  Seed {seed}:")
        print(f"    Shift point: task {shift}")
        print(f"    BatchHealthMonitor first alert: {batch_label}")
        print(f"    PredictiveHealthMonitor first alert: {pred_label}")

        if batch_alert is not None and pred_alert is not None:
            if pred_alert < batch_alert:
                print(f"    -> PredictiveHealthMonitor alerts EARLIER by {batch_alert - pred_alert} tasks")
            elif batch_alert < pred_alert:
                print(f"    -> BatchHealthMonitor alerts EARLIER by {pred_alert - batch_alert} tasks")
            else:
                print(f"    -> Both alert at the same task")

    batch_alerts = [r["batch_first_alert"] for r in results if r["batch_first_alert"] is not None]
    pred_alerts = [r["pred_first_alert"] for r in results if r["pred_first_alert"] is not None]
    shift_point = results[0]["shift_point"]

    print(f"\n{'=' * 60}")
    print(f"Summary ({len(args.seeds)} seeds):")
    print(f"  Shift point: task {shift_point}")

    if batch_alerts:
        batch_mean = np.mean(batch_alerts)
        print(f"  BatchHealthMonitor mean first alert: task {batch_mean:.1f} ({batch_mean - shift_point:+.1f} from shift)")
    if pred_alerts:
        pred_mean = np.mean(pred_alerts)
        print(f"  PredictiveHealthMonitor mean first alert: task {pred_mean:.1f} ({pred_mean - shift_point:+.1f} from shift)")

    if batch_alerts and pred_alerts:
        if np.mean(pred_alerts) < np.mean(batch_alerts):
            print(f"  -> PredictiveHealthMonitor has TEMPORAL ADVANTAGE (alerts {np.mean(batch_alerts) - np.mean(pred_alerts):.1f} tasks earlier)")
        else:
            print(f"  -> BatchHealthMonitor has TEMPORAL ADVANTAGE (alerts {np.mean(pred_alerts) - np.mean(batch_alerts):.1f} tasks earlier)")

    false_alarms_batch = sum(
        1 for r in results if r["batch_first_alert"] is not None and r["batch_first_alert"] < shift_point
    )
    false_alarms_pred = sum(
        1 for r in results if r["pred_first_alert"] is not None and r["pred_first_alert"] < shift_point
    )
    print(f"\n  False alarms (alert before shift):")
    print(f"    BatchHealthMonitor: {false_alarms_batch}/{len(results)}")
    print(f"    PredictiveHealthMonitor: {false_alarms_pred}/{len(results)}")

    output = {
        "experiment": "temporal_advantage_test",
        "seeds": args.seeds,
        "shift_point": shift_point,
        "per_seed_summary": [
            {
                "seed": r["seed"],
                "batch_first_alert": r["batch_first_alert"],
                "pred_first_alert": r["pred_first_alert"],
            }
            for r in results
        ],
        "summary": {
            "batch_mean_first_alert": float(np.mean(batch_alerts)) if batch_alerts else None,
            "pred_mean_first_alert": float(np.mean(pred_alerts)) if pred_alerts else None,
            "batch_false_alarms": false_alarms_batch,
            "pred_false_alarms": false_alarms_pred,
        },
    }

    out_path = Path("results/predictive_health_preflight")
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"temporal_advantage_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {fname}")


if __name__ == "__main__":
    main()
