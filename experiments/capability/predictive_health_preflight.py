"""
PredictiveHealthMonitor Preflight Experiment

Tests whether predictive residual (LeWM-inspired) can detect:
1. Domain shift (code -> reasoning)
2. Gradual shift (trivial -> harder)

Compares against BatchHealthMonitor baseline (27.2x / 4.1x separation).

Usage:
    python experiments/capability/predictive_health_preflight.py [--seeds 7 42 123] [--warmup 20]
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.capability_benchmark import build_task_suite
from core.predictive_health import PredictiveHealthMonitor, ControlContext
from topomem.embedding import EmbeddingManager


def encode_batch(embed_mgr, tasks):
    texts = []
    for t in tasks:
        parts = [t.prompt]
        if t.family == "code":
            fn = t.metadata.get("function_name", "")
            if fn:
                parts.append(fn)
        texts.append(" ".join(parts))
    return embed_mgr.encode_batch(texts)


def run_scenario(embed_mgr, tasks, warmup_windows=2, seed=42):
    embeddings = encode_batch(embed_mgr, tasks)
    mon = PredictiveHealthMonitor(
        latent_dim=embeddings.shape[1],
        warmup_windows=warmup_windows,
        window_size=10,
        learning_rate=0.05,
    )

    measured_residuals = []
    measured_z_scores = []
    measured_statuses = []
    for i, z in enumerate(embeddings):
        sig = mon.observe(z)
        if sig.status != "warmup":
            measured_residuals.append(sig.residual_mean)
            measured_z_scores.append(sig.residual_z_score)
            measured_statuses.append(sig.status)

    report = mon.health_report()

    return {
        "residual_mean": report.get("residual_mean", 0.0),
        "residual_var": report.get("residual_var", 0.0),
        "residual_trend": report.get("residual_trend", "unknown"),
        "status": report.get("status", "unknown"),
        "n_measured": len(measured_residuals),
        "measured_residuals": measured_residuals,
        "measured_z_scores": measured_z_scores,
        "measured_statuses": measured_statuses,
    }


def run_control(embed_mgr, seed, warmup_windows=2):
    tasks = build_task_suite("code", 40, seed=seed)
    first_half = tasks[:20]
    second_half = tasks[20:]
    all_tasks = first_half + second_half
    return run_scenario(embed_mgr, all_tasks, warmup_windows, seed)


def run_domain_shift(embed_mgr, seed, warmup_windows=2):
    code_tasks = build_task_suite("code", 20, seed=seed)
    reasoning_tasks = build_task_suite("reasoning", 20, seed=seed)
    all_tasks = code_tasks + reasoning_tasks
    return run_scenario(embed_mgr, all_tasks, warmup_windows, seed)


def run_gradual_shift(embed_mgr, seed, warmup_windows=2):
    tasks = build_task_suite("code", 40, seed=seed)
    trivial = [t for t in tasks if t.metadata.get("difficulty", "") == "trivial"]
    harder = [t for t in tasks if t.metadata.get("difficulty", "") != "trivial"]
    if len(trivial) < 3 or len(harder) < 3:
        return None
    all_tasks = trivial + harder
    return run_scenario(embed_mgr, all_tasks, warmup_windows, seed)


def compute_separation_ratio(shift_residuals, control_residuals):
    shift_mean = np.mean(shift_residuals) if shift_residuals else 0.0
    control_mean = np.mean(control_residuals) if control_residuals else 1e-10
    return shift_mean / max(control_mean, 1e-10)


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    data = np.array(data)
    n = len(data)
    if n < 2:
        return float(np.mean(data)), float(data[0]), float(data[0])
    boot_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        boot_means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(np.mean(data)), lo, hi


def main():
    parser = argparse.ArgumentParser(description="PredictiveHealthMonitor preflight")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    parser.add_argument("--warmup-windows", type=int, default=2)
    args = parser.parse_args()

    seeds = args.seeds
    warmup = args.warmup_windows

    print("=" * 60)
    print("PredictiveHealthMonitor Preflight Experiment")
    print("=" * 60)
    print(f"Seeds: {seeds}")
    print(f"Warmup windows: {warmup}")
    print()
    print("NOTE: All thresholds are provisional preflight targets,")
    print("not validated acceptance criteria.")
    print()

    print("[1/3] Loading embedding model...")
    embed_mgr = EmbeddingManager()

    print("[2/3] Running scenarios...")
    all_results = {}
    for seed in seeds:
        print(f"\n  Seed {seed}...")
        ctrl = run_control(embed_mgr, seed, warmup)
        shift = run_domain_shift(embed_mgr, seed, warmup)
        grad = run_gradual_shift(embed_mgr, seed, warmup)

        ctrl_mean = np.mean(ctrl["measured_residuals"]) if ctrl["measured_residuals"] else 0.0
        shift_mean = np.mean(shift["measured_residuals"]) if shift["measured_residuals"] else 0.0
        sep = shift_mean / max(ctrl_mean, 1e-10)

        print(f"    Control residual_mean:  {ctrl_mean:.4f}")
        print(f"    Shift residual_mean:    {shift_mean:.4f}")
        print(f"    Separation:             {sep:.1f}x")

        if grad:
            grad_mean = np.mean(grad["measured_residuals"]) if grad["measured_residuals"] else 0.0
            grad_sep = grad_mean / max(ctrl_mean, 1e-10)
            print(f"    Gradual residual_mean:  {grad_mean:.4f} ({grad_sep:.1f}x vs control)")
        else:
            grad_sep = 0.0

        all_results[seed] = {
            "control": ctrl,
            "domain_shift": shift,
            "gradual_shift": grad,
            "separation_domain": sep,
            "separation_gradual": grad_sep,
        }

    print("\n[3/3] Statistical analysis...")

    ctrl_residuals = [
        np.mean(all_results[s]["control"]["measured_residuals"])
        if all_results[s]["control"]["measured_residuals"]
        else 0.0
        for s in seeds
    ]
    shift_residuals = [
        np.mean(all_results[s]["domain_shift"]["measured_residuals"])
        if all_results[s]["domain_shift"]["measured_residuals"]
        else 0.0
        for s in seeds
    ]
    grad_residuals = [
        np.mean(all_results[s]["gradual_shift"]["measured_residuals"])
        if all_results[s]["gradual_shift"] and all_results[s]["gradual_shift"]["measured_residuals"]
        else 0.0
        for s in seeds
    ]

    domain_seps = [
        all_results[s]["separation_domain"] for s in seeds
    ]
    gradual_seps = [
        all_results[s]["separation_gradual"] for s in seeds
    ]

    ctrl_mean, ctrl_lo, ctrl_hi = bootstrap_ci(ctrl_residuals)
    shift_mean, shift_lo, shift_hi = bootstrap_ci(shift_residuals)
    sep_mean, sep_lo, sep_hi = bootstrap_ci(domain_seps)

    print(f"\n  Predictive Residual — Domain Shift ({len(seeds)} seeds, 95% bootstrap CI):")
    print(f"    Control:  {ctrl_mean:.4f} [{ctrl_lo:.4f}, {ctrl_hi:.4f}]")
    print(f"    Shift:    {shift_mean:.4f} [{shift_lo:.4f}, {shift_hi:.4f}]")
    print(f"    Separation: {sep_mean:.1f}x [{sep_lo:.1f}x, {sep_hi:.1f}x]")
    print(f"    Baseline (BatchHealthMonitor): 27.2x [18.4x, 37.3x]")

    valid_gradual = [s for s in gradual_seps if s > 0]
    if valid_gradual:
        grad_sep_mean, grad_sep_lo, grad_sep_hi = bootstrap_ci(valid_gradual)
        print(f"\n  Predictive Residual — Gradual Shift ({len(valid_gradual)} seeds):")
        print(f"    Separation vs control: {grad_sep_mean:.1f}x [{grad_sep_lo:.1f}x, {grad_sep_hi:.1f}x]")
        print(f"    Baseline (BatchHealthMonitor): 4.1x [2.6x, 5.9x]")
    else:
        grad_sep_mean, grad_sep_lo, grad_sep_hi = 0.0, 0.0, 0.0

    domain_verdict = "INCONCLUSIVE"
    if sep_lo > 2.0:
        domain_verdict = "CONFIRMED - domain shift detectable via predictive residual"
    elif sep_mean > 3.0:
        domain_verdict = "PROMISING - large separation but CI wide"
    else:
        domain_verdict = "INCONCLUSIVE - predictive residual cannot reliably detect domain shift"

    gradual_verdict = "INCONCLUSIVE"
    if valid_gradual:
        if grad_sep_lo > 1.0:
            gradual_verdict = "DETECTABLE - gradual drift separable from control"
        elif grad_sep_mean > 1.0:
            gradual_verdict = "MARGINAL - CI includes 1.0"
        else:
            gradual_verdict = "UNDETECTABLE"

    print(f"\n  Domain Shift Verdict: {domain_verdict}")
    print(f"  Gradual Shift Verdict: {gradual_verdict}")

    print(f"\n  Provisional Target Check (NOT validated acceptance criteria):")
    print(f"    Domain separation >= 10x (provisional): {'PASS' if sep_mean >= 10 else 'FAIL'} ({sep_mean:.1f}x)")
    print(f"    Gradual separation >= 5x (provisional):  {'PASS' if grad_sep_mean >= 5 else 'FAIL'} ({grad_sep_mean:.1f}x)")

    output = {
        "schema_version": "capbench.result.v1",
        "metadata": {
            "data_source": "verified_execution",
            "cost_model": "abstract_units_v1",
            "oracle_assumption": False,
            "verifier_policy": "predictive_health_preflight",
            "benchmark_suite": "predictive-health-preflight",
            "seeds": seeds,
            "warmup_windows": warmup,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "experiment": "predictive_health_preflight",
        "hypothesis": "Predictive residual (LeWM-inspired) can detect domain/gradual shifts",
        "per_seed_results": {str(s): all_results[s] for s in seeds},
        "analysis": {
            "domain_shift": {
                "control_mean_ci": [ctrl_mean, ctrl_lo, ctrl_hi],
                "shift_mean_ci": [shift_mean, shift_lo, shift_hi],
                "separation_mean_ci": [sep_mean, sep_lo, sep_hi],
                "verdict": domain_verdict,
            },
            "gradual_shift": {
                "separation_mean_ci": [grad_sep_mean, grad_sep_lo, grad_sep_hi],
                "verdict": gradual_verdict,
            },
            "baseline_comparison": {
                "batch_health_domain_separation": "27.2x [18.4x, 37.3x]",
                "batch_health_gradual_separation": "4.1x [2.6x, 5.9x]",
            },
            "provisional_target_check": {
                "domain_sep_target_10x": sep_mean >= 10,
                "gradual_sep_target_5x": grad_sep_mean >= 5,
                "note": "provisional targets, not validated acceptance criteria",
            },
        },
    }

    out_path = Path("results/predictive_health_preflight")
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"preflight_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {fname}")


if __name__ == "__main__":
    main()
