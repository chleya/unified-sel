"""
A1-fix6: 15-seed comparison - Oracle Distribution Signal Routing vs EWC

This uses the oracle evaluation built into no_boundary.py:
- For each seed, after training, we apply perfect routing at evaluation time
- Task 0: x[0]+x[1] > 0 → use snapshot expert
- Task 1: x[0]+x[1] < 0 → use current model

This should achieve the ORACLE routing upper bound!
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import get_results_path, save_json, timestamp


def _permutation_p_value(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na, nb = len(a), len(b)
    count = 0
    for _ in range(n_resamples):
        permuted = rng.permutation(pooled)
        diff = float(permuted[:na].mean() - permuted[na:].mean())
        if abs(diff) >= abs(observed):
            count += 1
    return float((count + 1) / (n_resamples + 1))


def _bootstrap_ci(values: np.ndarray, n_resamples: int = 5000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    resampled = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        resampled[i] = sample.mean()
    return float(np.quantile(resampled, 0.025)), float(np.quantile(resampled, 0.975))


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def run_unified_sel(seeds: List[int], ewc_lambda: float = 30.0,
                    snapshot: bool = False) -> List[Dict]:
    from experiments.continual.no_boundary import run_seed, NoBoundaryConfig

    config = NoBoundaryConfig()
    config.readout_mode = "hybrid_local"

    results = []
    for seed in seeds:
        print(f"  seed={seed}...", flush=True)
        result = run_seed(
            seed=seed, config=config, window_size=50,
            ewc_lambda=ewc_lambda, anchor_lambda=0.0,
            dual_path_alpha=0.0, snapshot_expert=snapshot,
            snapshot_surprise_threshold=0.0,
        )
        results.append(result)
    return results


def run_ewc(seeds: List[int]) -> List[Dict]:
    from experiments.baselines.ewc import run_experiment

    results = []
    for seed in seeds:
        print(f"  [EWC] seed={seed}...", flush=True)
        result = run_experiment(seed=seed)
        results.append(result)
    return results


def extract_unified_metrics(runs: List[Dict]) -> Dict[str, np.ndarray]:
    task_0_accs = []
    task_1_accs = []
    for run in runs:
        t0 = run.get("oracle_task_0_accuracy")
        t1 = run.get("oracle_task_1_accuracy")
        if t0 is not None and t1 is not None:
            task_0_accs.append(t0)
            task_1_accs.append(t1)
        else:
            task_0_accs.append(run.get("task_0_accuracy_final", 0.0))
            task_1_accs.append(run.get("task_1_accuracy_final", 0.0))
    
    task_0_accs = np.array(task_0_accs)
    task_1_accs = np.array(task_1_accs)
    
    t0_early = np.array([run.get("task_0_accuracy_after_early_stream", 0.0) for run in runs])
    forgettings = t0_early - task_0_accs
    
    return {
        "task_0_accuracy": task_0_accs,
        "task_1_accuracy": task_1_accs,
        "forgetting": forgettings,
        "avg_accuracy": (task_0_accs + task_1_accs) / 2,
    }


def extract_ewc_metrics(runs: List[Dict]) -> Dict[str, np.ndarray]:
    return {
        "task_0_accuracy": np.array([r["task_0_accuracy_after_task_1"] for r in runs]),
        "task_1_accuracy": np.array([r["task_1_accuracy_after_task_1"] for r in runs]),
        "forgetting": np.array([r["forgetting_task_0"] for r in runs]),
        "avg_accuracy": np.array([
            (r["task_0_accuracy_after_task_1"] + r["task_1_accuracy_after_task_1"]) / 2
            for r in runs
        ]),
    }


def main():
    SEEDS = list(range(7, 22))  # 15 seeds

    print("=== A1-fix6: Oracle Signal Routing (via no_boundary.py built-in oracle eval) ===")
    print(f"Running {len(SEEDS)} seeds...\n")

    print("1. Running Unified-SEL with snapshot and built-in oracle evaluation...")
    unified_runs = run_unified_sel(SEEDS, ewc_lambda=30.0, snapshot=True)

    print("\n2. Running EWC baseline...")
    ewc_runs = run_ewc(SEEDS)

    unified_metrics = extract_unified_metrics(unified_runs)
    ewc_metrics = extract_ewc_metrics(ewc_runs)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nUnified-SEL (Oracle Signal Routing):")
    print(f"  task_0: {unified_metrics['task_0_accuracy'].mean():.4f} "
          f"[{_bootstrap_ci(unified_metrics['task_0_accuracy'])[0]:.4f}, "
          f"{_bootstrap_ci(unified_metrics['task_0_accuracy'])[1]:.4f}]")
    print(f"  task_1: {unified_metrics['task_1_accuracy'].mean():.4f} "
          f"[{_bootstrap_ci(unified_metrics['task_1_accuracy'])[0]:.4f}, "
          f"{_bootstrap_ci(unified_metrics['task_1_accuracy'])[1]:.4f}]")
    print(f"  avg_acc: {unified_metrics['avg_accuracy'].mean():.4f} "
          f"[{_bootstrap_ci(unified_metrics['avg_accuracy'])[0]:.4f}, "
          f"{_bootstrap_ci(unified_metrics['avg_accuracy'])[1]:.4f}]")
    print(f"  forgetting: {unified_metrics['forgetting'].mean():.4f}")

    print("\nEWC Baseline:")
    print(f"  task_0: {ewc_metrics['task_0_accuracy'].mean():.4f} "
          f"[{_bootstrap_ci(ewc_metrics['task_0_accuracy'])[0]:.4f}, "
          f"{_bootstrap_ci(ewc_metrics['task_0_accuracy'])[1]:.4f}]")
    print(f"  task_1: {ewc_metrics['task_1_accuracy'].mean():.4f} "
          f"[{_bootstrap_ci(ewc_metrics['task_1_accuracy'])[0]:.4f}, "
          f"{_bootstrap_ci(ewc_metrics['task_1_accuracy'])[1]:.4f}]")
    print(f"  avg_acc: {ewc_metrics['avg_accuracy'].mean():.4f} "
          f"[{_bootstrap_ci(ewc_metrics['avg_accuracy'])[0]:.4f}, "
          f"{_bootstrap_ci(ewc_metrics['avg_accuracy'])[1]:.4f}]")
    print(f"  forgetting: {ewc_metrics['forgetting'].mean():.4f}")

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)

    p_value = _permutation_p_value(unified_metrics["avg_accuracy"], ewc_metrics["avg_accuracy"])
    cohen_d = _cohen_d(unified_metrics["avg_accuracy"], ewc_metrics["avg_accuracy"])

    print(f"\nAverage accuracy: Unified-SEL = {unified_metrics['avg_accuracy'].mean():.4f}, "
          f"EWC = {ewc_metrics['avg_accuracy'].mean():.4f}")
    print(f"Difference: {unified_metrics['avg_accuracy'].mean() - ewc_metrics['avg_accuracy'].mean():.4f}")
    print(f"p-value (permutation test): {p_value:.4f}")
    print(f"Cohen's d (effect size): {cohen_d:.4f}")

    if p_value < 0.05 and unified_metrics["avg_accuracy"].mean() > ewc_metrics["avg_accuracy"].mean():
        print("\n🎉 SUCCESS: Unified-SEL significantly beats EWC!")
    elif p_value < 0.05:
        print("\n❌ Unified-SEL is significantly WORSE than EWC")
    else:
        print("\n⚠️ No statistically significant difference")

    results = {
        "seeds": SEEDS,
        "unified_sel": {
            "metrics": {k: v.tolist() for k, v in unified_metrics.items()},
            "runs": unified_runs,
        },
        "ewc": {
            "metrics": {k: v.tolist() for k, v in ewc_metrics.items()},
            "runs": ewc_runs,
        },
        "statistics": {
            "p_value": p_value,
            "cohen_d": cohen_d,
        },
    }

    out_path = get_results_path("A1_fix6_oracle_signal") / f"{timestamp()}.json"
    save_json(results, out_path)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

