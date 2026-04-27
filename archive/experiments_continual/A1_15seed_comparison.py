"""
A1: 15-seed head-to-head comparison — Unified-SEL (no boundary) vs EWC (with boundary).

Statistical tests: permutation test + bootstrap CI + Cohen's d.
"""

from __future__ import annotations

import json
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


def run_unified_sel_seeds(seeds: List[int], steps: int = 600, checkpoint_step: int = 200,
                          ewc_lambda: float = 15.0, readout_mode: str = "hybrid_local") -> List[Dict]:
    from experiments.continual.no_boundary import run_seed, NoBoundaryConfig

    config = NoBoundaryConfig()
    config.steps = steps
    config.checkpoint_step = checkpoint_step
    config.readout_mode = readout_mode

    results = []
    for seed in seeds:
        print(f"  [Unified-SEL] seed={seed}...")
        result = run_seed(seed=seed, config=config, window_size=50, ewc_lambda=ewc_lambda)
        results.append(result)
    return results


def run_ewc_seeds(seeds: List[int]) -> List[Dict]:
    from experiments.baselines.ewc import run_experiment

    results = []
    for seed in seeds:
        print(f"  [EWC] seed={seed}...")
        result = run_experiment(seed=seed)
        results.append(result)
    return results


def extract_unified_metrics(runs: List[Dict]) -> Dict[str, np.ndarray]:
    task_0_accs = []
    task_1_accs = []
    forgettings = []
    for run in runs:
        task_0_final = run.get("task_0_accuracy_final", 0.0)
        task_1_final = run.get("task_1_accuracy_final", 0.0)
        forgetting = run.get("forgetting_task_0", 0.0)
        task_0_accs.append(task_0_final)
        task_1_accs.append(task_1_final)
        forgettings.append(forgetting)
    return {
        "task_0_accuracy": np.array(task_0_accs),
        "task_1_accuracy": np.array(task_1_accs),
        "forgetting": np.array(forgettings),
    }


def extract_ewc_metrics(runs: List[Dict]) -> Dict[str, np.ndarray]:
    return {
        "task_0_accuracy": np.array([r["task_0_accuracy_after_task_1"] for r in runs]),
        "task_1_accuracy": np.array([r["task_1_accuracy_after_task_1"] for r in runs]),
        "forgetting": np.array([r["forgetting_task_0"] for r in runs]),
    }


def main():
    num_seeds = 15
    start_seed = 7
    seeds = [start_seed + i for i in range(num_seeds)]

    print(f"=== A1: 15-seed comparison — Unified-SEL vs EWC ===")
    print(f"Seeds: {seeds}")

    print("\n[1/2] Running Unified-SEL (no boundary, hybrid_local, ewc_lambda=15.0)...")
    unified_runs = run_unified_sel_seeds(seeds, ewc_lambda=15.0, readout_mode="hybrid_local")

    print("\n[2/2] Running EWC baseline (with boundary, ewc_lambda=40.0)...")
    ewc_runs = run_ewc_seeds(seeds)

    unified = extract_unified_metrics(unified_runs)
    ewc = extract_ewc_metrics(ewc_runs)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    metrics = ["task_0_accuracy", "task_1_accuracy", "forgetting"]
    comparison = {}

    for metric in metrics:
        u_vals = unified[metric]
        e_vals = ewc[metric]

        u_mean = float(u_vals.mean())
        e_mean = float(e_vals.mean())
        u_ci = _bootstrap_ci(u_vals)
        e_ci = _bootstrap_ci(e_vals)
        p_value = _permutation_p_value(u_vals, e_vals)
        d = _cohen_d(u_vals, e_vals)

        comparison[metric] = {
            "unified_mean": u_mean,
            "unified_ci_95": list(u_ci),
            "ewc_mean": e_mean,
            "ewc_ci_95": list(e_ci),
            "diff": u_mean - e_mean,
            "p_value": p_value,
            "cohens_d": d,
            "significant_p05": p_value < 0.05,
            "large_effect": abs(d) >= 0.8,
        }

        print(f"\n--- {metric} ---")
        print(f"  Unified-SEL: {u_mean:.4f}  95% CI [{u_ci[0]:.4f}, {u_ci[1]:.4f}]")
        print(f"  EWC:         {e_mean:.4f}  95% CI [{e_ci[0]:.4f}, {e_ci[1]:.4f}]")
        print(f"  Diff:        {u_mean - e_mean:+.4f}")
        print(f"  p-value:     {p_value:.4f}  {'*' if p_value < 0.05 else ''}")
        print(f"  Cohen's d:   {d:.4f}  {'large' if abs(d) >= 0.8 else 'medium' if abs(d) >= 0.5 else 'small'}")

    unified_wins_task0 = comparison["task_0_accuracy"]["diff"] > 0 and comparison["task_0_accuracy"]["significant_p05"]
    less_forgetting = comparison["forgetting"]["diff"] < 0 and comparison["forgetting"]["significant_p05"]

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Unified-SEL task_0 accuracy > EWC: {unified_wins_task0}")
    print(f"  Unified-SEL forgetting < EWC:      {less_forgetting}")
    print(f"  HYPOTHESIS CONFIRMED: {unified_wins_task0 and less_forgetting}")

    final_result = {
        "experiment": "A1_15seed_comparison",
        "num_seeds": num_seeds,
        "seeds": seeds,
        "unified_config": {
            "readout_mode": "hybrid_local",
            "ewc_lambda": 15.0,
            "steps": 600,
            "checkpoint_step": 200,
            "boundary_signal": False,
        },
        "ewc_config": {
            "ewc_lambda": 40.0,
            "boundary_signal": True,
        },
        "comparison": comparison,
        "hypothesis_confirmed": unified_wins_task0 and less_forgetting,
        "data_source": "真实验证（非模拟）",
        "cost_model_note": "成本数字基于假设成本模型",
        "timestamp": timestamp(),
    }

    results_dir = get_results_path("A1_15seed_comparison")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(final_result, output_path)
    final_result["saved_to"] = str(output_path)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
