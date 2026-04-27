"""
A1-fix3: 15-seed comparison — Snapshot Expert vs EWC

Tests whether surprise-gated snapshot expert routing can match or beat EWC
on forgetting, while maintaining better task_1 performance.

Configs:
  1. Unified-SEL baseline (ewc_lambda=30, no snapshot)
  2. Snapshot expert (ewc_lambda=30, thresh=0.0 — always use snapshot)
  3. Snapshot expert (ewc_lambda=30, thresh=0.1 — surprise-gated)
  4. EWC baseline (ewc_lambda=40, with boundary signal)
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
                    snapshot: bool = False, surprise_thresh: float = 0.5) -> List[Dict]:
    from experiments.continual.no_boundary import run_seed, NoBoundaryConfig

    config = NoBoundaryConfig()
    config.readout_mode = "hybrid_local"

    results = []
    for seed in seeds:
        label = "snapshot" if snapshot else "baseline"
        print(f"  [{label}] seed={seed}...", flush=True)
        result = run_seed(
            seed=seed, config=config, window_size=50,
            ewc_lambda=ewc_lambda, anchor_lambda=0.0,
            dual_path_alpha=0.0, snapshot_expert=snapshot,
            snapshot_surprise_threshold=surprise_thresh,
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
    task_0_accs = [run.get("task_0_accuracy_final", 0.0) for run in runs]
    task_1_accs = [run.get("task_1_accuracy_final", 0.0) for run in runs]
    forgettings = [run.get("forgetting_task_0", 0.0) for run in runs]
    return {
        "task_0_accuracy": np.array(task_0_accs),
        "task_1_accuracy": np.array(task_1_accs),
        "forgetting": np.array(forgettings),
        "avg_accuracy": np.array([(t0 + t1) / 2 for t0, t1 in zip(task_0_accs, task_1_accs)]),
    }


def extract_ewc_metrics(runs: List[Dict]) -> Dict[str, np.ndarray]:
    return {
        "task_0_accuracy": np.array([r["task_0_accuracy_after_task_1"] for r in runs]),
        "task_1_accuracy": np.array([r["task_1_accuracy_after_task_1"] for r in runs]),
        "forgetting": np.array([r["forgetting_task_0"] for r in runs]),
        "avg_accuracy": np.array([(r["task_0_accuracy_after_task_1"] + r["task_1_accuracy_after_task_1"]) / 2 for r in runs]),
    }


def compare_vs_ewc(unified: Dict[str, np.ndarray], ewc: Dict[str, np.ndarray], label: str):
    print(f"\n{'='*60}")
    print(f"  {label} vs EWC")
    print(f"{'='*60}")

    for metric in ["task_0_accuracy", "task_1_accuracy", "forgetting", "avg_accuracy"]:
        u_vals = unified[metric]
        e_vals = ewc[metric]
        u_mean = float(u_vals.mean())
        e_mean = float(e_vals.mean())
        u_ci = _bootstrap_ci(u_vals)
        e_ci = _bootstrap_ci(e_vals)
        p_value = _permutation_p_value(u_vals, e_vals)
        d = _cohen_d(u_vals, e_vals)

        sig = "*" if p_value < 0.05 else ""
        eff = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"

        print(f"  {metric}:")
        print(f"    {label}: {u_mean:.4f}  95%CI [{u_ci[0]:.4f}, {u_ci[1]:.4f}]")
        print(f"    EWC:    {e_mean:.4f}  95%CI [{e_ci[0]:.4f}, {e_ci[1]:.4f}]")
        print(f"    diff={u_mean-e_mean:+.4f}  p={p_value:.4f}{sig}  d={d:.3f}({eff})")


def main():
    num_seeds = 15
    start_seed = 7
    seeds = [start_seed + i for i in range(num_seeds)]

    print(f"=== A1-fix3: 15-seed Snapshot Expert vs EWC ===")
    print(f"Seeds: {seeds}")

    print(f"\n[1/4] Unified-SEL baseline (ewc=30, no snapshot)...")
    baseline_runs = run_unified_sel(seeds, ewc_lambda=30, snapshot=False)

    print(f"\n[2/4] Snapshot expert (ewc=30, thresh=0.0 — always snapshot)...")
    snap0_runs = run_unified_sel(seeds, ewc_lambda=30, snapshot=True, surprise_thresh=0.0)

    print(f"\n[3/4] Snapshot expert (ewc=30, thresh=0.1 — surprise-gated)...")
    snap01_runs = run_unified_sel(seeds, ewc_lambda=30, snapshot=True, surprise_thresh=0.1)

    print(f"\n[4/4] EWC baseline (ewc=40, with boundary)...")
    ewc_runs = run_ewc(seeds)

    baseline = extract_unified_metrics(baseline_runs)
    snap0 = extract_unified_metrics(snap0_runs)
    snap01 = extract_unified_metrics(snap01_runs)
    ewc = extract_ewc_metrics(ewc_runs)

    compare_vs_ewc(baseline, ewc, "Baseline(ewc30)")
    compare_vs_ewc(snap0, ewc, "Snap(thresh=0.0)")
    compare_vs_ewc(snap01, ewc, "Snap(thresh=0.1)")

    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'task_0':>8} {'task_1':>8} {'forget':>8} {'avg':>8}")
    print("-" * 60)
    for name, data in [("Baseline(ewc30)", baseline), ("Snap(thresh=0.0)", snap0),
                       ("Snap(thresh=0.1)", snap01), ("EWC(ewc40)", ewc)]:
        print(f"{name:<25} {data['task_0_accuracy'].mean():8.4f} {data['task_1_accuracy'].mean():8.4f} "
              f"{data['forgetting'].mean():8.4f} {data['avg_accuracy'].mean():8.4f}")

    snap0_less_forget = snap0["forgetting"].mean() < ewc["forgetting"].mean()
    snap0_better_avg = snap0["avg_accuracy"].mean() > ewc["avg_accuracy"].mean()
    print(f"\nSnap(thresh=0.0) forgetting < EWC: {snap0_less_forget}")
    print(f"Snap(thresh=0.0) avg_acc > EWC:    {snap0_better_avg}")

    final_result = {
        "experiment": "A1_fix3_snapshot_expert_15seed",
        "num_seeds": num_seeds,
        "seeds": seeds,
        "baseline_mean": {k: float(v.mean()) for k, v in baseline.items()},
        "snap_thresh0_mean": {k: float(v.mean()) for k, v in snap0.items()},
        "snap_thresh01_mean": {k: float(v.mean()) for k, v in snap01.items()},
        "ewc_mean": {k: float(v.mean()) for k, v in ewc.items()},
        "data_source": "真实验证（非模拟）",
        "timestamp": timestamp(),
    }

    results_dir = get_results_path("A1_fix3_snapshot_expert")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(final_result, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
