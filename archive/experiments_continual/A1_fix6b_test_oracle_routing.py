"""
A1-fix6b: Test Oracle Routing Directly

This script applies perfect routing ONLY at evaluation time, without modifying the learner.
We use the known structure:
- Task 0: y = (x[0] + x[1] > 0.0) → use snapshot expert
- Task 1: y = (x[0] + x[1] < 0.0) → use current model
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


def run_unified_sel_with_oracle_eval(seeds: List[int], ewc_lambda: float = 30.0) -> List[Dict]:
    from experiments.continual.no_boundary import run_seed, NoBoundaryConfig, make_eval_task

    config = NoBoundaryConfig()
    config.readout_mode = "hybrid_local"

    results = []
    for seed in seeds:
        print(f"  seed={seed}...", flush=True)
        
        result = run_seed(
            seed=seed, config=config, window_size=50,
            ewc_lambda=ewc_lambda, anchor_lambda=0.0,
            dual_path_alpha=0.0, snapshot_expert=True,
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


def main():
    SEEDS = list(range(7, 22))  # 15 seeds

    print("=== A1-fix6b: Oracle Routing at Evaluation Time ===")
    print(f"Running {len(SEEDS)} seeds...\n")

    print("1. Running Unified-SEL with snapshot...")
    unified_runs = run_unified_sel_with_oracle_eval(SEEDS, ewc_lambda=30.0)

    print("\n2. Running EWC baseline...")
    ewc_runs = run_ewc(SEEDS)

    from experiments.continual.no_boundary import make_eval_task, NoBoundaryConfig

    config = NoBoundaryConfig()
    unified_task0_accs = []
    unified_task1_accs = []
    ewc_task0_accs = []
    ewc_task1_accs = []

    for i, seed in enumerate(SEEDS):
        run = unified_runs[i]
        
        X_task_0, y_task_0 = make_eval_task(0, config.eval_samples_per_task, seed + 1000, config.in_size)
        X_task_1, y_task_1 = make_eval_task(1, config.eval_samples_per_task, seed + 2000, config.in_size)
        
        from experiments.continual.no_boundary import run_seed
        import sys
        from core.learner import UnifiedSELClassifier
        
        clf = UnifiedSELClassifier(
            in_size=config.in_size,
            out_size=config.out_size,
            lr=config.lr,
            max_structures=config.pool.max_structures,
            evolve_every=config.evolve_every,
            pool_config=config.pool.to_pool_kwargs(),
            seed=seed,
            ewc_lambda=30.0,
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
            anchor_lambda=0.0,
        )
        
        from core.pool import StructurePool
        import pickle
        
        from experiments.continual.no_boundary import run_seed
        temp_result = run_seed(
            seed=seed, config=config, window_size=50,
            ewc_lambda=30.0, snapshot_expert=True,
            snapshot_surprise_threshold=0.0,
        )
        
        correct_t0 = 0
        for j in range(len(X_task_0)):
            x = X_task_0[j]
            y = int(y_task_0[j])
            boundary = x[0] + x[1]
            
            if boundary > 0.0:
                if len(temp_result.get("checkpoint_metrics", [])) > 0:
                    checkpoint = temp_result["checkpoint_metrics"][0]
                    
                    snapshot = None
                    if "checkpoint_metrics" in temp_result:
                        import copy
                        
                        snapshot_expert_state = None
                        clf_for_eval = UnifiedSELClassifier(
                            in_size=config.in_size,
                            out_size=config.out_size,
                            lr=config.lr,
                            max_structures=config.pool.max_structures,
                            evolve_every=config.evolve_every,
                            pool_config=config.pool.to_pool_kwargs(),
                            seed=seed,
                            ewc_lambda=30.0,
                            readout_mode=config.readout_mode,
                        )
                        
                        rng = np.random.default_rng(seed)
                        clf_for_eval.pool = StructurePool(
                            in_size=config.in_size,
                            out_size=config.out_size,
                            max_structures=config.pool.max_structures,
                            initial_structures=1,
                            seed=seed,
                            **config.pool.to_pool_kwargs(),
                        )
                        clf_for_eval.W_out = rng.normal(0.0, 0.1, size=(config.out_size, config.out_size))
                        clf_for_eval.W_out_fisher = np.zeros_like(clf_for_eval.W_out)
                        clf_for_eval.W_out_anchor = clf_for_eval.W_out.copy()
                        clf_for_eval.fisher_estimated = False
                        
                        from experiments.continual.no_boundary import stream_sample
                        
                        for step in range(config.steps):
                            if step < config.checkpoint_step:
                                progress = 0.0
                            else:
                                progress = (step - config.checkpoint_step) / max(config.steps - config.checkpoint_step - 1, 1)
                            x_step, y_step = stream_sample(progress, rng, in_size=config.in_size)
                            loss = clf_for_eval.fit_one(x_step, y_step)
                            
                            if step + 1 == config.checkpoint_step:
                                clf_for_eval.snapshot_expert(confidence_ratio_threshold=0.0)
                        
                        X_task_0_eval, y_task_0_eval = make_eval_task(0, config.eval_samples_per_task, seed + 1000, config.in_size)
                        X_task_1_eval, y_task_1_eval = make_eval_task(1, config.eval_samples_per_task, seed + 2000, config.in_size)
                        
                        correct_t0_seed = 0
                        for k in range(len(X_task_0_eval)):
                            x_eval = X_task_0_eval[k]
                            y_eval = int(y_task_0_eval[k])
                            boundary_eval = x_eval[0] + x_eval[1]
                            
                            if boundary_eval > 0.0 and clf_for_eval._snapshot_experts:
                                pred = int(np.argmax(clf_for_eval._predict_with_snapshot(x_eval, clf_for_eval._snapshot_experts[0])))
                            else:
                                pred = int(np.argmax(clf_for_eval.predict_proba_single(x_eval)))
                            
                            if pred == y_eval:
                                correct_t0_seed += 1
                        t0_acc = correct_t0_seed / len(X_task_0_eval)
                        
                        correct_t1_seed = 0
                        for k in range(len(X_task_1_eval)):
                            x_eval = X_task_1_eval[k]
                            y_eval = int(y_task_1_eval[k])
                            boundary_eval = x_eval[0] + x_eval[1]
                            
                            if boundary_eval > 0.0 and clf_for_eval._snapshot_experts:
                                pred = int(np.argmax(clf_for_eval._predict_with_snapshot(x_eval, clf_for_eval._snapshot_experts[0])))
                            else:
                                pred = int(np.argmax(clf_for_eval.predict_proba_single(x_eval)))
                            
                            if pred == y_eval:
                                correct_t1_seed += 1
                        t1_acc = correct_t1_seed / len(X_task_1_eval)
                        
                        unified_task0_accs.append(t0_acc)
                        unified_task1_accs.append(t1_acc)
                        print(f"    seed={seed}: t0={t0_acc:.4f}, t1={t1_acc:.4f}, avg={(t0_acc+t1_acc)/2:.4f}")
                        break

    ewc_task0_accs = [r["task_0_accuracy_after_task_1"] for r in ewc_runs]
    ewc_task1_accs = [r["task_1_accuracy_after_task_1"] for r in ewc_runs]

    unified_avg_accs = np.array([(t0 + t1) / 2 for t0, t1 in zip(unified_task0_accs, unified_task1_accs)])
    ewc_avg_accs = np.array([(t0 + t1) / 2 for t0, t1 in zip(ewc_task0_accs, ewc_task1_accs)])

    unified_forgetting = np.array([unified_task0_accs[0] - t0 for t0 in unified_task0_accs])
    ewc_forgetting = np.array([ewc_task0_accs[0] - t0 for t0 in ewc_task0_accs])

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nUnified-SEL (Oracle Routing at Eval Time):")
    print(f"  task_0: {np.mean(unified_task0_accs):.4f} "
          f"[{_bootstrap_ci(np.array(unified_task0_accs))[0]:.4f}, "
          f"{_bootstrap_ci(np.array(unified_task0_accs))[1]:.4f}]")
    print(f"  task_1: {np.mean(unified_task1_accs):.4f} "
          f"[{_bootstrap_ci(np.array(unified_task1_accs))[0]:.4f}, "
          f"{_bootstrap_ci(np.array(unified_task1_accs))[1]:.4f}]")
    print(f"  avg_acc: {np.mean(unified_avg_accs):.4f} "
          f"[{_bootstrap_ci(unified_avg_accs)[0]:.4f}, "
          f"{_bootstrap_ci(unified_avg_accs)[1]:.4f}]")
    print(f"  forgetting: {np.mean(unified_forgetting):.4f}")

    print("\nEWC Baseline:")
    print(f"  task_0: {np.mean(ewc_task0_accs):.4f} "
          f"[{_bootstrap_ci(np.array(ewc_task0_accs))[0]:.4f}, "
          f"{_bootstrap_ci(np.array(ewc_task0_accs))[1]:.4f}]")
    print(f"  task_1: {np.mean(ewc_task1_accs):.4f} "
          f"[{_bootstrap_ci(np.array(ewc_task1_accs))[0]:.4f}, "
          f"{_bootstrap_ci(np.array(ewc_task1_accs))[1]:.4f}]")
    print(f"  avg_acc: {np.mean(ewc_avg_accs):.4f} "
          f"[{_bootstrap_ci(ewc_avg_accs)[0]:.4f}, "
          f"{_bootstrap_ci(ewc_avg_accs)[1]:.4f}]")
    print(f"  forgetting: {np.mean(ewc_forgetting):.4f}")

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)

    p_value = _permutation_p_value(unified_avg_accs, ewc_avg_accs)
    cohen_d = _cohen_d(unified_avg_accs, ewc_avg_accs)

    print(f"\nAverage accuracy: Unified-SEL = {unified_avg_accs.mean():.4f}, "
          f"EWC = {ewc_avg_accs.mean():.4f}")
    print(f"Difference: {unified_avg_accs.mean() - ewc_avg_accs.mean():.4f}")
    print(f"p-value (permutation test): {p_value:.4f}")
    print(f"Cohen's d (effect size): {cohen_d:.4f}")

    if p_value < 0.05 and unified_avg_accs.mean() > ewc_avg_accs.mean():
        print("\n🎉 SUCCESS: Unified-SEL significantly beats EWC!")
    elif p_value < 0.05:
        print("\n❌ Unified-SEL is significantly WORSE than EWC")
    else:
        print("\n⚠️ No statistically significant difference")

    results = {
        "seeds": SEEDS,
        "unified_sel": {
            "task_0_accuracy": unified_task0_accs,
            "task_1_accuracy": unified_task1_accs,
            "avg_accuracy": unified_avg_accs.tolist(),
            "forgetting": unified_forgetting.tolist(),
        },
        "ewc": {
            "task_0_accuracy": ewc_task0_accs,
            "task_1_accuracy": ewc_task1_accs,
            "avg_accuracy": ewc_avg_accs.tolist(),
            "forgetting": ewc_forgetting.tolist(),
        },
        "statistics": {
            "p_value": p_value,
            "cohen_d": cohen_d,
        },
    }

    out_path = get_results_path("A1_fix6b_oracle_eval") / f"{timestamp()}.json"
    save_json(results, out_path)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

