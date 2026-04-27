"""
TopoMem OBD Multi-Seed Validation + Gradual Drift Test

Extends the single-seed preflight with:
1. 5 seeds for bootstrap CI on separation ratio
2. Gradual drift: code-trivial -> code-mixed (same domain, harder difficulty)

Usage:
    python experiments/capability/topomem_obd_multiseed.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.capability_benchmark import build_task_suite
from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine

SEEDS = [7, 42, 123, 256, 999, 1337, 2024, 3141, 4096, 65537]


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


def compute_centroid_drift(emb_a, emb_b):
    c_a = np.mean(emb_a, axis=0)
    c_b = np.mean(emb_b, axis=0)
    c_a_norm = c_a / (np.linalg.norm(c_a) + 1e-10)
    c_b_norm = c_b / (np.linalg.norm(c_b) + 1e-10)
    return float(1.0 - np.dot(c_a_norm, c_b_norm))


def compute_similarity_gap(embed_mgr, emb_a, emb_b):
    sim_matrix = embed_mgr.similarity_matrix(np.vstack([emb_a, emb_b]))
    n_a = len(emb_a)
    intra_a = sim_matrix[:n_a, :n_a]
    np.fill_diagonal(intra_a, 0)
    intra_b = sim_matrix[n_a:, n_a:]
    np.fill_diagonal(intra_b, 0)
    cross = sim_matrix[:n_a, n_a:]
    mean_intra = (np.mean(intra_a[intra_a > 0]) + np.mean(intra_b[intra_b > 0])) / 2
    mean_cross = np.mean(cross)
    return float(mean_intra - mean_cross)


def compute_wasserstein_drift(topo_engine, emb_a, emb_b, dim=0):
    try:
        diag_a = topo_engine.compute_persistence(emb_a)
        diag_b = topo_engine.compute_persistence(emb_b)
        return topo_engine.wasserstein_distance(diag_a, diag_b, dim=dim)
    except Exception:
        return -1.0


def run_single_seed(embed_mgr, topo_engine, seed):
    code_tasks = build_task_suite("code", 20, seed=seed)
    reasoning_tasks = build_task_suite("reasoning", 20, seed=seed)

    code_1 = code_tasks[:10]
    code_2 = code_tasks[10:]
    reasoning = reasoning_tasks[:20]

    code_trivial = [t for t in code_tasks if t.metadata.get("difficulty", "") == "trivial"]
    code_mixed = [t for t in code_tasks if t.metadata.get("difficulty", "") != "trivial"]

    emb_code_1 = encode_batch(embed_mgr, code_1)
    emb_code_2 = encode_batch(embed_mgr, code_2)
    emb_reasoning = encode_batch(embed_mgr, reasoning)

    results = {}

    results["control"] = {
        "centroid": compute_centroid_drift(emb_code_1, emb_code_2),
        "sim_gap": compute_similarity_gap(embed_mgr, emb_code_1, emb_code_2),
        "w_h0": compute_wasserstein_drift(topo_engine, emb_code_1, emb_code_2, dim=0),
        "w_h1": compute_wasserstein_drift(topo_engine, emb_code_1, emb_code_2, dim=1),
    }

    results["domain_shift"] = {
        "centroid": compute_centroid_drift(emb_code_1, emb_reasoning),
        "sim_gap": compute_similarity_gap(embed_mgr, emb_code_1, emb_reasoning),
        "w_h0": compute_wasserstein_drift(topo_engine, emb_code_1, emb_reasoning, dim=0),
        "w_h1": compute_wasserstein_drift(topo_engine, emb_code_1, emb_reasoning, dim=1),
    }

    if len(code_trivial) >= 5 and len(code_mixed) >= 5:
        emb_trivial = encode_batch(embed_mgr, code_trivial)
        emb_mixed = encode_batch(embed_mgr, code_mixed)
        results["gradual_shift"] = {
            "centroid": compute_centroid_drift(emb_trivial, emb_mixed),
            "sim_gap": compute_similarity_gap(embed_mgr, emb_trivial, emb_mixed),
            "w_h0": compute_wasserstein_drift(topo_engine, emb_trivial, emb_mixed, dim=0),
            "w_h1": compute_wasserstein_drift(topo_engine, emb_trivial, emb_mixed, dim=1),
            "n_trivial": len(code_trivial),
            "n_mixed": len(code_mixed),
        }
    else:
        results["gradual_shift"] = None

    return results


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
    print("=" * 60)
    print("TopoMem OBD Multi-Seed Validation + Gradual Drift")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")

    print("\n[1/3] Loading models...")
    embed_mgr = EmbeddingManager()
    topo_engine = TopologyEngine()

    print("[2/3] Running 5 seeds...")
    all_results = {}
    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        result = run_single_seed(embed_mgr, topo_engine, seed)
        all_results[seed] = result
        ctrl = result["control"]
        shift = result["domain_shift"]
        grad = result.get("gradual_shift")
        sep = shift["centroid"] / max(ctrl["centroid"], 1e-10)
        print(f"    Control centroid: {ctrl['centroid']:.4f}")
        print(f"    Shift centroid:   {shift['centroid']:.4f}")
        print(f"    Separation:       {sep:.1f}x")
        if grad:
            grad_sep = grad["centroid"] / max(ctrl["centroid"], 1e-10)
            print(f"    Gradual centroid: {grad['centroid']:.4f} ({grad_sep:.1f}x vs control)")

    print("\n[3/3] Statistical analysis...")

    control_centroids = [all_results[s]["control"]["centroid"] for s in SEEDS]
    shift_centroids = [all_results[s]["domain_shift"]["centroid"] for s in SEEDS]
    control_sim_gaps = [all_results[s]["control"]["sim_gap"] for s in SEEDS]
    shift_sim_gaps = [all_results[s]["domain_shift"]["sim_gap"] for s in SEEDS]
    control_w_h0 = [all_results[s]["control"]["w_h0"] for s in SEEDS]
    shift_w_h0 = [all_results[s]["domain_shift"]["w_h0"] for s in SEEDS]

    separations = [shift_centroids[i] / max(control_centroids[i], 1e-10) for i in range(len(SEEDS))]

    ctrl_mean, ctrl_lo, ctrl_hi = bootstrap_ci(control_centroids)
    shift_mean, shift_lo, shift_hi = bootstrap_ci(shift_centroids)
    sep_mean, sep_lo, sep_hi = bootstrap_ci(separations)
    sim_ctrl_mean, _, _ = bootstrap_ci(control_sim_gaps)
    sim_shift_mean, _, _ = bootstrap_ci(shift_sim_gaps)
    sim_separations = [shift_sim_gaps[i] / max(control_sim_gaps[i], 0.01) for i in range(len(SEEDS))]
    sim_sep_mean, sim_sep_lo, sim_sep_hi = bootstrap_ci(sim_separations)

    print(f"\n  Centroid Drift (5 seeds, 95% bootstrap CI):")
    print(f"    Control:  {ctrl_mean:.4f} [{ctrl_lo:.4f}, {ctrl_hi:.4f}]")
    print(f"    Shift:    {shift_mean:.4f} [{shift_lo:.4f}, {shift_hi:.4f}]")
    print(f"    Separation: {sep_mean:.1f}x [{sep_lo:.1f}x, {sep_hi:.1f}x]")

    print(f"\n  Similarity Gap (5 seeds, 95% bootstrap CI):")
    print(f"    Control:  {sim_ctrl_mean:.4f}")
    print(f"    Shift:    {sim_shift_mean:.4f}")
    print(f"    Separation: {sim_sep_mean:.1f}x [{sim_sep_lo:.1f}x, {sim_sep_hi:.1f}x]")

    gradual_results = []
    for s in SEEDS:
        g = all_results[s].get("gradual_shift")
        if g is not None:
            gradual_results.append(g)

    gradual_verdict = "NO DATA"
    if gradual_results:
        grad_centroids = [g["centroid"] for g in gradual_results]
        grad_sim_gaps = [g["sim_gap"] for g in gradual_results]
        grad_mean, grad_lo, grad_hi = bootstrap_ci(grad_centroids)
        grad_sep = [grad_centroids[i] / max(control_centroids[i], 1e-10) for i in range(len(gradual_results))]
        grad_sep_mean, grad_sep_lo, grad_sep_hi = bootstrap_ci(grad_sep)
        print(f"\n  Gradual Drift (code-trivial -> code-harder, {len(gradual_results)} seeds):")
        print(f"    Centroid: {grad_mean:.4f} [{grad_lo:.4f}, {grad_hi:.4f}]")
        print(f"    Separation vs control: {grad_sep_mean:.1f}x [{grad_sep_lo:.1f}x, {grad_sep_hi:.1f}x]")
        if grad_sep_lo > 1.0:
            gradual_verdict = "DETECTABLE - gradual drift separable from control"
        elif grad_sep_mean > 1.0:
            gradual_verdict = "MARGINAL - gradual drift barely separable, CI includes 1.0"
        else:
            gradual_verdict = "UNDETECTABLE - gradual drift indistinguishable from control"
        print(f"    Verdict: {gradual_verdict}")

    from scipy import stats as sp_stats
    t_stat, p_value = sp_stats.ttest_rel(shift_centroids, control_centroids)
    cohens_d = (shift_mean - ctrl_mean) / max(np.std(control_centroids + shift_centroids, ddof=1), 1e-10)

    print(f"\n  Paired t-test (shift vs control centroid):")
    print(f"    t = {t_stat:.4f}, p = {p_value:.6f}")
    print(f"    Cohen's d = {cohens_d:.2f}")

    domain_verdict = "INCONCLUSIVE"
    if p_value < 0.05 and sep_lo > 2.0:
        domain_verdict = "CONFIRMED - domain shift clearly detectable across seeds"
    elif p_value < 0.05:
        domain_verdict = "CONFIRMED (weak) - domain shift detectable but separation variable"
    elif sep_mean > 3.0:
        domain_verdict = "PROMISING - large separation but not statistically significant (need more seeds)"
    else:
        domain_verdict = "INCONCLUSIVE - cannot confirm drift detection"

    print(f"\n  Domain Shift Verdict: {domain_verdict}")

    output = {
        "schema_version": "capbench.result.v1",
        "metadata": {
            "data_source": "verified_execution",
            "cost_model": "abstract_units_v1",
            "oracle_assumption": False,
            "verifier_policy": "topomem_obd_multiseed",
            "benchmark_suite": "obd-multiseed-5seed",
            "task_count": 60,
            "seeds": SEEDS,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "experiment": "topomem_obd_multiseed",
        "hypothesis": "Batch-level drift signals can detect domain distribution shifts (multi-seed validation)",
        "per_seed_results": {str(s): all_results[s] for s in SEEDS},
        "analysis": {
            "centroid_drift": {
                "control_mean_ci": [ctrl_mean, ctrl_lo, ctrl_hi],
                "shift_mean_ci": [shift_mean, shift_lo, shift_hi],
                "separation_mean_ci": [sep_mean, sep_lo, sep_hi],
            },
            "similarity_gap": {
                "control_mean": sim_ctrl_mean,
                "shift_mean": sim_shift_mean,
                "separation_mean_ci": [sim_sep_mean, sim_sep_lo, sim_sep_hi],
            },
            "paired_t_test": {"t": t_stat, "p": p_value},
            "cohens_d": cohens_d,
            "domain_shift_verdict": domain_verdict,
            "gradual_drift_verdict": gradual_verdict,
        },
    }

    if gradual_results:
        output["analysis"]["gradual_drift"] = {
            "centroid_mean_ci": [grad_mean, grad_lo, grad_hi],
            "separation_mean_ci": [grad_sep_mean, grad_sep_lo, grad_sep_hi],
        }

    out_path = Path("results/topomem_obd_preflight")
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"obd_multiseed_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {fname}")


if __name__ == "__main__":
    main()
