"""
TopoMem OBD Preflight: Batch-Level Drift Detection Experiment

Hypothesis: TopoMem's batch-level drift signals (Wasserstein distance, H1/H2
metrics) can detect when the task distribution shifts from one domain to another.

This is DIFFERENT from the failed per-task routing experiment:
- Per-task routing: "Is THIS task surprising?" -> Failed (embedding novelty != answer correctness)
- Batch-level OBD: "Has the distribution SHIFTED over the last N tasks?" -> Unvalidated

Experiment design:
  Batch A: 20 code tasks (baseline)
  Batch B: 20 code tasks (same distribution, control)
  Batch C: 20 reasoning tasks (different distribution, shift)
  Batch D: 20 mixed tasks (partial shift)

Expected:
  A->B drift: LOW (same distribution)
  B->C drift: HIGH (domain shift)
  C->D drift: MEDIUM (partial shift back)

Usage:
    python experiments/capability/topomem_obd_preflight.py
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


def compute_batch_wasserstein(embed_mgr, emb_a, emb_b):
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


def compute_persistence_stats(topo_engine, embeddings, label):
    result = topo_engine.compute_persistence(embeddings)
    h0 = result[0] if len(result) > 0 else np.array([]).reshape(0, 2)
    h1 = result[1] if len(result) > 1 else np.array([]).reshape(0, 2)
    h0_finite = h0[(h0[:, 1] - h0[:, 0]) > 1e-10] if len(h0) > 0 else np.array([]).reshape(0, 2)
    h1_finite = h1[(h1[:, 1] - h1[:, 0]) > 1e-10] if len(h1) > 0 else np.array([]).reshape(0, 2)
    h0_persistences = (h0_finite[:, 1] - h0_finite[:, 0]) if len(h0_finite) > 0 else np.array([])
    h1_persistences = (h1_finite[:, 1] - h1_finite[:, 0]) if len(h1_finite) > 0 else np.array([])
    stats = {
        "label": label,
        "n_points": len(embeddings),
        "h0_n_features": len(h0_finite),
        "h0_mean_persistence": float(np.mean(h0_persistences)) if len(h0_persistences) > 0 else 0.0,
        "h0_max_persistence": float(np.max(h0_persistences)) if len(h0_persistences) > 0 else 0.0,
        "h1_n_features": len(h1_finite),
        "h1_mean_persistence": float(np.mean(h1_persistences)) if len(h1_persistences) > 0 else 0.0,
        "h1_max_persistence": float(np.max(h1_persistences)) if len(h1_persistences) > 0 else 0.0,
    }
    return stats, result


def compute_wasserstein_drift(topo_engine, diag_a, diag_b, dim=0):
    try:
        return topo_engine.wasserstein_distance(diag_a, diag_b, dim=dim)
    except Exception:
        return -1.0


def compute_centroid_drift(emb_a, emb_b):
    c_a = np.mean(emb_a, axis=0)
    c_b = np.mean(emb_b, axis=0)
    c_a_norm = c_a / (np.linalg.norm(c_a) + 1e-10)
    c_b_norm = c_b / (np.linalg.norm(c_b) + 1e-10)
    return float(1.0 - np.dot(c_a_norm, c_b_norm))


def compute_covariance_shift(emb_a, emb_b):
    cov_a = np.cov(emb_a.T)
    cov_b = np.cov(emb_b.T)
    diff = cov_a - cov_b
    return float(np.linalg.norm(diff, "fro"))


def main():
    print("=" * 60)
    print("TopoMem OBD Preflight: Batch-Level Drift Detection")
    print("=" * 60)

    print("\n[1/5] Loading embedding model...")
    embed_mgr = EmbeddingManager()
    topo_engine = TopologyEngine()

    print("[2/5] Building task batches...")
    code_tasks = build_task_suite("code", 20, seed=7)
    reasoning_tasks = build_task_suite("reasoning", 20, seed=7)
    mixed_tasks = build_task_suite("mixed", 40, seed=7)
    code_batch_1 = code_tasks[:10]
    code_batch_2 = code_tasks[10:]
    reasoning_batch = reasoning_tasks[:20]
    mixed_code = [t for t in mixed_tasks if t.family == "code"][:10]
    mixed_reasoning = [t for t in mixed_tasks if t.family == "reasoning"][:10]
    mixed_batch = mixed_code + mixed_reasoning

    batches = {
        "A_code_1": code_batch_1,
        "B_code_2": code_batch_2,
        "C_reasoning": reasoning_batch,
        "D_mixed": mixed_batch,
    }

    print(f"  Batch A (code baseline): {len(code_batch_1)} tasks")
    print(f"  Batch B (code control):  {len(code_batch_2)} tasks")
    print(f"  Batch C (reasoning):     {len(reasoning_batch)} tasks")
    print(f"  Batch D (mixed):         {len(mixed_batch)} tasks")

    print("\n[3/5] Encoding task prompts...")
    embeddings = {}
    for name, tasks in batches.items():
        emb = encode_batch(embed_mgr, tasks)
        embeddings[name] = emb
        print(f"  {name}: shape {emb.shape}")

    print("\n[4/5] Computing drift signals...")
    transitions = [
        ("A_code_1", "B_code_2", "CONTROL (same domain)"),
        ("A_code_1", "C_reasoning", "SHIFT (code -> reasoning)"),
        ("B_code_2", "C_reasoning", "SHIFT (code -> reasoning)"),
        ("C_reasoning", "D_mixed", "PARTIAL (reasoning -> mixed)"),
        ("A_code_1", "D_mixed", "PARTIAL (code -> mixed)"),
    ]

    drift_results = []
    for src, dst, desc in transitions:
        emb_src = embeddings[src]
        emb_dst = embeddings[dst]

        centroid_drift = compute_centroid_drift(emb_src, emb_dst)
        cov_shift = compute_covariance_shift(emb_src, emb_dst)
        sim_drift = compute_batch_wasserstein(embed_mgr, emb_src, emb_dst)

        stats_src, diag_src = compute_persistence_stats(topo_engine, emb_src, src)
        stats_dst, diag_dst = compute_persistence_stats(topo_engine, emb_dst, dst)

        w0_drift = compute_wasserstein_drift(topo_engine, diag_src, diag_dst, dim=0)
        w1_drift = compute_wasserstein_drift(topo_engine, diag_src, diag_dst, dim=1)

        result = {
            "transition": f"{src} -> {dst}",
            "description": desc,
            "centroid_drift": centroid_drift,
            "covariance_shift": cov_shift,
            "similarity_gap": sim_drift,
            "wasserstein_H0": w0_drift,
            "wasserstein_H1": w1_drift,
            "src_h0_features": stats_src["h0_n_features"],
            "src_h1_features": stats_src["h1_n_features"],
            "dst_h0_features": stats_dst["h0_n_features"],
            "dst_h1_features": stats_dst["h1_n_features"],
            "src_h0_mean_persist": stats_src["h0_mean_persistence"],
            "dst_h0_mean_persist": stats_dst["h0_mean_persistence"],
            "src_h1_mean_persist": stats_src["h1_mean_persistence"],
            "dst_h1_mean_persist": stats_dst["h1_mean_persistence"],
        }
        drift_results.append(result)

        print(f"\n  {src} -> {dst} ({desc}):")
        print(f"    Centroid drift:      {centroid_drift:.4f}")
        print(f"    Covariance shift:    {cov_shift:.4f}")
        print(f"    Similarity gap:      {sim_drift:.4f}")
        print(f"    Wasserstein H0:      {w0_drift:.4f}")
        print(f"    Wasserstein H1:      {w1_drift:.4f}")
        print(f"    H0 features:         {stats_src['h0_n_features']} -> {stats_dst['h0_n_features']}")
        print(f"    H1 features:         {stats_src['h1_n_features']} -> {stats_dst['h1_n_features']}")

    print("\n[5/5] Analysis...")

    control = drift_results[0]
    shifts = [r for r in drift_results if "SHIFT" in r["description"]]
    partials = [r for r in drift_results if "PARTIAL" in r["description"]]

    print("\n  Drift Signal Comparison:")
    print(f"  {'Transition':<35} {'Centroid':>10} {'CovShift':>10} {'SimGap':>10} {'W-H0':>10} {'W-H1':>10}")
    print(f"  {'-'*85}")
    for r in drift_results:
        print(f"  {r['transition'] + ' (' + r['description'][:7] + ')':<35} "
              f"{r['centroid_drift']:>10.4f} {r['covariance_shift']:>10.4f} "
              f"{r['similarity_gap']:>10.4f} {r['wasserstein_H0']:>10.4f} "
              f"{r['wasserstein_H1']:>10.4f}")

    shift_centroids = [r["centroid_drift"] for r in shifts]
    control_centroid = control["centroid_drift"]
    separation = min(shift_centroids) / max(control_centroid, 1e-10)

    print(f"\n  Key Metric: Centroid Drift")
    print(f"    Control (A->B): {control_centroid:.4f}")
    print(f"    Shift min:      {min(shift_centroids):.4f}")
    print(f"    Separation:     {separation:.1f}x")

    verdict = "INCONCLUSIVE"
    if separation > 5.0 and all(r["centroid_drift"] > 3 * control_centroid for r in shifts):
        verdict = "PROMISING - drift signals clearly separate domain shift from control"
    elif separation > 2.0:
        verdict = "WEAK POSITIVE - drift signals show some separation but overlap possible"
    elif separation > 1.0:
        verdict = "MARGINAL - drift signals barely distinguish shift from control"
    else:
        verdict = "NEGATIVE - drift signals cannot distinguish shift from control"

    print(f"\n  Verdict: {verdict}")

    output = {
        "schema_version": "capbench.result.v1",
        "metadata": {
            "data_source": "verified_execution",
            "cost_model": "abstract_units_v1",
            "oracle_assumption": False,
            "verifier_policy": "topomem_obd_preflight",
            "benchmark_suite": "obd-preflight-4batch",
            "task_count": 60,
            "seeds": [7],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "experiment": "topomem_obd_preflight",
        "hypothesis": "Batch-level drift signals can detect domain distribution shifts",
        "batches": {k: len(v) for k, v in batches.items()},
        "drift_results": drift_results,
        "analysis": {
            "control_centroid_drift": control_centroid,
            "shift_min_centroid_drift": min(shift_centroids),
            "separation_ratio": separation,
            "verdict": verdict,
        },
    }

    out_path = Path("results/topomem_obd_preflight")
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"obd_preflight_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {fname}")


if __name__ == "__main__":
    main()
