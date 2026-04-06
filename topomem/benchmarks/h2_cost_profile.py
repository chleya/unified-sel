"""
P1: H2 Computational Cost Profiling

Measures the overhead of enabling H2 (max_homology_dim=2) vs H1 only (dim=1)
across different node counts.

Key question: Is H2 computation cost acceptable for default enablement?
"""

import sys, os, time, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "data" / "models" / "hf_cache")
warnings.filterwarnings("ignore")

import numpy as np
from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.config import EmbeddingConfig, TopologyConfig


def benchmark(embeddings, dim, n_runs=3):
    """Time compute_persistence for given dimension and embeddings."""
    times = []
    for _ in range(n_runs):
        cfg = TopologyConfig(max_homology_dim=dim, metric="cosine")
        topo = TopologyEngine(cfg)
        t0 = time.time()
        result = topo.compute_persistence(embeddings)
        elapsed = time.time() - t0
        times.append(elapsed)
    return np.mean(times), np.std(times), result


def main():
    print("=== P1: H2 Computational Cost Profiling ===\n")

    # Load and encode embeddings
    corp_path = Path(__file__).parent.parent / "data" / "test_corpus"
    import json as j
    prog = j.loads((corp_path / "programming.json").read_text(encoding="utf-8"))
    phys = j.loads((corp_path / "physics.json").read_text(encoding="utf-8"))
    geo = j.loads((corp_path / "geography.json").read_text(encoding="utf-8"))

    print("Encoding embeddings...")
    t0 = time.time()
    emb_cfg = EmbeddingConfig()
    emb_mgr = EmbeddingManager(emb_cfg)
    emb_all = emb_mgr.encode_batch([item["content"][:512] for item in prog + phys + geo])
    print(f"Encoded {len(emb_all)} items in {time.time()-t0:.1f}s\n")

    # Node counts to test
    node_counts = [10, 15, 20, 30, 40, 50, 60, 80, 100]
    n_runs = 3

    results = []

    print(f"{'n':>5} | {'H1 time':>10} | {'H2 time':>10} | {'Overhead':>10} | {'Ratio':>7} | {'H1 dim=1':>12} | {'H2 dim=2':>12}")
    print("-" * 80)

    for n in node_counts:
        # Sample n random embeddings
        np.random.seed(42)
        idx = np.random.permutation(len(emb_all))[:n]
        vecs = emb_all[idx]

        t_h1, s_h1, r1 = benchmark(vecs, dim=1, n_runs=n_runs)
        t_h2, s_h2, r2 = benchmark(vecs, dim=2, n_runs=n_runs)

        overhead_ms = (t_h2 - t_h1) * 1000
        overhead_ratio = t_h2 / t_h1 if t_h1 > 0 else float('inf')
        betti_1 = len(r1[1]) if len(r1) > 1 else 0
        betti_2 = len(r2[2]) if len(r2) > 2 else 0

        print(f"{n:>5} | {t_h1*1000:>9.1f}ms | {t_h2*1000:>9.1f}ms | {overhead_ms:>9.1f}ms | {overhead_ratio:>6.2f}x | betti_1={betti_1:>5} | betti_2={betti_2:>5}")

        results.append({
            "n": n,
            "h1_time_ms": round(t_h1 * 1000, 2),
            "h2_time_ms": round(t_h2 * 1000, 2),
            "overhead_ms": round(overhead_ms, 2),
            "ratio": round(overhead_ratio, 2),
            "betti_1": betti_1,
            "betti_2": betti_2,
        })

    print()

    # Analysis
    avg_ratio = np.mean([r["ratio"] for r in results])
    max_ratio = np.max([r["ratio"] for r in results])
    avg_overhead_ms = np.mean([r["overhead_ms"] for r in results])
    max_overhead_ms = np.max([r["overhead_ms"] for r in results])

    print("=== ANALYSIS ===\n")
    print(f"Average H2/H1 time ratio: {avg_ratio:.2f}x")
    print(f"Maximum H2/H1 time ratio: {max_ratio:.2f}x")
    print(f"Average overhead: {avg_overhead_ms:.1f}ms")
    print(f"Maximum overhead: {max_overhead_ms:.1f}ms")

    # Practical thresholds
    print("\n--- Practical Assessment ---")
    if avg_overhead_ms < 50:
        print("[PASS] H2 overhead is NEGLIGIBLE (< 50ms avg) - no excuse not to enable")
    elif avg_overhead_ms < 200:
        print("[WARN] H2 overhead is ACCEPTABLE (50-200ms) - acceptable for quality-critical use")
    else:
        print("[FAIL] H2 overhead is HIGH (> 200ms avg) - consider adaptive dim selection")

    if max_overhead_ms < 500:
        print("[PASS] Max overhead is LOW (< 500ms) - acceptable even for large n")
    elif max_overhead_ms < 2000:
        print("[WARN] Max overhead is MODERATE (500-2000ms) - acceptable with caching")
    else:
        print("[FAIL] Max overhead is HIGH (> 2s) - definitely need adaptive approach")

    # Decision
    print("\n--- Recommendation ---")
    if avg_ratio < 1.5 and max_ratio < 2.5:
        print("[RECOMMEND] Enable H2 by default (ratio < 1.5x, max < 2.5x)")
    elif avg_ratio < 2.0 and max_ratio < 4.0:
        print("[CONDITIONAL] Enable H2 with adaptive fallback for large n")
    else:
        print("[NOT RECOMMENDED] H2 overhead too high - keep disabled")

    # Also show: does H2 provide any signal at small n?
    print("\n--- H2 Signal Availability ---")
    for r in results:
        has_h2 = r["betti_2"] > 0
        has_h1 = r["betti_1"] > 0
        status = "[H2]" if has_h2 else "[NO-H2]"
        print(f"  n={r['n']:>3}: {status} betti_2={r['betti_2']:>2}, overhead: {r['overhead_ms']:.1f}ms")

    # Save
    out = {
        "results": results,
        "summary": {
            "avg_ratio": round(avg_ratio, 2),
            "max_ratio": round(max_ratio, 2),
            "avg_overhead_ms": round(avg_overhead_ms, 1),
            "max_overhead_ms": round(max_overhead_ms, 1),
        }
    }
    out_path = Path(__file__).parent / "results" / f"h2_cost_profile_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    return out


if __name__ == "__main__":
    main()
