"""
Phase 1: Within-Domain Semantic Clustering
Claude's recommended design: Test if H1 encodes within-domain semantic structure.

Design:
- Use programming domain (20 items)
- Create 2 semantic sub-clusters via embedding similarity (k-means on embeddings)
- Compare H1 characteristics within semantically similar vs dissimilar groups
- Key question: Do items in the same semantic cluster have more similar H1 structure?
"""

import os, sys, json, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

_SCRIPT_DIR = Path(__file__).parent
_PKG_DIR = _SCRIPT_DIR.parent
_ROOT_DIR = _PKG_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.config import EmbeddingConfig, TopologyConfig


def load_corpus(domain, limit=None):
    path = _PKG_DIR / "data" / "test_corpus" / f"{domain}.json"
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return items[:limit] if limit else items


def h1_pers_stats(emb_mgr, topo_engine, vecs):
    """Return list of H1 persistence values for a set of vectors."""
    diagrams = topo_engine.compute_persistence(np.array(vecs, dtype=np.float64))
    if len(diagrams) > 1:
        pers = [float(r[1]-r[0]) for r in diagrams[1] if np.isfinite(r[1])]
        return pers
    return []


def semantic_coherence(emb_mgr, vecs, labels):
    """Compute average cosine similarity within clusters vs between clusters."""
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(vecs)
    n = len(labels)
    
    within_sum = 0.0
    within_count = 0
    between_sum = 0.0
    between_count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] == labels[j]:
                within_sum += sims[i, j]
                within_count += 1
            else:
                between_sum += sims[i, j]
                between_count += 1
    
    within_mean = within_sum / within_count if within_count > 0 else 0
    between_mean = between_sum / between_count if between_count > 0 else 0
    
    return float(within_mean), float(between_mean)


def main():
    print("=" * 60)
    print("PHASE 1: Within-Domain Semantic Clustering")
    print("Testing if H1 encodes within-domain semantic structure")
    print("=" * 60)
    
    emb_mgr = EmbeddingManager(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    topo_engine = TopologyEngine(TopologyConfig(max_homology_dim=1, filtration_steps=30, metric="cosine"))
    
    prog = load_corpus("programming", limit=20)
    prog_texts = [x["content"] for x in prog]
    
    print(f"\nDomain: Programming, {len(prog_texts)} items")
    
    # Encode all items
    print("Encoding...")
    vecs = emb_mgr.encode_batch(prog_texts)
    vecs = np.array(vecs, dtype=np.float64)
    
    # Create 2 semantic clusters via k-means on embeddings
    print("K-means clustering (k=2) on embeddings...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vecs)
    
    cluster0_items = [i for i, l in enumerate(labels) if l == 0]
    cluster1_items = [i for i, l in enumerate(labels) if l == 1]
    
    print(f"  Cluster 0: {len(cluster0_items)} items: {[prog[i]['id'] for i in cluster0_items]}")
    print(f"  Cluster 1: {len(cluster1_items)} items: {[prog[i]['id'] for i in cluster1_items]}")
    
    # Semantic coherence check
    within_sim, between_sim = semantic_coherence(emb_mgr, vecs, labels)
    print(f"\n  Within-cluster similarity: {within_sim:.4f}")
    print(f"  Between-cluster similarity: {between_sim:.4f}")
    print(f"  Separation ratio: {within_sim/between_sim:.3f}")
    
    # H1 analysis: within each cluster
    print(f"\n[H1 Analysis]")
    
    # Full set H1
    full_pers = h1_pers_stats(emb_mgr, topo_engine, vecs)
    print(f"  Full set (n={len(prog_texts)}): {len(full_pers)} cycles, mean={np.mean(full_pers):.4f}")
    
    # Each cluster individually
    c0_vecs = vecs[cluster0_items]
    c1_vecs = vecs[cluster1_items]
    
    c0_pers = h1_pers_stats(emb_mgr, topo_engine, c0_vecs)
    c1_pers = h1_pers_stats(emb_mgr, topo_engine, c1_vecs)
    
    print(f"  Cluster 0: {len(c0_pers)} cycles, mean={np.mean(c0_pers):.4f}, median={np.median(c0_pers):.4f}")
    print(f"  Cluster 1: {len(c1_pers)} cycles, mean={np.mean(c1_pers):.4f}, median={np.median(c1_pers):.4f}")
    
    # Combined clusters = full set? (check if they're additive)
    # The key test: do clusters have more or less persistence than random groups?
    
    # Random grouping test: compare H1 of true clusters vs random clusters
    print(f"\n[Random Grouping Comparison]")
    n_perm = 1000
    random_cluster_diffs = []
    
    for _ in range(n_perm):
        rand_labels = np.random.permutation(len(labels))
        c0_rand = [vecs[i] for i in range(len(labels)) if rand_labels[i] == 0]
        c1_rand = [vecs[i] for i in range(len(labels)) if rand_labels[i] == 1]
        
        if len(c0_rand) >= 3 and len(c1_rand) >= 3:
            p0 = h1_pers_stats(emb_mgr, topo_engine, np.array(c0_rand, dtype=np.float64))
            p1 = h1_pers_stats(emb_mgr, topo_engine, np.array(c1_rand, dtype=np.float64))
            if p0 and p1 and np.mean(p0) > 0 and np.mean(p1) > 0:
                random_cluster_diffs.append(abs(np.mean(p0) - np.mean(p1)))
    
    true_diff = abs(np.mean(c0_pers) - np.mean(c1_pers)) if c0_pers and c1_pers and np.mean(c0_pers) > 0 and np.mean(c1_pers) > 0 else 0
    random_diffs = np.array(random_cluster_diffs) if random_cluster_diffs else np.array([0.0])
    
    p_val = float(np.mean(random_diffs >= true_diff)) if len(random_diffs) > 0 else 0.5
    
    print(f"  True cluster mean persistence diff: {true_diff:.4f}")
    print(f"  Random grouping mean diff: {np.mean(random_diffs):.4f} ± {np.std(random_diffs):.4f}")
    print(f"  Random grouping min/max: {np.min(random_diffs):.4f} / {np.max(random_diffs):.4f}")
    print(f"  p (true diff >= random diff): {p_val:.4f}")
    
    if p_val < 0.05:
        print(f"  => Semantic clusters have SIGNIFICANTLY different H1 structure")
    else:
        print(f"  => No significant difference between semantic and random clusters")
    
    # Persistence distribution within clusters: are they clustered or spread out?
    print(f"\n[Persistence Distribution per Cluster]")
    
    def stats(pers_list):
        if not pers_list:
            return "no cycles"
        return f"min={min(pers_list):.4f}, max={max(pers_list):.4f}, std={np.std(pers_list):.4f}"
    
    print(f"  Cluster 0: {stats(c0_pers)}")
    print(f"  Cluster 1: {stats(c1_pers)}")
    print(f"  Full set:  {stats(full_pers)}")
    
    # Key test: within-cluster persistence variance
    # If H1 encodes semantic structure, items in same cluster should have similar persistence
    # Measure: coefficient of variation (CV) within true clusters vs random clusters
    
    print(f"\n[Within-Cluster Persistence Homogeneity]")
    # True clusters
    c0_cv = np.std(c0_pers) / np.mean(c0_pers) if c0_pers and np.mean(c0_pers) > 0 else 0
    c1_cv = np.std(c1_pers) / np.mean(c1_pers) if c1_pers and np.mean(c1_pers) > 0 else 0
    true_avg_cv = (c0_cv + c1_cv) / 2
    
    print(f"  True clusters CV: cluster0={c0_cv:.3f}, cluster1={c1_cv:.3f}, avg={true_avg_cv:.3f}")
    
    # Random clusters CV
    random_cvs = []
    for _ in range(n_perm):
        rand_labels = np.random.permutation(len(labels))
        for cluster_id in [0, 1]:
            cluster_vecs = [vecs[i] for i in range(len(labels)) if rand_labels[i] == cluster_id]
            if len(cluster_vecs) >= 2:
                p = h1_pers_stats(emb_mgr, topo_engine, np.array(cluster_vecs, dtype=np.float64))
                if p and np.mean(p) > 0:
                    random_cvs.append(np.std(p) / np.mean(p))
    
    random_cv = np.mean(random_cvs) if random_cvs else 0
    print(f"  Random clusters CV: {random_cv:.3f}")
    print(f"  True clusters MORE homogeneous than random: {true_avg_cv < random_cv}")
    
    results = {
        "n_clusters": 2,
        "cluster_sizes": [len(cluster0_items), len(cluster1_items)],
        "cluster0_cycles": len(c0_pers),
        "cluster0_mean_pers": float(np.mean(c0_pers)) if c0_pers else 0,
        "cluster1_cycles": len(c1_pers),
        "cluster1_mean_pers": float(np.mean(c1_pers)) if c1_pers else 0,
        "full_cycles": len(full_pers),
        "full_mean_pers": float(np.mean(full_pers)),
        "within_cluster_sim": within_sim,
        "between_cluster_sim": between_sim,
        "separation_ratio": float(within_sim/between_sim),
        "true_diff_vs_random_p": p_val,
        "true_avg_cv": true_avg_cv,
        "random_avg_cv": float(random_cv),
    }
    
    out_path = _SCRIPT_DIR / "results" / f"phase1_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    
    print(f"\n{'='*50}")
    print(f"PHASE 1 INTERPRETATION")
    print(f"{'='*50}")
    if results["true_diff_vs_random_p"] < 0.05:
        print(f"[PASS] Semantic clusters have significantly different H1")
        print(f"  => H1 encodes semantic sub-structure")
    else:
        print(f"[FAIL] No significant H1 difference between semantic clusters")
        print(f"  => H1 does NOT encode within-domain semantic sub-structure")
    
    if results["separation_ratio"] > 1.5:
        print(f"[OK] Embedding clusters are well-separated (ratio={results['separation_ratio']:.2f})")
    else:
        print(f"[WARN] Embedding clusters NOT well-separated (ratio={results['separation_ratio']:.2f})")


if __name__ == "__main__":
    main()
