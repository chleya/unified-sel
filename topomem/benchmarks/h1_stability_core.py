"""
H1 Stability Core Benchmark - Phase 1
Measures: H1 persistence ratio + diagram distance under domain invasion
不再看 purity（已经被验证是 trivial），聚焦 H1 拓扑稳定性

Design:
  Phase 1: Store 20 programming items (A), measure baseline H1
  Phase 2: Add 20 physics items (B), measure H1 drift
  Phase 3: Add 10 geography items (C), measure H1 drift
  
Core metrics:
  - H1 persistence ratio: H1_mean_pers_after / H1_mean_pers_before
  - H1 diagram distance: L2 distance between persistence diagrams before/after
  - H0 cluster count change
  - H1 cycle count change
"""

import os, sys, json, time
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).parent
_PKG_DIR = _SCRIPT_DIR.parent
_ROOT_DIR = _PKG_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

from topomem.system import TopoMemSystem
from topomem.config import TopoMemConfig, MemoryConfig, EmbeddingConfig, TopologyConfig


def load_corpus(domain, limit=None):
    path = _PKG_DIR / "data" / "test_corpus" / f"{domain}.json"
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return items[:limit] if limit else items


def persistence_diagram_stats(diagrams, topo_engine, vecs):
    """Compute H0/H1 stats from persistence diagrams."""
    h0 = diagrams[0] if len(diagrams) > 0 else np.array([])
    h1 = diagrams[1] if len(diagrams) > 1 else np.array([])
    
    def pers(diagram):
        return [float(r[1] - r[0]) for r in diagram if np.isfinite(r[1])]
    
    h0_vals = pers(h0)
    h1_vals = pers(h1)
    labels = topo_engine.compute_full_result(vecs).cluster_labels
    
    return {
        "h0_clusters": int(len(set(labels))) if labels is not None else 0,
        "h0_n": len(h0_vals),
        "h0_total_pers": float(np.sum(h0_vals)),
        "h0_mean_pers": float(np.mean(h0_vals)) if h0_vals else 0.0,
        "h1_n": len(h1_vals),
        "h1_total_pers": float(np.sum(h1_vals)),
        "h1_mean_pers": float(np.mean(h1_vals)) if h1_vals else 0.0,
        "h1_max_pers": float(np.max(h1_vals)) if h1_vals else 0.0,
        "h1_min_pers": float(np.min(h1_vals)) if h1_vals else 0.0,
    }


def diagram_l2_distance(diagram1, diagram2):
    """Approximate bottleneck/Wasserstein L2 distance between two diagrams."""
    if len(diagram1) == 0:
        diagram1 = np.array([[0.0, 0.0]])
    if len(diagram2) == 0:
        diagram2 = np.array([[0.0, 0.0]])
    arr1 = np.array(diagram1, dtype=np.float64)
    arr2 = np.array(diagram2, dtype=np.float64)
    # Pairwise Euclidean distance matrix, take min per row
    dists = []
    for p1 in arr1:
        min_d = min(np.linalg.norm(p1 - p2) for p2 in arr2)
        dists.append(min_d)
    return float(np.mean(dists))


def run():
    print("=" * 60)
    print("H1 STABILITY CORE BENCHMARK")
    print("Metrics: H1 persistence ratio + diagram distance")
    print("=" * 60)
    
    prog = load_corpus("programming", limit=20)
    phys = load_corpus("physics", limit=20)
    geo = load_corpus("geography", limit=10)
    print(f"Corpus: {len(prog)} prog(A) + {len(phys)} phys(B) + {len(geo)} geo(C)")
    
    tmpdir = "F:\\tmp\\h1_core_bench"
    os.makedirs(tmpdir, exist_ok=True)
    
    config = TopoMemConfig(
        memory=MemoryConfig(chroma_persist_dir=tmpdir),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        topology=TopologyConfig(max_homology_dim=1, filtration_steps=30, metric="cosine"),
        engine=None,
    )
    system = TopoMemSystem(config=config)
    
    # Phase 1: Baseline A
    print("\n[Phase 1] Store programming (A) - baseline H1...")
    for item in prog:
        system.add_knowledge(item["content"], metadata={"domain": "A", "id": item["id"]})
    system.memory.update_topology(system.topology)
    
    a_vecs = system.embedding.encode_batch([x["content"] for x in prog])
    a_diagrams = system.topology.compute_persistence(np.array(a_vecs))
    phase1 = persistence_diagram_stats(a_diagrams, system.topology, a_vecs)
    print(f"  H0 clusters: {phase1['h0_clusters']}")
    print(f"  H1 cycles: {phase1['h1_n']}, mean_pers={phase1['h1_mean_pers']:.4f}, max_pers={phase1['h1_max_pers']:.4f}")
    
    # Phase 2: A + B
    print("\n[Phase 2] Store physics (B) - measure H1 drift...")
    for item in phys:
        system.add_knowledge(item["content"], metadata={"domain": "B", "id": item["id"]})
    system.memory.update_topology(system.topology)
    
    ab_vecs = system.embedding.encode_batch([x["content"] for x in prog + phys])
    ab_diagrams = system.topology.compute_persistence(np.array(ab_vecs))
    phase2 = persistence_diagram_stats(ab_diagrams, system.topology, ab_vecs)
    
    # H1 diagram distance A vs A+B
    h1_dist_AB = diagram_l2_distance(a_diagrams[1] if len(a_diagrams) > 1 else np.array([]),
                                      ab_diagrams[1] if len(ab_diagrams) > 1 else np.array([]))
    
    # H1 stability ratio A -> A+B
    h1_ratio_AB = phase1["h1_mean_pers"] / (phase2["h1_mean_pers"] + 1e-10)
    
    print(f"  H0 clusters: {phase1['h0_clusters']} -> {phase2['h0_clusters']}")
    print(f"  H1 cycles: {phase1['h1_n']} -> {phase2['h1_n']}")
    print(f"  H1 mean_pers: {phase1['h1_mean_pers']:.4f} -> {phase2['h1_mean_pers']:.4f}")
    print(f"  H1 stability ratio: {h1_ratio_AB:.3f}")
    print(f"  H1 diagram distance: {h1_dist_AB:.4f}")
    
    # Phase 3: A + B + C
    print("\n[Phase 3] Store geography (C) - measure H1 drift...")
    for item in geo:
        system.add_knowledge(item["content"], metadata={"domain": "C", "id": item["id"]})
    system.memory.update_topology(system.topology)
    
    abc_vecs = system.embedding.encode_batch([x["content"] for x in prog + phys + geo])
    abc_diagrams = system.topology.compute_persistence(np.array(abc_vecs))
    phase3 = persistence_diagram_stats(abc_diagrams, system.topology, abc_vecs)
    
    h1_dist_ABC = diagram_l2_distance(a_diagrams[1] if len(a_diagrams) > 1 else np.array([]),
                                        abc_diagrams[1] if len(abc_diagrams) > 1 else np.array([]))
    h1_ratio_ABC = phase1["h1_mean_pers"] / (phase3["h1_mean_pers"] + 1e-10)
    
    print(f"  H0 clusters: {phase2['h0_clusters']} -> {phase3['h0_clusters']}")
    print(f"  H1 cycles: {phase2['h1_n']} -> {phase3['h1_n']}")
    print(f"  H1 mean_pers: {phase2['h1_mean_pers']:.4f} -> {phase3['h1_mean_pers']:.4f}")
    print(f"  H1 stability ratio: {h1_ratio_ABC:.3f}")
    print(f"  H1 diagram distance: {h1_dist_ABC:.4f}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"H1 STABILITY SUMMARY")
    print(f"{'='*50}")
    print(f"Baseline:     {phase1['h1_n']} cycles, pers={phase1['h1_mean_pers']:.4f}")
    print(f"After B:      {phase2['h1_n']} cycles, pers={phase2['h1_mean_pers']:.4f}, ratio={h1_ratio_AB:.3f}")
    print(f"After C:      {phase3['h1_n']} cycles, pers={phase3['h1_mean_pers']:.4f}, ratio={h1_ratio_ABC:.3f}")
    print(f"H1 dist(A->B): {h1_dist_AB:.4f}")
    print(f"H1 dist(A->C): {h1_dist_ABC:.4f}")
    
    # Key insight
    if h1_ratio_AB > 0.8:
        print(f"\n==> H1 is HIGHLY STABLE (ratio={h1_ratio_AB:.3f} > 0.8)")
    elif h1_ratio_AB > 0.5:
        print(f"\n==> H1 is MODERATELY STABLE (ratio={h1_ratio_AB:.3f})")
    else:
        print(f"\n==> H1 is UNSTABLE (ratio={h1_ratio_AB:.3f} < 0.5)")
    
    results = {
        "phase1_baseline": phase1,
        "phase2_AB": phase2,
        "phase3_ABC": phase3,
        "h1_stability_ratio_AB": float(h1_ratio_AB),
        "h1_stability_ratio_ABC": float(h1_ratio_ABC),
        "h1_diagram_distance_AB": float(h1_dist_AB),
        "h1_diagram_distance_ABC": float(h1_dist_ABC),
        "summary": {
            "h1_stable": bool(h1_ratio_AB > 0.8),
            "h1_ratio": float(h1_ratio_AB),
            "h1_dist_AB": float(h1_dist_AB),
            "h1_dist_ABC": float(h1_dist_ABC),
        }
    }
    
    out_path = _SCRIPT_DIR / "results" / f"h1_stability_core_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
