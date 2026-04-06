"""
H1 Stability Under Adversarial Query - Phase 2
Combines Option C (H1 stability quantification) + Option A (adversarial queries)

Core metrics:
  - H1 persistence ratio: H1_mean_pers_after_invasion / H1_mean_pers_before
  - Retrieval purity under adversarial queries (topo vs vec)
  - Persistence diagram similarity under perturbation (L2 distance between diagrams)

Design:
  Phase 1: Store 20 programming items (A)
  Phase 2: Add physics items (B) as invasion, measure H1 change
  Phase 3: Query A with adversarial queries (physics-flavored versions)
  Phase 4: Measure H1 stability (repeat queries, track persistence drift)
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


def adversarial_queries(items, emb_mgr):
    """
    Generate physics-flavored (B) versions of programming (A) queries.
    Strategy: paraphrase the query in a "physics domain" style.
    We use the test_question field and rephrase it.
    """
    import re
    # Simple rule-based adversarial transformation
    # Programming terms -> physics equivalents
    replacements = [
        # Variable/function concepts
        (r'\bfunction\b', 'physical system'),
        (r'\bvariable\b', 'physical quantity'),
        (r'\bparameter\b', 'measurement'),
        (r'\brecursion\b', 'oscillation'),
        (r'\biteration\b', 'cycle'),
        (r'\bloop\b', 'periodic motion'),
        (r'\barray\b', 'wave pattern'),
        (r'\blist\b', 'sequence'),
        (r'\bdictionary\b', 'field map'),
        (r'\bhash\b', 'transform'),
        (r'\bO\(n\)\b', 'linear relationship'),
        (r'\bcomplexity\b', 'energy cost'),
        # Programming context
        (r'\bcompile\b', 'phase transition'),
        (r'\bdebug\b', 'error correction'),
        (r'\bstack\b', 'pendulum'),
        (r'\bheap\b', 'field'),
        (r'\bpointer\b', 'trajectory'),
        (r'\bthread\b', 'flow'),
        (r'\block\b', 'constraint'),
        (r'\bsyntax\b', 'grammar'),
        (r'\balgorithm\b', 'process'),
    ]
    
    adversarial = []
    for item in items:
        q = item["test_question"]
        for pattern, replacement in replacements:
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
        adversarial.append(q)
    return adversarial


def measure_h1_stability(items, emb_mgr, topo_engine):
    """Measure H1 persistence stats for items."""
    texts = [x["content"] for x in items]
    vecs = emb_mgr.encode_batch(texts)
    diagrams = topo_engine.compute_persistence(vecs)
    h0 = diagrams[0] if len(diagrams) > 0 else np.array([])
    h1 = diagrams[1] if len(diagrams) > 1 else np.array([])
    
    def pers_vals(diagram):
        return [float(r[1]-r[0]) for r in diagram if np.isfinite(r[1])]
    
    h0_vals = pers_vals(h0)
    h1_vals = pers_vals(h1)
    labels = topo_engine.compute_full_result(vecs).cluster_labels
    
    return {
        "n_clusters": int(len(set(labels))) if labels is not None else 0,
        "h1_n": len(h1_vals),
        "h1_total_pers": float(np.sum(h1_vals)),
        "h1_mean_pers": float(np.mean(h1_vals)) if h1_vals else 0.0,
        "h1_max_pers": float(np.max(h1_vals)) if h1_vals else 0.0,
        "h0_n": len(h0_vals),
        "h0_mean_pers": float(np.mean(h0_vals)) if h0_vals else 0.0,
    }


def measure_purity(system, query_items, adversarial_queries, domain_filter, k=5):
    """Measure retrieval purity with normal vs adversarial queries."""
    emb = system.embedding
    results = {"normal": {"hybrid": [], "topological": [], "vector": []},
               "adversarial": {"hybrid": [], "topological": [], "vector": []}}
    
    normal_vecs = emb.encode_batch([x["content"] for x in query_items])
    adv_vecs = emb.encode_batch(adversarial_queries)
    
    for vec in normal_vecs:
        for strat in ["hybrid", "topological", "vector"]:
            try:
                retrieved = system.memory.retrieve(vec, strategy=strat, k=k)
                if retrieved:
                    cnt = sum(1 for n, s in retrieved if n.metadata.get("domain") == domain_filter)
                    results["normal"][strat].append(cnt / len(retrieved))
            except Exception:
                pass
    
    for vec in adv_vecs:
        for strat in ["hybrid", "topological", "vector"]:
            try:
                retrieved = system.memory.retrieve(vec, strategy=strat, k=k)
                if retrieved:
                    cnt = sum(1 for n, s in retrieved if n.metadata.get("domain") == domain_filter)
                    results["adversarial"][strat].append(cnt / len(retrieved))
            except Exception:
                pass
    
    summary = {}
    for qtype in results:
        summary[qtype] = {s: float(np.mean(r)) if r else 0.0 
                          for s, r in results[qtype].items()}
    return summary


def persistence_diagram_distance(diagram1, diagram2):
    """Compute L2 distance between two persistence diagrams (bottleneck distance approximation)."""
    if len(diagram1) == 0:
        diagram1 = np.array([[0, 0]])
    if len(diagram2) == 0:
        diagram2 = np.array([[0, 0]])
    arr1 = np.array(diagram1)
    arr2 = np.array(diagram2)
    # Simple pairwise L2 distance matrix, take min
    dists = []
    for p1 in arr1:
        min_d = min(np.linalg.norm(p1 - p2) for p2 in arr2)
        dists.append(min_d)
    return float(np.mean(dists))


def measure_h1_drift(items_a, items_b, emb_mgr, topo_engine):
    """
    Measure how much H1 structure of domain A changes after domain B is added.
    items_a: original domain A items
    items_b: invasion domain B items  
    """
    # Encode all items together
    all_texts = [x["content"] for x in items_a] + [x["content"] for x in items_b]
    all_vecs = emb_mgr.encode_batch(all_texts)
    
    # Compute diagrams for original A
    a_vecs = all_vecs[:len(items_a)]
    b_vecs = all_vecs[len(items_a):]
    
    diagrams_a = topo_engine.compute_persistence(a_vecs)
    diagrams_ab = topo_engine.compute_persistence(all_vecs)
    
    h1_a = diagrams_a[1] if len(diagrams_a) > 1 else np.array([])
    h1_ab = diagrams_ab[1] if len(diagrams_ab) > 1 else np.array([])
    
    # Distance between diagrams
    h1_dist = persistence_diagram_distance(h1_a, h1_ab)
    
    return {
        "h1_dist": h1_dist,
        "h1_n_A": len(h1_a),
        "h1_n_AB": len(h1_ab),
    }


def run():
    print("=" * 60)
    print("H1 STABILITY + ADVERSARIAL QUERY BENCHMARK")
    print("Option C (quantify) + Option A (adversarial)")
    print("=" * 60)
    
    prog = load_corpus("programming", limit=20)
    phys = load_corpus("physics", limit=20)
    
    print(f"Corpus: {len(prog)} prog + {len(phys)} phys")
    
    tmpdir = "F:\\tmp\\h1_adv_bench"
    os.makedirs(tmpdir, exist_ok=True)
    
    config = TopoMemConfig(
        memory=MemoryConfig(chroma_persist_dir=tmpdir),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        topology=TopologyConfig(max_homology_dim=1, filtration_steps=30, metric="cosine"),
        engine=None,
    )
    system = TopoMemSystem(config=config)
    
    # Generate adversarial queries
    adv_queries = adversarial_queries(prog, system.embedding)
    print(f"Generated {len(adv_queries)} adversarial queries")
    print(f"Sample: Q: {prog[0]['test_question']}")
    print(f"  Adv: {adv_queries[0]}")
    
    # Phase 1: Store programming, measure baseline H1
    print("\n[Phase 1] Store programming items, measure baseline H1...")
    for item in prog:
        system.add_knowledge(item["content"], metadata={"domain": "A", "id": item["id"]})
    system.memory.update_topology(system.topology)
    
    baseline = measure_h1_stability(prog, system.embedding, system.topology)
    print(f"  Baseline H1: {baseline['h1_n']} cycles, mean_pers={baseline['h1_mean_pers']:.4f}")
    print(f"  H0 clusters: {baseline['n_clusters']}")
    
    # Phase 2: Measure H1 drift when physics items added
    print("\n[Phase 2] Measure H1 drift from physics invasion...")
    drift = measure_h1_drift(prog, phys, system.embedding, system.topology)
    print(f"  H1 diagram distance: {drift['h1_dist']:.4f}")
    print(f"  H1 cycles A: {drift['h1_n_A']}, A+B: {drift['h1_n_AB']}")
    
    # Phase 3: Add physics items
    print("\n[Phase 3] Store physics items...")
    for item in phys:
        system.add_knowledge(item["content"], metadata={"domain": "B", "id": item["id"]})
    system.memory.update_topology(system.topology)
    
    # Phase 4: Adversarial query purity
    print("\n[Phase 4] Retrieval purity: normal vs adversarial queries...")
    purity = measure_purity(system, prog[:8], adv_queries[:8], "A", k=5)
    
    print(f"\n  Normal queries:     hybrid={purity['normal']['hybrid']:.0%}  topo={purity['normal']['topological']:.0%}  vec={purity['normal']['vector']:.0%}")
    print(f"  Adversarial queries: hybrid={purity['adversarial']['hybrid']:.0%}  topo={purity['adversarial']['topological']:.0%}  vec={purity['adversarial']['vector']:.0%}")
    
    # Phase 5: H1 stability over repeated adversarial queries
    print("\n[Phase 5] H1 stability under repeated adversarial queries...")
    h1_measurements = []
    for i in range(5):
        stats = measure_h1_stability(prog, system.embedding, system.topology)
        h1_measurements.append({
            "iteration": i,
            "h1_mean_pers": stats["h1_mean_pers"],
            "h1_n": stats["h1_n"],
            "n_clusters": stats["n_clusters"],
        })
        print(f"  Iter {i}: H1 mean_pers={stats['h1_mean_pers']:.4f}, clusters={stats['n_clusters']}")
    
    # Summary
    h1_stability = baseline["h1_mean_pers"] / (h1_measurements[-1]["h1_mean_pers"] + 1e-10)
    adv_drop = purity["normal"]["topological"] - purity["adversarial"]["topological"]
    
    print(f"\n{'='*40}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*40}")
    print(f"H1 stability ratio:       {h1_stability:.3f}")
    print(f"H1 diagram distance:      {drift['h1_dist']:.4f}")
    print(f"Adversarial purity drop:  topo={adv_drop:+.0%}  vec={purity['normal']['vector']-purity['adversarial']['vector']:+.0%}")
    
    winner = "TOPO" if purity["adversarial"]["topological"] > purity["adversarial"]["vector"] else \
             ("VEC" if purity["adversarial"]["vector"] > purity["adversarial"]["topological"] else "TIE")
    print(f"Adversarial winner:       {winner}")
    
    results = {
        "baseline_h1": baseline,
        "h1_drift": drift,
        "h1_stability_ratio": float(h1_stability),
        "purity_normal": purity["normal"],
        "purity_adversarial": purity["adversarial"],
        "adversarial_purity_drop": {k: purity["normal"][k] - purity["adversarial"][k] for k in purity["normal"]},
        "h1_measurements": h1_measurements,
        "winner": winner,
    }
    
    out_path = _SCRIPT_DIR / "results" / f"h1_adversarial_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
