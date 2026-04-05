"""
TopoMem Domain Forgetting Benchmark
=====================================
Tests: does topological retrieval prevent domain forgetting better than vector?

Design:
  Phase 1: Add programming 20 items  → measure accuracy
  Phase 2: Add physics 20 items      → re-test programming (forgetting)
  Phase 3: Add geography 20 items    → re-test programming (continued forgetting)

Comparisons:
  TM-Topo: topological retrieval (cluster-based)
  TM-Vec:  pure vector retrieval (ChromaDB ANN)
"""

import json, sys, os, tempfile, time
from pathlib import Path

# topomem package lives at F:/unified-sel/topomem/ (memory.py, __init__.py, etc.)
# script is at F:/unified-sel/topomem/benchmarks/
# we need F:/unified-sel in sys.path so 'from topomem.memory' resolves
# Use cwd (F:/unified-sel/topomem when run via 'python benchmarks/...') instead of __file__
# to avoid relative-path resolution issues across different working directories.
_PKG = Path.cwd()  # F:/unified-sel/topomem  (must run script from this dir)
sys.path.insert(0, str(_PKG.parent))  # F:/unified-sel
sys.path.insert(0, str(_PKG.parent.parent))  # F:/unified-sel

os.environ["HF_HOME"] = str(_PKG / "data" / "models" / "hf_cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(_PKG / "data" / "models" / "hf_cache")

from topomem.memory import MemoryGraph, MemoryConfig
from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine


def load_corpus(domain):
    path = _PKG / "data" / "test_corpus" / f"{domain}.json"
    return json.load(open(path, encoding="utf-8"))


def make_graph():
    tmp = tempfile.mkdtemp(prefix="topomem_dom_")
    cfg = MemoryConfig(chroma_persist_dir=tmp, max_nodes=500)
    emb = EmbeddingManager()
    graph = MemoryGraph(config=cfg, embedding_mgr=emb)
    return graph, emb  # return both so callers can encode queries


def encode(graph, query):
    """Encode text query using the graph's embedding manager."""
    return graph._embedding_mgr.encode(query)


def retrieve_vec(graph, emb_mgr, query, k=5):
    emb = encode(graph, query)
    return graph.retrieve(emb, strategy="vector", k=k)


def retrieve_topo(graph, emb_mgr, query, k=5):
    emb = encode(graph, query)
    return graph.retrieve(emb, strategy="topological", k=k)


def recall_ok(results_with_scores, keywords):
    """Top-3 results contain at least one keyword → 1, else 0."""
    nodes = [n for n, _ in results_with_scores]
    for n in nodes:
        cl = n.content.lower()
        if any(kw.lower() in cl for kw in keywords):
            return 1
    return 0


def add_items(graph, items):
    """Add all items as memories."""
    for item in items:
        graph.add_memory_from_text(item["content"])


def run_phase(graph, emb_mgr, topo_engine, test_items, label):
    """Test both retrieval strategies on test_items. Returns (topo_acc, vec_acc)."""
    topo_ok = vec_ok = 0
    for item in test_items:
        q = item["test_question"]
        kw = item["expected_keywords"]
        topo_ok += recall_ok(retrieve_topo(graph, emb_mgr, q, k=3), kw)
        vec_ok  += recall_ok(retrieve_vec(graph, emb_mgr, q, k=3), kw)
    n = len(test_items)
    topo_acc = topo_ok / n
    vec_acc  = vec_ok  / n
    print(f"  {label:<30} TM-Topo={topo_acc:.2%}  TM-Vec={vec_acc:.2%}")
    return topo_acc, vec_acc


def run_benchmark():
    print("=" * 60)
    print("TopoMem Domain Forgetting Benchmark")
    print("=" * 60)

    prog  = load_corpus("programming")
    phys  = load_corpus("physics")
    geo   = load_corpus("geography")
    print(f"Corpora loaded: programming={len(prog)}, physics={len(phys)}, geography={len(geo)}")

    results = {}

    # Phase 1: Programming only
    print("\n--- Phase 1: Programming only ---")
    graph, emb_mgr = make_graph()
    topo  = TopologyEngine()
    add_items(graph, prog)
    graph.update_topology(topo)
    p1_topo, p1_vec = run_phase(graph, emb_mgr, topo, prog, "Baseline (programming only)")
    results["phase1"] = {"topo": p1_topo, "vec": p1_vec}

    # Phase 2: + Physics
    print("\n--- Phase 2: Add Physics ---")
    add_items(graph, phys)
    graph.update_topology(topo)
    p2_topo, p2_vec = run_phase(graph, emb_mgr, topo, prog, "After physics (forgetting)")
    results["phase2"] = {
        "topo": p2_topo, "vec": p2_vec,
        "topo_forgetting": p1_topo - p2_topo,
        "vec_forgetting":  p1_vec  - p2_vec,
    }

    # Phase 3: + Geography
    print("\n--- Phase 3: Add Geography ---")
    add_items(graph, geo)
    graph.update_topology(topo)
    p3_topo, p3_vec = run_phase(graph, emb_mgr, topo, prog, "After geography (total forgetting)")
    results["phase3"] = {
        "topo": p3_topo, "vec": p3_vec,
        "topo_forgetting": p1_topo - p3_topo,
        "vec_forgetting":  p1_vec  - p3_vec,
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Domain Forgetting (programming accuracy)")
    print("=" * 60)
    print(f"  {'Phase':<35} {'TM-Topo':>9} {'TM-Vec':>9}  {'Winner':>6}")
    print(f"  {'-'*35} {'-'*9} {'-'*9}  {'-'*6}")
    for label, topo, vec in [
        ("Programming only",    p1_topo, p1_vec),
        ("After Physics added", p2_topo, p2_vec),
        ("After Geography added",p3_topo, p3_vec),
    ]:
        w = "Topo" if topo > vec else "Vec" if vec > topo else "Tie"
        print(f"  {label:<35} {topo:>8.2%}  {vec:>8.2%}  {w:>6}")
    print()
    print(f"  Total forgetting (TM-Topo): {p1_topo-p3_topo:+.2%}")
    print(f"  Total forgetting (TM-Vec):  {p1_vec-p3_vec:+.2%}")
    print(f"\n  Nodes: {graph.node_count()}, clusters: {len(graph.get_cluster_centers())}")

    out = _PKG / "benchmarks" / "results" / f"domain_forgetting_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out}")
    return results


if __name__ == "__main__":
    run_benchmark()
