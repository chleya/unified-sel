"""
Cluster Formation + Retrieval Quality Benchmark
==============================================
Tests: Do multi-node clusters actually form? Does consolidation merge them?
      Does cluster structure help or hurt retrieval quality?

Design:
  Use programming corpus (20 items, known to form 2-3 clusters via TDA).
  Measure: recall@5, MRR, and cluster structure BEFORE and AFTER consolidation.
"""

import json, sys, os, tempfile, time
from pathlib import Path

_PKG = Path.cwd()
sys.path.insert(0, str(_PKG.parent))
os.environ["HF_HOME"] = str(_PKG / "data" / "models" / "hf_cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(_PKG / "data" / "models" / "hf_cache")

from topomem.system import TopoMemSystem
from topomem.config import TopoMemConfig, MemoryConfig


def load_corpus(name):
    return json.load(open(_PKG / "data" / "test_corpus" / f"{name}.json", encoding="utf-8"))


def get_all_nodes(memory_graph):
    """Get all MemoryNode objects from the NetworkX graph."""
    return [data["node"] for _, data in memory_graph._graph.nodes(data=True)]


def cluster_stats(all_nodes):
    """Compute cluster size distribution."""
    sizes = {}
    for node in all_nodes:
        cid = node.cluster_id if node.cluster_id is not None else -1
        sizes[cid] = sizes.get(cid, 0) + 1
    return sorted(sizes.values(), reverse=True)


def main():
    import numpy as np

    print("=" * 70)
    print("Cluster Formation + Retrieval Quality Benchmark")
    print("=" * 70)

    prog = load_corpus("programming")
    print(f"\nLoaded programming corpus: {len(prog)} items")

    # Create TopoMemSystem with temp ChromaDB
    tmp = tempfile.mkdtemp(prefix="topomem_cluster_")
    topo_config = TopoMemConfig(
        memory=MemoryConfig(chroma_persist_dir=tmp, max_nodes=200),
    )
    system = TopoMemSystem(config=topo_config)

    print("\nAdding 20 programming items...")
    for item in prog[:20]:
        system.add_knowledge(item["content"])
    print(f"  Nodes: {system.get_status().memory_node_count}")

    # Cluster structure BEFORE consolidation
    all_nodes_before = get_all_nodes(system.memory)
    sizes_before = cluster_stats(all_nodes_before)
    n_clusters_before = len([s for s in sizes_before if s > 1])  # multi-node clusters
    n_orphans_before = sum(1 for n in all_nodes_before if n.cluster_id is None)

    print("\n--- BEFORE consolidation_pass ---")
    print(f"  Total nodes: {len(all_nodes_before)}")
    print(f"  Multi-node clusters: {n_clusters_before}")
    print(f"  Orphan nodes (cluster_id=None): {n_orphans_before}")
    print(f"  Cluster sizes: {sizes_before}")

    # Retrieval evaluation BEFORE
    print("\n--- Retrieval BEFORE consolidation ---")
    k = 5
    topo_correct = vec_correct = 0
    topo_mrr = vec_mrr = 0.0

    for item in prog[:20]:
        q = item["test_question"]
        emb = system.memory._embedding_mgr.encode(q)

        topo_res = system.memory.retrieve(emb, strategy="topological", k=k)
        vec_res = system.memory.retrieve(emb, strategy="vector", k=k)

        # Target: node that best matches this item's content
        target_emb = system.memory._embedding_mgr.encode(item["content"])
        best_node = max(all_nodes_before, key=lambda n: np.dot(n.embedding, target_emb))
        target_id = best_node.id

        topo_ids = [node.id for node, _ in topo_res]
        vec_ids = [node.id for node, _ in vec_res]

        if target_id in topo_ids:
            topo_correct += 1
            topo_mrr += 1.0 / (topo_ids.index(target_id) + 1)
        if target_id in vec_ids:
            vec_correct += 1
            vec_mrr += 1.0 / (vec_ids.index(target_id) + 1)

    topo_r1 = topo_correct / 20
    vec_r1 = vec_correct / 20
    topo_m1 = topo_mrr / 20
    vec_m1 = vec_mrr / 20
    print(f"  TM-Topo: recall@{k}={topo_r1:.1%}  MRR={topo_m1:.3f}")
    print(f"  TM-Vec:  recall@{k}={vec_r1:.1%}  MRR={vec_m1:.3f}")

    # Run consolidation_pass
    print("\n--- Running consolidation_pass ---")
    report = system.consolidation_pass(update_topology=True)
    print(f"  Orphans detected: {report.get('orphans', '?')}")
    print(f"  Merge candidates: {report.get('merge_candidates', '?')}")
    print(f"  Cluster count: {report.get('cluster_count', '?')}")

    # Cluster structure AFTER consolidation
    all_nodes_after = get_all_nodes(system.memory)
    sizes_after = cluster_stats(all_nodes_after)
    n_clusters_after = len([s for s in sizes_after if s > 1])
    n_orphans_after = sum(1 for n in all_nodes_after if n.cluster_id is None)

    print(f"\n--- AFTER consolidation_pass ---")
    print(f"  Total nodes: {len(all_nodes_after)}")
    print(f"  Multi-node clusters: {n_clusters_after} (was {n_clusters_before})")
    print(f"  Orphan nodes: {n_orphans_after} (was {n_orphans_before})")
    print(f"  Cluster sizes: {sizes_after}")

    # Retrieval evaluation AFTER
    print("\n--- Retrieval AFTER consolidation ---")
    topo_correct2 = vec_correct2 = 0
    topo_mrr2 = vec_mrr2 = 0.0

    for item in prog[:20]:
        q = item["test_question"]
        emb = system.memory._embedding_mgr.encode(q)

        topo_res2 = system.memory.retrieve(emb, strategy="topological", k=k)
        vec_res2 = system.memory.retrieve(emb, strategy="vector", k=k)

        target_emb2 = system.memory._embedding_mgr.encode(item["content"])
        best_node2 = max(all_nodes_after, key=lambda n: np.dot(n.embedding, target_emb2))
        target_id2 = best_node2.id

        topo_ids2 = [node.id for node, _ in topo_res2]
        vec_ids2 = [node.id for node, _ in vec_res2]

        if target_id2 in topo_ids2:
            topo_correct2 += 1
            topo_mrr2 += 1.0 / (topo_ids2.index(target_id2) + 1)
        if target_id2 in vec_ids2:
            vec_correct2 += 1
            vec_mrr2 += 1.0 / (vec_ids2.index(target_id2) + 1)

    topo_r2 = topo_correct2 / 20
    vec_r2 = vec_correct2 / 20
    topo_m2 = topo_mrr2 / 20
    vec_m2 = vec_mrr2 / 20
    print(f"  TM-Topo: recall@{k}={topo_r2:.1%}  MRR={topo_m2:.3f}")
    print(f"  TM-Vec:  recall@{k}={vec_r2:.1%}  MRR={vec_m2:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Clusters: {n_clusters_before} -> {n_clusters_after} multi-node clusters")
    print(f"  Orphans:  {n_orphans_before} -> {n_orphans_after}")
    print()
    print(f"  {'Metric':<20} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*52}")
    for name, before, after in [
        ("TM-Topo recall@5", topo_r1, topo_r2),
        ("TM-Vec recall@5", vec_r1, vec_r2),
        ("TM-Topo MRR", topo_m1, topo_m2),
        ("TM-Vec MRR", vec_m1, vec_m2),
    ]:
        delta = after - before
        sign = '+' if delta > 0 else ''
        print(f"  {name:<20} {before:>9.1%} {after:>9.1%} {sign}{delta:>8.1%}")

    # Save
    results = {
        'clusters_multi_before': n_clusters_before,
        'clusters_multi_after': n_clusters_after,
        'orphans_before': n_orphans_before,
        'orphans_after': n_orphans_after,
        'cluster_sizes_before': sizes_before,
        'cluster_sizes_after': sizes_after,
        'topo_recall_before': topo_r1,
        'vec_recall_before': vec_r1,
        'topo_mrr_before': topo_m1,
        'vec_mrr_before': vec_m1,
        'topo_recall_after': topo_r2,
        'vec_recall_after': vec_r2,
        'topo_mrr_after': topo_m2,
        'vec_mrr_after': vec_m2,
        'consolidation_report': {k: v for k, v in report.items() if not k.startswith('_')},
    }
    out = _PKG / "benchmarks" / "results" / f"cluster_recall_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults: {out}")
    return results


if __name__ == "__main__":
    main()
