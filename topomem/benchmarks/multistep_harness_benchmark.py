"""
Multi-Step Memory Harness Benchmark
===================================
Meta-Harness insight: The harness determines what context the model sees at each step.

This benchmark tests: Can TopoMem maintain coherent multi-step memory context
better than a flat vector store?

Design (inspired by Meta-Harness's full-trace approach):
  Simulate a research conversation with 5 sequential steps on the SAME project topic.
  - Step 1: Add project background knowledge
  - Step 2: Add implementation details  
  - Step 3: Add debugging session
  - Step 4: Add optimization discussion
  - Step 5: Query: "What was the key problem we discussed in step 2?"

  Key metric: retrieval quality of early-step knowledge AFTER subsequent steps are added.
  If topological helps, early knowledge should remain accessible (not buried by new items).
"""

import json, sys, os, tempfile, time
from pathlib import Path

_PKG = Path.cwd()
sys.path.insert(0, str(_PKG.parent))
os.environ["HF_HOME"] = str(_PKG / "data" / "models" / "hf_cache")
os.environ["SENTENCE_TRANSORMERS_HOME"] = str(_PKG / "data" / "models" / "hf_cache")

from topomem.system import TopoMemSystem
from topomem.config import TopoMemConfig, MemoryConfig


def get_all_nodes(memory_graph):
    return [data["node"] for _, data in memory_graph._graph.nodes(data=True)]


def cosine_sim(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_items(topic, n=5):
    """Create n pieces of knowledge about a topic, ordered chronologically.
    Each item is a 'memory' that would be added in a different conversation step."""
    templates = {
        "research": [
            "Project {i}: We chose {topic} because existing approaches failed on edge cases.",
            "Project {i}: The key challenge is that {topic} requires balancing accuracy vs speed.",
            "Project {i}: We tried three approaches: A failed due to X, B was too slow, C worked.",
            "Project {i}: The breakthrough came when we realized the problem was in step 2 of the pipeline.",
            "Project {i}: Final result: 94% accuracy at 3x speed improvement over baseline.",
        ],
        "debugging": [
            "Bug {i}: The crash happened in {topic} because null values weren't handled.",
            "Bug {i}: Stack trace pointed to {topic} but root cause was upstream in the config.",
            "Bug {i}: Reproduced by adding a 10MB file - {topic} had no size limit.",
            "Bug {i}: The fix required changing 3 modules, not just {topic} directly.",
            "Bug {i}: Added regression test after {topic} fix; 4 similar bugs caught pre-launch.",
        ],
        "optimization": [
            "Opt {i}: {topic} was 200ms; profiling showed 80% time in serialization.",
            "Opt {i}: Switching from JSON to msgpack reduced {topic} from 200ms to 15ms.",
            "Opt {i}: Cache invalidation in {topic} was too aggressive - relaxed to 60s TTL.",
            "Opt {i}: Batch processing {topic} reduced API calls from 1000 to 10 per request.",
            "Opt {i}: Current {topic} at 12ms p99, well under 50ms SLA.",
        ],
    }
    templates = templates.get(topic, templates["research"])
    items = []
    for i in range(n):
        t = templates[i % len(templates)]
        content = t.format(i=i+1, topic=f"the {topic} component")
        items.append({
            "content": content,
            "step": i + 1,
            "topic": topic,
        })
    return items


def evaluate_recall_after_steps(system, early_items, late_items, test_query, k=5):
    """After adding both early and late items, test if early items are still retrievable."""
    import numpy as np

    # Query embedding
    emb = system.memory._embedding_mgr.encode(test_query)

    # Get topological and vector retrieval results
    topo_res = system.memory.retrieve(emb, strategy="topological", k=k)
    vec_res = system.memory.retrieve(emb, strategy="vector", k=k)

    topo_ids = {node.id for node, _ in topo_res}
    vec_ids = {node.id for node, _ in vec_res}

    # Check which early items are in top-k
    topo_hits = vec_hits = 0
    topo_ranks = vec_ranks = []
    
    all_nodes = get_all_nodes(system.memory)
    
    for item in early_items:
        # Find the node that best matches this item's content
        item_emb = system.memory._embedding_mgr.encode(item["content"])
        best = max(all_nodes, key=lambda n: cosine_sim(n.embedding, item_emb))
        target_id = best.id

        if target_id in topo_ids:
            topo_hits += 1
            rank = [r.id for r, _ in topo_res].index(target_id) + 1
            topo_ranks.append(rank)
        else:
            topo_ranks.append(k + 1)
        
        if target_id in vec_ids:
            vec_hits += 1
            rank = [r.id for r, _ in vec_res].index(target_id) + 1
            vec_ranks.append(rank)
        else:
            vec_ranks.append(k + 1)

    topo_recall = topo_hits / len(early_items)
    vec_recall = vec_hits / len(early_items)
    topo_mrr = sum(1/r for r in topo_ranks) / len(topo_ranks)
    vec_mrr = sum(1/r for r in vec_ranks) / len(vec_ranks)

    return topo_recall, vec_recall, topo_mrr, vec_mrr


def run_topic_experiment(system_factory, topic, n_early=5, n_late=10):
    """Run experiment for one topic: early items vs late items interference."""
    import numpy as np

    early_items = make_items(topic, n_early)
    late_topic = list(set(["research", "debugging", "optimization"]) - {topic})[0]
    late_items = make_items(late_topic, n_late)
    test_query = f"What problem did we encounter with the {topic} component?"
    
    # Fresh system
    tmp = tempfile.mkdtemp(prefix=f"topomem_harness_{topic[:3]}_")
    config = TopoMemConfig(memory=MemoryConfig(chroma_persist_dir=tmp, max_nodes=300))
    system = system_factory(config)
    
    # Phase 1: Add early items only
    for item in early_items:
        system.add_knowledge(item["content"])
    # CRITICAL: update topology so cluster structure forms
    system.memory.update_topology(system.topology)
    
    # Phase 2: Add late items (interference)
    for item in late_items:
        system.add_knowledge(item["content"])
    # Update topology again after interference items
    system.memory.update_topology(system.topology)
    
    # Evaluate
    topo_r, vec_r, topo_m, vec_m = evaluate_recall_after_steps(
        system, early_items, late_items, test_query, k=5
    )
    
    status = system.get_status()
    n_clusters = status.memory_cluster_count
    n_nodes = status.memory_node_count
    
    return {
        "topic": topic,
        "early_items": n_early,
        "late_items": n_late,
        "total_nodes": n_nodes,
        "clusters": n_clusters,
        "topo_recall": topo_r,
        "vec_recall": vec_r,
        "topo_mrr": topo_m,
        "vec_mrr": vec_m,
        "winner": "Topo" if topo_r > vec_r else "Vec" if vec_r > topo_r else "Tie",
    }


def main():
    print("=" * 70)
    print("Multi-Step Memory Harness Benchmark")
    print("(Inspired by Stanford Meta-Harness: harness > model)")
    print("=" * 70)

    topics = ["research", "debugging", "optimization"]
    results = []

    for topic in topics:
        print(f"\n--- Topic: {topic.upper()} ---")
        result = run_topic_experiment(TopoMemSystem, topic, n_early=5, n_late=10)
        print(f"  Total nodes: {result['total_nodes']}, Clusters: {result['clusters']}")
        print(f"  TM-Topo: recall@5={result['topo_recall']:.1%}  MRR={result['topo_mrr']:.3f}")
        print(f"  TM-Vec:  recall@5={result['vec_recall']:.1%}  MRR={result['vec_mrr']:.3f}")
        print(f"  Winner: {result['winner']}")
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Topic':<15} {'Nodes':>6} {'Clus':>5} {'TM-Topo R@5':>12} {'TM-Vec R@5':>11} {'Winner':>6}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['topic']:<15} {r['total_nodes']:>6} {r['clusters']:>5} "
              f"{r['topo_recall']:>11.1%} {r['vec_recall']:>10.1%} {r['winner']:>6}")

    topo_avg = sum(r['topo_recall'] for r in results) / len(results)
    vec_avg = sum(r['vec_recall'] for r in results) / len(results)
    topo_mrr_avg = sum(r['topo_mrr'] for r in results) / len(results)
    vec_mrr_avg = sum(r['vec_mrr'] for r in results) / len(results)

    print(f"  {'-'*60}")
    print(f"  {'AVERAGE':<15} {'':>6} {'':>5} "
          f"{topo_avg:>11.1%} {vec_avg:>10.1%} "
          f"{'Topo' if topo_avg > vec_avg else 'Vec' if vec_avg > topo_avg else 'Tie':>6}")
    print(f"\n  TM-Topo avg MRR: {topo_mrr_avg:.3f}")
    print(f"  TM-Vec avg MRR:  {vec_mrr_avg:.3f}")

    # Save
    out = _PKG / "benchmarks" / "results" / f"harness_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "results": results,
        "summary": {
            "topo_avg_recall": topo_avg,
            "vec_avg_recall": vec_avg,
            "topo_avg_mrr": topo_mrr_avg,
            "vec_avg_mrr": vec_mrr_avg,
        }
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults: {out}")
    return summary


if __name__ == "__main__":
    main()
