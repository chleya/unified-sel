"""
Pollution Resistance Benchmark - H1 Stability Under Geometric Invasion

Hypothesis: H1 topology is MORE STABLE than VEC geometry when embedding
space is polluted by geometrically similar but semantically different items.
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


def load_corpus(domain):
    path = _PKG_DIR / "data" / "test_corpus" / f"{domain}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_overlap(items_a, items_b, emb_mgr):
    texts_a = [x["content"] for x in items_a]
    texts_b = [x["content"] for x in items_b]
    va = emb_mgr.encode_batch(texts_a)
    vb = emb_mgr.encode_batch(texts_b)
    overlaps = []
    for a in va:
        sims = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)) for b in vb]
        overlaps.append(max(sims))
    return {"mean": float(np.mean(overlaps)), "max": float(np.max(overlaps))}


def measure_topo(items, emb_mgr, topo_engine):
    texts = [x["content"] for x in items]
    vecs = emb_mgr.encode_batch(texts)
    diagrams = topo_engine.compute_persistence(vecs)
    h0 = diagrams[0] if len(diagrams) > 0 else np.array([])
    h1 = diagrams[1] if len(diagrams) > 1 else np.array([])
    h0_vals = [float(r[1]-r[0]) for r in h0 if np.isfinite(r[1])]
    h1_vals = [float(r[1]-r[0]) for r in h1 if np.isfinite(r[1])]
    labels = topo_engine.compute_full_result(vecs).cluster_labels
    return {
        "n_clusters": int(len(set(labels))) if labels is not None else 0,
        "h1_n": len(h1_vals), "h1_mean_pers": float(np.mean(h1_vals)) if h1_vals else 0.0,
        "h0_n": len(h0_vals), "h0_mean_pers": float(np.mean(h0_vals)) if h0_vals else 0.0,
    }


def measure_purity(system, query_items, domain_filter, k=5):
    emb = system.embedding
    query_vecs = emb.encode_batch([x["content"] for x in query_items])
    results = {"hybrid": [], "topological": [], "vector": []}
    for vec in query_vecs:
        for strat in ["hybrid", "topological", "vector"]:
            try:
                retrieved = system.memory.retrieve(vec, strategy=strat, k=k)
                if retrieved:
                    cnt = sum(1 for n, s in retrieved if n.metadata.get("domain") == domain_filter)
                    results[strat].append(cnt / len(retrieved))
            except Exception:
                pass
    return {s: float(np.mean(r)) if r else 0.0 for s, r in results.items()}


def run():
    print("=" * 60)
    print("POLLUTION RESISTANCE BENCHMARK")
    print("=" * 60)

    prog = load_corpus("programming")[:20]
    phys = load_corpus("physics")[:20]
    geo = load_corpus("geography")[:10]
    print(f"Corpus: {len(prog)} prog + {len(phys)} phys + {len(geo)} geo")

    tmpdir = "F:\\tmp\\topomem_prb"
    os.makedirs(tmpdir, exist_ok=True)

    config = TopoMemConfig(
        memory=MemoryConfig(chroma_persist_dir=tmpdir),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        topology=TopologyConfig(max_homology_dim=1, filtration_steps=30, metric="cosine"),
        engine=None,
    )
    system = TopoMemSystem(config=config)

    # Phase 1: Store programming
    print("\n[Phase 1] Storing programming items...")
    for item in prog:
        system.add_knowledge(item["content"], metadata={"domain": "A", "id": item["id"]})
    system.memory.update_topology(system.topology)

    s1 = measure_topo(prog, system.embedding, system.topology)
    pre = measure_purity(system, prog[:8], "A", k=5)
    print(f"  Clusters: {s1['n_clusters']}, H1: {s1['h1_n']} cycles, H1 mean pers: {s1['h1_mean_pers']:.4f}")
    print(f"  Pre-invasion A-purity: hybrid={pre['hybrid']:.0%} topo={pre['topological']:.0%} vec={pre['vector']:.0%}")

    # Phase 2: Store physics (invasion)
    print("\n[Phase 2] Storing physics items (invasion)...")
    for item in phys:
        system.add_knowledge(item["content"], metadata={"domain": "B", "id": item["id"]})
    system.memory.update_topology(system.topology)

    overlap = compute_overlap(prog, phys, system.embedding)
    s2 = measure_topo(prog, system.embedding, system.topology)
    print(f"  Overlap A<->B: mean={overlap['mean']:.4f} max={overlap['max']:.4f}")
    print(f"  A's H1 after invasion: {s2['h1_n']} cycles, H1 mean pers: {s2['h1_mean_pers']:.4f}")

    # Phase 3: Store geography
    print("\n[Phase 3] Storing geography items...")
    for item in geo:
        system.add_knowledge(item["content"], metadata={"domain": "C", "id": item["id"]})
    system.memory.update_topology(system.topology)

    post = measure_purity(system, prog[:8], "A", k=5)
    print(f"\n[Results]")
    print(f"  Pre-invasion purity:  hybrid={pre['hybrid']:.0%}  topo={pre['topological']:.0%}  vec={pre['vector']:.0%}")
    print(f"  Post-invasion purity: hybrid={post['hybrid']:.0%}  topo={post['topological']:.0%}  vec={post['vector']:.0%}")
    print(f"  Purity drop: hybrid={pre['hybrid']-post['hybrid']:+.0%}  topo={pre['topological']-post['topological']:+.0%}  vec={pre['vector']-post['vector']:+.0%}")
    h1_stab = s1["h1_mean_pers"] / (s2["h1_mean_pers"] + 1e-10)
    print(f"  H1 stability ratio: {s1['h1_mean_pers']:.4f} -> {s2['h1_mean_pers']:.4f} = {h1_stab:.2f}")

    winner = "TOPO" if post["topological"] > post["vector"] else ("VEC" if post["vector"] > post["topological"] else "TIE")
    print(f"  Winner: {winner}")

    results = {
        "pre_purity": pre, "post_purity": post,
        "purity_drops": {k: pre[k]-post[k] for k in pre},
        "phase1_topo": s1, "phase2_topo": s2,
        "h1_stability": float(h1_stab),
        "geometric_overlap": overlap,
    }
    out_path = _SCRIPT_DIR / "results" / f"pollution_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
