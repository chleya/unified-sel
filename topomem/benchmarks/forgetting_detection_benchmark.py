#!/usr/bin/env python3
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
Forgetting Detection Benchmark -- v2

重新设计（基于 Phase 2 的关键洞察）：
  - Intra-A similarity 不变 -> 不够敏感
  - 测量更有意义的指标：
    1. 跨域最近邻干扰：A 的邻居被 B/C 节点占据的比例
    2. 域间分离度：A vs B vs C 的 centroid 距离随入侵的变化
    3. A 的检索排名下降：A 节点对 A 查询的排序位置变化

设计（embedding 层，无需 LLM）：
  Phase 0: A (20 items) -> baseline
  Phase 1: +B (20 items) -> 测量 A 的邻居干扰
  Phase 2: +C (10 items) -> 继续测量
  Phase 3: consolidation_pass -> 验证能否恢复 A 的结构
"""

import os, sys, json, time
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).parent
_PKG_DIR = _SCRIPT_DIR.parent
_ROOT_DIR = _PKG_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.config import (
    TopoMemConfig, EmbeddingConfig, TopologyConfig,
    MemoryConfig as TopoMemMemoryConfig,
)
from topomem.system import TopoMemSystem


def load_corpus(domain):
    path = _PKG_DIR / "data" / "test_corpus" / f"{domain}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_h1_metrics(vecs, topo_engine):
    """Compute H1 metrics from embedding matrix."""
    if len(vecs) < 3:
        return {"h1_cycles": 0, "mean_persistence": 0.0, "total_persistence": 0.0}
    diagrams = topo_engine.compute_persistence(np.array(vecs, dtype=np.float64))
    if isinstance(diagrams, dict):
        h1_diagrams = diagrams.get(1, [])
    else:
        h1_diagrams = diagrams[1] if len(diagrams) > 1 else []
    if len(h1_diagrams) == 0:
        return {"h1_cycles": 0, "mean_persistence": 0.0, "total_persistence": 0.0}
    persistences = [
        float(d[1] - d[0])
        for d in h1_diagrams
        if np.isfinite(d[1]) and np.isfinite(d[0])
    ]
    return {
        "h1_cycles": len(persistences),
        "mean_persistence": float(np.mean(persistences)) if persistences else 0.0,
        "total_persistence": float(np.sum(persistences)) if persistences else 0.0,
    }


def normalize(vecs):
    """Row-normalize for cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms


def cosine_sim(a, b):
    """Row vectors a and b: cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def nearest_neighbor_interference(vecs_A, vecs_B, vecs_C, k=5):
    """
    For each A node in combined A+B+C space: what fraction of top-k neighbors are B or C?
    """
    all_vecs = np.vstack([vecs_A, vecs_B, vecs_C]) if vecs_C is not None else np.vstack([vecs_A, vecs_B])
    all_norm = normalize(all_vecs)
    n_A = len(vecs_A)
    interference_fractions = []
    for i in range(n_A):
        query = all_norm[i]
        sims = all_norm @ query
        sims[i] = -1
        top_k_idx = np.argsort(-sims)[:k]
        # top_k_idx >= n_A means it's from B or C
        frac = float(np.sum(top_k_idx >= n_A)) / k
        interference_fractions.append(frac)
    return float(np.mean(interference_fractions))


def centroid_distance(vecs_A, vecs_B):
    """Mean pairwise centroid distance (inter-domain separation)."""
    cent_A = np.mean(vecs_A, axis=0)
    cent_B = np.mean(vecs_B, axis=0)
    return float(cosine_sim(cent_A, cent_B))  # cosine similarity (1 = identical)


def domain_coherence(vecs):
    """Mean within-domain pairwise cosine similarity."""
    if len(vecs) < 2:
        return 0.0
    normed = normalize(vecs)
    sim_matrix = normed @ normed.T
    n = len(vecs)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0


def precision_at_k_A(vecs_A, k=5):
    """
    For each A node, what's the fraction of top-k neighbors that are also A?
    Measures A's self-coherence in retrieval.
    """
    if len(vecs_A) < 2:
        return 0.0
    normed = normalize(vecs_A)
    n = len(vecs_A)
    precisions = []
    for i in range(n):
        query = normed[i]
        sims = normed @ query
        sims[i] = -1  # exclude self
        top_k = np.argsort(-sims)[:k]
        # top_k contains indices in [0, n). Is it < n? (means from A)
        n_A_in_topk = np.sum(top_k < n)
        precisions.append(n_A_in_topk / k)
    return float(np.mean(precisions))


def main():
    print("=" * 70)
    print("Forgetting Detection Benchmark v2")
    print("测量：域间干扰 + H1 稳定性")
    print("=" * 70)

    corpus_A = load_corpus("programming")
    corpus_B = load_corpus("physics")
    corpus_C = load_corpus("geography")

    texts_A = [item["content"] for item in corpus_A]
    texts_B = [item["content"] for item in corpus_B]
    texts_C = [item["content"] for item in corpus_C]

    print(f"\nDomains: A=programming({len(texts_A)}), B=physics({len(texts_B)}), C=geography({len(texts_C)})")

    emb_cfg = EmbeddingConfig()
    topo_cfg = TopologyConfig()
    emb_mgr = EmbeddingManager(emb_cfg)
    topo_engine = TopologyEngine(topo_cfg)

    results = {"timestamp": time.strftime("%Y%m%d_%H%M%S"), "phases": {}}

    # ===== PHASE 0: Baseline A =====
    print("\n" + "=" * 70)
    print("PHASE 0: A (programming) baseline")
    print("=" * 70)

    vecs_A = np.array(emb_mgr.encode_batch(texts_A), dtype=np.float64)
    h1_A = compute_h1_metrics(vecs_A, topo_engine)
    coh_A = domain_coherence(vecs_A)
    prec_A_A = precision_at_k_A(vecs_A)

    print(f"  H1 cycles: {h1_A['h1_cycles']}, mean_pers: {h1_A['mean_persistence']:.4f}")
    print(f"  A coherence: {coh_A:.4f}")
    print(f"  A MRR (nearest A neighbor): {prec_A_A:.2f}")

    results["phases"]["phase0_A_baseline"] = {
        "domain": "A (programming)",
        "n_items": len(texts_A),
        "h1_cycles": h1_A["h1_cycles"],
        "mean_persistence": h1_A["mean_persistence"],
        "coherence_A": coh_A,
        "prec_A_A": prec_A_A,
    }

    # ===== PHASE 1: B invades =====
    print("\n" + "=" * 70)
    print("PHASE 1: B (physics) -- invasion")
    print("=" * 70)

    vecs_B = np.array(emb_mgr.encode_batch(texts_B), dtype=np.float64)
    vecs_AB = np.vstack([vecs_A, vecs_B])

    h1_AB = compute_h1_metrics(vecs_AB, topo_engine)
    interference_AB = nearest_neighbor_interference(vecs_A, vecs_B, None, k=5)
    coh_A_AB = domain_coherence(vecs_A)  # A's internal after B added
    cent_dist_AB = centroid_distance(vecs_A, vecs_B)
    prec_A_A_AB = precision_at_k_A(vecs_A)

    print(f"  A+B: {len(vecs_AB)} items, H1 cycles: {h1_AB['h1_cycles']} (ratio: {h1_AB['h1_cycles']/h1_A['h1_cycles']:.2f}x)")
    print(f"  Interference (B in A's neighborhood): {interference_AB:.1%}")
    print(f"  A coherence (post-invasion): {coh_A_AB:.4f} (baseline: {coh_A:.4f})")
    print(f"  A->B centroid cosine: {cent_dist_AB:.4f} (1=identical, 0=orthogonal)")
    print(f"  A retrieval rank (nearest A neighbor): {prec_A_A_AB:.2f} (baseline: {prec_A_A:.2f})")

    results["phases"]["phase1_B_invasion"] = {
        "domain": "B (physics) added",
        "n_items_AB": len(vecs_AB),
        "h1_cycles_AB": h1_AB["h1_cycles"],
        "h1_cycles_ratio": h1_AB["h1_cycles"] / h1_A["h1_cycles"],
        "interference_B_in_A_neighbors": interference_AB,
        "coherence_A": coh_A_AB,
        "coherence_A_drop": coh_A - coh_A_AB,
        "centroid_AB": cent_dist_AB,
        "prec_A_A": prec_A_A_AB,
        "prec_A_change": prec_A_A_AB - prec_A_A,
    }

    # ===== PHASE 2: C continues invasion =====
    print("\n" + "=" * 70)
    print("PHASE 2: C (geography) -- continued invasion")
    print("=" * 70)

    vecs_C = np.array(emb_mgr.encode_batch(texts_C), dtype=np.float64)
    vecs_ABC = np.vstack([vecs_A, vecs_B, vecs_C])

    h1_ABC = compute_h1_metrics(vecs_ABC, topo_engine)
    interference_ABC = nearest_neighbor_interference(vecs_A, vecs_B, vecs_C, k=5)
    coh_A_ABC = domain_coherence(vecs_A)
    cent_dist_AC = centroid_distance(vecs_A, vecs_C)
    cent_dist_BC = centroid_distance(vecs_B, vecs_C)
    prec_A_A_ABC = precision_at_k_A(vecs_A)

    print(f"  A+B+C: {len(vecs_ABC)} items, H1 cycles: {h1_ABC['h1_cycles']} (ratio: {h1_ABC['h1_cycles']/h1_A['h1_cycles']:.2f}x)")
    print(f"  Interference (B+C in A's neighborhood): {interference_ABC:.1%}")
    print(f"  A coherence (post-invasion): {coh_A_ABC:.4f} (baseline: {coh_A:.4f})")
    print(f"  A->C centroid cosine: {cent_dist_AC:.4f}")
    print(f"  B->C centroid cosine: {cent_dist_BC:.4f}")
    print(f"  A retrieval rank: {prec_A_A_ABC:.2f} (baseline: {prec_A_A:.2f})")

    results["phases"]["phase2_C_invasion"] = {
        "domain": "C (geography) added",
        "n_items_ABC": len(vecs_ABC),
        "h1_cycles_ABC": h1_ABC["h1_cycles"],
        "h1_cycles_ratio": h1_ABC["h1_cycles"] / h1_A["h1_cycles"],
        "interference_B+C_in_A_neighbors": interference_ABC,
        "coherence_A": coh_A_ABC,
        "coherence_A_drop": coh_A - coh_A_ABC,
        "centroid_AC": cent_dist_AC,
        "centroid_BC": cent_dist_BC,
        "prec_A_A": prec_A_A_ABC,
        "prec_A_change": prec_A_A_ABC - prec_A_A,
    }

    # ===== PHASE 3: TopoMemSystem consolidation =====
    print("\n" + "=" * 70)
    print("PHASE 3: TopoMemSystem -- consolidation_pass")
    print("=" * 70)

    tmp_dir = _ROOT_DIR / "tmp_forgetting_v2_chroma"
    tmp_dir.mkdir(exist_ok=True)

    mem_cfg = TopoMemMemoryConfig(
        max_nodes=500,
        chroma_persist_dir=str(tmp_dir),
        similarity_top_k=5,
    )
    sys_cfg = TopoMemConfig(
        embedding=EmbeddingConfig(),
        topology=TopologyConfig(),
        memory=mem_cfg,
    )
    system = TopoMemSystem(config=sys_cfg)

    # Store all domains
    for i, item in enumerate(corpus_A):
        emb = emb_mgr.encode(item["content"])
        system.memory.add_memory(item["content"], np.array(emb, dtype=np.float64),
                                 metadata={"domain": "A", "item_id": i})
    system.memory.update_topology(topo_engine=system.topology)

    for i, item in enumerate(corpus_B):
        emb = emb_mgr.encode(item["content"])
        system.memory.add_memory(item["content"], np.array(emb, dtype=np.float64),
                                 metadata={"domain": "B", "item_id": i})
    system.memory.update_topology(topo_engine=system.topology)

    for i, item in enumerate(corpus_C):
        emb = emb_mgr.encode(item["content"])
        system.memory.add_memory(item["content"], np.array(emb, dtype=np.float64),
                                 metadata={"domain": "C", "item_id": i})
    system.memory.update_topology(topo_engine=system.topology)

    # Run consolidation
    report = system.consolidation_pass(orphan_threshold=0.05, merge_centroid_threshold=0.92)
    system.memory.update_topology(topo_engine=system.topology)

    # Get A nodes after consolidation
    nodes_A = [n['node'] for n in system.memory._graph.nodes.values() if n.get('node', {}).metadata.get('domain') == 'A']
    texts_A_post = [n.content for n in nodes_A]

    if len(texts_A_post) >= 3:
        vecs_A_post = np.array(emb_mgr.encode_batch(texts_A_post), dtype=np.float64)
        h1_A_post = compute_h1_metrics(vecs_A_post, topo_engine)
        coh_A_post = domain_coherence(vecs_A_post)
        prec_A_A_post = precision_at_k_A(vecs_A_post)
    else:
        h1_A_post = {"h1_cycles": 0, "mean_persistence": 0.0}
        coh_A_post = 0.0
        prec_A_A_post = 0.0

    print(f"  Consolidation: orphans={report.get('orphans_detected', 0)}, "
          f"merges={report.get('merge_candidates_found', 0)}")
    print(f"  A nodes post-consolidation: {len(nodes_A)}/{len(texts_A)}")
    print(f"  A H1 cycles: {h1_A_post['h1_cycles']}, mean_pers: {h1_A_post['mean_persistence']:.4f}")
    print(f"  A coherence: {coh_A_post:.4f} (baseline: {coh_A:.4f})")
    print(f"  A retrieval rank: {prec_A_A_post:.2f} (baseline: {prec_A_A:.2f})")

    results["phases"]["phase3_consolidation"] = {
        "consolidation_report": {k: v for k, v in report.items() if k not in ["_private"]},
        "nodes_A_post_consol": len(nodes_A),
        "h1_cycles_A": h1_A_post["h1_cycles"],
        "mean_persistence_A": h1_A_post["mean_persistence"],
        "coherence_A": coh_A_post,
        "coherence_A_recovery": coh_A_post - coh_A_ABC,
        "prec_A_A": prec_A_A_post,
        "prec_A_recovery": prec_A_A_ABC - prec_A_A_post,
    }

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY -- Interference + H1 Stability")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Phase0':<12} {'Phase1 (+B)':<12} {'Phase2 (+C)':<12} {'Phase3 (consol.)':<15}")
    print("-" * 85)
    print(f"{'H1 cycles':<35} {h1_A['h1_cycles']:<12} {h1_AB['h1_cycles']:<12} {h1_ABC['h1_cycles']:<12} {h1_A_post['h1_cycles']:<15}")
    print(f"{'H1 cycles ratio (vs A)':<35} {'1.00':<12} {h1_AB['h1_cycles']/h1_A['h1_cycles']:<12.2f}x {h1_ABC['h1_cycles']/h1_A['h1_cycles']:<12.2f}x "
          f"{h1_A_post['h1_cycles']/max(h1_A['h1_cycles'],1):<15.2f}x")
    print(f"{'Mean persistence':<35} {h1_A['mean_persistence']:<12.4f} {h1_AB['mean_persistence']:<12.4f} {h1_ABC['mean_persistence']:<12.4f} "
          f"{h1_A_post['mean_persistence']:<15.4f}")
    print(f"{'Interference (B+C in A neigh.)':<35} {'0.0%':<12} {'--':<12} {interference_ABC:<12.1%} {'--':<15}")
    print(f"{'A coherence':<35} {coh_A:<12.4f} {coh_A_AB:<12.4f} {coh_A_ABC:<12.4f} {coh_A_post:<15.4f}")
    print(f"{'A coherence drop':<35} {'0':<12} {coh_A-coh_A_AB:<12.4f} {coh_A-coh_A_ABC:<12.4f} {coh_A_ABC-coh_A_post:<15.4f}")
    print(f"{'A MRR (nearest A)':<35} {prec_A_A:<12.2f} {prec_A_A_AB:<12.2f} {prec_A_A_ABC:<12.2f} {prec_A_A_post:<15.2f}")
    print(f"{'A MRR change':<35} {'0':<12} {prec_A_A_AB-prec_A_A:<12.2f} {prec_A_A_ABC-prec_A_A:<12.2f} "
          f"{prec_A_A_ABC-prec_A_A_post:<15.2f}")

    # ===== KEY FINDINGS =====
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    interference_significant = interference_ABC > 0.2
    coh_drop_significant = (coh_A - coh_A_ABC) > 0.05
    rank_degraded = prec_A_A_ABC > prec_A_A
    consolidation_recovered_coh = coh_A_post > coh_A_ABC
    consolidation_recovered_rank = prec_A_A_post < prec_A_A_ABC

    print(f"\n[{'PASS' if interference_significant else 'FAIL'}] Interference > 20%: {interference_ABC:.1%}")
    print(f"    -> {'B+C 干扰了 A 的邻居结构' if interference_significant else 'A 的邻居结构未被显著干扰'}")
    print(f"[{'PASS' if coh_drop_significant else 'FAIL'}] A coherence 下降 > 0.05: {coh_A - coh_A_ABC:.4f}")
    print(f"    -> {'A 的内聚结构被破坏' if coh_drop_significant else 'A 的内聚结构保持稳定'}")
    print(f"[{'PASS' if rank_degraded else 'FAIL'}] A retrieval rank 下降: {prec_A_A:.2f} -> {prec_A_A_ABC:.2f}")
    print(f"    -> {'A 的自我区分度下降' if rank_degraded else 'A 的自我区分度不变'}")
    print(f"\n[{'PASS' if consolidation_recovered_coh else 'FAIL'}] Consolidation 恢复了 coherence: {coh_A_post:.4f} vs {coh_A_ABC:.4f}")
    print(f"[{'PASS' if consolidation_recovered_rank else 'FAIL'}] Consolidation 恢复了 retrieval rank: {prec_A_A_post:.2f} vs {prec_A_A_ABC:.2f}")

    # Verdict
    if consolidation_recovered_rank and consolidation_recovered_coh:
        verdict = "TOPOMEM_CONSOLIDATION_RESTORES_MEMORY_STRUCTURE"
        print(f"\n[RESULT] Verdict: consolidation_pass 能恢复 A 的结构 PASS")
    elif consolidation_recovered_coh:
        verdict = "CONSOLIDATION_PARTIALLY_EFFECTIVE"
        print(f"\n[RESULT] Verdict: consolidation 部分有效（coherence 恢复）")
    elif consolidation_recovered_rank:
        verdict = "CONSOLIDATION_RANK_RECOVERY_ONLY"
        print(f"\n[RESULT] Verdict: consolidation recovers RANK only")
    elif interference_significant:
        verdict = "INTERFERENCE_DETECTED_BUT_CONSOLIDATION_INEFFECTIVE"
        print(f"\n[RESULT] Verdict: interference detected, consolidation INEFFECTIVE")
    else:
        verdict = "NO_SIGNIFICANT_FORGETTING_THIS_CORPUS"
        print(f"\n[RESULT] Verdict: 在此 corpus 上，A 对 B+C 入侵具有强抗干扰性")

    results["summary"] = {
        "verdict": verdict,
        "interference_ABC": interference_ABC,
        "coherence_series": [coh_A, coh_A_AB, coh_A_ABC, coh_A_post],
        "prec_A_series": [prec_A_A, prec_A_A_AB, prec_A_A_ABC, prec_A_A_post],
        "consolidation_helped_coh": consolidation_recovered_coh,
        "consolidation_helped_rank": consolidation_recovered_rank,
    }

    # Save
    results_dir = _SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = int(time.time())
    results_path = results_dir / f"forgetting_detection_v2_{ts}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
