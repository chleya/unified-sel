#!/usr/bin/env python3
"""
Isolation Test — TDA Cluster Validity Diagnostic

Purpose: Validate whether TDA's H0 persistence captures semantic domain structure.
5 domains x 10 items. After TDA clustering, measure node isolation and retrieval purity.

Key metrics:
- Node isolation: fraction of domain nodes that are alone in their cluster
- Cluster purity: fraction of clusters containing only one domain
- Retrieval purity: for a domain query, what fraction of top-5 are from that domain?

Verdict:
- strong signal: isolation >70%, purity >70%
- weak signal: isolation 40-70%
- no signal: isolation <40% or >70% single-node clusters
"""
import os
import sys
import json
import shutil

# File: .../unified-sel/topomem/benchmarks/isolation_test.py
# We need '.../unified-sel' on sys.path so 'from topomem import ...' resolves correctly.
# __file__ = .../benchmarks/isolation_test.py
# dirname(__file__) = .../benchmarks
# dirname(dirname(__file__)) = .../topomem  (package root)
# dirname(dirname(dirname(__file__))) = .../unified-sel  (project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from topomem.system import TopoMemSystem
from topomem.config import TopoMemConfig


DOMAINS = {
    "python_programming": [
        "def quicksort(arr): return sorted(arr)",
        "class Foo: pass",
        "import numpy as np",
        "for i in range(10): print(i)",
        "with open('f.txt') as f: data = f.read()",
        "try: x = 1 except: pass",
        "lambda x: x * 2",
        "[x**2 for x in range(5)]",
        "def fib(n): return fib(n-1) + fib(n-2) if n > 1 else n",
        "dict.get('key', default=None)",
    ],
    "classical_mechanics": [
        "F = ma — Newton's second law",
        "KE = 0.5 * m * v^2 — kinetic energy",
        "PE = mgh — gravitational potential energy",
        "p = mv — momentum conservation",
        "Impulse J = F * dt",
        "Work W = F * d * cos(theta)",
        "Angular momentum L = I * omega",
        "Simple harmonic motion: x(t) = A * cos(omega*t)",
        "Centripetal force F = m*v^2/r",
        "Elastic collision: v1' = ((m1-m2)*v1 + 2*m2*v2)/(m1+m2)",
    ],
    "molecular_biology": [
        "DNA replication occurs in the S phase",
        "mRNA is transcribed from DNA by RNA polymerase",
        "tRNA carries amino acids to the ribosome",
        "CRISPR-Cas9 enables precise genome editing",
        "PCR amplifies specific DNA sequences",
        "Histone proteins package DNA into nucleosomes",
        "Oncogenes promote cell division when mutated",
        "Telomerase maintains chromosome ends",
        "Epigenetic marks regulate gene expression",
        "Cell cycle: G1 -> S -> G2 -> M",
    ],
    "econometrics": [
        "OLS: beta_hat = (X'X)^-1 * X'y",
        "Heteroskedasticity: Var(u|x) != constant",
        "Durbin-Watson tests for autocorrelation",
        "ADF test checks for unit root in time series",
        "Johansen test for cointegration rank",
        "GARCH models volatility clustering",
        "Instrumental variables address endogeneity",
        "Difference-in-differences estimation",
        "Propensity score matching reduces selection bias",
        "Panel data: fixed effects removes time-invariant heterogeneity",
    ],
    "art_history": [
        "Renaissance: perspective and humanism in art",
        "Impressionism captures light and movement",
        "Cubism: multiple viewpoints simultaneously",
        "Surrealism: dream-like imagery and the unconscious",
        "Baroque: dramatic lighting and emotional intensity",
        "Minimalism: reduction to essential forms",
        "Abstract expressionism: action painting and gesture",
        "Art Nouveau: organic forms and decorative patterns",
        "Pop Art: mass media and consumer culture",
        "Dada: anti-art and political protest",
    ],
}

DOMAIN_QUERIES = {
    "python_programming": "Python programming concepts and code patterns",
    "classical_mechanics": "Force and energy in classical physics",
    "molecular_biology": "DNA and cellular biology mechanisms",
    "econometrics": "Regression analysis and statistical econometrics",
    "art_history": "Art movements and artistic styles through history",
}


def run_test() -> dict:
    tmpdir = os.path.join(os.environ.get('TEMP', '/tmp'),
                          f'isolation_test_{os.getpid()}')
    os.makedirs(tmpdir, exist_ok=True)

    try:
        # --- TOPO SYSTEM ---
        cfg_topo = TopoMemConfig()
        cfg_topo.memory.chroma_persist_dir = os.path.join(tmpdir, "chroma_topo")
        system = TopoMemSystem(config=cfg_topo)

        # --- VECTOR-ONLY SYSTEM ---
        cfg_vec = TopoMemConfig()
        cfg_vec.memory.chroma_persist_dir = os.path.join(tmpdir, "chroma_vec")
        system_vec = TopoMemSystem(config=cfg_vec)

        # Step 1: Add memories
        print("\n[Step 1] Adding 5 domains x 10 items = 50 memories...")
        domain_node_map = {}
        for domain, items in DOMAINS.items():
            metas = [{"domain": domain, "item_idx": i} for i in range(len(items))]
            added = system.add_knowledge_batch(items, metas)
            # ChromaDB uses UUIDs, so we collect actual node IDs from the graph
            domain_node_map[domain] = []
            for nid, data in system.memory._graph.nodes(data=True):
                if data["node"].metadata.get("domain") == domain:
                    domain_node_map[domain].append(nid)
            print(f"  {domain}: {len(domain_node_map[domain])} added")

            metas_v = [{"domain": domain} for _ in items]
            system_vec.add_knowledge_batch(items, metas_v)
        total = system.memory.node_count()
        print(f"  Total: {total} nodes")

        # Step 2: TDA update
        print("\n[Step 2] Running TDA update_topology...")
        system.memory.update_topology(system.topology)
        report = system.consolidation_pass()
        print(f"  orphans={report.get('orphans_detected', '?')}, "
              f"merge_candidates={report.get('merge_candidates_found', '?')}")

        # Step 3: Analyze cluster structure
        print("\n[Step 3] Analyzing TDA cluster structure...")
        graph = system.memory._graph
        cluster_contents = {}
        for nid, data in graph.nodes(data=True):
            node = data["node"]
            cid = node.cluster_id
            if cid not in cluster_contents:
                cluster_contents[cid] = []
            cluster_contents[cid].append(node.metadata.get("domain", "unknown"))

        n_clusters = len(cluster_contents)
        avg_size = total / max(n_clusters, 1)
        print(f"  {n_clusters} clusters, avg size={avg_size:.1f}")
        csizes = sorted([(c, len(d)) for c, d in cluster_contents.items()], key=lambda x: -x[1])
        singles = sum(1 for _, s in csizes if s == 1)
        print(f"  Single-node clusters: {singles}/{n_clusters}")
        print(f"  Top 5 cluster sizes: {[s for _, s in csizes[:5]]}")

        # Node isolation per domain
        node_isolation = {}
        for domain, ids in domain_node_map.items():
            cid_counts = {}
            for nid in ids:
                node = graph.nodes[nid]["node"]
                cid = node.cluster_id
                cid_counts[cid] = cid_counts.get(cid, 0) + 1
            # fraction of domain's nodes that are the ONLY node from this domain in their cluster
            isolated = sum(1 for sz in cid_counts.values() if sz == 1)
            purity = isolated / len(ids)
            node_isolation[domain] = {"isolated": isolated, "total": len(ids), "purity": purity}
            print(f"  {domain}: {isolated}/{len(ids)} isolated ({purity:.0%})")

        avg_isolation = sum(node_isolation[d]["purity"] for d in node_isolation) / len(node_isolation)

        # Cluster domain purity
        cluster_purities = []
        for cid, doms in cluster_contents.items():
            if doms:
                mc = max(set(doms), key=doms.count)
                cluster_purities.append(doms.count(mc) / len(doms))
        avg_cluster_purity = sum(cluster_purities) / max(len(cluster_purities), 1)
        print(f"  Avg cluster purity: {avg_cluster_purity:.1%}")

        # Step 4: Retrieval purity
        print("\n[Step 4] Testing retrieval purity...")
        topo_ret = {}
        vec_ret = {}
        topo_wins = 0
        vec_wins = 0
        for domain, query in DOMAIN_QUERIES.items():
            q_emb = system.embedding.encode(query)
            r_topo = system.memory.retrieve(q_emb, strategy="topological", k=5)
            topo_doms = [node.metadata.get("domain", "?") for node, _ in r_topo]
            topo_pct = topo_doms.count(domain) / 5

            q_emb_v = system_vec.embedding.encode(query)
            r_vec = system_vec.memory.retrieve(q_emb_v, strategy="vector", k=5)
            vec_doms = [node.metadata.get("domain", "?") for node, _ in r_vec]
            vec_pct = vec_doms.count(domain) / 5

            topo_ret[domain] = {"correct": topo_doms.count(domain), "purity": topo_pct, "retrieved": topo_doms}
            vec_ret[domain] = {"correct": vec_doms.count(domain), "purity": vec_pct, "retrieved": vec_doms}
            w = "TIE" if abs(topo_pct - vec_pct) < 0.01 else ("TOPO" if topo_pct > vec_pct else "VEC")
            if topo_pct > vec_pct:
                topo_wins += 1
            elif vec_pct > topo_pct:
                vec_wins += 1
            print(f"  [{domain}] TOPO={topo_pct:.0%}  VEC={vec_pct:.0%}  -> {w}")

        avg_topo_purity = sum(topo_ret[d]["purity"] for d in topo_ret) / len(topo_ret)
        avg_vec_purity = sum(vec_ret[d]["purity"] for d in vec_ret) / len(vec_ret)
        print(f"\n  Avg: TOPO={avg_topo_purity:.1%}  VEC={avg_vec_purity:.1%}")

        # Step 5: Concentration
        print("\n[Step 5] Domain concentration...")
        rand_base = 0.2
        concentrations = {}
        for domain, ids in domain_node_map.items():
            cc = {}
            for nid in ids:
                node = graph.nodes[nid]["node"]
                cc[node.cluster_id] = cc.get(node.cluster_id, 0) + 1
            max_c = max(cc.values()) if cc else 0
            conc = max_c / len(ids)
            concentrations[domain] = conc
            print(f"  {domain}: {max_c}/{len(ids)} in largest cluster ({conc:.0%})")
        avg_conc = sum(concentrations.values()) / len(concentrations)
        print(f"  Avg concentration: {avg_conc:.1%} ({avg_conc/rand_base:.2f}x random)")

        # VERDICT
        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)
        print(f"  Node isolation: {avg_isolation:.1%}")
        print(f"  Cluster purity: {avg_cluster_purity:.1%}")
        print(f"  Single-node clusters: {singles}/{n_clusters} ({singles/max(n_clusters,1):.0%})")
        print(f"  Concentration: {avg_conc/rand_base:.2f}x random")
        print(f"  Retrieval: TOPO {topo_wins} wins, VEC {vec_wins} wins")
        print(f"  Avg retrieval: TOPO={avg_topo_purity:.1%} VEC={avg_vec_purity:.1%}")

        if avg_isolation > 0.7 and avg_cluster_purity > 0.7:
            verdict = "STRONG TDA SIGNAL — domains are topologically separated"
            signal = "strong"
        elif avg_isolation > 0.4:
            verdict = "WEAK TDA SIGNAL — some topological separation"
            signal = "weak"
        elif singles / max(n_clusters, 1) > 0.7:
            verdict = "NO TDA SIGNAL — all nodes fragmented into single-element clusters"
            signal = "none"
        else:
            verdict = "MIXED — domains partially mixed"
            signal = "mixed"

        print(f"\n  => {verdict}")
        return {
            "n_domains": 5, "items_per_domain": 10,
            "total_nodes": total, "n_clusters": n_clusters,
            "single_node_clusters": singles,
            "avg_cluster_size": avg_size,
            "avg_node_isolation": float(avg_isolation),
            "avg_cluster_purity": float(avg_cluster_purity),
            "concentration_ratio": float(avg_conc / rand_base),
            "avg_topo_purity": float(avg_topo_purity),
            "avg_vec_purity": float(avg_vec_purity),
            "topo_wins": topo_wins, "vec_wins": vec_wins,
            "verdict": verdict, "signal": signal,
            "node_isolation": node_isolation,
            "topo_retrieval": topo_ret,
            "vec_retrieval": vec_ret,
            "cluster_sizes": dict(csizes[:15]),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    r = run_test()
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"isolation_test_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print(f"\nResults: {out_path}")
