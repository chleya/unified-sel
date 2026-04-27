"""
topomem/benchmarks/cross_domain_interference_benchmark.py

方向3: 更难数据集的检索 benchmark

核心假设：
- 在简单同质语料（deer-flow）上，所有方法打平，因为 cosine similarity 已经足够
- 在跨域干扰场景下，H0 persistence weighting 应该帮助筛选更核心/稳定的记忆

设计：
1. 构造 3 个语义差异大的域（programming, physics, cooking）
2. 每个域插入 N 个记忆
3. 查询是跨域的 multi-hop 问题
4. 添加干扰项：与查询表面相似但来自错误域的记忆
5. 对比：pure vector vs hybrid with persistence weighting
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

import warnings
warnings.filterwarnings("ignore")

HF_CACHE = PROJECT / "topomem" / "data" / "models" / "hf_cache"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(HF_CACHE)

from topomem.embedding import EmbeddingManager, EmbeddingConfig
from topomem.topology import TopologyEngine, TopologyConfig
from topomem.memory import MemoryGraph, MemoryConfig


# ------------------------------------------------------------------
# Domain corpora (curated for maximum semantic distance)
# ------------------------------------------------------------------

DOMAINS = {
    "programming": [
        "Python uses dynamic typing and interprets code at runtime for flexibility",
        "JavaScript's event loop handles asynchronous operations in a single thread",
        "Rust's ownership system prevents memory leaks without garbage collection",
        "Go's goroutines provide efficient concurrent programming with minimal overhead",
        "SQL JOIN operations combine rows from multiple tables based on related columns",
        "REST APIs use HTTP methods to perform CRUD operations on resources",
        "Git branches allow parallel development without affecting the main codebase",
        "Docker containers package applications with their dependencies for portability",
        "Machine learning models train on data to recognize patterns and make predictions",
        "Graph databases store relationships between entities using edges and nodes",
        "Blockchain maintains a distributed ledger of transactions across peer networks",
        "Kubernetes orchestrates containerized applications across clusters of machines",
        "React components use virtual DOM to efficiently update the actual DOM",
        "HTTP/2 multiplexes multiple requests over a single TCP connection",
        "NoSQL databases sacrifice ACID properties for horizontal scalability and flexibility",
    ],
    "physics": [
        "Quantum entanglement correlates particle states regardless of distance between them",
        "General relativity describes gravity as the curvature of spacetime around massive objects",
        "The Standard Model classifies fundamental particles into fermions and bosons",
        "Thermodynamics studies energy transfer and entropy changes in physical systems",
        "Maxwell's equations unify electricity and magnetism into electromagnetic theory",
        "Wave-particle duality explains how light exhibits both wave and particle characteristics",
        "Entropy measures the disorder or randomness in a closed thermodynamic system",
        "The photoelectric effect demonstrates that light can knock electrons from metals",
        "Nuclear fusion combines atomic nuclei to release enormous amounts of energy",
        "Special relativity shows that time dilates and length contracts at high velocities",
        "Chaos theory studies how small changes in initial conditions create unpredictable outcomes",
        "Superconductivity allows electric current to flow without resistance at low temperatures",
        "The Heisenberg uncertainty principle limits precision in measuring particle position and momentum",
        "String theory proposes that fundamental particles are one-dimensional vibrating strings",
        "Dark matter interacts gravitationally but does not emit or absorb electromagnetic radiation",
    ],
    "cooking": [
        "Sous vide cooking uses precise temperature control in a water bath for consistent results",
        "Maillard reaction creates browning and flavor when amino acids react with sugars at high heat",
        "Emulsification suspends oil droplets in water using an emulsifier like lecithin",
        "Gluten formation gives bread dough its elastic structure through protein networks",
        "Caramelization breaks down sugar molecules at high temperatures creating sweet flavors",
        "Umami taste comes from glutamate and provides savory depth in fermented foods",
        "Confit cooking preserves food by submerging it in fat at low temperatures for long periods",
        "Deglazing dissolves caramelized bits from a pan with liquid to make sauces",
        "Proofing allows yeast dough to rest and rise before baking for optimal texture",
        "Basting coats food with pan drippings to add moisture and flavor during cooking",
        "Roux thickening agents cook together flour and fat to thicken sauces and soups",
        "Brining soaks meat in salt solution to improve moisture retention and flavor",
        "Fermentation uses microorganisms to preserve food and create complex flavors",
        "Degassing releases air bubbles from batter before baking to prevent tunneling",
        "Searing creates a flavorful crust through the Maillard reaction at high temperatures",
    ],
}

# Multi-hop queries that require combining knowledge from multiple domains
MULTIHOP_QUERIES = [
    {
        "query": "How might quantum entanglement principles relate to secure cryptographic systems?",
        "keywords": ["quantum", "entanglement", "cryptographic", "secure"],
        "bridge_domain": "programming",
    },
    {
        "query": "What cooking technique analogy explains the Maillard reaction in terms of chemical bonds?",
        "keywords": ["Maillard", "chemical", "bonds", "reaction"],
        "bridge_domain": "physics",
    },
    {
        "query": "How could thermodynamic efficiency principles optimize computational resource allocation?",
        "keywords": ["thermodynamic", "efficiency", "computational", "resource"],
        "bridge_domain": "programming",
    },
    {
        "query": "Explain protein folding in terms of how gluten networks form in bread dough.",
        "keywords": ["protein", "folding", "gluten", "network"],
        "bridge_domain": "cooking",
    },
    {
        "query": "How does wave interference relate to noise-canceling headphone technology?",
        "keywords": ["wave", "interference", "noise", "canceling"],
        "bridge_domain": "programming",
    },
]

# Interference items: semantically similar to query but from wrong domain
INTERFERENCE_ITEMS = {
    "quantum_programming": [
        "Python quantum computing libraries like Qiskit and Cirq simulate quantum circuits",
        "JavaScript promises handle asynchronous quantum API calls in web applications",
    ],
    "maillard_cooking": [
        "Caramelized onions add sweetness to French onion soup through browning reactions",
        "Searing steak creates a flavorful crust via high-temperature chemical reactions",
    ],
    "thermodynamic_programming": [
        "Cloud computing dynamically allocates computational resources based on demand",
        "Load balancing distributes processing tasks across multiple servers efficiently",
    ],
}


# ------------------------------------------------------------------
# Retrieval systems
# ------------------------------------------------------------------

class PureVectorRetrieval:
    """纯向量检索：只使用 cosine similarity。"""
    def __init__(self, memory: MemoryGraph):
        self.memory = memory

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        results = self.memory.retrieve(query_embedding, strategy="vector", k=k)
        return [(node.content[:100], score) for node, score in results]


class HybridPersistenceRetrieval:
    """混合检索 + H0 persistence weighting（方向4的新功能）。"""
    def __init__(self, memory: MemoryGraph):
        self.memory = memory

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        results = self.memory.retrieve(query_embedding, strategy="hybrid", k=k)
        return [(node.content[:100], score) for node, score in results]


# ------------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------------

def keyword_match_score(content: str, keywords: List[str]) -> float:
    """检查内容中包含多少关键词。"""
    content_lower = content.lower()
    matches = sum(1 for kw in keywords if kw.lower() in content_lower)
    return matches / len(keywords) if keywords else 0.0


def domain_relevance_score(content: str, expected_domain: str) -> float:
    """检查内容是否与目标域相关。"""
    domain_keywords = {
        "programming": ["code", "program", "software", "api", "data", "system", "database", "algorithm"],
        "physics": ["quantum", "energy", "particle", "wave", "field", "mass", "force", "relativity"],
        "cooking": ["cook", "heat", "temperature", "flavor", "food", "bake", "sauce", "ingredient"],
    }
    keywords = domain_keywords.get(expected_domain.lower(), [])
    if not keywords:
        return 0.5
    content_lower = content.lower()
    matches = sum(1 for kw in keywords if kw.lower() in content_lower)
    return matches / len(keywords)


def evaluate_retrieval(
    retrieved: List[Tuple[str, float]],
    query_info: dict,
    k: int = 5
) -> dict:
    """评估检索结果的质量。"""
    keywords = query_info["keywords"]
    bridge_domain = query_info["bridge_domain"]

    top_k = retrieved[:k]
    contents = [c for c, _ in top_k]

    # Keyword match
    keyword_scores = [keyword_match_score(c, keywords) for c in contents]
    keyword_recall = max(keyword_scores) if keyword_scores else 0.0

    # Domain relevance
    domain_scores = [domain_relevance_score(c, bridge_domain) for c in contents]
    domain_recall = max(domain_scores) if domain_scores else 0.0

    # Combined: prioritize both keyword match AND domain relevance
    combined_score = 0.6 * keyword_recall + 0.4 * domain_recall

    return {
        "keyword_recall": keyword_recall,
        "domain_recall": domain_recall,
        "combined_score": combined_score,
        "top_contents": contents[:3],
    }


# ------------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------------

def run_benchmark():
    print("=" * 70)
    print("Cross-Domain Interference Benchmark: Direction 3")
    print("=" * 70)

    # Initialize components
    print("\n[1/4] Initializing system components...")
    emb_config = EmbeddingConfig()
    topo_config = TopologyConfig(max_homology_dim=2)
    mem_config = MemoryConfig(
        topo_recompute_interval=5,  # More frequent updates for small corpus
    )

    embedding_manager = EmbeddingManager(emb_config)
    topology_engine = TopologyEngine(topo_config)
    memory = MemoryGraph(mem_config, embedding_manager)

    print(f"    Embedding: {emb_config.model_name} ({emb_config.dimension}D)")
    print(f"    Metric: {topo_config.metric}")

    # Add memories from each domain
    print("\n[2/4] Building memory corpus...")
    all_memories = []
    for domain, contents in DOMAINS.items():
        for content in contents:
            all_memories.append({
                "content": content,
                "domain": domain,
                "embedding": embedding_manager.encode(content),
            })

    # Add all memories
    for item in all_memories:
        memory.add_memory(
            content=item["content"],
            embedding=item["embedding"],
            metadata={"domain": item["domain"]},
            topo_engine=topology_engine,
        )

    print(f"    Total memories: {len(all_memories)}")
    print(f"    Domains: {list(DOMAINS.keys())}")

    # Add interference items
    print("\n[3/4] Adding interference items...")
    interference_count = 0
    for category, items in INTERFERENCE_ITEMS.items():
        for content in items:
            embedding = embedding_manager.encode(content)
            memory.add_memory(
                content=content,
                embedding=embedding,
                metadata={"domain": "interference", "category": category},
                topo_engine=topology_engine,
            )
            interference_count += 1
    print(f"    Added {interference_count} interference items")
    print(f"    Total nodes in memory: {memory.node_count()}")

    # Run evaluation
    print("\n[4/4] Running retrieval evaluation...")

    purevec_system = PureVectorRetrieval(memory)
    hybrid_system = HybridPersistenceRetrieval(memory)

    results = {
        "pure_vector": [],
        "hybrid_persistence": [],
    }

    for query_info in MULTIHOP_QUERIES:
        query = query_info["query"]
        query_embedding = embedding_manager.encode(query)

        # Pure vector retrieval
        pv_results = purevec_system.retrieve(query_embedding, k=5)
        pv_eval = evaluate_retrieval(pv_results, query_info)
        results["pure_vector"].append({
            "query": query,
            "eval": pv_eval,
        })

        # Hybrid with persistence
        hp_results = hybrid_system.retrieve(query_embedding, k=5)
        hp_eval = evaluate_retrieval(hp_results, query_info)
        results["hybrid_persistence"].append({
            "query": query,
            "eval": hp_eval,
        })

    # Compute aggregate metrics
    pv_keyword = np.mean([r["eval"]["keyword_recall"] for r in results["pure_vector"]])
    pv_domain = np.mean([r["eval"]["domain_recall"] for r in results["pure_vector"]])
    pv_combined = np.mean([r["eval"]["combined_score"] for r in results["pure_vector"]])

    hp_keyword = np.mean([r["eval"]["keyword_recall"] for r in results["hybrid_persistence"]])
    hp_domain = np.mean([r["eval"]["domain_recall"] for r in results["hybrid_persistence"]])
    hp_combined = np.mean([r["eval"]["combined_score"] for r in results["hybrid_persistence"]])

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'PureVec':>12} {'Hybrid+Persist':>15} {'Delta':>10}")
    print("-" * 65)
    print(f"{'Keyword Recall':<25} {pv_keyword:>12.3f} {hp_keyword:>15.3f} {hp_keyword - pv_keyword:>+10.3f}")
    print(f"{'Domain Recall':<25} {pv_domain:>12.3f} {hp_domain:>15.3f} {hp_domain - pv_domain:>+10.3f}")
    print(f"{'Combined Score':<25} {pv_combined:>12.3f} {hp_combined:>15.3f} {hp_combined - pv_combined:>+10.3f}")

    # Per-query breakdown
    print("\n" + "-" * 70)
    print("PER-QUERY RESULTS")
    print("-" * 70)
    for i, query_info in enumerate(MULTIHOP_QUERIES):
        q_short = query_info["query"][:50] + "..."
        pv = results["pure_vector"][i]["eval"]
        hp = results["hybrid_persistence"][i]["eval"]
        delta = hp["combined_score"] - pv["combined_score"]
        winner = "H+P" if delta > 0 else "PV" if delta < 0 else "TIE"
        print(f"\nQuery {i+1}: {q_short}")
        print(f"  PureVec: kw={pv['keyword_recall']:.2f} dom={pv['domain_recall']:.2f} comb={pv['combined_score']:.2f}")
        print(f"  Hybrid+P: kw={hp['keyword_recall']:.2f} dom={hp['domain_recall']:.2f} comb={hp['combined_score']:.2f}")
        print(f"  Winner: {winner} (delta={delta:+.3f})")

    # Statistical significance (simple sign test)
    wins = sum(
        1 for i in range(len(MULTIHOP_QUERIES))
        if results["hybrid_persistence"][i]["eval"]["combined_score"] > results["pure_vector"][i]["eval"]["combined_score"]
    )
    total = len(MULTIHOP_QUERIES)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if abs(hp_combined - pv_combined) < 0.05:
        verdict = "TIE: No significant difference between methods"
    elif hp_combined > pv_combined:
        verdict = f"Hybrid+Persistence WINS (delta={hp_combined-pv_combined:.3f})"
    else:
        verdict = f"PureVector WINS (delta={pv_combined-hp_combined:.3f})"

    print(f"\n{verdict}")
    print(f"Win count: {wins}/{total} queries")

    # Save results
    timestamp = int(time.time())
    outpath = Path(__file__).parent / "results" / f"cross_domain_interference_{timestamp}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "experiment": "Direction 3: Cross-Domain Interference Benchmark",
        "timestamp": timestamp,
        "n_memories": len(all_memories),
        "n_interference": interference_count,
        "n_queries": len(MULTIHOP_QUERIES),
        "aggregate": {
            "pure_vector": {
                "keyword_recall": float(pv_keyword),
                "domain_recall": float(pv_domain),
                "combined_score": float(pv_combined),
            },
            "hybrid_persistence": {
                "keyword_recall": float(hp_keyword),
                "domain_recall": float(hp_domain),
                "combined_score": float(hp_combined),
            },
            "delta_combined": float(hp_combined - pv_combined),
        },
        "per_query": results,
        "verdict": verdict,
        "win_count": wins,
    }

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {outpath}")
    return output_data


if __name__ == "__main__":
    run_benchmark()
