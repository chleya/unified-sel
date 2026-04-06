"""
topomem/benchmarks/real_task_comparison.py

真实任务对比基准：TopoMem Hybrid vs 纯向量检索 vs k-NN
核心问题：TopoMem 在真实下游任务上有没有增量价值？

方案：三种检索都用相同底层向量，但检索策略不同
- PureVec: 纯 cosine similarity
- TopoMem-Hybrid: H0 cluster boundary boosted + vector
- k-NN: k-nearest neighbors
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

import warnings
warnings.filterwarnings("ignore")
HF_CACHE = PROJECT / "topomem" / "data" / "models" / "hf_cache"
os.environ["HF_HOME"] = str(HF_CACHE)

from topomem.embedding import EmbeddingManager, EmbeddingConfig
from topomem.topology import TopologyEngine, TopologyConfig
from topomem.memory import MemoryGraph, MemoryConfig


# ------------------------------------------------------------------
# 1. Load corpus
# ------------------------------------------------------------------

def load_corpus(base_path: str) -> List[dict]:
    base = Path(base_path)
    items = []
    for py_file in base.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if len(content) < 100:
                continue
            rel = py_file.relative_to(base).as_posix()
            items.append({"id": rel, "content": content, "source": "deer-flow"})
        except Exception:
            pass
    return items


def sample_items(items: List[dict], n: int, seed: int = 42) -> List[dict]:
    rng = np.random.RandomState(seed)
    if len(items) <= n:
        return items
    idx = rng.choice(len(items), n, replace=False)
    return [items[i] for i in idx]


# ------------------------------------------------------------------
# 2. Systems
# ------------------------------------------------------------------

class PureVecSystem:
    """Pure cosine similarity retrieval."""
    def __init__(self, items: List[dict]):
        self.items = items
        self.emb = EmbeddingManager(EmbeddingConfig())
        texts = [item["content"][:512] for item in items]
        self.embs = self.emb.encode_batch(texts)  # (N, D) normalized

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        q = self.emb.encode(query)  # (D,) normalized
        sims = self.embs @ q  # (N,)
        top = np.argsort(sims)[::-1][:k]
        return [(self.items[i], float(sims[i])) for i in top]


class TopoMemHybridSystem:
    """TopoMem-style: H0 cluster boundary boosting + vector retrieval.

    Uses H0 TDA clustering to boost retrieval:
    - Within top-k cluster: boost similarity
    - Across cluster boundary: penalize
    - If cluster has < k members, expand to neighboring clusters
    """
    def __init__(self, items: List[dict]):
        self.items = items
        self.item_by_id = {item["id"]: item for item in items}
        self.emb = EmbeddingManager(EmbeddingConfig())
        self.topo = TopologyEngine(TopologyConfig())

        # Encode all items
        texts = [item["content"][:512] for item in items]
        self.embs = self.emb.encode_batch(texts)  # (N, D)
        self.n = len(items)

        # Compute H0 clustering (pass FULL diagram list, not just diagram[0])
        diagram = self.topo.compute_persistence(self.embs)
        self.h0_diagram = diagram[0] if len(diagram) > 0 else np.array([])
        self.h1_diagram = diagram[1] if len(diagram) > 1 else np.array([])

        # cluster_labels_from_h0 expects: (diagram_list, points)
        self.cluster_labels = self.topo.cluster_labels_from_h0(diagram, self.embs)

        # Compute cluster-level centroid similarities
        self._compute_cluster_stats()

    def _compute_cluster_stats(self):
        """Precompute cluster centroids and within-cluster coherence."""
        unique_clusters = np.unique(self.cluster_labels)
        self.cluster_centroids = {}
        self.cluster_coherence = {}

        for c in unique_clusters:
            mask = self.cluster_labels == c
            members = self.embs[mask]
            centroid = members.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            self.cluster_centroids[c] = centroid
            # Within-cluster coherence: mean pairwise similarity
            if len(members) > 1:
                sims = members @ centroid
                self.cluster_coherence[c] = float(np.mean(sims))
            else:
                self.cluster_coherence[c] = 1.0

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        q = self.emb.encode(query)  # (D,)

        # Base cosine similarities
        base_sims = self.embs @ q  # (N,)

        # Cluster-based boosted scores
        boosted_sims = np.zeros(self.n)
        cluster_scores = np.zeros(self.n)

        for i in range(self.n):
            c = self.cluster_labels[i]
            centroid_sim = float(self.cluster_centroids.get(c, q) @ q)
            coherence = self.cluster_coherence.get(c, 0.5)
            # Boost: within-cluster coherence + query-centroid similarity
            cluster_scores[i] = coherence * (0.5 + 0.5 * centroid_sim)

        # Combine: base similarity weighted by cluster health
        # Cluster health = coherence * centroid-query sim
        boosted_sims = base_sims * (0.6 + 0.4 * cluster_scores)

        top = np.argsort(boosted_sims)[::-1][:k]
        return [(self.items[i], float(boosted_sims[i])) for i in top]


class kNNSystem:
    """Simple k-NN."""
    def __init__(self, items: List[dict], k: int = 5):
        self.items = items
        self.k = k
        self.emb = EmbeddingManager(EmbeddingConfig())
        texts = [item["content"][:512] for item in items]
        self.embs = self.emb.encode_batch(texts)

    def retrieve(self, query: str, k: int = None) -> List[Tuple[dict, float]]:
        k = k or self.k
        q = self.emb.encode(query)
        sims = self.embs @ q
        top = np.argsort(sims)[::-1][:k]
        return [(self.items[i], float(sims[i])) for i in top]


# ------------------------------------------------------------------
# 3. Evaluation
# ------------------------------------------------------------------

def safe_text(item: dict) -> str:
    content = item.get("content", "")
    if isinstance(content, list):
        return " ".join(str(c) for c in content)
    return str(content)


def recall_at_k(retrieved: List[Tuple[dict, float]], keywords: List[str], k: int = 5) -> float:
    if not retrieved or not keywords:
        return 0.0
    hits = 0
    for item, _ in retrieved[:k]:
        text = safe_text(item).lower() + " " + str(item.get("id", "")).lower()
        for kw in keywords:
            if kw.lower() in text:
                hits += 1
                break
    return hits / len(keywords)


def mrr(retrieved: List[Tuple[dict, float]], keywords: List[str]) -> float:
    for rank, (item, _) in enumerate(retrieved, 1):
        text = safe_text(item).lower() + " " + str(item.get("id", "")).lower()
        for kw in keywords:
            if kw.lower() in text:
                return 1.0 / rank
    return 0.0


# ------------------------------------------------------------------
# 4. Tasks
# ------------------------------------------------------------------

def get_tasks() -> List[dict]:
    return [
        {"query": "How does DeerFlow handle user authentication and session management?", "keywords": ["auth", "session", "user", "login"]},
        {"query": "What is the interface for creating a new research agent in DeerFlow?", "keywords": ["agent", "research", "create", "interface"]},
        {"query": "How does the multi-agent orchestration work?", "keywords": ["orchestrat", "agent", "workflow", "multi"]},
        {"query": "What sandboxing mechanism is used for code execution?", "keywords": ["sandbox", "exec", "security", "isolate"]},
        {"query": "How are tool outputs parsed and fed back to agents?", "keywords": ["tool", "output", "parse", "feedback"]},
        {"query": "What is the prompt template structure for the flow coordinator?", "keywords": ["prompt", "template", "coordinator", "flow"]},
        {"query": "How does DeerFlow handle rate limiting and quota management?", "keywords": ["rate", "limit", "quota", "throttle"]},
        {"query": "What data structures are used for agent state persistence?", "keywords": ["state", "persist", "checkpoint", "memory"]},
    ]


# ------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------

def main():
    print("=== Real Task Comparison: TopoMem vs Vector Retrieval ===\n")

    items = load_corpus(r"F:\workspace-ideas\deer-flow")
    print(f"Loaded {len(items)} files")
    sampled = sample_items(items, 100, seed=42)
    print(f"Using 100 items (seed=42)\n")

    print("Building systems...")
    t0 = time.time()
    systems = {
        "PureVec": PureVecSystem(sampled),
        "TopoMem-Hybrid": TopoMemHybridSystem(sampled),
        "kNN(k=3)": kNNSystem(sampled, k=3),
        "kNN(k=5)": kNNSystem(sampled, k=5),
    }
    print(f"  Done in {time.time()-t0:.1f}s\n")

    tasks = get_tasks()
    results = {name: {"r5": [], "r10": [], "mrr": []} for name in systems}

    for task in tasks:
        q, kws = task["query"], task["keywords"]
        for name, sys in systems.items():
            try:
                ret = sys.retrieve(q, k=10)
                r5 = recall_at_k(ret, kws, 5)
                r10 = recall_at_k(ret, kws, 10)
                m = mrr(ret, kws)
                results[name]["r5"].append(r5)
                results[name]["r10"].append(r10)
                results[name]["mrr"].append(m)
            except Exception as e:
                print(f"  [{name}] Error: {e}")
                results[name]["r5"].append(0.0)
                results[name]["r10"].append(0.0)
                results[name]["mrr"].append(0.0)

    summary = {}
    for name, m in results.items():
        summary[name] = {
            "R@5": float(np.mean(m["r5"])) if m["r5"] else 0.0,
            "R@10": float(np.mean(m["r10"])) if m["r10"] else 0.0,
            "MRR": float(np.mean(m["mrr"])) if m["mrr"] else 0.0,
        }

    print(f"{'System':<16} {'R@5':>6} {'R@10':>6} {'MRR':>6}")
    print("-" * 40)
    for name, m in sorted(summary.items(), key=lambda x: -x[1]["R@5"]):
        print(f"{name:<16} {m['R@5']:>6.2f} {m['R@10']:>6.2f} {m['MRR']:>6.2f}")

    print("\n--- Per-Task R@5 ---")
    print(f"{'Query':<42} {'PVec':>6} {'TM':>6} {'k3':>6} {'k5':>6}")
    print("-" * 68)
    names = list(systems.keys())
    for i, task in enumerate(tasks):
        vals = []
        for name in names:
            v = results[name]["r5"][i] if i < len(results[name]["r5"]) else 0
            vals.append(f"{v:>6.2f}")
        q_short = task["query"][:40]
        print(f"{q_short:<42} " + " ".join(vals))

    # Verdict
    best = sorted(summary.items(), key=lambda x: -x[1]["R@5"])[0]
    second = sorted(summary.items(), key=lambda x: -x[1]["R@5"])[1]
    diff = best[1]["R@5"] - second[1]["R@5"]

    print(f"\n=== VERDICT ===")
    if abs(diff) < 0.05:
        print(f"TIE: {best[0]} vs {second[0]} (diff={diff:.2f})")
    elif diff > 0:
        print(f"{best[0]} wins by {diff:.2f}")
    else:
        print(f"{second[0]} wins")

    # TopoMem-Hybrid analysis
    tm_mrr = summary.get("TopoMem-Hybrid", {}).get("MRR", 0)
    pv_mrr = summary.get("PureVec", {}).get("MRR", 0)
    if tm_mrr > pv_mrr + 0.05:
        print(f"ACTION: TopoMem-Hybrid has measurable advantage (MRR +{tm_mrr-pv_mrr:.2f})")
    elif tm_mrr < pv_mrr - 0.05:
        print(f"ACTION: PureVec outperforms TopoMem-Hybrid (MRR -{pv_mrr-tm_mrr:.2f})")
    else:
        print(f"ACTION: No significant difference - H0 topology adds no retrieval value")

    # Save
    out_path = Path(__file__).parent / "results" / f"real_task_comparison_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_task": results, "tasks": tasks}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
