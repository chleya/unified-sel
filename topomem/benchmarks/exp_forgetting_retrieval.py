"""
benchmarks/exp_forgetting_retrieval.py

P0 实验：检索策略抗遗忘能力对比

核心问题：当新知识入侵时，vector检索和topological检索
对旧知识的recall谁更稳定？

设计：
  1. 存入 Domain A（N items）
  2. 测 baseline recall_A（query A items，看 top-K 中有多少是 A）
  3. 存入 Domain B（语义相似的 domain）
  4. 测 post-invasion recall_A
  5. 对比：vector recall drop vs topological recall drop

数据集：20 Newsgroups
  - 相似domain对: comp.sys.ibm.pc.hardware vs comp.sys.mac.hardware (computing)
  - 相异domain对: comp.sys.ibm.pc.hardware vs rec.sport.baseball
"""

import sys, io, json, time
from pathlib import Path
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_SCRIPT_DIR = Path(__file__).resolve().parent          # .../benchmarks/
_PKG_DIR = Path(__file__).resolve().parents[1]          # .../topomem/
_ROOT_DIR = Path(__file__).resolve().parents[2]          # .../unified-sel/
sys.path.insert(0, str(_ROOT_DIR))    # 让 from topomem 能找到
sys.path.insert(0, str(_SCRIPT_DIR))  # 让 from exp1_data_loader 能找到

from topomem.system import TopoMemSystem
from topomem.config import TopoMemConfig, MemoryConfig, EmbeddingConfig, TopologyConfig

# 20 Newsgroups 数据加载器
from exp1_data_loader import load_20newsgroups, encode_corpus


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms


def compute_domain_similarity(items_A, items_B, emb_manager):
    """计算两个 domain 之间的语义相似度。"""
    texts_A = [x["text"] for x in items_A]
    texts_B = [x["text"] for x in items_B]
    vecs_A = normalize(np.array(emb_manager.encode_batch(texts_A)))
    vecs_B = normalize(np.array(emb_manager.encode_batch(texts_B)))
    sim_matrix = vecs_A @ vecs_B.T
    return {
        "mean": float(np.mean(sim_matrix)),
        "max": float(np.max(sim_matrix)),
        "median": float(np.median(sim_matrix)),
    }


def measure_recall_by_domain(
    system,
    domain_items,
    domain_name,
    k=10,
):
    """测某个 domain 的 items 作为 query 时，top-K 中同 domain 物品的占比。

    Returns:
        {strategy: recall_fraction}
        recall_fraction = 在 top-K 中有多少比例是同 domain 的
    """
    texts = [item["text"] for item in domain_items]
    vecs = system.embedding.encode_batch(texts)

    results = {}
    for strategy in ["vector", "topological", "hybrid"]:
        all_recalls = []
        for i, (vec, item) in enumerate(zip(vecs, domain_items)):
            try:
                retrieved = system.memory.retrieve(vec, strategy=strategy, k=k)
                if not retrieved:
                    all_recalls.append(0.0)
                    continue
                # retrieved: List[Tuple[MemoryNode, float]]
                retrieved_domains = []
                for node_score in retrieved:
                    node = node_score[0] if isinstance(node_score, tuple) else node_score
                    retrieved_domains.append(node.metadata.get("domain"))

                # recall = 同 domain 在 top-K 中的比例
                same_domain_count = sum(1 for d in retrieved_domains if d == domain_name)
                recall = same_domain_count / k
                all_recalls.append(recall)
            except Exception as e:
                all_recalls.append(0.0)

        results[strategy] = {
            "mean": float(np.mean(all_recalls)) if all_recalls else 0.0,
            "std": float(np.std(all_recalls)) if len(all_recalls) > 1 else 0.0,
        }
    return results


def run_experiment_pair(
    items_pool_A,
    items_pool_B,
    domain_A_name,
    domain_B_name,
    n_runs=5,
    seed=42,
):
    """对一对 domain 运行实验。items_pool_A/B 是完整候选池，每 run 随机抽样100篇。"""
    print(f"\n{'='*70}")
    print(f"Domain A={domain_A_name} ({len(items_pool_A)} docs pool) vs Domain B={domain_B_name} ({len(items_pool_B)} docs pool)")
    print(f"{'='*70}")

    all_runs = []

    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        tmp_dir = _PKG_DIR / "tmp_exp_forgetting_v2" / f"run_{run_idx}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 每 run 随机选100篇，保证A/B内部不重叠
        rng = np.random.RandomState(run_seed)
        idx_A = rng.permutation(len(items_pool_A))[:100]
        idx_B = rng.permutation(len(items_pool_B))[:100]
        items_A = [items_pool_A[i] for i in idx_A]
        items_B = [items_pool_B[i] for i in idx_B]

        config = TopoMemConfig(
            memory=MemoryConfig(chroma_persist_dir=str(tmp_dir), similarity_top_k=15),
            embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            topology=TopologyConfig(max_homology_dim=1, metric="cosine"),
        )
        system = TopoMemSystem(config=config)

        # === Phase 1: 存 A ===
        for item in items_A:
            system.add_knowledge(item["text"], metadata={
                "domain": domain_A_name,
                "item_id": item["doc_id"],
            })
        system.memory.update_topology(system.topology)

        # === Phase 2: 测 baseline recall ===
        baseline = measure_recall_by_domain(system, items_A, domain_A_name, k=10)

        # === Phase 3: 存 B（入侵）===
        for item in items_B:
            system.add_knowledge(item["text"], metadata={
                "domain": domain_B_name,
                "item_id": item["doc_id"],
            })
        system.memory.update_topology(system.topology)

        # === Phase 4: 测 post-invasion recall ===
        post_invasion = measure_recall_by_domain(system, items_A, domain_A_name, k=10)

        # 计算 drop
        drops = {}
        for s in baseline:
            drops[s] = {
                k: baseline[s][k] - post_invasion[s][k]
                for k in baseline[s]
            }

        run_data = {
            "run": run_idx,
            "baseline": baseline,
            "post_invasion": post_invasion,
            "drops": drops,
        }
        all_runs.append(run_data)

        print(f"\n  [Run {run_idx+1}/{n_runs}]")
        for s in ["vector", "topological", "hybrid"]:
            b = baseline[s]["mean"]
            p = post_invasion[s]["mean"]
            d = drops[s]["mean"]
            print(f"    {s:12s}: baseline={b:.1%}  post={p:.1%}  drop={d:+.1%}")

    # ============================================================
    # 汇总
    # ============================================================
    print(f"\n  AGGREGATE ({n_runs} runs):")
    summary = {}
    for s in ["vector", "topological", "hybrid"]:
        base_vals = [r["baseline"][s]["mean"] for r in all_runs]
        post_vals = [r["post_invasion"][s]["mean"] for r in all_runs]
        drop_vals = [r["drops"][s]["mean"] for r in all_runs]

        summary[s] = {
            "baseline_mean": float(np.mean(base_vals)),
            "baseline_std": float(np.std(base_vals)),
            "post_invasion_mean": float(np.mean(post_vals)),
            "post_invasion_std": float(np.std(post_vals)),
            "drop_mean": float(np.mean(drop_vals)),
            "drop_std": float(np.std(drop_vals)),
        }
        print(f"    {s:12s}: drop={summary[s]['drop_mean']:+.1%} (±{summary[s]['drop_std']:.1%})  "
              f"base={summary[s]['baseline_mean']:.1%}  post={summary[s]['post_invasion_mean']:.1%}")

    winner = min(summary.keys(), key=lambda s: summary[s]["drop_mean"])
    print(f"\n  Winner (least recall drop): {winner.upper()}")

    return {"domain_A": domain_A_name, "domain_B": domain_B_name, "runs": all_runs, "summary": summary, "winner": winner}


def run():
    print("="*70)
    print("P0: Retrieval Strategy vs Forgetting (20 Newsgroups)")
    print("="*70)

    # 加载 20 Newsgroups 数据
    print("\nLoading 20 Newsgroups corpus...")
    all_items = load_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    print(f"  Total: {len(all_items)} documents")

    # 按 domain 分组
    from collections import defaultdict
    by_domain = defaultdict(list)
    for item in all_items:
        by_domain[item["label_name"]].append(item)

    # 选择实验对
    SIMILAR_PAIRS = [
        ("comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"),  # 高度相似（都是PC硬件）
        ("rec.sport.baseball", "rec.sport.hockey"),              # 高度相似（都是体育）
    ]
    DISSIMILAR_PAIRS = [
        ("comp.sys.ibm.pc.hardware", "rec.sport.baseball"),     # 完全不同
        ("sci.crypt", "rec.autos"),                              # 完全不同
    ]

    results = {}

    # 1. 相似 domain 对（高干扰 → 预期拓扑优势明显）
    print("\n" + "#"*70)
    print("# SIMILAR DOMAIN PAIRS (high interference expected)")
    print("#"*70)
    for pair in SIMILAR_PAIRS:
        dA, dB = pair
        if dA in by_domain and dB in by_domain:
            r = run_experiment_pair(by_domain[dA], by_domain[dB], dA, dB, n_runs=3, seed=42)
            results[f"similar_{dA}_vs_{dB}"] = r

    # 2. 相异 domain 对（低干扰 → 预期差异小）
    print("\n" + "#"*70)
    print("# DISSIMILAR DOMAIN PAIRS (low interference expected)")
    print("#"*70)
    for pair in DISSIMILAR_PAIRS:
        dA, dB = pair
        if dA in by_domain and dB in by_domain:
            r = run_experiment_pair(by_domain[dA], by_domain[dB], dA, dB, n_runs=3, seed=42)
            results[f"dissimilar_{dA}_vs_{dB}"] = r

    # 保存
    out_path = _SCRIPT_DIR / "results" / f"exp_forgetting_retrieval_{int(time.time())}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # 打印最终结论
    print("\n" + "="*70)
    print("FINAL CONCLUSION")
    print("="*70)
    similar_drops = {s: [] for s in ["vector", "topological", "hybrid"]}
    dissimilar_drops = {s: [] for s in ["vector", "topological", "hybrid"]}

    for key, r in results.items():
        target = similar_drops if "similar" in key else dissimilar_drops
        for s in ["vector", "topological", "hybrid"]:
            target[s].append(r["summary"][s]["drop_mean"])

    print("\n  Avg recall drop in SIMILAR pairs (high interference):")
    for s in ["vector", "topological", "hybrid"]:
        print(f"    {s}: {np.mean(similar_drops[s]):+.1%}")

    print("\n  Avg recall drop in DISSIMILAR pairs (low interference):")
    for s in ["vector", "topological", "hybrid"]:
        print(f"    {s}: {np.mean(dissimilar_drops[s]):+.1%}")

    return results


if __name__ == "__main__":
    run()
