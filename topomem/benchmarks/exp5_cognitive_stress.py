"""
topomem/benchmarks/exp5_cognitive_stress.py

实验5：认知压力测试

核心问题：
- TopoMem 在极限条件下表现如何？
- 记忆容量、并发压力、长期运行的退化

实验设计：
1. 记忆容量测试：最大记忆条目数
2. 并发压力：同时处理多个检索请求
3. 长期退化：重复访问导致的质量衰减
4. 干扰抵抗：无关信息混入时的选择性记忆
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from topomem.embedding import EmbeddingManager, EmbeddingConfig
from topomem.topology import TopologyEngine, TopologyConfig
from topomem.memory import MemoryGraph, MemoryConfig

sys.path.insert(0, str(Path(__file__).parent))
from exp1_data_loader import load_or_fetch_corpus, get_topic_array
from exp1_metrics import compute_h0_purity, compute_rand_index


# ------------------------------------------------------------------
# 测试1: 记忆容量测试
# ------------------------------------------------------------------

def test_memory_capacity(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    memory_sizes: List[int] = [100, 500, 1000, 2000, 5000],
    seed: int = 42,
) -> Dict:
    """测试记忆容量对检索质量的影响。

    逐步增加记忆条目，观察检索精度变化。
    """
    print("\n  [Test 1] Memory Capacity...")

    results = []
    rng = np.random.RandomState(seed)

    for size in memory_sizes:
        if size > len(embeddings):
            continue

        # 采样
        indices = rng.choice(len(embeddings), size=size, replace=False)
        sample_emb = embeddings[indices]
        sample_labels = true_labels[indices]

        # K-Means 聚类质量作为"记忆检索"代理
        n_clusters = min(6, len(np.unique(sample_labels)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(sample_emb)

        ari = adjusted_rand_score(sample_labels, labels)
        purity = compute_h0_purity(labels, sample_labels)

        results.append({
            "memory_size": size,
            "ari": float(ari),
            "purity": float(purity),
        })

        print(f"    Size={size}: ARI={ari:.4f}, Purity={purity:.4f}")

    return {
        "test": "memory_capacity",
        "results": results,
    }


# ------------------------------------------------------------------
# 测试2: 并发压力测试
# ------------------------------------------------------------------

def test_concurrent_pressure(
    items: List[Dict],
    embeddings: np.ndarray,
    query_counts: List[int] = [1, 10, 50, 100],
    seed: int = 42,
) -> Dict:
    """测试并发检索压力。

    模拟多个并发查询，观察响应时间变化。
    """
    print("\n  [Test 2] Concurrent Pressure...")

    results = []
    rng = np.random.RandomState(seed)

    for n_queries in query_counts:
        # 准备查询
        query_indices = rng.choice(len(embeddings), size=n_queries, replace=False)
        queries = [embeddings[i] for i in query_indices]

        # 串行执行（模拟并发）
        start = time.time()
        for query_emb in queries:
            # 简单的余弦相似度计算作为"检索"
            similarities = np.dot(embeddings, query_emb)
            _ = np.argsort(similarities)[-5:]  # top-5
        elapsed = time.time() - start

        avg_time = elapsed / n_queries * 1000  # ms per query

        results.append({
            "n_queries": n_queries,
            "total_time_sec": float(elapsed),
            "avg_time_ms": float(avg_time),
            "throughput_qps": float(n_queries / elapsed) if elapsed > 0 else 0,
        })

        print(f"    Queries={n_queries}: {avg_time:.2f}ms/query, {n_queries/elapsed:.1f} QPS")

    return {
        "test": "concurrent_pressure",
        "results": results,
    }


# ------------------------------------------------------------------
# 测试3: 长期退化测试
# ------------------------------------------------------------------

def test_long_term_degradation(
    items: List[Dict],
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    n_cycles: int = 10,
    seed: int = 42,
) -> Dict:
    """测试重复访问/更新导致的长期退化。

    模拟多次"学习-检索"循环，观察质量变化。
    """
    print("\n  [Test 3] Long-term Degradation...")

    results = []
    rng = np.random.RandomState(seed)

    # 初始状态
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    initial_labels = kmeans.fit_predict(embeddings)
    initial_ari = adjusted_rand_score(true_labels, initial_labels)

    results.append({
        "cycle": 0,
        "ari": float(initial_ari),
        "note": "initial",
    })
    print(f"    Cycle=0 (initial): ARI={initial_ari:.4f}")

    # 模拟每次"学习"后的状态
    for cycle in range(1, n_cycles + 1):
        # 模拟噪声累积（每次学习引入微小噪声）
        noise_scale = 0.01 * cycle  # 递增噪声
        noise = rng.randn(*embeddings.shape) * noise_scale
        degraded_emb = embeddings + noise

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(degraded_emb)
        ari = adjusted_rand_score(true_labels, labels)

        degradation = initial_ari - ari

        results.append({
            "cycle": cycle,
            "ari": float(ari),
            "degradation": float(degradation),
            "noise_scale": float(noise_scale),
        })

        print(f"    Cycle={cycle}: ARI={ari:.4f} (degradation: {degradation:+.4f})")

    # 检查是否有显著退化
    final_ari = results[-1]["ari"]
    max_degradation = max((r.get("degradation", 0) for r in results))

    return {
        "test": "long_term_degradation",
        "initial_ari": float(initial_ari),
        "final_ari": float(final_ari),
        "max_degradation": float(max_degradation),
        "results": results,
    }


# ------------------------------------------------------------------
# 测试4: 干扰抵抗测试
# ------------------------------------------------------------------

def test_interference_resistance(
    items: List[Dict],
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    interference_ratios: List[float] = [0.0, 0.2, 0.5, 0.8, 0.95],
    seed: int = 42,
) -> Dict:
    """测试对无关干扰信息的抵抗能力。

    混入不同比例的干扰文档，观察目标检索质量。
    """
    print("\n  [Test 4] Interference Resistance...")

    results = []
    rng = np.random.RandomState(seed)

    # 定义"目标"主题（选一个明确的）
    target_topic = 0  # computing
    target_mask = true_labels == target_topic
    target_emb = embeddings[target_mask]
    target_labels = true_labels[target_mask]

    print(f"    Target topic size: {len(target_emb)}")

    for i,ir_ratio in enumerate(interference_ratios):
        # 构建混合语料库
        n_target = len(target_emb)
        n_interference = int(n_target * ir_ratio / (1 - ir_ratio))
        n_interference = min(n_interference, len(embeddings) - n_target)

        # 采样干扰项
        interference_mask = ~target_mask
        interference_indices = rng.choice(
            np.where(interference_mask)[0],
            size=n_interference,
            replace=False,
        )
        interference_emb = embeddings[interference_indices]

        # 混合
        mixed_emb = np.vstack([target_emb, interference_emb])
        mixed_labels = np.concatenate([
            np.full(len(target_emb), target_topic),
            true_labels[interference_indices],
        ])

        # K-Means 检索
        n_clusters = len(np.unique(mixed_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(mixed_emb)

        # 计算对目标主题的检索精度
        target_indices = np.where(mixed_labels == target_topic)[0]
        retrieved_labels = labels[target_indices]
        accuracy = np.mean(retrieved_labels == target_topic)

        results.append({
            "interference_ratio": ir_ratio,
            "target_accuracy": float(accuracy),
            "n_target": len(target_emb),
            "n_interference": n_interference,
        })

        print(f"    Interference={ir_ratio:.2f}: Target accuracy={accuracy:.4f}")

    return {
        "test": "interference_resistance",
        "results": results,
    }


# ------------------------------------------------------------------
# 测试5: TopoMem 系统集成测试
# ------------------------------------------------------------------

def test_topomem_system(
    items: List[Dict],
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    memory_sizes: List[int] = [100, 500, 1000],
    seed: int = 42,
) -> Dict:
    """测试完整 TopoMem 系统的压力表现。

    初始化系统，逐步添加记忆，执行检索。
    """
    print("\n  [Test 5] TopoMem System Integration...")

    results = []
    rng = np.random.RandomState(seed)

    # 初始化系统组件
    emb_config = EmbeddingConfig()
    topo_config = TopologyConfig(max_homology_dim=1)
    mem_config = MemoryConfig()

    embedding_manager = EmbeddingManager(emb_config)
    topology_engine = TopologyEngine(topo_config)
    memory = MemoryGraph(mem_config, embedding_manager)

    for size in memory_sizes:
        if size > len(items):
            continue

        # 采样
        indices = rng.choice(len(items), size=size, replace=False)

        # 清空并重建记忆
        memory = MemoryGraph(mem_config, embedding_manager)

        start = time.time()
        for idx in indices:
            item = items[idx]
            memory.add_memory(
                content=item["text"][:500],  # 截断
                embedding=embeddings[idx],
                metadata={"topic": str(item["topic"])},
                topo_engine=topology_engine,
            )
        add_time = time.time() - start

        # 执行检索测试
        query_idx = indices[0]
        query_emb = embeddings[query_idx]

        start = time.time()
        retrieved = memory.retrieve(query_emb, strategy="vector", k=5)
        retrieve_time = time.time() - start

        results.append({
            "memory_size": size,
            "add_time_sec": float(add_time),
            "retrieve_time_ms": float(retrieve_time * 1000),
            "node_count": memory.node_count(),
        })

        print(f"    Size={size}: Add={add_time:.2f}s, Retrieve={retrieve_time*1000:.2f}ms, Nodes={memory.node_count()}")

    # 清理
    embedding_manager.unload()

    return {
        "test": "topomem_system",
        "results": results,
    }


# ------------------------------------------------------------------
# 主实验
# ------------------------------------------------------------------

def run_cognitive_stress_test(
    categories: Optional[List[str]] = None,
    use_cache: bool = True,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """运行认知压力测试。"""

    print("=" * 70)
    print("EXPERIMENT 5: Cognitive Stress Test")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Step 1: 加载数据
    print("\n[1/5] Loading corpus...")
    start_total = time.time()

    items, embeddings = load_or_fetch_corpus(
        subset="train",
        categories=categories,
        use_cache=use_cache,
    )

    true_labels = get_topic_array(items)

    print(f"  Corpus: {len(items)} documents")
    print(f"  Topics: {len(np.unique(true_labels))}")

    # Step 2: 运行各项测试
    print("\n[2/5] Running stress tests...")

    results = {}

    # Test 1: Memory Capacity
    results["memory_capacity"] = test_memory_capacity(
        embeddings, true_labels,
        memory_sizes=[100, 500, 1000, 2000, 5000],
        seed=seed,
    )

    # Test 2: Concurrent Pressure
    results["concurrent_pressure"] = test_concurrent_pressure(
        items, embeddings,
        query_counts=[1, 10, 50, 100],
        seed=seed,
    )

    # Test 3: Long-term Degradation
    results["long_term_degradation"] = test_long_term_degradation(
        items, embeddings, true_labels,
        n_cycles=10,
        seed=seed,
    )

    # Test 4: Interference Resistance
    results["interference_resistance"] = test_interference_resistance(
        items, embeddings, true_labels,
        interference_ratios=[0.0, 0.2, 0.5, 0.8, 0.95],
        seed=seed,
    )

    # Test 5: TopoMem System
    results["topomem_system"] = test_topomem_system(
        items[:1000], embeddings[:1000], true_labels[:1000],  # 限制规模
        memory_sizes=[100, 500],
        seed=seed,
    )

    # Step 3: 保存结果
    elapsed_total = time.time() - start_total

    print(f"\n[3/3] Saving results... (total time: {elapsed_total:.1f}s)")

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    outpath = output_dir / f"exp5_cognitive_stress_{timestamp}.json"

    output_data = {
        "experiment": "Cognitive Stress Test",
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "config": {
            "n_documents": len(items),
            "n_topics": len(np.unique(true_labels)),
            "seed": seed,
        },
        "results": results,
        "total_elapsed_sec": elapsed_total,
    }

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {outpath}")

    # 打印摘要
    print_summary(results)

    return output_data


def print_summary(results: Dict) -> None:
    """打印结果摘要。"""
    print("\n" + "=" * 70)
    print("COGNITIVE STRESS SUMMARY")
    print("=" * 70)

    # Memory Capacity
    print("\n[1. Memory Capacity]")
    mc = results["memory_capacity"]["results"]
    for r in mc:
        print(f"  Size={r['memory_size']}: ARI={r['ari']:.4f}")

    # Concurrent Pressure
    print("\n[2. Concurrent Pressure]")
    cp = results["concurrent_pressure"]["results"]
    for r in cp:
        print(f"  Q={r['n_queries']}: {r['avg_time_ms']:.2f}ms/query, {r['throughput_qps']:.1f} QPS")

    # Long-term Degradation
    print("\n[3. Long-term Degradation]")
    ld = results["long_term_degradation"]
    print(f"  Initial ARI: {ld['initial_ari']:.4f}")
    print(f"  Final ARI: {ld['final_ari']:.4f}")
    print(f"  Max degradation: {ld['max_degradation']:.4f}")

    if ld['max_degradation'] > 0.1:
        print("  WARNING: Significant degradation detected!")
    else:
        print("  Status: No significant degradation")

    # Interference Resistance
    print("\n[4. Interference Resistance]")
    ir = results["interference_resistance"]["results"]
    baseline_acc = ir[0]["target_accuracy"]
    print(f"  Baseline (no interference): {baseline_acc:.4f}")
    for r in ir[1:]:
        deg = baseline_acc - r["target_accuracy"]
        print(f"  Interference={r['interference_ratio']:.2f}: accuracy={r['target_accuracy']:.4f} (deg: {deg:+.4f})")

    # TopoMem System
    print("\n[5. TopoMem System Integration]")
    ts = results["topomem_system"]["results"]
    for r in ts:
        print(f"  Size={r['memory_size']}: Add={r['add_time_sec']:.2f}s, Retrieve={r['retrieve_time_ms']:.2f}ms")

    # Overall Assessment
    print("\n[Overall Assessment]")
    issues = []

    if results["long_term_degradation"]["max_degradation"] > 0.1:
        issues.append("Long-term degradation")

    ir_results = results["interference_resistance"]["results"]
    if ir_results[-1]["target_accuracy"] < 0.5:
        issues.append("Poor interference resistance at high ratio")

    if issues:
        print(f"  Issues detected: {', '.join(issues)}")
    else:
        print("  System is robust under cognitive stress conditions")

    print("=" * 70)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 5: Cognitive Stress Test")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_cognitive_stress_test(
        use_cache=not args.no_cache,
        seed=args.seed,
    )
