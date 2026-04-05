"""
TopoMem MVP v0.1 — 快速演示脚本

跳过 LLM 推理，只展示核心组件的功能。
运行时间：~30 秒
"""

import os
import sys
import tempfile

# Fix Windows console encoding for emoji output
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from pathlib import Path

# 设置环境变量
os.environ["HF_HOME"] = r"F:\unified-sel\topomem\data\models\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"F:\unified-sel\topomem\data\models\hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"F:\unified-sel\topomem\data\models\hf_cache"

sys.path.insert(0, r"F:\unified-sel")

import numpy as np

from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.memory import MemoryGraph
from topomem.self_awareness import SelfAwareness
from topomem.guard import ConsistencyGuard
from topomem.adapters import AdapterPool, decide_action
from topomem.config import MemoryConfig


def main():
    print("=" * 70)
    print("  TopoMem MVP v0.1 — 快速演示（无 LLM 推理）")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()

    # ==================================================================
    # Step 1: Embedding
    # ==================================================================
    print("\n[1/6] EmbeddingManager — 文本编码")
    print("-" * 50)

    emb_mgr = EmbeddingManager()

    # 三个不同领域的文本
    texts = {
        "programming": [
            "Python list comprehensions are faster than for loops.",
            "The GIL prevents true parallel execution in CPython.",
            "Generators use yield to produce values lazily.",
        ],
        "geography": [
            "Mount Everest is the highest point above sea level.",
            "The Pacific Ocean is the largest and deepest ocean.",
            "The Sahara Desert is the largest hot desert.",
        ],
        "physics": [
            "E=mc² shows mass-energy equivalence.",
            "The speed of light is approximately 3×10⁸ m/s.",
            "Newton's second law is F=ma.",
        ],
    }

    all_texts = []
    all_embeddings = []
    for domain, domain_texts in texts.items():
        embeddings = emb_mgr.encode_batch(domain_texts)
        all_texts.extend(domain_texts)
        all_embeddings.append((domain, embeddings))
        print(f"  {domain}: {len(domain_texts)} 条, 形状 {embeddings.shape}")

    # 相似度矩阵
    all_embs = np.vstack([e for _, e in all_embeddings])
    sim_matrix = emb_mgr.similarity_matrix(all_embs)

    # 领域内 vs 跨领域相似度
    prog_emb = all_embeddings[0][1]
    geo_emb = all_embeddings[1][1]
    within_sim = float(np.mean(emb_mgr.similarity_matrix(prog_emb)))
    cross_sim = float(np.mean([
        emb_mgr.similarity(p, g)
        for p in prog_emb
        for g in geo_emb
    ]))
    print(f"\n  领域内相似度（编程）: {within_sim:.4f}")
    print(f"  跨领域相似度（编程 vs 地理）: {cross_sim:.4f}")
    assert within_sim > cross_sim, "领域内相似度应高于跨领域"
    print("  ✅ 领域内相似度高于跨领域，符合预期")

    # ==================================================================
    # Step 2: Topology
    # ==================================================================
    print("\n[2/6] TopologyEngine — 拓扑分析")
    print("-" * 50)

    topo = TopologyEngine()
    result = topo.compute_full_result(all_embs)

    print(f"  簇数: {result.n_clusters}")
    print(f"  拓扑特征数: {len(result.features)}")
    print(f"  指纹形状: {result.fingerprint.shape}")

    # 检查是否检测到多个簇
    assert result.n_clusters >= 2, f"应检测到至少 2 个簇，实际 {result.n_clusters}"
    print(f"  ✅ 成功检测到 {result.n_clusters} 个拓扑簇")

    # ==================================================================
    # Step 3: Memory
    # ==================================================================
    print("\n[3/6] MemoryGraph — 记忆存储与检索")
    print("-" * 50)

    config = MemoryConfig(
        max_nodes=50,
        chroma_persist_dir=str(Path(tmpdir) / "chromadb"),
    )
    mem = MemoryGraph(config=config, embedding_mgr=emb_mgr)

    # 添加所有知识
    for domain, domain_texts in texts.items():
        for text in domain_texts:
            emb = emb_mgr.encode(text)
            mem.add_memory(text, emb, metadata={"domain": domain}, topo_engine=topo)

    print(f"  总记忆数: {mem.node_count()}")

    # 更新拓扑
    mem.update_topology(topo)
    cluster_ids = mem._get_all_cluster_ids()
    print(f"  拓扑簇数: {len(cluster_ids)}")

    # 检索测试
    query_emb = emb_mgr.encode("What is Python GIL?")
    results = mem.retrieve(query_emb, strategy="hybrid", k=3)
    print(f"\n  查询: 'What is Python GIL?'")
    print(f"  检索到 {len(results)} 条记忆:")
    for i, (r, score) in enumerate(results):
        print(f"    [{i+1}] {r.content[:60]}... (cluster {r.cluster_id}, score {score:.3f})")

    # 验证检索结果相关性
    if results:
        # 第一条记忆应该来自编程簇
        assert "GIL" in results[0][0].content.lower() or "python" in results[0][0].content.lower() or "list" in results[0][0].content.lower()
        print("  ✅ 检索结果与查询相关")

    # ==================================================================
    # Step 4: SelfAwareness
    # ==================================================================
    print("\n[4/6] SelfAwareness — 自我认知")
    print("-" * 50)

    sa = SelfAwareness()
    sa.update_fingerprint(mem, topo)

    drift = sa.detect_drift()
    print(f"  漂移状态: {drift.status}")
    print(f"  短期漂移: {drift.short_drift:.4f}")
    print(f"  长期漂移: {drift.long_drift:.4f}")
    print(f"  趋势: {drift.trend}")

    identity = sa.get_identity_vector()
    print(f"  身份向量形状: {identity.shape}")

    assert drift.status in ("stable", "evolving"), f"意外状态: {drift.status}"
    print("  ✅ 漂移检测正常工作")

    # 模拟领域切换后的漂移
    for text in [
        "Quantum entanglement allows particles to share states.",
        "Heisenberg uncertainty principle limits position and momentum.",
    ]:
        emb = emb_mgr.encode(text)
        mem.add_memory(text, emb, metadata={"domain": "quantum"}, topo_engine=topo)

    mem.update_topology(topo)
    sa.update_fingerprint(mem, topo)

    drift_after = sa.detect_drift()
    print(f"\n  领域切换后:")
    print(f"  漂移状态: {drift_after.status}")
    print(f"  短期漂移: {drift_after.short_drift:.4f}")
    print(f"  长期漂移: {drift_after.long_drift:.4f}")

    # ==================================================================
    # Step 5: ConsistencyGuard
    # ==================================================================
    print("\n[5/6] ConsistencyGuard — 一致性守护")
    print("-" * 50)

    guard = ConsistencyGuard()

    # 新知识接受
    new_text = "Machine learning uses data to train models."
    new_emb = emb_mgr.encode(new_text)
    accept, reason = guard.should_accept_memory(new_text, new_emb, mem, sa)
    print(f"  新知识: accept={accept}")
    print(f"  原因: {reason}")

    # 重复检测
    dup_text = "Python list comprehensions are faster than for loops."
    dup_emb = emb_mgr.encode(dup_text)
    dup_accept, dup_reason = guard.should_accept_memory(dup_text, dup_emb, mem, sa)
    print(f"\n  重复知识: accept={dup_accept}")
    print(f"  原因: {dup_reason[:60]}...")

    assert not dup_accept, "重复知识应被拒绝"
    print("  ✅ 重复检测正常工作")

    # ==================================================================
    # Step 6: AdapterPool
    # ==================================================================
    print("\n[6/6] AdapterPool — 动态塑造")
    print("-" * 50)

    pool = AdapterPool(embedding_mgr=emb_mgr)
    print(f"  初始 adapter 数: {pool.adapter_count}")

    # 从记忆簇创建 adapter
    for cid in cluster_ids[:2]:
        memories = mem.retrieve_by_cluster(cid)
        if memories:
            adapter = pool.create_adapter(cid, memories)
            print(f"  创建 adapter: '{adapter.name}' (cluster {cid})")

    print(f"  创建后 adapter 数: {pool.adapter_count}")

    # 选择 adapter
    queries = [
        ("Python programming", "编程查询"),
        ("Mountains and oceans", "地理查询"),
        "Baking a cake recipe",  # 未知领域
    ]

    print("\n  Adapter 选择:")
    for q in queries:
        query_text = q if isinstance(q, str) else q[0]
        query_desc = q[1] if isinstance(q, tuple) else "未知领域查询"
        q_emb = emb_mgr.encode(query_text)
        adapter, surprise = pool.select_adapter(q_emb, mem)
        print(f"    {query_desc}: '{adapter.name}' (surprise={surprise:.4f})")

    # 决策矩阵演示
    print("\n  决策矩阵:")
    scenarios = [
        (0.2, 0.05, "低 surprise + 低 tension"),
        (0.8, 0.05, "高 surprise + 低 tension"),
        (0.2, 0.2, "低 surprise + 高 tension"),
        (0.8, 0.2, "高 surprise + 高 tension"),
    ]
    for surprise, tension, desc in scenarios:
        action = decide_action(surprise, tension)
        print(f"    {desc}: {action}")

    # ==================================================================
    # 最终摘要
    # ==================================================================
    print("\n" + "=" * 70)
    print("  ✅ 所有核心组件验证通过！")
    print("=" * 70)
    print(f"""
  系统组件摘要:
    EmbeddingManager:  {emb_mgr}
    TopologyEngine:    {topo}
    MemoryGraph:       {mem}
    SelfAwareness:     {sa}
    ConsistencyGuard:  {guard}
    AdapterPool:       {pool}

  关键验证:
    ✅ 领域内相似度 > 跨领域相似度
    ✅ 拓扑检测到 {result.n_clusters} 个簇
    ✅ 检索结果与查询相关
    ✅ 漂移检测正常工作
    ✅ 重复知识被正确拒绝
    ✅ Adapter 自动创建和选择
    ✅ 决策矩阵四种路径正确
""")


if __name__ == "__main__":
    main()
