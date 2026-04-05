import sys
import tempfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from topomem.config import TopoMemConfig, MemoryConfig, EmbeddingConfig
from topomem.memory import MemoryGraph
from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine

# 用干净临时目录
tmpdir = tempfile.mkdtemp(prefix="quick_demo_check_")
print(f"Temp dir: {tmpdir}")

config = MemoryConfig(
    max_nodes=50,
    chroma_persist_dir=str(Path(tmpdir) / "chromadb"),
)
emb_mgr = EmbeddingManager(EmbeddingConfig())
mem = MemoryGraph(config=config, embedding_mgr=emb_mgr)

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

topo = TopologyEngine()

# 添加知识
for domain, domain_texts in texts.items():
    for text in domain_texts:
        emb = emb_mgr.encode(text)
        mem.add_memory(text, emb, metadata={"domain": domain}, topo_engine=topo)

print(f"\n添加后 node_count(): {mem.node_count()}")
coll = mem._collection
all_data = coll.get(include=["metadatas", "documents"])
print(f"ChromaDB 节点数: {len(all_data['ids'])}")

# 打印所有节点
from collections import Counter
domains = Counter(m.get("domain") for m in all_data["metadatas"])
print(f"Domain分布: {dict(domains)}")
print(f"\n所有节点内容:")
for i, (doc, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
    print(f"  [{i}] domain={meta.get('domain')} cluster={meta.get('cluster_id')} persist={meta.get('persistence_score')} content={doc[:50]}")

print(f"\nNetworkX node_count: {mem.node_count()}")
print(f"pending_topo_update: {mem._inserts_since_topo_update}")

# 触发 topology update
print("\n触发 topology update...")
result = mem.update_topology(topo)
print(f"n_clusters: {result.n_clusters}")

# 再次检查
all_data2 = coll.get(include=["metadatas"])
domains2 = Counter(m.get("cluster_id", -1) for m in all_data2["metadatas"])
print(f"\n更新后 cluster分布: {dict(domains2)}")
print(f"更新后 ChromaDB 节点数: {len(all_data2['ids'])}")
