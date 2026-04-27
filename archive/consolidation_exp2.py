"""
consolidation_pass merge 检测实验
添加语义相近的知识，看是否能检测到可合并的簇
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import tempfile
from pathlib import Path
from collections import Counter
from topomem.config import TopoMemConfig, MemoryConfig
from topomem.system import TopoMemSystem

print("=" * 60)
print("consolidation_pass — Merge 检测实验")
print("=" * 60)

tmpdir = tempfile.mkdtemp(prefix="topomem_merge_exp_")
print(f"\n临时目录: {tmpdir}")

config = TopoMemConfig(
    memory=MemoryConfig(chroma_persist_dir=str(Path(tmpdir) / "chromadb"))
)
system = TopoMemSystem(config=config)

# 添加语义相近的知识（同一领域）
python_knowledge = [
    "Python is a programming language with clean syntax",
    "Python supports list comprehensions for concise code",
    "Python has a Global Interpreter Lock that limits parallelism",
    "Python decorators allow modifying function behavior",
    "Python generators use yield to produce sequences lazily",
    "Python virtual environments help manage dependencies",
]

ml_knowledge = [
    "Machine learning trains models on data to make predictions",
    "Neural networks are inspired by biological neuron structures",
    "Deep learning uses multiple layers to learn representations",
    "Gradient descent optimizes model parameters iteratively",
]

geo_knowledge = [
    "Beijing is the capital city of China in East Asia",
    "Shanghai is the largest metropolitan area in China",
    "The Great Wall stretches across northern China for thousands of kilometers",
]

all_knowledge = [
    ("Python programming", python_knowledge),
    ("Machine learning", ml_knowledge),
    ("Chinese geography", geo_knowledge),
]

added_count = 0
for domain, items in all_knowledge:
    for text in items:
        if system.add_knowledge(text, {"domain": domain}):
            added_count += 1

print(f"\n添加 {added_count} 条知识")

# 触发 topology update
print("\n触发 topology update...")
report = system.consolidation_pass(update_topology=True)
print(f"  n_clusters: {report['cluster_count']}")

# 检查 ChromaDB
coll = system.memory._collection
all_data = coll.get(include=["metadatas"])
cluster_dist = Counter(m.get("cluster_id", -1) for m in all_data["metadatas"])
print(f"\n  簇分布: {dict(cluster_dist)}")
print(f"  ChromaDB 节点数: {len(all_data['ids'])}")

# 运行诊断
print("\n" + "=" * 60)
print("consolidation_pass 诊断")
print("=" * 60)
report = system.consolidation_pass()

print(f"\n  节点数: {report['node_count']}")
print(f"  孤立节点: {report['orphan_count']}")
print(f"  簇数: {report['cluster_count']}")
print(f"  可合并簇对: {report['merge_count']}")

if report["merge_candidates"]:
    print(f"\n  发现 {len(report['merge_candidates'])} 对可合并簇:")
    for pair in report["merge_candidates"]:
        print(f"    {pair}")

# 检查具体哪些节点在哪些簇
print("\n  各簇节点内容:")
coll2 = system.memory._collection
all_data2 = coll2.get(include=["metadatas", "documents"])
by_cluster = {}
for i, meta in enumerate(all_data2["metadatas"]):
    cid = meta.get("cluster_id", -1)
    by_cluster.setdefault(cid, []).append(all_data2["documents"][i][:40])

for cid in sorted(by_cluster.keys()):
    print(f"    簇 {cid}: {by_cluster[cid]}")

print("\n实验完成！")
