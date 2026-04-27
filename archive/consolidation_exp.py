"""
consolidation_pass 实验脚本
测试 orphan detection 和 merge candidates
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import tempfile
from pathlib import Path
from topomem.config import TopoMemConfig, MemoryConfig
from topomem.system import TopoMemSystem

print("=" * 60)
print("consolidation_pass 实验")
print("=" * 60)

# 使用临时目录避免残留数据
tmpdir = tempfile.mkdtemp(prefix="topomem_consolidation_")
print(f"\n临时目录: {tmpdir}")

# 创建配置（使用临时 ChromaDB）
config = TopoMemConfig(
    memory=MemoryConfig(chroma_persist_dir=str(Path(tmpdir) / "chromadb"))
)
system = TopoMemSystem(config=config)
print("系统初始化完成")

# 添加真实知识
knowledge_items = [
    # 编程 cluster
    ("Python is a programming language with elegant syntax", {"domain": "programming"}),
    ("Python list comprehensions are faster than for loops", {"domain": "programming"}),
    ("Python GIL prevents true parallel execution in CPython", {"domain": "programming"}),
    ("Machine learning uses data to train predictive models", {"domain": "ml"}),
    ("Neural networks are inspired by biological neurons", {"domain": "ml"}),
    # 地理 cluster
    ("Beijing is the capital of China", {"domain": "geography"}),
    ("Shanghai is the largest city in China", {"domain": "geography"}),
    # 物理 cluster
    ("Quantum mechanics describes matter at atomic scales", {"domain": "physics"}),
    ("Einstein's relativity transforms space and time", {"domain": "physics"}),
]

print(f"\n添加 {len(knowledge_items)} 条知识...")
accepted_count = 0
for text, metadata in knowledge_items:
    result = system.add_knowledge(text, metadata)
    status = "OK" if result else "REJECTED"
    print(f"  [{status}] {text[:45]}")
    if result:
        accepted_count += 1

print(f"\n接受率: {accepted_count}/{len(knowledge_items)}")

# 第一次 consolidation_pass 诊断
print("\n" + "=" * 60)
print("consolidation_pass 初始诊断")
print("=" * 60)
report1 = system.consolidation_pass()
print(f"\n  节点数（NetworkX图）: {report1['node_count']}")
print(f"  孤立节点数: {report1['orphan_count']}")
print(f"  拓扑簇数: {report1['cluster_count']}")
print(f"  可合并簇对: {report1['merge_count']}")

if report1['orphans']:
    print(f"\n  孤立节点: {report1['orphans']}")

# 检查 ChromaDB 实际数据
print("\n" + "=" * 60)
print("ChromaDB 数据检查")
print("=" * 60)
coll = system.memory._collection
all_data = coll.get(include=["metadatas"])
print(f"  ChromaDB 节点数: {len(all_data['ids'])}")

# 按 cluster_id 分组
from collections import Counter
cluster_ids = [m.get('cluster_id', -1) for m in all_data['metadatas']]
cluster_dist = Counter(cluster_ids)
print(f"  簇分布: {dict(cluster_dist)}")

# 检查 pending topo update
pending_count = sum(1 for m in all_data['metadatas'] if m.get('pending_topo_update', False))
print(f"  待更新拓扑节点: {pending_count}")

# 触发 topology update
print("\n" + "=" * 60)
print("触发 update_topology=True")
print("=" * 60)
report2 = system.consolidation_pass(update_topology=True)
print(f"\n  topology_updated: {report2['topology_updated']}")
print(f"  簇数: {report2['cluster_count']}")
print(f"  孤立节点: {report2['orphan_count']}")

# 再次诊断
print("\n" + "=" * 60)
print("最终状态")
print("=" * 60)
report3 = system.consolidation_pass()
coll3 = system.memory._collection
all3 = coll3.get(include=["metadatas"])
cluster_dist3 = Counter(m.get('cluster_id', -1) for m in all3['metadatas'])
print(f"  ChromaDB 节点数: {len(all3['ids'])}")
print(f"  簇分布: {dict(cluster_dist3)}")
print(f"  孤立节点: {report3['orphan_count']}")
print(f"  可合并簇对: {report3['merge_count']}")

print("\n实验完成！")
