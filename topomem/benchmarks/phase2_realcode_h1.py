"""
Phase 2: Real Code H1 Analysis - LIGHT VERSION
Test if TDA/H1 can distinguish code from different projects.
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
from topomem.config import EmbeddingConfig, TopologyConfig


def extract_py_content(file_path, max_lines=80):
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[:max_lines]
        content = []
        for line in lines:
            s = line.strip()
            if s and not s.startswith('#') and '"""' not in s and "'''" not in s:
                content.append(s)
        return '\n'.join(content[:20])
    except:
        return ""


def sample_files(root, subdir, n=15):
    dp = Path(root) / subdir
    if not dp.exists():
        return []
    py_files = [f for f in dp.glob("**/*.py")
                 if '__pycache__' not in str(f) and 'test' not in f.name.lower()]
    sampled = py_files[:min(n, len(py_files))]
    items = []
    for f in sampled:
        c = extract_py_content(f)
        if len(c) > 50:
            items.append({"content": c, "source": f"{subdir}/{f.name}"})
    return items


def h1_stats(vecs):
    diagrams = topo_engine.compute_persistence(np.array(vecs, dtype=np.float64))
    if len(diagrams) > 1:
        pers = [float(r[1]-r[0]) for r in diagrams[1] if np.isfinite(r[1])]
        return {"n_cycles": len(pers), "mean_pers": float(np.mean(pers)) if pers else 0.0,
                "sum_pers": float(np.sum(pers)) if pers else 0.0}
    return {"n_cycles": 0, "mean_pers": 0.0, "sum_pers": 0.0}


print("=" * 50)
print("PHASE 2: Real Code H1 (LIGHT)")
print("=" * 50)

emb_mgr = EmbeddingManager(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
topo_engine = TopologyEngine(TopologyConfig(max_homology_dim=1, filtration_steps=30, metric="cosine"))

workspace = Path("F:/workspace-ideas")

# 5 domains
domains_cfg = {
    "A1_deerflow_skills": (workspace/"deer-flow", "skills", 15),
    "A2_deerflow_backend": (workspace/"deer-flow", "backend", 15),
    "B1_hermes_agent":    (workspace/"hermes-agent", "agent", 15),
    "B2_hermes_tools":     (workspace/"hermes-agent", "tools", 15),
    "C1_cognee":          (workspace/"cognee", "cognee", 15),
}

print("\n[Sampling]")
corpus = {}
for name, (proj, sub, n) in domains_cfg.items():
    items = sample_files(proj, sub, n)
    corpus[name] = items
    print(f"  {name}: {len(items)} files")

# Individual H1
print("\n[Individual Domain H1]")
results = {}
for name, items in corpus.items():
    texts = [x["content"] for x in items]
    vecs = emb_mgr.encode_batch(texts)
    r = h1_stats(np.array(vecs, dtype=np.float64))
    r["n_files"] = len(items)
    results[name] = r
    print(f"  {name}: {r['n_cycles']} cycles, mean={r['mean_pers']:.4f}")

# Cross-domain pairs
print("\n[Cross-Domain Pairs]")
pairs = [
    ("A1+A2_deerflow", corpus["A1_deerflow_skills"] + corpus["A2_deerflow_backend"]),
    ("B1+B2_hermes", corpus["B1_hermes_agent"] + corpus["B2_hermes_tools"]),
    ("A1+B1_agenttype", corpus["A1_deerflow_skills"] + corpus["B1_hermes_agent"]),
    ("A1+C1_diverse", corpus["A1_deerflow_skills"] + corpus["C1_cognee"]),
]

for pair_name, items in pairs:
    texts = [x["content"] for x in items]
    vecs = emb_mgr.encode_batch(texts)
    r = h1_stats(np.array(vecs, dtype=np.float64))
    print(f"  {pair_name}: n={len(items)}, {r['n_cycles']} cycles, mean={r['mean_pers']:.4f}")
    results[pair_name] = r

# Redundancy analysis
print("\n[Redundancy: Within vs Cross]")
a1_n = results["A1_deerflow_skills"]["n_cycles"]
a2_n = results["A2_deerflow_backend"]["n_cycles"]
b1_n = results["B1_hermes_agent"]["n_cycles"]

within_deer_n = results["A1+A2_deerflow"]["n_cycles"]
within_hermes_n = results["B1+B2_hermes"]["n_cycles"]
cross_agent_n = results["A1+B1_agenttype"]["n_cycles"]
cross_diverse_n = results["A1+C1_diverse"]["n_cycles"]

deer_redundancy = (a1_n + a2_n) - within_deer_n
hermes_redundancy = (b1_n + results["B2_hermes_tools"]["n_cycles"]) - within_hermes_n
cross_redundancy = (a1_n + b1_n) - cross_agent_n

print(f"  Deer-flow within redundancy: {deer_redundancy}")
print(f"  Hermes within redundancy: {hermes_redundancy}")
print(f"  Cross-agent redundancy: {cross_redundancy}")
print(f"  Cross-diverse redundancy: {cross_diverse_n - (a1_n + results['C1_cognee']['n_cycles'])}")

# Key question: does cross have MORE or LESS redundancy than within?
if cross_redundancy > deer_redundancy:
    print(f"  => Cross-project has MORE shared H1 than deer-flow within!")
    print(f"     H1 does NOT distinguish project origin!")
elif deer_redundancy > 0:
    ratio = cross_redundancy / deer_redundancy
    print(f"  => Cross/Within ratio: {ratio:.2f} (cross < within = expected)")
else:
    print(f"  => Cannot determine (within redundancy = 0)")

# Save
out_path = _SCRIPT_DIR / "results" / f"phase2_{int(time.time())}.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {out_path}")

print("\n" + "=" * 50)
print("DONE - Phase 2 complete")
print("=" * 50)
