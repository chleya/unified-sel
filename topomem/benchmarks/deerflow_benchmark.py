"""
DeerFlow Retrieval + Interference Benchmark
==========================================
Real code corpus from deer-flow (289 Python files).

Design:
  Phase 1: Add deer-flow backend code (memory/channels/tools) → measure recall@5
  Phase 2: Add deer-flow frontend/skills docs → test backend recall (interference)
  Phase 3: Add MORE mixed code → test recall@5 degradation

Key metric: recall@5 for backend queries BEFORE vs AFTER adding interference domains.
If topological helps, backend recall should drop less than vector-only recall.
"""

import json, sys, os, tempfile, time
from pathlib import Path

_PKG = Path.cwd()  # F:/unified-sel/topomem
sys.path.insert(0, str(_PKG.parent))
os.environ["HF_HOME"] = str(_PKG / "data" / "models" / "hf_cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(_PKG / "data" / "models" / "hf_cache")

from topomem.memory import MemoryGraph, MemoryConfig
from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine


def chunk_python_file(filepath, max_chunks=8):
    """Extract meaningful chunks from a Python file (class/func definitions)."""
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return []
    
    lines = content.split('\n')
    chunks = []
    current = []
    in_class = in_func = False
    
    for line in lines:
        stripped = line.strip()
        # Start of class or top-level function
        if stripped.startswith('class ') or (stripped.startswith('def ') and not in_func and not in_class):
            if current:
                chunk_text = '\n'.join(current).strip()
                if chunk_text and len(chunk_text) > 100:
                    chunks.append(chunk_text)
                current = []
            if stripped.startswith('class '):
                in_class = True
                in_func = False
            else:
                in_func = True
        elif stripped.startswith('class '):
            in_class = True
            in_func = False
            if current:
                chunk_text = '\n'.join(current).strip()
                if chunk_text and len(chunk_text) > 100:
                    chunks.append(chunk_text)
                current = []
        elif stripped.startswith('def ') and (in_class or in_func):
            in_func = True
            if current:
                chunk_text = '\n'.join(current).strip()
                if chunk_text and len(chunk_text) > 100:
                    chunks.append(chunk_text)
                current = []
        current.append(line)
        if len(chunks) >= max_chunks:
            break
    
    if current:
        chunk_text = '\n'.join(current).strip()
        if chunk_text and len(chunk_text) > 100:
            chunks.append(chunk_text)
    
    return chunks


def load_deerflow_corpus(base_path='F:/workspace-ideas/deer-flow', max_per_dir=15):
    """Load code chunks from deer-flow, grouped by domain."""
    domains = {
        'backend_memory': ['backend/packages/harness/deerflow/agents/memory'],
        'backend_channels': ['backend/app/channels'],
        'backend_tools': ['backend/packages/harness/deerflow/tools/builtins'],
        'backend_config': ['backend/packages/harness/deerflow/config'],
        'frontend': ['frontend/src'],
        'backend_sandbox': ['backend/packages/harness/deerflow/sandbox'],
        'skills': ['skills/public'],
    }
    
    all_items = {}
    
    for domain, dirs in domains.items():
        items = []
        for d in dirs:
            full_dir = Path(base_path) / d
            if not full_dir.exists():
                continue
            py_files = list(full_dir.glob('**/*.py'))
            for pf in py_files[:max_per_dir]:
                chunks = chunk_python_file(str(pf), max_chunks=4)
                for i, chunk in enumerate(chunks):
                    # Extract a meaningful name from the chunk
                    name = chunk.split('\n')[0].strip()
                    if name.startswith('def '):
                        name = name.replace('def ', '').split('(')[0].strip()
                    elif name.startswith('class '):
                        name = name.replace('class ', '').split('(')[0].strip()
                    else:
                        name = pf.stem
                    
                    # Create a test question from the name
                    test_q = f"How does the {name} work?"
                    keywords = [name.split('_')[0] if '_' in name else name.split('.')[0]]
                    
                    items.append({
                        'id': f'{domain}_{pf.stem}_{i}',
                        'content': chunk[:500],  # Truncate for embedding size
                        'domain': domain,
                        'test_question': test_q,
                        'expected_keywords': keywords,
                        'filepath': str(pf.relative_to(base_path)),
                    })
        all_items[domain] = items
        print(f"  {domain}: {len(items)} items from {len([d for d in dirs if (Path(base_path)/d).exists()])} dirs")
    
    return all_items


def recall_at_k(results_with_scores, keywords, k=5):
    """Top-k results contain at least one keyword → 1, else 0."""
    nodes = [n for n, _ in results_with_scores[:k]]
    for n in nodes:
        content_lower = n.content.lower()
        for kw in keywords:
            if kw.lower() in content_lower:
                return 1
    return 0


def make_graph():
    tmp = tempfile.mkdtemp(prefix="topomem_deer_")
    cfg = MemoryConfig(chroma_persist_dir=tmp, max_nodes=1000)
    emb = EmbeddingManager()
    graph = MemoryGraph(config=cfg, embedding_mgr=emb)
    return graph, emb


def run_recall_test(graph, emb_mgr, test_items, label, k=5):
    """Test recall@5 for both strategies."""
    topo_ok = vec_ok = 0
    n = len(test_items)
    for item in test_items:
        q = item['test_question']
        kw = item['expected_keywords']
        emb = emb_mgr.encode(q)
        topo_res = graph.retrieve(emb, strategy='topological', k=k)
        vec_res = graph.retrieve(emb, strategy='vector', k=k)
        topo_ok += recall_at_k(topo_res, kw, k)
        vec_ok += recall_at_k(vec_res, kw, k)
    topo_acc = topo_ok / n
    vec_acc = vec_ok / n
    print(f"  {label:<35} TM-Topo={topo_acc:.2%} ({topo_ok}/{n})  TM-Vec={vec_acc:.2%} ({vec_ok}/{n})")
    return topo_acc, vec_acc


def run_benchmark():
    print("=" * 70)
    print("DeerFlow Retrieval + Interference Benchmark")
    print("=" * 70)
    
    print("\nLoading deer-flow corpus...")
    corpus = load_deerflow_corpus()
    
    # Use backend_memory + backend_channels as "core domain" (what we want to protect)
    # Use frontend + skills as "interference domains"
    core_items = corpus.get('backend_memory', []) + corpus.get('backend_channels', [])
    interference_items = corpus.get('frontend', []) + corpus.get('skills', []) + corpus.get('backend_sandbox', [])
    
    if len(core_items) < 5:
        print("ERROR: Not enough core items!")
        return
    
    print(f"\nCore domain (backend): {len(core_items)} items")
    print(f"Interference domains: {len(interference_items)} items")
    
    results = {}
    
    # Phase 1: Core domain only
    print("\n--- Phase 1: Core domain only ---")
    graph, emb_mgr = make_graph()
    topo = TopologyEngine()
    for item in core_items:
        graph.add_memory_from_text(item['content'])
    graph.update_topology(topo)
    p1_topo, p1_vec = run_recall_test(graph, emb_mgr, core_items[:20], "Baseline (core only)", k=5)
    results['phase1'] = {'topo': p1_topo, 'vec': p1_vec, 'n_items': len(core_items)}
    
    # Phase 2: + interference (small)
    print("\n--- Phase 2: Core + interference (small) ---")
    interference_small = interference_items[:len(interference_items)//3]
    for item in interference_small:
        graph.add_memory_from_text(item['content'])
    graph.update_topology(topo)
    p2_topo, p2_vec = run_recall_test(graph, emb_mgr, core_items[:20], "After interference (+1/3)", k=5)
    results['phase2'] = {
        'topo': p2_topo, 'vec': p2_vec,
        'topo_interference': p1_topo - p2_topo,
        'vec_interference': p1_vec - p2_vec,
    }
    
    # Phase 3: + interference (full)
    print("\n--- Phase 3: Core + all interference ---")
    interference_rest = interference_items[len(interference_small):]
    for item in interference_rest:
        graph.add_memory_from_text(item['content'])
    graph.update_topology(topo)
    p3_topo, p3_vec = run_recall_test(graph, emb_mgr, core_items[:20], "After interference (all)", k=5)
    results['phase3'] = {
        'topo': p3_topo, 'vec': p3_vec,
        'topo_interference': p1_topo - p3_topo,
        'vec_interference': p1_vec - p3_vec,
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Interference Protection (recall@5 on core domain)")
    print("=" * 70)
    print(f"  {'Phase':<40} {'TM-Topo':>8} {'TM-Vec':>8} {'Winner':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for label, topo, vec in [
        ("Core only (baseline)",    p1_topo, p1_vec),
        ("+ 1/3 interference",       p2_topo, p2_vec),
        ("+ all interference",       p3_topo, p3_vec),
    ]:
        w = "Topo" if topo > vec else "Vec" if vec > topo else "Tie"
        print(f"  {label:<40} {topo:>7.1%}  {vec:>7.1%}  {w:>8}")
    print()
    print(f"  Interference effect (TM-Topo): {p1_topo-p3_topo:+.1%}")
    print(f"  Interference effect (TM-Vec):  {p1_vec-p3_vec:+.1%}")
    print(f"\n  Total nodes: {graph.node_count()}")
    print(f"  Clusters: {len(graph.get_cluster_centers())}")
    
    # Save
    out = _PKG / "benchmarks" / "results" / f"deerflow_interference_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults: {out}")
    return results


if __name__ == "__main__":
    run_benchmark()
