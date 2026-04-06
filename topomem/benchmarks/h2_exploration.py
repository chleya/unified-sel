"""
H2 Exploration: 2D Cavities as Structural Signatures

核心问题：H2 (2D holes) 编码什么？它和 H0/H1 有什么关系？

实验设计：
1. Basic H2 existence check: 在不同规模数据上测 H2
2. H2 vs H1 comparison: 哪个先出现/更强？
3. Domain invasion: H2 如何响应域入侵？
4. Cross-domain H2: 不同领域的 H2 模式

假设：
- H1 = 循环/依赖环
- H2 = 空腔/高阶连接模式（可能 = 域间桥梁结构）
- 如果 H2 在跨域入侵时增加 → H2 是"域间整合度"指标
"""

import sys, os, time, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "data" / "models" / "hf_cache")
warnings.filterwarnings("ignore")

import numpy as np
from topomem.topology import TopologyEngine, TopologyConfig
from topomem.embedding import EmbeddingManager, EmbeddingConfig
from topomem.memory import MemoryGraph, MemoryConfig


def get_h012_stats(embs: np.ndarray, topo: TopologyEngine) -> dict:
    """Compute H0, H1, H2 stats from a point cloud."""
    diagram = topo.compute_persistence(embs)
    
    h0 = diagram[0] if len(diagram) > 0 else np.array([]).reshape(0, 2)
    h1 = diagram[1] if len(diagram) > 1 else np.array([]).reshape(0, 2)
    h2 = diagram[2] if len(diagram) > 2 else np.array([]).reshape(0, 2)
    
    def stats(arr):
        if len(arr) == 0:
            return {"count": 0, "mean_pers": 0.0, "max_pers": 0.0, "total_pers": 0.0}
        pers = arr[:, 1] - arr[:, 0] if arr.ndim == 2 and arr.shape[1] == 2 else np.array([])
        pers = pers[np.isfinite(pers)]
        return {
            "count": len(arr),
            "mean_pers": float(np.mean(pers)) if len(pers) > 0 else 0.0,
            "max_pers": float(np.max(pers)) if len(pers) > 0 else 0.0,
            "total_pers": float(np.sum(pers)) if len(pers) > 0 else 0.0,
        }
    
    return {
        "n": len(embs),
        "H0": stats(h0),
        "H1": stats(h1),
        "H2": stats(h2),
    }


def run_basic_h2_check():
    """Check H2 existence at different scales."""
    print("=== 1. Basic H2 Existence Check ===\n")
    
    topo = TopologyEngine(TopologyConfig(max_homology_dim=2))
    emb_mgr = EmbeddingManager(EmbeddingConfig())
    
    # Generate synthetic clusters
    np.random.seed(42)
    n_range = [10, 15, 20, 30, 50, 80, 100]
    
    print(f"{'n':>4} {'H0':>6} {'H1':>6} {'H2':>6} | {'H1 mean':>10} {'H2 mean':>10}")
    print("-" * 60)
    
    results = {}
    for n in n_range:
        # 3 well-separated clusters
        c1 = np.random.randn(n//3, 384).astype(np.float32)
        c2 = np.random.randn(n//3, 384).astype(np.float32) + 5.0
        c3 = np.random.randn(n//3, 384).astype(np.float32) + 10.0
        embs = np.concatenate([c1, c2, c3])
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        
        stats = get_h012_stats(embs, topo)
        h0_c = stats["H0"]["count"]
        h1_c = stats["H1"]["count"]
        h2_c = stats["H2"]["count"]
        h1_m = stats["H1"]["mean_pers"]
        h2_m = stats["H2"]["mean_pers"]
        
        print(f"{n:>4} {h0_c:>6} {h1_c:>6} {h2_c:>6} | {h1_m:>10.4f} {h2_m:>10.4f}")
        results[n] = stats
    
    return results


def run_domain_invasion_h2():
    """Check H2 response to domain invasion."""
    print("\n\n=== 2. Domain Invasion: H2 Response ===\n")
    
    corp_path = Path(__file__).parent.parent / "data" / "test_corpus"
    import json as j
    prog_items = j.loads((corp_path / "programming.json").read_text(encoding="utf-8"))[:20]
    phys_items = j.loads((corp_path / "physics.json").read_text(encoding="utf-8"))[:20]
    
    topo = TopologyEngine(TopologyConfig(max_homology_dim=2))
    emb_mgr = EmbeddingManager(EmbeddingConfig())
    
    # Phase A: Programming only
    print("Phase A: Programming domain (20 items)")
    prog_embs = np.array([emb_mgr.encode(item["content"][:512]) for item in prog_items])
    prog_embs = prog_embs / np.linalg.norm(prog_embs, axis=1, keepdims=True)
    s_a = get_h012_stats(prog_embs, topo)
    print(f"  H0={s_a['H0']['count']}, H1={s_a['H1']['count']}, H2={s_a['H2']['count']}")
    print(f"  H1 mean_pers={s_a['H1']['mean_pers']:.4f}, H2 mean_pers={s_a['H2']['mean_pers']:.4f}")
    
    # Phase B: Add physics
    print("\nPhase B: + Physics domain (20 more items)")
    phys_embs = np.array([emb_mgr.encode(item["content"][:512]) for item in phys_items])
    phys_embs = phys_embs / np.linalg.norm(phys_embs, axis=1, keepdims=True)
    combined_embs = np.concatenate([prog_embs, phys_embs])
    combined_embs = combined_embs / np.linalg.norm(combined_embs, axis=1, keepdims=True)
    s_b = get_h012_stats(combined_embs, topo)
    print(f"  H0={s_b['H0']['count']}, H1={s_b['H1']['count']}, H2={s_b['H2']['count']}")
    print(f"  H1 mean_pers={s_b['H1']['mean_pers']:.4f}, H2 mean_pers={s_b['H2']['mean_pers']:.4f}")
    
    # Cross-domain similarity
    all_embs = np.concatenate([prog_embs, phys_embs])
    prog_centroid = prog_embs.mean(axis=0)
    prog_centroid = prog_centroid / np.linalg.norm(prog_centroid)
    phys_centroid = phys_embs.mean(axis=0)
    phys_centroid = phys_centroid / np.linalg.norm(phys_centroid)
    cross_sim = float(prog_centroid @ phys_centroid)
    print(f"\n  Cross-domain centroid sim: {cross_sim:.4f}")
    
    print("\n  Changes:")
    print(f"    H1: {s_a['H1']['count']} -> {s_b['H1']['count']} ({s_b['H1']['count']-s_a['H1']['count']:+d})")
    print(f"    H2: {s_a['H2']['count']} -> {s_b['H2']['count']} ({s_b['H2']['count']-s_a['H2']['count']:+d})")
    print(f"    H1 mean_pers: {s_a['H1']['mean_pers']:.4f} -> {s_b['H1']['mean_pers']:.4f}")
    print(f"    H2 mean_pers: {s_a['H2']['mean_pers']:.4f} -> {s_b['H2']['mean_pers']:.4f}")
    
    # H2 interpretation
    h2_change = s_b['H2']['count'] - s_a['H2']['count']
    if h2_change > 0:
        interp = "H2 INCREASED: new 2D cavities emerge at domain boundary (域间空腔)"
    elif h2_change < 0:
        interp = "H2 DECREASED: domain overlap fills in 2D cavities"
    else:
        if s_a['H2']['count'] > 0:
            interp = "H2 stable: domains don't create/remove 2D cavities"
        else:
            interp = "H2 absent in both: no 2D cavities in this configuration"
    print(f"\n  Interpretation: {interp}")
    
    return s_a, s_b


def run_real_codebase_h2():
    """Check H2 on real codebases (deer-flow vs cognee)."""
    print("\n\n=== 3. Real Codebase H2 Profiles ===\n")
    
    from pathlib import Path as P
    base = P(r"F:\workspace-ideas")
    
    def load_py_files(path: P, max_files: int = 30) -> np.ndarray:
        """Load embeddings from Python files."""
        emb_mgr = EmbeddingManager(EmbeddingConfig())
        files = list(path.rglob("*.py"))
        files = [f for f in files if "__pycache__" not in str(f)][:max_files]
        embs = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")[:512]
                if len(content) > 100:
                    v = emb_mgr.encode(content)
                    embs.append(v)
            except Exception:
                pass
        if not embs:
            return np.array([])
        embs = np.array(embs)
        return embs / np.linalg.norm(embs, axis=1, keepdims=True)
    
    topo = TopologyEngine(TopologyConfig(max_homology_dim=2))
    
    projects = {
        "deer-flow (skills)": P(base) / "deer-flow" / "skills" / "public",
        "deer-flow (backend)": P(base) / "deer-flow" / "backend" / "packages" / "harness",
        "hermes-agent": P(base) / "hermes-agent" / "skills",
        "cognee": P(base) / "cognee" / "cognee",
    }
    
    print(f"{'Project':<25} {'n':>4} {'H1':>6} {'H2':>6} {'H1/m':>8} {'H2/m':>8}")
    print("-" * 65)
    
    results = {}
    for name, path in projects.items():
        if not path.exists():
            print(f"{name:<25} NOT FOUND")
            continue
        embs = load_py_files(path, 30)
        if len(embs) < 5:
            print(f"{name:<25} too few files")
            continue
        s = get_h012_stats(embs, topo)
        h1_per_node = s['H1']['count'] / s['n']
        h2_per_node = s['H2']['count'] / s['n']
        print(f"{name:<25} {s['n']:>4} {s['H1']['count']:>6} {s['H2']['count']:>6} {h1_per_node:>8.2f} {h2_per_node:>8.2f}")
        results[name] = s
    
    return results


def run_mix_ratio_h2():
    """Check H2 as a function of domain mixing ratio."""
    print("\n\n=== 4. H2 vs Domain Mix Ratio ===\n")
    
    corp_path = Path(__file__).parent.parent / "data" / "test_corpus"
    import json as j
    prog_items = j.loads((corp_path / "programming.json").read_text(encoding="utf-8"))[:20]
    phys_items = j.loads((corp_path / "physics.json").read_text(encoding="utf-8"))[:20]
    
    topo = TopologyEngine(TopologyConfig(max_homology_dim=2))
    emb_mgr = EmbeddingManager(EmbeddingConfig())
    
    prog_embs = np.array([emb_mgr.encode(item["content"][:512]) for item in prog_items])
    prog_embs = prog_embs / np.linalg.norm(prog_embs, axis=1, keepdims=True)
    phys_embs = np.array([emb_mgr.encode(item["content"][:512]) for item in phys_items])
    phys_embs = phys_embs / np.linalg.norm(phys_embs, axis=1, keepdims=True)
    
    print(f"{'Mix ratio (B/A)':>18} {'n':>4} {'H0':>6} {'H1':>6} {'H2':>6} | {'H1/m':>8} {'H2/m':>8}")
    print("-" * 68)
    
    results = {}
    # Mix ratios: 0 (pure A), 0.25, 0.5, 1.0, 2.0
    ratios = [(0, 20), (5, 20), (10, 20), (20, 20), (20, 5)]
    
    for n_phys, n_prog in ratios:
        embs_a = prog_embs[:n_prog]
        embs_b = phys_embs[:n_phys] if n_phys > 0 else np.array([]).reshape(0, 384)
        if n_phys > 0:
            embs = np.concatenate([embs_a, embs_b])
        else:
            embs = embs_a
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        s = get_h012_stats(embs, topo)
        ratio = n_phys / n_prog if n_prog > 0 else 0.0
        h1pn = s['H1']['count'] / s['n']
        h2pn = s['H2']['count'] / s['n']
        print(f"{ratio:>18.2f} {s['n']:>4} {s['H0']['count']:>6} {s['H1']['count']:>6} {s['H2']['count']:>6} | {h1pn:>8.2f} {h2pn:>8.2f}")
        results[ratio] = s
    
    return results


def main():
    print("=== H2 Exploration: 2D Cavities as Structural Signatures ===\n")
    t0 = time.time()
    
    r1 = run_basic_h2_check()
    r2_a, r2_b = run_domain_invasion_h2()
    r3 = run_real_codebase_h2()
    r4 = run_mix_ratio_h2()
    
    total_time = time.time() - t0
    
    # Verdict
    print("\n" + "=" * 60)
    print("=== H2 VERDICT ===\n")
    
    # Check if H2 is detectable
    h2_detected = any(r["H2"]["count"] > 0 for r in r1.values())
    print(f"  H2 detectable: {h2_detected}")
    
    # H2 domain invasion response
    h2_a = r2_a["H2"]["count"]
    h2_b = r2_b["H2"]["count"]
    h2_change = h2_b - h2_a
    print(f"  H2 domain invasion response: {h2_a} -> {h2_b} ({h2_change:+d})")
    
    if h2_change > 0:
        print("  -> H2 INCREASES with domain mixing: H2 encodes inter-domain bridging")
    elif h2_detected:
        print("  -> H2 STABLE or DECREASES: H2 may encode intra-domain structure")
    else:
        print("  -> H2 NOT DETECTED in this configuration")
    
    # H1/H2 comparison
    h1_v_h2 = []
    for n, r in r1.items():
        if r["H1"]["count"] > 0 and r["H2"]["count"] > 0:
            h1_v_h2.append((n, r["H1"]["count"], r["H2"]["count"]))
    
    if h1_v_h2:
        print(f"\n  H1 vs H2 in synthetic clusters:")
        for n, h1, h2 in h1_v_h2:
            print(f"    n={n}: H1={h1}, H2={h2}, H1/H2 ratio={h1/h2:.1f}")
    
    print(f"\n  Total time: {total_time:.1f}s")
    print("\n  Next steps:")
    print("  1. If H2 detects domain boundaries: H2 = inter-domain scaffolding metric")
    print("  2. If H2 stable within domain: H2 = domain structural fingerprint")
    print("  3. If H2 absent: H2 needs larger point clouds or lower-dimensional embeddings")
    
    # Save
    out_path = Path(__file__).parent / "results" / f"h2_exploration_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    def to_serializable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, np.integer, int):
            return int(v)
        return v
    
    save_data = {
        "basic_check": {str(k): v for k, v in r1.items()},
        "domain_invasion": {"A": r2_a, "B": r2_b},
        "real_codebase": r3,
        "mix_ratio": {str(k): v for k, v in r4.items()},
        "h2_detected": h2_detected,
        "total_time_s": total_time,
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, default=to_serializable, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
