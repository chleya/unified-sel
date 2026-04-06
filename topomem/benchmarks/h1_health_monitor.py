"""
H1 Health Monitoring Experiment

核心问题：H1 能否检测 embedding 空间的几何变化？

实验设计：
1. Phase A: 添加编程域 20 条 → 记录 H1 health baseline
2. Phase B: 添加高斯噪声扰动的同一批数据 → 测 H1 health 变化
3. Phase C: 添加完全不相关域 → 测 H1 health 变化
4. Phase D: 添加跨域混合内容 → 测 H1 的敏感性

指标：
- h1_health_score = mean_persistence / (1 + fragmentation_index)
- betti_1_count
- mean_h1_persistence
- fragmentation_index = betti_1 / n_nodes
"""

import sys, os, time, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "data" / "models" / "hf_cache")
warnings.filterwarnings("ignore")

import numpy as np
from topomem.system import TopoMemSystem
from topomem.config import TopoMemConfig


def get_h1_info(system):
    """从 system 提取 H1 健康指标."""
    metrics = system.get_metrics()
    status = system.get_status()
    return {
        "h1_health": metrics.get("h1_health", None),
        "h1_drift": metrics.get("h1_drift", None),
        "node_count": metrics.get("current_node_count", None),
        "cluster_count": metrics.get("current_cluster_count", None),
        "tda_cache_hits": metrics.get("tda_cache_hits", 0),
        "tda_cache_misses": metrics.get("tda_cache_misses", 0),
        "drift_status": getattr(status, "drift_status", None),
    }


def phase_report(phase: str, system: TopoMemSystem, elapsed: float):
    # Manually calibrate to get H1 metrics
    h1_metrics = None
    try:
        report = system.self_aware.calibrate(system.memory, system.topology, system.engine)
        h1_metrics = system.self_aware.get_h1_health()  # returns H1Metrics dataclass or float
        if hasattr(h1_metrics, 'betti_1_count'):
            print(f"  (Calibration: betti_1={h1_metrics.betti_1_count}, "
                  f"mean_pers={h1_metrics.mean_h1_persistence:.4f}, "
                  f"frag={h1_metrics.fragmentation_index:.3f}, "
                  f"health={h1_metrics.h1_health_score:.4f})")
        else:
            print(f"  (H1 health={h1_metrics:.4f})")
    except Exception as e:
        print(f"  (Calibration error: {e})")
    
    info = get_h1_info(system)
    print(f"\n  [{phase}] after {elapsed:.1f}s")
    print(f"    Nodes: {info['node_count']}, Clusters: {info['cluster_count']}")
    print(f"    H1 health: {info['h1_health']}, H1 drift: {info['h1_drift']}")
    print(f"    TDA cache: {info['tda_cache_hits']} hits / {info['tda_cache_misses']} misses")
    print(f"    Drift status: {info['drift_status']}")
    if h1_metrics and hasattr(h1_metrics, 'betti_1_count'):
        print(f"    H1 detail: betti={h1_metrics.betti_1_count}, "
              f"frag_idx={h1_metrics.fragmentation_index:.4f}, "
              f"health={h1_metrics.h1_health_score:.4f}, "
              f"stability={h1_metrics.stability_score:.4f}")
    # Use detailed H1 health score if available
    if h1_metrics and hasattr(h1_metrics, 'h1_health_score'):
        info['h1_health_detailed'] = h1_metrics.h1_health_score
        info['betti_1_count'] = h1_metrics.betti_1_count
        info['fragmentation_index'] = h1_metrics.fragmentation_index
    return info


def run():
    print("=== H1 Health Monitoring Experiment ===\n")

    # Load corpus
    corp_path = Path(__file__).parent.parent / "data" / "test_corpus"
    import json as j

    prog_items = j.loads((corp_path / "programming.json").read_text(encoding="utf-8"))
    phys_items = j.loads((corp_path / "physics.json").read_text(encoding="utf-8"))
    geo_items = j.loads((corp_path / "geography.json").read_text(encoding="utf-8"))

    # Take 20 from each
    prog = prog_items[:20]
    phys = phys_items[:20]

    print(f"Loaded: {len(prog)} programming, {len(phys)} physics, {len(geo_items)} geography items\n")

    results = {}
    t0_total = time.time()

    # === Phase A: Baseline - programming domain ===
    print("=== Phase A: Baseline (programming domain, 20 items) ===")
    system = TopoMemSystem(TopoMemConfig())
    t0 = time.time()
    for item in prog:
        system.add_knowledge(item["content"][:512], metadata={"domain": "programming"})
    elapsed = time.time() - t0
    info_a = phase_report("A", system, elapsed)
    results["A_baseline"] = info_a

    # === Phase B: Add noise-corrupted version of SAME items ===
    print("\n=== Phase B: Noise-corrupted programming items (+20 noisy copies) ===")
    t0 = time.time()
    # Add same content with slight noise (simulate embedding degradation)
    for item in prog[:10]:  # Add 10 noisy versions
        noisy = item["content"][:400] + " [modified variation]"  # Simple variation
        system.add_knowledge(noisy, metadata={"domain": "programming_noise"})
    elapsed = time.time() - t0
    info_b = phase_report("B", system, elapsed)
    results["B_noise"] = info_b

    # === Phase C: Add physics domain (geometrically distinct) ===
    print("\n=== Phase C: Physics domain invasion (+20 physics items) ===")
    t0 = time.time()
    for item in phys:
        system.add_knowledge(item["content"][:512], metadata={"domain": "physics"})
    elapsed = time.time() - t0
    info_c = phase_report("C", system, elapsed)
    results["C_physics"] = info_c

    # === Phase D: Consolidation pass ===
    print("\n=== Phase D: Consolidation pass ===")
    t0 = time.time()
    report = system.consolidation_pass()
    elapsed = time.time() - t0
    info_d = phase_report("D", system, elapsed)
    print(f"    Consolidation: orphans={report.get('orphans_detected', '?')}, "
          f"merges={report.get('merge_candidates_found', '?')}")
    results["D_post_consolidation"] = info_d

    # === Analysis ===
    print("\n" + "=" * 50)
    print("=== H1 HEALTH ANALYSIS ===")

    h1_a = info_a.get('h1_health_detailed', info_a['h1_health'])
    h1_b = info_b.get('h1_health_detailed', info_b['h1_health'])
    h1_c = info_c.get('h1_health_detailed', info_c['h1_health'])
    h1_d = info_d.get('h1_health_detailed', info_d['h1_health'])

    print(f"\n  H1 health scores:")
    print(f"    A (baseline, 20 prog):    {h1_a}")
    print(f"    B (+10 noisy prog):       {h1_b} (Δ={h1_b-h1_a:+.4f})")
    print(f"    C (+20 physics):          {h1_c} (Δ={h1_c-h1_a:+.4f})")
    print(f"    D (post-consolidation):   {h1_d} (Δ={h1_d-h1_a:+.4f})")

    print(f"\n  Betti-1 counts:")
    print(f"    A: {info_a.get('betti_1', 'N/A')}")
    print(f"    B: {info_b.get('betti_1', 'N/A')}")
    print(f"    C: {info_c.get('betti_1', 'N/A')}")
    print(f"    D: {info_d.get('betti_1', 'N/A')}")

    # Verdict
    print("\n  === VERDICT ===")
    has_signal = False

    # Check if noise phase changed H1 health
    if abs(h1_b - h1_a) > 0.01:
        print(f"  [YES] H1 detects NOISE perturbation: delta={h1_b-h1_a:+.4f}")
        has_signal = True
    else:
        print(f"  [NO]  H1 does NOT detect noise perturbation: delta={h1_b-h1_a:+.4f}")

    # Check if physics invasion changed H1 health
    if abs(h1_c - h1_a) > 0.01:
        print(f"  [YES] H1 detects DOMAIN INVASION: delta={h1_c-h1_a:+.4f}")
        has_signal = True
    else:
        print(f"  [NO]  H1 does NOT detect domain invasion: delta={h1_c-h1_a:+.4f}")

    # Check if consolidation changed H1 health
    if abs(h1_d - h1_c) > 0.01:
        print(f"  [~] H1 changes after consolidation: delta={h1_d-h1_c:+.4f}")
    else:
        print(f"  [=] H1 stable after consolidation: delta={h1_d-h1_c:+.4f}")

    print(f"\n  Node counts: A={info_a['node_count']}, B={info_b['node_count']}, "
          f"C={info_c['node_count']}, D={info_d['node_count']}")

    if not has_signal:
        print("\n  WARNING: H1 shows no significant signal in this experiment.")

    # Save
    out_path = Path(__file__).parent / "results" / f"h1_health_monitor_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        def to_dict(v):
            if hasattr(v, '__dataclass_fields__'):
                return {k: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for k, vv in v.__dict__.items()}
            elif isinstance(v, dict):
                return {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()}
            return float(v) if isinstance(v, (np.floating, float)) else v
        json.dump({
            "results": {k: {kk: to_dict(vv) for kk, vv in v.items()} for k, v in results.items()},
            "verdict": "H1 signal detected" if has_signal else "No H1 signal",
            "total_time_s": time.time() - t0_total,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
