"""
H1+H2 Health Monitoring Dashboard

Combined H1 (geometric quality) + H2 (domain bridging) monitoring.

Core metrics:
  H1: betti_1_count, fragmentation_index, h1_health_score
  H2: betti_2_count, h2_to_h1_ratio, cavitation_rate, h2_health_score

Signals:
  H1 health < 0.3: embedding space degraded (noise/fragmentation)
  H2/H1 > 0.3: domain mixing detected (cross-domain content)
  H2/H1 < 0.15: domain purity maintained
"""

import sys, os, time, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "data" / "models" / "hf_cache")
warnings.filterwarnings("ignore")

import numpy as np
from topomem.system import TopoMemSystem
from topomem.config import TopoMemConfig


def run_health_monitor():
    print("=== H1+H2 Combined Health Monitoring ===\n")

    corp_path = Path(__file__).parent.parent / "data" / "test_corpus"
    import json as j
    prog = j.loads((corp_path / "programming.json").read_text(encoding="utf-8"))[:20]
    phys = j.loads((corp_path / "physics.json").read_text(encoding="utf-8"))[:20]
    geo = j.loads((corp_path / "geography.json").read_text(encoding="utf-8"))[:10]

    print(f"Loaded: {len(prog)} programming, {len(phys)} physics, {len(geo)} geography\n")

    results = {}
    t0 = time.time()

    # Phase A: Baseline (programming domain only)
    print("Phase A: Baseline (20 programming items)")
    sys_a = TopoMemSystem(TopoMemConfig())
    for item in prog:
        sys_a.add_knowledge(item["content"][:512], metadata={"domain": "programming"})
    # Manually calibrate to compute TDA diagram
    sys_a.self_aware.calibrate(sys_a.memory, sys_a.topology, sys_a.engine)
    m_a = sys_a.get_metrics()
    print(f"  H1: betti_1={m_a['current_cluster_count']} clusters, h1_health={m_a['h1_health']:.3f}")
    print(f"  H2: betti_2={m_a['betti_2_count']}, h2/h1={m_a['h2_to_h1_ratio']:.3f}, cavitation={m_a['cavitation_rate']:.4f}")
    results["A_baseline"] = {k: m_a[k] for k in ["current_node_count", "current_cluster_count", "h1_health", "h2_to_h1_ratio", "betti_2_count", "cavitation_rate", "h2_health"]}

    # Phase B: Domain invasion (add physics)
    print("\nPhase B: Domain invasion (+20 physics items)")
    for item in phys:
        sys_a.add_knowledge(item["content"][:512], metadata={"domain": "physics"})
    sys_a.self_aware.calibrate(sys_a.memory, sys_a.topology, sys_a.engine)
    m_b = sys_a.get_metrics()
    print(f"  H1: betti_1={m_b['current_cluster_count']} clusters, h1_health={m_b['h1_health']:.3f}")
    print(f"  H2: betti_2={m_b['betti_2_count']}, h2/h1={m_b['h2_to_h1_ratio']:.3f}, cavitation={m_b['cavitation_rate']:.4f}")
    results["B_physics_invasion"] = {k: m_b[k] for k in ["current_node_count", "current_cluster_count", "h1_health", "h2_to_h1_ratio", "betti_2_count", "cavitation_rate", "h2_health"]}

    # Phase C: More domains (add geography)
    print("\nPhase C: Third domain (+10 geography items)")
    for item in geo:
        sys_a.add_knowledge(item["content"][:512], metadata={"domain": "geography"})
    sys_a.self_aware.calibrate(sys_a.memory, sys_a.topology, sys_a.engine)
    m_c = sys_a.get_metrics()
    print(f"  H1: betti_1={m_c['current_cluster_count']} clusters, h1_health={m_c['h1_health']:.3f}")
    print(f"  H2: betti_2={m_c['betti_2_count']}, h2/h1={m_c['h2_to_h1_ratio']:.3f}, cavitation={m_c['cavitation_rate']:.4f}")
    results["C_three_domains"] = {k: m_c[k] for k in ["current_node_count", "current_cluster_count", "h1_health", "h2_to_h1_ratio", "betti_2_count", "cavitation_rate", "h2_health"]}

    # Phase D: Consolidation
    print("\nPhase D: Consolidation pass")
    cons = sys_a.consolidation_pass()
    sys_a.self_aware.calibrate(sys_a.memory, sys_a.topology, sys_a.engine)
    m_d = sys_a.get_metrics()
    print(f"  H1: h1_health={m_d['h1_health']:.3f}")
    print(f"  H2: betti_2={m_d['betti_2_count']}, h2/h1={m_d['h2_to_h1_ratio']:.3f}")
    print(f"  Consolidation: orphans={cons.get('orphans_detected',0)}, merges={cons.get('merge_candidates_found',0)}")
    results["D_post_consolidation"] = {k: m_d[k] for k in ["current_node_count", "current_cluster_count", "h1_health", "h2_to_h1_ratio", "betti_2_count", "cavitation_rate", "h2_health"]}
    results["D_consolidation_meta"] = {"orphans": cons.get("orphans_detected", 0), "merges": cons.get("merge_candidates_found", 0)}

    # Verdict
    print("\n" + "=" * 50)
    print("=== H1+H2 HEALTH VERDICT ===\n")

    h1_signals = []
    h2_signals = []

    # H1 analysis
    h1_a = m_a["h1_health"]
    h1_b = m_b["h1_health"]
    h1_c = m_c["h1_health"]

    if h1_b < h1_a - 0.05:
        h1_signals.append(f"H1 WARNING: Health dropped {h1_a:.3f} -> {h1_b:.3f} (domain invasion)")
    elif h1_c < h1_a - 0.05:
        h1_signals.append(f"H1 WARNING: Health dropped after third domain")
    else:
        h1_signals.append("H1 STABLE: Geometric quality maintained across domains")

    # H2 analysis
    h2_b = m_b["h2_to_h1_ratio"]
    h2_c = m_c["h2_to_h1_ratio"]

    if h2_b > 0.3:
        h2_signals.append(f"H2 DOMAIN MIXING: h2/h1={h2_b:.3f} > 0.3 (strong mixing)")
    elif h2_b > 0.15:
        h2_signals.append(f"H2 MINOR MIXING: h2/h1={h2_b:.3f} (0.15-0.3)")
    elif h2_b == 0 and m_b["betti_2_count"] == 0:
        h2_signals.append(f"H2 SUPPRESSED: No 2D cavities detected (h2/h1=0, betti_2=0)")
    else:
        h2_signals.append(f"H2 DOMAIN PURITY: h2/h1={h2_b:.3f} < 0.15")

    # C domain analysis
    if h2_c > h2_b + 0.05:
        h2_signals.append(f"H2 INCREASED with third domain: {h2_b:.3f} -> {h2_c:.3f}")
    elif h2_c == h2_b:
        h2_signals.append("H2 STABLE across third domain")

    print("H1 Signals:")
    for s in h1_signals:
        print(f"  - {s}")

    print("\nH2 Signals:")
    for s in h2_signals:
        print(f"  - {s}")

    print("\n=== COMBINED INTERPRETATION ===")
    if any("MIXING" in s for s in h2_signals) and any("WARNING" in s for s in h1_signals):
        print("  -> MULTI-DOMAIN DETECTED: H1 degraded + H2 mixing = genuine domain invasion")
    elif any("MIXING" in s for s in h2_signals):
        print("  -> DOMAIN BOUNDARY DETECTED: H2 mixing without H1 degradation")
    elif any("STABLE" in s for s in h1_signals) and any("PURITY" in s for s in h2_signals):
        print("  -> DOMAIN PURITY MAINTAINED: H1 stable + H2 pure = single domain")
    else:
        print("  -> INCONCLUSIVE: Check individual signals above")

    # Metrics table
    print("\n=== METRICS TABLE ===")
    print(f"{'Phase':<25} {'Nodes':>6} {'Clusters':>9} {'H1 Health':>10} {'Betti2':>8} {'H2/H1':>8} {'Cavitation':>10}")
    print("-" * 82)
    for phase, d in results.items():
        if phase.startswith("D_consolidation"):
            continue
        print(f"{phase:<25} {d['current_node_count']:>6} {d['current_cluster_count']:>9} {d['h1_health']:>10.3f} {d['betti_2_count']:>8} {d['h2_to_h1_ratio']:>8.3f} {d['cavitation_rate']:>10.4f}")

    # Save
    out_path = Path(__file__).parent / "results" / f"h1h2_health_monitor_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")

    return results


if __name__ == "__main__":
    run_health_monitor()
