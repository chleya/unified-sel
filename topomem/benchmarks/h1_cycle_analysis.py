"""
H1 Cycle Deep Analysis
Separates A-only cycles from B/C invasion cycles
Compares persistence distributions statistically
Tests: Are new cycles from B/C domain statistically different from A's cycles?
"""

import os, sys, json, time
from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats

_SCRIPT_DIR = Path(__file__).parent
_PKG_DIR = _SCRIPT_DIR.parent
_ROOT_DIR = _PKG_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

from topomem.system import TopoMemSystem
from topomem.config import TopoMemConfig, MemoryConfig, EmbeddingConfig, TopologyConfig


def load_corpus(domain, limit=None):
    path = _PKG_DIR / "data" / "test_corpus" / f"{domain}.json"
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return items[:limit] if limit else items


def pers_from_diagram(diagram):
    return [float(r[1] - r[0]) for r in diagram if np.isfinite(r[1])]


def run():
    print("=" * 60)
    print("H1 CYCLE DEEP ANALYSIS")
    print("Separating domain-A cycles from invasion cycles")
    print("=" * 60)
    
    prog = load_corpus("programming", limit=20)
    phys = load_corpus("physics", limit=20)
    geo = load_corpus("geography", limit=10)
    
    tmpdir = "F:\\tmp\\h1_cycle_analysis"
    os.makedirs(tmpdir, exist_ok=True)
    
    config = TopoMemConfig(
        memory=MemoryConfig(chroma_persist_dir=tmpdir),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        topology=TopologyConfig(max_homology_dim=1, filtration_steps=30, metric="cosine"),
        engine=None,
    )
    system = TopoMemSystem(config=config)
    
    # Store A
    for item in prog:
        system.add_knowledge(item["content"], metadata={"domain": "A", "id": item["id"]})
    system.memory.update_topology(system.topology)
    
    # Get A-only H1
    a_vecs = system.embedding.encode_batch([x["content"] for x in prog])
    a_diag = system.topology.compute_persistence(np.array(a_vecs))
    a_pers = pers_from_diagram(a_diag[1] if len(a_diag) > 1 else np.array([]))
    
    print(f"\n[Domain A - Programming]")
    print(f"  H1 cycles: {len(a_pers)}")
    print(f"  Persistence: min={min(a_pers):.4f}, max={max(a_pers):.4f}")
    print(f"  Mean={np.mean(a_pers):.4f}, Std={np.std(a_pers):.4f}, Median={np.median(a_pers):.4f}")
    print(f"  Values: {[round(p,4) for p in sorted(a_pers)]}")
    
    # Store B
    for item in phys:
        system.add_knowledge(item["content"], metadata={"domain": "B", "id": item["id"]})
    system.memory.update_topology(system.topology)
    
    # Get A+B H1
    ab_vecs = system.embedding.encode_batch([x["content"] for x in prog + phys])
    ab_diag = system.topology.compute_persistence(np.array(ab_vecs))
    ab_pers = pers_from_diagram(ab_diag[1] if len(ab_diag) > 1 else np.array([]))
    
    # B-only cycles = A+B minus A cycles (best matching)
    # Strategy: match A's cycles by persistence, remaining are B's
    a_sorted = sorted(a_pers)
    ab_sorted = sorted(ab_pers)
    
    # A's 8 cycles are preserved (by stability ratio), remaining are B's
    b_pers = ab_sorted[len(a_sorted):]  # Remove A's cycles from A+B
    new_ab = ab_sorted[:len(a_sorted)]  # The "first 8" of A+B
    
    print(f"\n[Domain A+B Combined]")
    print(f"  Total H1 cycles: {len(ab_pers)}")
    print(f"  A-preserved (first {len(new_ab)}): mean={np.mean(new_ab):.4f}")
    print(f"  B-new cycles: {len(b_pers)}")
    if b_pers:
        print(f"  B persistence: min={min(b_pers):.4f}, max={max(b_pers):.4f}")
        print(f"  Mean={np.mean(b_pers):.4f}, Std={np.std(b_pers):.4f}, Median={np.median(b_pers):.4f}")
        print(f"  Values: {[round(p,4) for p in sorted(b_pers)]}")
    
    # Statistical test: A cycles vs B cycles
    print(f"\n[Statistical Tests]")
    if len(a_pers) > 0 and len(b_pers) > 0:
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = scipy_stats.mannwhitneyu(a_pers, b_pers, alternative='two-sided')
        print(f"  Mann-Whitney U: stat={u_stat:.1f}, p={u_pval:.4f}")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = scipy_stats.ks_2samp(a_pers, b_pers)
        print(f"  KS 2-sample: stat={ks_stat:.4f}, p={ks_pval:.4f}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(a_pers)-1)*np.var(a_pers) + (len(b_pers)-1)*np.var(b_pers)) / (len(a_pers)+len(b_pers)-2))
        cohens_d = (np.mean(b_pers) - np.mean(a_pers)) / (pooled_std + 1e-10)
        print(f"  Cohen's d: {cohens_d:.4f}")
        
        # Interpretation
        if u_pval < 0.05:
            print(f"  => SIGNIFICANT difference (p={u_pval:.4f} < 0.05)")
            print(f"     H1 cycles of B domain are DISTINCT from A's cycles")
        else:
            print(f"  => NO significant difference (p={u_pval:.4f} >= 0.05)")
            print(f"     A and B cycles are NOT statistically distinguishable")
        
        if abs(cohens_d) < 0.2:
            print(f"  => Negligible effect size (d={cohens_d:.3f})")
        elif abs(cohens_d) < 0.5:
            print(f"  => Small effect size (d={cohens_d:.3f})")
        elif abs(cohens_d) < 0.8:
            print(f"  => Medium effect size (d={cohens_d:.3f})")
        else:
            print(f"  => Large effect size (d={cohens_d:.3f})")
    else:
        print(f"  Not enough data: A={len(a_pers)}, B={len(b_pers)}")
    
    # Store C and do full analysis
    print(f"\n[Domain A+B+C]")
    for item in geo:
        system.add_knowledge(item["content"], metadata={"domain": "C", "id": item["id"]})
    system.memory.update_topology(system.topology)
    
    abc_vecs = system.embedding.encode_batch([x["content"] for x in prog + phys + geo])
    abc_diag = system.topology.compute_persistence(np.array(abc_vecs))
    abc_pers = pers_from_diagram(abc_diag[1] if len(abc_diag) > 1 else np.array([]))
    
    print(f"  Total H1 cycles: {len(abc_pers)}")
    
    # Summary table
    print(f"\n{'='*50}")
    print(f"CYCLE DISTRIBUTION SUMMARY")
    print(f"{'='*50}")
    print(f"Phase      | Cycles | Mean Pers | Std     | Min    | Max")
    print(f"-----------|--------|-----------|---------|--------|-------")
    print(f"A (prog)   |   {len(a_pers):2d}  |   {np.mean(a_pers):.4f}  | {np.std(a_pers):.4f} | {min(a_pers):.4f} | {max(a_pers):.4f}")
    print(f"A+B        |   {len(ab_pers):2d}  |   {np.mean(ab_pers):.4f} | {np.std(ab_pers):.4f} | {min(ab_pers):.4f} | {max(ab_pers):.4f}")
    print(f"A+B+C      |   {len(abc_pers):2d}  |   {np.mean(abc_pers):.4f} | {np.std(abc_pers):.4f} | {min(abc_pers):.4f} | {max(abc_pers):.4f}")
    
    if len(a_pers) > 0 and len(b_pers) > 0:
        # New vs old persistence comparison
        print(f"\nNew cycles added:")
        print(f"  B (phys) added {len(ab_pers) - len(a_pers)} cycles, mean pers = {np.mean(b_pers):.4f}")
        print(f"  C (geo)  added {len(abc_pers) - len(ab_pers)} cycles, mean pers = {np.mean(abc_pers[len(ab_pers):]):.4f}")
        
        # Are B's cycles more or less persistent than A's?
        ratio = np.mean(b_pers) / (np.mean(a_pers) + 1e-10)
        print(f"\nB/A persistence ratio: {ratio:.3f}")
        if ratio > 1.1:
            print(f"  => B cycles are MORE persistent than A (stronger topology)")
        elif ratio < 0.9:
            print(f"  => B cycles are LESS persistent than A (weaker topology)")
        else:
            print(f"  => B cycles are SIMILAR to A (comparable topology)")
    
    results = {
        "A_cycles": len(a_pers),
        "A_pers": a_pers,
        "A_mean_pers": float(np.mean(a_pers)),
        "A_std_pers": float(np.std(a_pers)),
        "AB_cycles": len(ab_pers),
        "AB_pers": ab_pers,
        "AB_mean_pers": float(np.mean(ab_pers)),
        "B_new_cycles": len(b_pers),
        "B_pers": b_pers,
        "B_mean_pers": float(np.mean(b_pers)) if b_pers else 0.0,
        "ABC_cycles": len(abc_pers),
        "ABC_pers": abc_pers,
        "ABC_mean_pers": float(np.mean(abc_pers)),
        "Mann_Whitney_U": float(u_stat) if len(a_pers) > 0 and len(b_pers) > 0 else None,
        "Mann_Whitney_p": float(u_pval) if len(a_pers) > 0 and len(b_pers) > 0 else None,
        "KS_stat": float(ks_stat) if len(a_pers) > 0 and len(b_pers) > 0 else None,
        "KS_p": float(ks_pval) if len(a_pers) > 0 and len(b_pers) > 0 else None,
        "Cohens_d": float(cohens_d) if len(a_pers) > 0 and len(b_pers) > 0 else None,
    }
    
    out_path = _SCRIPT_DIR / "results" / f"h1_cycle_analysis_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
