"""
Phase 0: Immediate experiments (zero cost, uses existing data)
1. Permutation test - is B/A ratio=1.374 statistically significant?
2. Domain C control - geography as domain C
"""

import os, sys, json, time
from pathlib import Path
import numpy as np

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


def pers_from_diagram(diagrams):
    return [float(r[1] - r[0]) for r in diagrams if np.isfinite(r[1])]


def measure_h1(config_dict, items):
    """Measure H1 mean persistence for a set of items."""
    tmpdir = config_dict["tmpdir"]
    os.makedirs(tmpdir, exist_ok=True)
    
    config = TopoMemConfig(
        memory=MemoryConfig(chroma_persist_dir=tmpdir),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        topology=TopologyConfig(max_homology_dim=1, filtration_steps=30, metric="cosine"),
        engine=None,
    )
    system = TopoMemSystem(config=config)
    
    for item in items:
        system.add_knowledge(item["content"], metadata={"domain": item.get("domain", "?"), "id": item.get("id", "")})
    system.memory.update_topology(system.topology)
    
    vecs = system.embedding.encode_batch([x["content"] for x in items])
    diagrams = system.topology.compute_persistence(np.array(vecs))
    h1_pers = pers_from_diagram(diagrams[1] if len(diagrams) > 1 else np.array([]))
    
    return {
        "n": len(items),
        "h1_n": len(h1_pers),
        "h1_mean": float(np.mean(h1_pers)) if h1_pers else 0.0,
        "h1_median": float(np.median(h1_pers)) if h1_pers else 0.0,
        "h1_sum": float(np.sum(h1_pers)) if h1_pers else 0.0,
    }


def permutation_test(items_a, items_b, config_dict, n_permutations=10000):
    """
    Permutation test: 
    If B/A ratio=1.374 is real, it should be rare under permutation.
    """
    print(f"\n[Permutation Test] {n_permutations} permutations...")
    
    # Observed B/A ratio
    all_items = items_a + items_b
    
    # First compute all-item H1
    tmpdir = config_dict["tmpdir"] + "_perm"
    os.makedirs(tmpdir, exist_ok=True)
    
    # Method: for each permutation, randomly split all_items into two groups
    # and compute the mean_pers ratio
    ratios = []
    for i in range(n_permutations):
        perm = np.random.permutation(len(all_items))
        n_a = len(items_a)
        group1 = [all_items[j] for j in perm[:n_a]]
        group2 = [all_items[j] for j in perm[n_a:]]
        
        # Quick H1 for this permutation
        r1 = measure_h1({"tmpdir": tmpdir + f"_p{i}"}, group1)
        r2 = measure_h1({"tmpdir": tmpdir + f"_p{i}"}, group2)
        
        if r1["h1_mean"] > 0 and r2["h1_mean"] > 0:
            ratio = r2["h1_mean"] / r1["h1_mean"]
            ratios.append(ratio)
    
    ratios = np.array(ratios)
    
    # Observed ratio
    observed_ratio = 1.374  # B/A
    
    # Two-tailed p-value: what fraction of permutations have |ratio| >= observed?
    # For one-tailed (B > A): what fraction have ratio >= observed?
    p_two = float(np.mean(np.abs(ratios - 1.0) >= np.abs(observed_ratio - 1.0)))
    p_one = float(np.mean(ratios >= observed_ratio))
    
    # Percentile of observed ratio
    percentile = float(np.mean(ratios < observed_ratio)) * 100
    
    print(f"  Permutation distribution:")
    print(f"    Mean ratio: {np.mean(ratios):.3f}")
    print(f"    Std: {np.std(ratios):.3f}")
    print(f"    Min: {np.min(ratios):.3f}, Max: {np.max(ratios):.3f}")
    print(f"  Observed B/A ratio: {observed_ratio:.3f}")
    print(f"  Percentile: {percentile:.1f}%")
    print(f"  One-tailed p (B>A): {p_one:.4f}")
    print(f"  Two-tailed p: {p_two:.4f}")
    
    if p_one < 0.05:
        print(f"  => SIGNIFICANT at p={p_one:.4f} < 0.05")
    else:
        print(f"  => NOT significant at p={p_one:.4f}")
    
    return {
        "n_permutations": n_permutations,
        "observed_ratio": observed_ratio,
        "perm_mean": float(np.mean(ratios)),
        "perm_std": float(np.std(ratios)),
        "percentile": percentile,
        "p_one_tailed": p_one,
        "p_two_tailed": p_two,
    }


def domain_c_control(config_dict):
    """
    Domain C control: use geography as domain C.
    Compare B/C ratio vs B/A ratio vs A/C ratio.
    If domain specificity is real: B/C and A/C should be 
    distinguishable from B/A (which should be ~1.0 for unrelated domains).
    """
    print(f"\n[Domain C Control] Geography as domain C...")
    
    prog = load_corpus("programming", limit=20)
    phys = load_corpus("physics", limit=20)
    geo = load_corpus("geography", limit=10)
    
    # A vs C
    print(f"  Computing A (prog) vs C (geo)...")
    r_ac = measure_h1(config_dict, prog + geo)
    ratio_ac = r_ac["h1_mean"] / r_ac["h1_mean"]  # same dataset... no
    
    # Separate measurements
    r_a = measure_h1(config_dict, prog)
    r_b = measure_h1(config_dict, phys)
    r_c = measure_h1(config_dict, geo)
    
    print(f"  A (prog): {r_a['h1_n']} cycles, mean={r_a['h1_mean']:.4f}")
    print(f"  B (phys): {r_b['h1_n']} cycles, mean={r_b['h1_mean']:.4f}")
    print(f"  C (geo):  {r_c['h1_n']} cycles, mean={r_c['h1_mean']:.4f}")
    
    # B/C ratio: are B and C more different than B and A?
    # Since we can't directly compare cross-domain,
    # we use the items within each domain as their own baseline
    # and look at the ratio of ratios
    # B/A = 0.0277/0.0201 = 1.374
    # C/A = 0.0204/0.0201 (if C had 20 items, but C only has 10)
    # Better: compare effect sizes
    
    # Key insight: if domains are truly different, then
    # (B_mean - A_mean) should be non-zero
    # But we need a better metric: use the combined AB ratio vs AC
    
    # Combined: measure A+B, A+C, B+C
    print(f"\n  Measuring domain pairs...")
    r_ab = measure_h1(config_dict, prog + phys)
    r_ac2 = measure_h1(config_dict, prog + geo)
    r_bc = measure_h1(config_dict, phys + geo)
    
    print(f"  A+B: {r_ab['h1_n']} cycles, mean={r_ab['h1_mean']:.4f}")
    print(f"  A+C: {r_ac2['h1_n']} cycles, mean={r_ac2['h1_mean']:.4f}")
    print(f"  B+C: {r_bc['h1_n']} cycles, mean={r_bc['h1_mean']:.4f}")
    
    # Key metric: how many new cycles does each domain ADD?
    # A alone: 8 cycles
    # A+B: 31 cycles -> B adds 23
    # A+C: ? cycles -> C adds ?
    # B+C: ? cycles -> C adds ? and B adds ?
    
    cycles_A = r_a["h1_n"]
    cycles_AB = r_ab["h1_n"]
    cycles_AC = r_ac2["h1_n"]
    cycles_BC = r_bc["h1_n"]
    cycles_B = r_b["h1_n"]
    cycles_C = r_c["h1_n"]
    
    print(f"\n  Cycle count analysis:")
    print(f"    A adds: {cycles_A} cycles (baseline)")
    print(f"    B adds when A present: {cycles_AB - cycles_A} cycles")
    print(f"    C adds when A present: {cycles_AC - cycles_A} cycles")
    print(f"    C adds when B present: {cycles_BC - cycles_B} cycles")
    
    b_adds = cycles_AB - cycles_A
    c_adds_ac = cycles_AC - cycles_A
    c_adds_bc = cycles_BC - cycles_B
    
    # If C is unrelated to both A and B, it should add similar number of cycles regardless of what's there
    print(f"\n  C adds to A: {c_adds_ac}")
    print(f"  C adds to B: {c_adds_bc}")
    print(f"  Difference: {abs(c_adds_ac - c_adds_bc)}")
    
    # If domains are equivalent, C should add similar amount to both
    if cycles_A > 0:
        c_add_ratio = c_adds_ac / cycles_A if cycles_A > 0 else 0
        print(f"  C/A add ratio: {c_adds_ac}/{cycles_A} = {c_add_ratio:.2f}")
    
    return {
        "A_h1_mean": r_a["h1_mean"],
        "B_h1_mean": r_b["h1_mean"],
        "C_h1_mean": r_c["h1_mean"],
        "A_cycles": cycles_A,
        "B_cycles": cycles_B,
        "C_cycles": cycles_C,
        "AB_cycles": cycles_AB,
        "AC_cycles": cycles_AC,
        "BC_cycles": cycles_BC,
        "B_adds_to_A": b_adds,
        "C_adds_to_A": c_adds_ac,
        "C_adds_to_B": c_adds_bc,
        "B_A_ratio": float(r_b["h1_mean"] / r_a["h1_mean"]) if r_a["h1_mean"] > 0 else 0,
    }


def main():
    print("=" * 60)
    print("PHASE 0: Permutation Test + Domain C Control")
    print("=" * 60)
    
    tmpdir = "F:\\tmp\\phase0"
    os.makedirs(tmpdir, exist_ok=True)
    config_dict = {"tmpdir": tmpdir}
    
    prog = load_corpus("programming", limit=20)
    phys = load_corpus("physics", limit=20)
    
    print(f"\nCorpus: {len(prog)} prog + {len(phys)} phys")
    
    # First: raw measurements
    print(f"\n[Raw H1 Measurements]")
    r_a = measure_h1(config_dict, prog)
    r_b = measure_h1(config_dict, phys)
    print(f"  A (prog): {r_a['h1_n']} cycles, mean={r_a['h1_mean']:.4f}")
    print(f"  B (phys): {r_b['h1_n']} cycles, mean={r_b['h1_mean']:.4f}")
    print(f"  B/A ratio: {r_b['h1_mean']/r_a['h1_mean']:.3f}")
    
    # Phase 0a: Permutation test
    perm_result = permutation_test(prog, phys, config_dict, n_permutations=5000)
    
    # Phase 0b: Domain C control
    domainc_result = domain_c_control(config_dict)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"PHASE 0 SUMMARY")
    print(f"{'='*50}")
    print(f"B/A mean persistence ratio: {r_b['h1_mean']/r_a['h1_mean']:.3f}")
    print(f"Permutation test: p(one-tailed)={perm_result['p_one_tailed']:.4f}")
    print(f"Percentile of observed ratio: {perm_result['percentile']:.1f}%")
    
    results = {
        "raw_A": r_a,
        "raw_B": r_b,
        "permutation": perm_result,
        "domain_C": domainc_result,
    }
    
    out_path = _SCRIPT_DIR / "results" / f"phase0_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
