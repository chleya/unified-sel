"""
Phase 0: Fast Permutation Test + Domain C Control
Uses direct embeddings + TDA (no TopoMemSystem overhead)
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


def load_corpus(domain, limit=None):
    path = _PKG_DIR / "data" / "test_corpus" / f"{domain}.json"
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return items[:limit] if limit else items


def h1_pers_mean(emb_mgr, topo_engine, texts):
    """Fast H1 mean persistence from texts only."""
    vecs = emb_mgr.encode_batch(texts)
    diagrams = topo_engine.compute_persistence(np.array(vecs, dtype=np.float64))
    if len(diagrams) > 1:
        pers = [float(r[1]-r[0]) for r in diagrams[1] if np.isfinite(r[1])]
        return float(np.mean(pers)) if pers else 0.0, len(pers)
    return 0.0, 0


def permutation_test_fast(prog_texts, phys_texts, emb_mgr, topo_engine, n_perm=5000):
    """
    Fast permutation test.
    Observed: phys_mean / prog_mean
    Null: ratio under random permutation of labels.
    """
    all_texts = prog_texts + phys_texts
    n_a = len(prog_texts)
    
    print(f"[Permutation Test] {n_perm} permutations (fast mode)...")
    
    # Observed ratio
    r_prog = h1_pers_mean(emb_mgr, topo_engine, prog_texts)
    r_phys = h1_pers_mean(emb_mgr, topo_engine, phys_texts)
    observed_ratio = r_phys[0] / r_prog[0] if r_prog[0] > 0 else 0
    print(f"  Observed: prog_mean={r_prog[0]:.4f} ({r_prog[1]} cycles), phys_mean={r_phys[0]:.4f} ({r_phys[1]} cycles)")
    print(f"  Observed B/A ratio: {observed_ratio:.3f}")
    
    # Pre-compute all embeddings
    print(f"  Pre-computing {len(all_texts)} embeddings...")
    all_vecs = emb_mgr.encode_batch(all_texts)
    
    ratios = []
    for i in range(n_perm):
        perm = np.random.permutation(len(all_texts))
        group1_vecs = np.array([all_vecs[j] for j in perm[:n_a]], dtype=np.float64)
        group2_vecs = np.array([all_vecs[j] for j in perm[n_a:]], dtype=np.float64)
        
        diag1 = topo_engine.compute_persistence(group1_vecs)
        diag2 = topo_engine.compute_persistence(group2_vecs)
        
        p1 = [float(r[1]-r[0]) for r in diag1[1] if np.isfinite(r[1])] if len(diag1) > 1 else []
        p2 = [float(r[1]-r[0]) for r in diag2[1] if np.isfinite(r[1])] if len(diag2) > 1 else []
        
        m1 = float(np.mean(p1)) if p1 else 0.0
        m2 = float(np.mean(p2)) if p2 else 0.0
        
        if m1 > 0 and m2 > 0:
            ratios.append(m2 / m1)
        
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n_perm} done...")
    
    ratios = np.array(ratios)
    
    p_one = float(np.mean(ratios >= observed_ratio))
    percentile = float(np.mean(ratios < observed_ratio)) * 100
    
    print(f"\n  Permutation distribution:")
    print(f"    Mean: {np.mean(ratios):.3f}, Std: {np.std(ratios):.3f}")
    print(f"    Min: {np.min(ratios):.3f}, Max: {np.max(ratios):.3f}")
    print(f"    25th: {np.percentile(ratios, 25):.3f}, 50th: {np.percentile(ratios, 50):.3f}, 75th: {np.percentile(ratios, 75):.3f}")
    print(f"  Observed ratio: {observed_ratio:.3f} (percentile: {percentile:.1f}%)")
    print(f"  One-tailed p: {p_one:.4f}")
    
    if p_one < 0.05:
        print(f"  => SIGNIFICANT (p={p_one:.4f} < 0.05)")
    elif p_one < 0.10:
        print(f"  => BORDERLINE (p={p_one:.4f} < 0.10)")
    else:
        print(f"  => NOT significant (p={p_one:.4f})")
    
    return {
        "observed_ratio": observed_ratio,
        "n_permutations": n_perm,
        "perm_mean": float(np.mean(ratios)),
        "perm_std": float(np.std(ratios)),
        "percentile": percentile,
        "p_one_tailed": p_one,
    }


def domain_c_fast(emb_mgr, topo_engine):
    """Domain C control using geography."""
    print(f"\n[Domain C Control] Using geography as domain C...")
    
    prog = load_corpus("programming", limit=20)
    phys = load_corpus("physics", limit=20)
    geo = load_corpus("geography", limit=10)
    
    prog_texts = [x["content"] for x in prog]
    phys_texts = [x["content"] for x in phys]
    geo_texts = [x["content"] for x in geo]
    
    # Individual domain measurements
    m_a, n_a = h1_pers_mean(emb_mgr, topo_engine, prog_texts)
    m_b, n_b = h1_pers_mean(emb_mgr, topo_engine, phys_texts)
    m_c, n_c = h1_pers_mean(emb_mgr, topo_engine, geo_texts)
    
    print(f"  A (prog {len(prog_texts)} items): {n_a} cycles, mean={m_a:.4f}")
    print(f"  B (phys {len(phys_texts)} items): {n_b} cycles, mean={m_b:.4f}")
    print(f"  C (geo {len(geo_texts)} items): {n_c} cycles, mean={m_c:.4f}")
    
    # Combined pairs
    ab_texts = prog_texts + phys_texts
    ac_texts = prog_texts + geo_texts
    bc_texts = phys_texts + geo_texts
    
    m_ab, n_ab = h1_pers_mean(emb_mgr, topo_engine, ab_texts)
    m_ac, n_ac = h1_pers_mean(emb_mgr, topo_engine, ac_texts)
    m_bc, n_bc = h1_pers_mean(emb_mgr, topo_engine, bc_texts)
    
    print(f"\n  Combined pairs:")
    print(f"    A+B ({len(ab_texts)} items): {n_ab} cycles, mean={m_ab:.4f}")
    print(f"    A+C ({len(ac_texts)} items): {n_ac} cycles, mean={m_ac:.4f}")
    print(f"    B+C ({len(bc_texts)} items): {n_bc} cycles, mean={m_bc:.4f}")
    
    # New cycles added by each domain
    # Use C's 10 items vs A's 20 items - normalize
    # C adds to A: (n_ac - n_a)
    # C adds to B: (n_bc - n_b)
    c_adds_a = n_ac - n_a
    c_adds_b = n_bc - n_b
    
    print(f"\n  New H1 cycles added:")
    print(f"    C adds to A: {c_adds_a} cycles")
    print(f"    C adds to B: {c_adds_b} cycles")
    print(f"    Difference: {abs(c_adds_a - c_adds_b)}")
    
    # Normalize by item count
    if n_a > 0:
        c_adds_a_norm = c_adds_a / len(geo_texts)
        c_adds_b_norm = c_adds_b / len(geo_texts)
        print(f"    C adds per item (to A): {c_adds_a_norm:.1f}")
        print(f"    C adds per item (to B): {c_adds_b_norm:.1f}")
    
    # B adds per item vs C adds per item
    b_adds_a = n_ab - n_a
    b_adds_a_norm = b_adds_a / len(phys_texts)
    print(f"\n  B adds per item (to A): {b_adds_a_norm:.1f}")
    
    # If C is truly "unrelated" to A and B, its per-item contribution should be similar
    # If it's more related to one, it would add fewer unique cycles
    
    # Interpretation
    if abs(c_adds_a - c_adds_b) <= 2:
        print(f"\n  => C adds SIMILAR amount to both A and B")
        print(f"     Suggests C's H1 structure is INDEPENDENT of A/B context")
    elif c_adds_a > c_adds_b:
        print(f"\n  => C adds MORE to A than to B (+{c_adds_a - c_adds_b})")
        print(f"     Suggests C is MORE similar to A in H1 structure")
    else:
        print(f"\n  => C adds MORE to B than to A (+{c_adds_b - c_adds_a})")
        print(f"     Suggests C is MORE similar to B in H1 structure")
    
    return {
        "A_mean": m_a, "A_cycles": n_a,
        "B_mean": m_b, "B_cycles": n_b,
        "C_mean": m_c, "C_cycles": n_c,
        "AB_cycles": n_ab, "AC_cycles": n_ac, "BC_cycles": n_bc,
        "C_adds_to_A": c_adds_a,
        "C_adds_to_B": c_adds_b,
        "B_adds_to_A": b_adds_a,
        "B_A_per_item": b_adds_a_norm,
        "C_A_per_item": c_adds_a_norm if n_a > 0 else 0,
        "C_B_per_item": c_adds_b_norm if n_b > 0 else 0,
    }


def main():
    print("=" * 60)
    print("PHASE 0: FAST Permutation Test + Domain C Control")
    print("=" * 60)
    
    emb_mgr = EmbeddingManager(EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    topo_engine = TopologyEngine(TopologyConfig(max_homology_dim=1, filtration_steps=30, metric="cosine"))
    
    prog = load_corpus("programming", limit=20)
    phys = load_corpus("physics", limit=20)
    
    prog_texts = [x["content"] for x in prog]
    phys_texts = [x["content"] for x in phys]
    
    print(f"\nCorpus: {len(prog_texts)} prog + {len(phys_texts)} phys")
    
    # Permutation test
    perm_result = permutation_test_fast(prog_texts, phys_texts, emb_mgr, topo_engine, n_perm=5000)
    
    # Domain C control
    domainc_result = domain_c_fast(emb_mgr, topo_engine)
    
    print(f"\n{'='*50}")
    print(f"PHASE 0 SUMMARY")
    print(f"{'='*50}")
    print(f"B/A ratio: {perm_result['observed_ratio']:.3f}")
    print(f"Permutation p(one-tailed): {perm_result['p_one_tailed']:.4f}")
    print(f"Percentile: {perm_result['percentile']:.1f}%")
    print(f"C adds to A: {domainc_result['C_adds_to_A']} cycles")
    print(f"C adds to B: {domainc_result['C_adds_to_B']} cycles")
    
    results = {
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
