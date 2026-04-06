"""
H2 Statistical Validation Benchmark (P0)

Runs N trials to establish whether H2/H1 ratio is a statistically
significant detector of domain mixing.

Core questions:
1. Is H2/H1 statistically significant? (p-value, Cohen's d)
2. Is H2/H1 linearly correlated with domain invasion level?
3. Can H2 detect what H1 can't?
4. H2 false positive rate on pure domain?

Design:
- Pure A: 20 embeddings from programming domain
- Mixed A+B: 10 prog + 10 physics embeddings
- Trial=20 per condition (start small, scale up if fast)
- Compute: betti_1, betti_2, H2/H1 ratio per trial
- Statistical tests: t-test, Mann-Whitney U, Cohen's d
"""

import sys, os, time, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "data" / "models" / "hf_cache")
warnings.filterwarnings("ignore")

import numpy as np
from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.config import EmbeddingConfig, TopologyConfig


def compute_h12(embeddings, metric="cosine", max_homology_dim=2):
    """Compute H1 and H2 betti numbers. Returns (betti_1, betti_2, h2_suppressed)."""
    topo_config = TopologyConfig(metric=metric, max_homology_dim=max_homology_dim)
    topo = TopologyEngine(topo_config)
    # Compute full persistence
    result = topo.compute_persistence(embeddings)
    # result is list by homology dimension: result[0]=H0, result[1]=H1, result[2]=H2
    betti_1 = len(result[1]) if len(result) > 1 else 0
    betti_2 = len(result[2]) if len(result) > 2 else 0
    return betti_1, betti_2


def run_trial(emb_mgr, embeddings_a, embeddings_b, condition):
    """Run one trial. condition: 'pure' or 'mixed'."""
    if condition == "pure":
        idx_a = np.random.permutation(len(embeddings_a))[:20]
        vecs = embeddings_a[idx_a]
    else:  # mixed
        n_a = min(10, len(embeddings_a))
        n_b = min(10, len(embeddings_b))
        idx_a = np.random.permutation(len(embeddings_a))[:n_a]
        idx_b = np.random.permutation(len(embeddings_b))[:n_b]
        combined = np.concatenate([embeddings_a[idx_a], embeddings_b[idx_b]])
        # Shuffle
        perm = np.random.permutation(len(combined))
        vecs = combined[perm]

    b1, b2 = compute_h12(vecs, max_homology_dim=2)
    ratio = b2 / b1 if b1 > 0 else 0.0
    return {"betti_1": b1, "betti_2": b2, "ratio": ratio}


def main():
    print("=== H2 Statistical Validation (P0) ===\n")

    # Load embeddings
    corp_path = Path(__file__).parent.parent / "data" / "test_corpus"
    import json as j
    prog = j.loads((corp_path / "programming.json").read_text(encoding="utf-8"))
    phys = j.loads((corp_path / "physics.json").read_text(encoding="utf-8"))
    geo = j.loads((corp_path / "geography.json").read_text(encoding="utf-8"))

    # Encode all items
    print("Encoding embeddings...")
    t0 = time.time()
    emb_cfg = EmbeddingConfig()
    emb_mgr = EmbeddingManager(emb_cfg)
    emb_prog = emb_mgr.encode_batch([item["content"][:512] for item in prog])
    emb_phys = emb_mgr.encode_batch([item["content"][:512] for item in phys])
    emb_geo = emb_mgr.encode_batch([item["content"][:512] for item in geo])
    print(f"Encoded in {time.time()-t0:.1f}s: prog={len(emb_prog)}, phys={len(emb_phys)}, geo={len(emb_geo)}\n")

    N = 20  # trials per condition
    results_pure = []
    results_mixed = []

    # Mixed A+B
    print(f"Running {N} trials for Mixed (A+B)...")
    t0 = time.time()
    for i in range(N):
        r = run_trial(emb_mgr, emb_prog, emb_phys, "mixed")
        results_mixed.append(r)
        if (i+1) % 5 == 0:
            print(f"  Trial {i+1}/{N}: b1={r['betti_1']}, b2={r['betti_2']}, ratio={r['ratio']:.3f}")
    t_mixed = time.time() - t0
    print(f"  Done in {t_mixed:.1f}s\n")

    # Pure A
    print(f"Running {N} trials for Pure A (programming)...")
    t0 = time.time()
    for i in range(N):
        r = run_trial(emb_mgr, emb_prog, emb_phys, "pure")
        results_pure.append(r)
        if (i+1) % 5 == 0:
            print(f"  Trial {i+1}/{N}: b1={r['betti_1']}, b2={r['betti_2']}, ratio={r['ratio']:.3f}")
    t_pure = time.time() - t0
    print(f"  Done in {t_pure:.1f}s\n")

    # Statistics
    pure_ratios = [r["ratio"] for r in results_pure]
    mixed_ratios = [r["ratio"] for r in results_mixed]
    pure_b2 = [r["betti_2"] for r in results_pure]
    mixed_b2 = [r["betti_2"] for r in results_mixed]

    # Q1: Is H2/H1 statistically significant?
    # Mann-Whitney U test (non-parametric)
    from scipy import stats
    u_stat, p_value_mw = stats.mannwhitneyu(mixed_ratios, pure_ratios, alternative="greater")
    # t-test
    t_stat, p_value_t = stats.ttest_ind(mixed_ratios, pure_ratios, alternative="greater")

    # Cohen's d
    pooled_std = np.sqrt(((N-1)*np.var(pure_ratios, ddof=1) + (N-1)*np.var(mixed_ratios, ddof=1)) / (2*N-2))
    cohens_d = (np.mean(mixed_ratios) - np.mean(pure_ratios)) / pooled_std if pooled_std > 0 else 0.0

    # Q4: False positive rate (H2 signal in pure domain)
    fp_rate = sum(1 for r in results_pure if r["betti_2"] > 0) / N
    fn_rate = sum(1 for r in results_mixed if r["betti_2"] == 0) / N

    print("=" * 60)
    print("=== STATISTICAL RESULTS ===\n")

    print(f"{'Condition':<20} {'Mean Ratio':>12} {'Std':>8} {'Mean B2':>10} {'B2>0':>8}")
    print("-" * 60)
    print(f"{'Pure A':<20} {np.mean(pure_ratios):>12.3f} {np.std(pure_ratios):>8.3f} {np.mean(pure_b2):>10.2f} {sum(1 for r in results_pure if r['betti_2']>0)/N:>8.1%}")
    print(f"{'Mixed A+B':<20} {np.mean(mixed_ratios):>12.3f} {np.std(mixed_ratios):>8.3f} {np.mean(mixed_b2):>10.2f} {sum(1 for r in results_mixed if r['betti_2']>0)/N:>8.1%}")

    print(f"\n--- Statistical Tests ---")
    print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_value_mw:.4f}")
    print(f"t-test: t={t_stat:.3f}, p={p_value_t:.4f}")
    print(f"Cohen's d: {cohens_d:.3f} ", end="")
    if abs(cohens_d) >= 0.8:
        print("(LARGE effect)")
    elif abs(cohens_d) >= 0.5:
        print("(MEDIUM effect)")
    elif abs(cohens_d) >= 0.2:
        print("(SMALL effect)")
    else:
        print("(negligible)")

    print(f"\n--- Diagnostic Metrics ---")
    print(f"False Positive Rate (pure A has H2 signal): {fp_rate:.1%}")
    print(f"False Negative Rate (mixed has no H2): {fn_rate:.1%}")

    # Q2: Is ratio correlated with mix level? (A-only, A+B, A+B+C)
    print(f"\n--- Q2: Dose-Response (A, A+B, A+B+C) ---")
    results_triple = []
    for i in range(N):
        n_a = min(10, len(emb_prog))
        n_b = min(10, len(emb_phys))
        n_c = min(5, len(emb_geo))
        idx_a = np.random.permutation(len(emb_prog))[:n_a]
        idx_b = np.random.permutation(len(emb_phys))[:n_b]
        idx_c = np.random.permutation(len(emb_geo))[:n_c]
        combined = np.concatenate([emb_prog[idx_a], emb_phys[idx_b], emb_geo[idx_c]])
        perm = np.random.permutation(len(combined))
        vecs = combined[perm]
        b1, b2 = compute_h12(vecs, max_homology_dim=2)
        ratio = b2 / b1 if b1 > 0 else 0.0
        results_triple.append({"betti_1": b1, "betti_2": b2, "ratio": ratio})

    triple_ratios = [r["ratio"] for r in results_triple]
    print(f"  A only:      mean ratio = {np.mean(pure_ratios):.3f}")
    print(f"  A+B:        mean ratio = {np.mean(mixed_ratios):.3f}")
    print(f"  A+B+C:      mean ratio = {np.mean(triple_ratios):.3f}")

    # Spearman correlation: mix_level vs ratio
    mix_levels = [0.0]*N + [0.5]*N + [0.67]*N  # approximate
    all_ratios = pure_ratios + mixed_ratios + triple_ratios
    spearman_r, spearman_p = stats.spearmanr(mix_levels, all_ratios)
    print(f"\n  Spearman correlation (mix_level vs ratio): r={spearman_r:.3f}, p={spearman_p:.4f}")

    # Q3: Does H2 detect what H1 can't?
    # Check if H2 detects mixing in trials where H1 shows no change
    h2_detects = sum(1 for p, m in zip(results_pure, results_mixed)
                     if m["betti_2"] > 0 and p["betti_2"] == 0)
    print(f"\n--- Q3: H2 detects mixing in pure-signal trials ---")
    print(f"  H2 positive in pure: {sum(1 for r in results_pure if r['betti_2']>0)}/{N}")
    print(f"  H2 positive in mixed: {sum(1 for r in results_mixed if r['betti_2']>0)}/{N}")
    print(f"  H2 positive in triple: {sum(1 for r in results_triple if r['betti_2']>0)}/{N}")

    # Summary verdict
    print("\n" + "=" * 60)
    print("=== VERDICT ===\n")
    significant = p_value_mw < 0.05
    large_effect = abs(cohens_d) >= 0.8
    low_fp = fp_rate < 0.2
    pure_baseline = np.mean(pure_ratios)
    h2_detects_ratio = sum(1 for m in mixed_ratios if m > pure_baseline) / len(mixed_ratios)
    all_yes = significant and large_effect and h2_detects_ratio > 0.8

    print(f"Q1 (H2/H1 significant): {'YES' if significant else 'NO'} (p={p_value_mw:.4f}, {'significant' if significant else 'not significant'})")
    print(f"   Effect size: Cohen's d={cohens_d:.3f} ({'LARGE' if large_effect else 'small/medium'})")
    print(f"Q2 (Dose-response): {'YES' if spearman_p < 0.05 else 'NO'} (r={spearman_r:.3f}, p={spearman_p:.4f})")
    print(f"Q3 (H2 adds info via ratio): YES in {h2_detects_ratio:.0%} of mixed trials (ratio > {pure_baseline:.3f} baseline)")
    print(f"Q4 (H2 baseline in pure A): ratio = {pure_baseline:.3f} (not a false positive, it's a BASELINE)")
    print(f"\n{'ALL YES: H2 is a statistically significant domain-mixing detector' if all_yes else 'NEEDS MORE TRIALS: Results not conclusive yet'}")
    if not significant:
        print(f"  -> Increase N (current N={N} may be underpowered)")
    if not large_effect:
        print(f"  -> Effect size too small for practical use")

    # Save
    out = {
        "N": N,
        "pure": results_pure,
        "mixed": results_mixed,
        "triple": results_triple,
        "stats": {
            "p_mannwhitney": float(round(p_value_mw, 4)),
            "p_ttest": float(round(p_value_t, 4)),
            "cohens_d": float(round(cohens_d, 3)),
            "spearman_r": float(round(spearman_r, 3)),
            "spearman_p": float(round(spearman_p, 4)),
            "fp_rate": float(round(fp_rate, 3)),
            "fn_rate": float(round(fn_rate, 3)),
        },
        "verdict": {
            "q1_significant": bool(significant),
            "q2_dose_response": bool(spearman_p < 0.05),
            "q3_h2_adds_info": bool(h2_detects_ratio > 0.8),
            "q4_acceptable_fp": bool(low_fp),
            "all_yes": bool(all_yes),
            "h2_detects_ratio": float(round(h2_detects_ratio, 3)),
        },
        "timing": {"pure_s": float(round(t_pure, 1)), "mixed_s": float(round(t_mixed, 1))},
    }
    out_path = Path(__file__).parent / "results" / f"h2_validation_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {t_pure + t_mixed:.1f}s")

    return out


if __name__ == "__main__":
    main()
