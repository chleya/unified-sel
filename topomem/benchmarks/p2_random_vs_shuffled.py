#!/usr/bin/env python3
"""
P2: Random vs Shuffled vs Real Vectors Comparison

Three-way control to determine what makes real embedding H1 special:
1. REAL vectors: actual MiniLM-L6-v2 embeddings (preserve geometry)
2. SHUFFLED: real embeddings with shuffled node labels (same geometry, no semantics)
3. RANDOM: uniformly sampled from unit hypersphere (no geometry, no semantics)

Hypothesis:
- If RANDOM ≈ SHUFFLED: H1 is purely geometric (supports our P0 conclusion)
- If RANDOM << SHUFFLED ≈ REAL: H1 captures genuine geometric structure beyond random
- If SHUFFLED ≈ REAL: gamma/persistence differences are noise

This validates direction #1 (gamma interpretation reversal).
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPOMEM_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(TOPOMEM_DIR)
sys.path.insert(0, PROJECT_DIR)

from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.config import EmbeddingConfig, TopologyConfig

def load_corpus(domain):
    path = os.path.join(TOPOMEM_DIR, 'data', 'test_corpus', f'{domain}.json')
    with open(path, encoding='utf-8') as f:
        return [item['content'] for item in json.load(f)]

def normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return v / norms

def sample_random_sphere(n, d):
    """Sample n points uniformly from d-dimensional unit hypersphere."""
    # Box-Muller for Gaussian, then normalize
    x = np.random.randn(n, d)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

def shuffle_vectors(vecs):
    """Shuffle vector order (preserves pairwise geometry, destroys sequence semantics)."""
    indices = np.random.permutation(len(vecs))
    return vecs[indices]

def compute_h1_pairs(vecs, topo_engine):
    result = topo_engine.compute_persistence(vecs)
    if len(result) <= 1 or len(result[1]) == 0:
        return []
    pairs = []
    for pair in result[1]:
        birth, death = float(pair[0]), float(pair[1])
        if death > birth:
            pairs.append((birth, death))
    return pairs

def betti1_growth(vecs, topo_engine, max_n=25):
    n = min(max_n, len(vecs))
    bettis = []
    for k in range(1, n + 1):
        idx = np.linspace(0, len(vecs)-1, k, dtype=int)
        sub = vecs[idx]
        pairs = compute_h1_pairs(sub, topo_engine)
        bettis.append(len(pairs))
    return bettis

def fit_gamma(betti_curve):
    xs, ys = [], []
    for i, b in enumerate(betti_curve):
        if b > 0:
            xs.append(i + 1)
            ys.append(b)
    if len(xs) < 3:
        return 0.0
    log_x, log_y = np.log(np.array(xs, dtype=float)), np.log(np.array(ys, dtype=float))
    return float(np.polyfit(log_x, log_y, 1)[0])

def pers_stats(pairs):
    if not pairs:
        return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'q1': 0.0, 'q3': 0.0}
    pers = np.array([d - b for b, d in pairs])
    return {
        'count': len(pairs),
        'mean': float(np.mean(pers)),
        'std': float(np.std(pers)),
        'min': float(np.min(pers)),
        'max': float(np.max(pers)),
        'q1': float(np.percentile(pers, 25)),
        'q3': float(np.percentile(pers, 75)),
    }

print("=" * 60)
print("P2: Random vs Shuffled vs Real Vectors")
print("=" * 60)

# Load real data
print("\n[1/5] Loading corpus...")
prog = load_corpus('programming')
phys = load_corpus('physics')
print(f"  Programming: {len(prog)}, Physics: {len(phys)}")

# Encode
print("\n[2/5] Encoding real embeddings...")
emb = EmbeddingManager(EmbeddingConfig())
vecs_prog = normalize(emb.encode_batch(prog))
vecs_phys = normalize(emb.encode_batch(phys))
vecs_all = np.vstack([vecs_prog, vecs_phys])

dim = vecs_prog.shape[1]
print(f"  Dimension: {dim}")

# Generate controls
print("\n[3/5] Generating control vectors...")
np.random.seed(42)
shuffled_prog = shuffle_vectors(vecs_prog)
shuffled_phys = shuffle_vectors(vecs_phys)
shuffled_all = np.vstack([shuffled_prog, shuffled_phys])

np.random.seed(42)
random_prog = sample_random_sphere(len(vecs_prog), dim)
random_phys = sample_random_sphere(len(vecs_phys), dim)
random_all = np.vstack([random_prog, random_phys])

print(f"  SHUFFLED: node labels permuted, geometry preserved")
print(f"  RANDOM: uniform hypersphere sampling, no geometry")

# Init topology
topo = TopologyEngine(TopologyConfig())

# ===== TEST 1: Within-domain comparison (A=programming 20 items) =====
print("\n[4/5] TEST 1: Within-domain comparison (A only, 20 items)...")

print("\n  --- Programming domain (20 items) ---")
print(f"  {'Metric':<25} {'REAL':<12} {'SHUFFLED':<12} {'RANDOM':<12}")
print("  " + "-" * 65)

pairs_real = compute_h1_pairs(vecs_prog, topo)
pairs_shuff = compute_h1_pairs(shuffled_prog, topo)
pairs_rand = compute_h1_pairs(random_prog, topo)

s_real = pers_stats(pairs_real)
s_shuff = pers_stats(pairs_shuff)
s_rand = pers_stats(pairs_rand)

g_real = fit_gamma(betti1_growth(vecs_prog, topo))
g_shuff = fit_gamma(betti1_growth(shuffled_prog, topo))
g_rand = fit_gamma(betti1_growth(random_prog, topo))

growth_real = betti1_growth(vecs_prog, topo)
growth_shuff = betti1_growth(shuffled_prog, topo)
growth_rand = betti1_growth(random_prog, topo)

print(f"  {'Betti-1 count':<25} {s_real['count']:<12} {s_shuff['count']:<12} {s_rand['count']:<12}")
print(f"  {'Gamma (power-law exp)':<25} {g_real:<12.3f} {g_shuff:<12.3f} {g_rand:<12.3f}")
print(f"  {'Mean persistence':<25} {s_real['mean']:<12.4f} {s_shuff['mean']:<12.4f} {s_rand['mean']:<12.4f}")
print(f"  {'Std persistence':<25} {s_real['std']:<12.4f} {s_shuff['std']:<12.4f} {s_rand['std']:<12.4f}")
print(f"  {'Min persistence':<25} {s_real['min']:<12.4f} {s_shuff['min']:<12.4f} {s_rand['min']:<12.4f}")
print(f"  {'Max persistence':<25} {s_real['max']:<12.4f} {s_shuff['max']:<12.4f} {s_rand['max']:<12.4f}")
print(f"  {'Q1 persistence':<25} {s_real['q1']:<12.4f} {s_shuff['q1']:<12.4f} {s_rand['q1']:<12.4f}")
print(f"  {'Q3 persistence':<25} {s_real['q3']:<12.4f} {s_shuff['q3']:<12.4f} {s_rand['q3']:<12.4f}")

print(f"\n  Growth curves (n=5..{len(growth_real)}):")
print(f"  {'n':<4} {'REAL':<8} {'SHUFFLED':<10} {'RANDOM':<8}")
print("  " + "-" * 35)
for i in range(min(8, len(growth_real))):
    print(f"  {i+5:<4} {growth_real[i]:<8} {growth_shuff[i]:<10} {growth_rand[i]:<8}")

# Statistical comparison
print(f"\n  Statistical tests (Shuffled vs Real):")
if s_real['count'] > 0 and s_shuff['count'] > 0:
    diff_betti = abs(s_real['count'] - s_shuff['count'])
    diff_pct = diff_betti / max(s_real['count'], s_shuff['count']) * 100
    print(f"  Betti-1 difference: {diff_betti} ({diff_pct:.1f}%)")
    gamma_diff = abs(g_real - g_shuff)
    print(f"  Gamma difference: {gamma_diff:.3f}")

# ===== TEST 2: Cross-domain A+B (40 items) =====
print("\n[5/5] TEST 2: Cross-domain A+B (40 items)...")

print("\n  --- Programming + Physics (40 items) ---")
print(f"  {'Metric':<25} {'REAL':<12} {'SHUFFLED':<12} {'RANDOM':<12}")
print("  " + "-" * 65)

pairs_real_ab = compute_h1_pairs(vecs_all, topo)
pairs_shuff_ab = compute_h1_pairs(shuffled_all, topo)
pairs_rand_ab = compute_h1_pairs(random_all, topo)

s_real_ab = pers_stats(pairs_real_ab)
s_shuff_ab = pers_stats(pairs_shuff_ab)
s_rand_ab = pers_stats(pairs_rand_ab)

g_real_ab = fit_gamma(betti1_growth(vecs_all, topo))
g_shuff_ab = fit_gamma(betti1_growth(shuffled_all, topo))
g_rand_ab = fit_gamma(betti1_growth(random_all, topo))

growth_real_ab = betti1_growth(vecs_all, topo)
growth_shuff_ab = betti1_growth(shuffled_all, topo)
growth_rand_ab = betti1_growth(random_all, topo)

print(f"  {'Betti-1 count':<25} {s_real_ab['count']:<12} {s_shuff_ab['count']:<12} {s_rand_ab['count']:<12}")
print(f"  {'Gamma':<25} {g_real_ab:<12.3f} {g_shuff_ab:<12.3f} {g_rand_ab:<12.3f}")
print(f"  {'Mean persistence':<25} {s_real_ab['mean']:<12.4f} {s_shuff_ab['mean']:<12.4f} {s_rand_ab['mean']:<12.4f}")
print(f"  {'Std persistence':<25} {s_real_ab['std']:<12.4f} {s_shuff_ab['std']:<12.4f} {s_rand_ab['std']:<12.4f}")

# A+B ratio
ratio_real = s_real_ab['count'] / max(s_real['count'], 1)
ratio_shuff = s_shuff_ab['count'] / max(s_shuff['count'], 1)
ratio_rand = s_rand_ab['count'] / max(s_rand['count'], 1)
print(f"  {'A+B / A ratio':<25} {ratio_real:<12.3f} {ratio_shuff:<12.3f} {ratio_rand:<12.3f}")

print(f"\n  Growth curves (n=5..{len(growth_real_ab)}):")
print(f"  {'n':<4} {'REAL':<8} {'SHUFFLED':<10} {'RANDOM':<8}")
print("  " + "-" * 35)
for i in range(min(10, len(growth_real_ab))):
    print(f"  {i+5:<4} {growth_real_ab[i]:<8} {growth_shuff_ab[i]:<10} {growth_rand_ab[i]:<8}")

# ===== FINAL VERDICT =====
print("\n" + "=" * 60)
print("P2 VERDICT: What does H1 actually measure?")
print("=" * 60)

# Compare within-domain
real_vs_shuff_betti = abs(s_real['count'] - s_shuff['count']) / max(s_real['count'], s_shuff['count'], 1)
shuff_vs_rand_betti = abs(s_shuff['count'] - s_rand['count']) / max(s_shuff['count'], s_rand['count'], 1)

print(f"\n  Within-domain differences (A=20 items):")
print(f"  REAL vs SHUFFLED Betti-1: {real_vs_shuff_betti*100:.1f}%")
print(f"  SHUFFLED vs RANDOM Betti-1: {shuff_vs_rand_betti*100:.1f}%")
print(f"  REAL vs RANDOM Betti-1: {abs(s_real['count'] - s_rand['count']) / max(s_real['count'], s_rand['count'], 1)*100:.1f}%")

verdict_key = None
if shuff_vs_rand_betti < 0.2 and real_vs_shuff_betti > 0.3:
    verdict_key = "GEOMETRY_BEYOND_RANDOM"
    verdict_text = "SHUFFLED ≈ RANDOM but REAL >> BOTH. H1 captures genuine geometric structure beyond random."
elif shuff_vs_rand_betti < 0.2 and real_vs_shuff_betti < 0.2:
    verdict_key = "PURELY_GEOMETRIC"
    verdict_text = "RANDOM ≈ SHUFFLED ≈ REAL. H1 is purely geometric, no semantic signal."
elif shuff_vs_rand_betti > 0.3 and real_vs_shuff_betti < 0.2:
    verdict_key = "GEOMETRY_DESTROYED_BY_SHUFFLE"
    verdict_text = "SHUFFLED >> RANDOM, SHUFFLED ≈ REAL. Shuffling destroys real geometric structure."
elif s_rand['count'] == 0 and s_real['count'] > 0:
    verdict_key = "RANDOM_GENERATES_NO_CYCLES"
    verdict_text = "RANDOM produces zero H1 cycles. All H1 cycles are geometric structure from real embeddings."
else:
    verdict_key = "MIXED"
    verdict_text = f"Complex pattern - real={s_real['count']}, shuff={s_shuff['count']}, rand={s_rand['count']}"

print(f"\n  {'=' * 50}")
print(f"  VERDICT: {verdict_key}")
print(f"  {'=' * 50}")
print(f"  {verdict_text}")

print(f"\n  Gamma comparison:")
print(f"  REAL gamma={g_real:.3f}, SHUFFLED gamma={g_shuff:.3f}, RANDOM gamma={g_rand:.3f}")
if g_real > g_shuff > g_rand:
    print(f"  Pattern: REAL > SHUFFLED > RANDOM → Gamma confirms semantic geometry")
elif g_shuff > g_real > g_rand:
    print(f"  Pattern: SHUFFLED > REAL > RANDOM → Shuffling amplifies some geometric effect")
else:
    print(f"  Pattern: {g_real:.3f} / {g_shuff:.3f} / {g_rand:.3f}")

# Save results
results_dir = os.path.join(TOPOMEM_DIR, 'benchmarks', 'results')
os.makedirs(results_dir, exist_ok=True)
ts = int(time.time())
result_file = os.path.join(results_dir, f'p2_random_comparison_{ts}.json')

save_data = {
    'verdict': verdict_key,
    'within_A': {
        'real': s_real, 'shuffled': s_shuff, 'random': s_rand,
        'gamma_real': g_real, 'gamma_shuff': g_shuff, 'gamma_rand': g_rand,
        'growth_real': growth_real, 'growth_shuff': growth_shuff, 'growth_rand': growth_rand,
    },
    'within_AB': {
        'real': s_real_ab, 'shuffled': s_shuff_ab, 'random': s_rand_ab,
        'gamma_real': g_real_ab, 'gamma_shuff': g_shuff_ab, 'gamma_rand': g_rand_ab,
        'growth_real': growth_real_ab, 'growth_shuff': growth_shuff_ab, 'growth_rand': growth_rand_ab,
    },
}

with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)

print(f"\nResults saved: {result_file}")
