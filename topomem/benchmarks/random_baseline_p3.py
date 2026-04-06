#!/usr/bin/env python3
"""
Priority P0: Random Baseline -- Is H1 structure semantic or geometric?

Test: Shuffle node labels in embedding space, re-run TEST 1-4.
If gamma, entropy, super-linear emergence are similar to real data -> GEOMETRIC (Qwen)
If they are different -> SEMANTIC (Claude)

Also tests: A+B super-linear emergence mechanism (cross-domain edges → C(k,2) extra cycles)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json, os, sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(PKG_DIR)
sys.path.insert(0, ROOT_DIR)

from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.config import EmbeddingConfig, TopologyConfig

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1, norms)
    return vecs / norms

def load_corpus(domain):
    corpus_file = os.path.join(PKG_DIR, 'data', 'test_corpus', f'{domain}.json')
    with open(corpus_file, encoding='utf-8') as f:
        data = json.load(f)
    return [item['content'] for item in data]

def entropy_of_persistence(pers_values, n_bins=10):
    if len(pers_values) < 2:
        return 0.0
    hist, _ = np.histogram(pers_values, bins=n_bins, range=(0, max(pers_values) + 1e-8))
    hist = hist / max(sum(hist), 1)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist + 1e-12))

def gini_coefficient(values):
    if len(values) < 2:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1) * sorted_vals))) / (n * cum[-1]) - (n + 1) / n

def compute_h1_pers_values(vecs, topo_engine):
    result = topo_engine.compute_persistence(vecs.astype(np.float64))
    h1_pairs = result[1] if len(result) > 1 else np.array([]).reshape(0, 2)
    pers_values = []
    for p in h1_pairs:
        if len(p) >= 2:
            pers = max(0, float(p[1]) - float(p[0]))
            if pers > 1e-8:
                pers_values.append(pers)
    return pers_values

print("=" * 60)
print("Priority P0: Random Baseline")
print("Is H1 structure GEOMETRIC or SEMANTIC?")
print("=" * 60)

# Load real data
prog = load_corpus('programming')
phys = load_corpus('physics')
geo = load_corpus('geography')
emb_mgr = EmbeddingManager(EmbeddingConfig())

# Encode real embeddings
vecs_prog_real = normalize(emb_mgr.encode_batch(prog)).astype(np.float64)
vecs_phys_real = normalize(emb_mgr.encode_batch(phys)).astype(np.float64)
vecs_geo_real = normalize(emb_mgr.encode_batch(geo)).astype(np.float64)

n_trials = 5  # Multiple random shuffles for robustness

# ----- TEST 1: Betti-1 Growth Curve (SHUFFLED) -----
print("\n" + "=" * 60)
print("TEST 1: Betti-1 Growth Curve (Random Baseline)")
print("=" * 60)

n_points_range = list(range(5, len(vecs_prog_real) + 1, 2))
topo_engine = TopologyEngine(TopologyConfig())

all_shuffled_growth = []
for trial in range(n_trials):
    np.random.seed(trial * 42 + 7)
    idx = np.random.permutation(len(vecs_prog_real))
    vecs_shuffled = vecs_prog_real[idx]  # same geometry, shuffled labels
    
    growth_counts = []
    for n in n_points_range:
        pv = compute_h1_pers_values(vecs_shuffled[:n], topo_engine)
        growth_counts.append(len(pv))
    all_shuffled_growth.append(growth_counts)

# Average across trials
avg_growth = np.mean(all_shuffled_growth, axis=0)

print(f"\n{'n':>4} {'Real B1':>10} {'Shuffled B1 (avg)':>20}")
print("-" * 38)
real_growth = [0, 0, 0, 0, 1, 4, 5, 6]  # from Priority 3
for i, n in enumerate(n_points_range):
    print(f"{n:>4} {real_growth[i]:>10} {avg_growth[i]:>20.1f}")

# Fit power law on shuffled data
valid_points = [(n_points_range[i], max(avg_growth[i], 0.1)) 
                for i in range(len(n_points_range)) if avg_growth[i] > 0.1]
if len(valid_points) >= 2:
    log_n = np.log([p[0] for p in valid_points])
    log_b = np.log([p[1] for p in valid_points])
    gamma_shuffled = np.polyfit(log_n, log_b, 1)[0] if np.std(log_b) > 0 else 0
else:
    gamma_shuffled = 0

gamma_real = 3.678
gamma_diff_pct = abs(gamma_shuffled - gamma_real) / max(gamma_real, 0.1) * 100

print(f"\nGamma (real):     {gamma_real:.3f}")
print(f"Gamma (shuffled): {gamma_shuffled:.3f}")
print(f"Difference:       {gamma_diff_pct:.1f}%")

if gamma_diff_pct < 20:
    print("\n[GEOMETRIC] gamma is similar -> H1 growth is GEOMETRIC (Qwen)")
    gamma_verdict = "GEOMETRIC"
elif gamma_shuffled < 1.0:
    print("\n[GEOMETRIC-SHUFFLED] shuffled drops to ~1.0 -> gamma is GEOMETRIC (Qwen)")
    gamma_verdict = "GEOMETRIC"
else:
    print(f"\n[UNCLEAR] shuffled gamma={gamma_shuffled:.3f}, need more trials")
    gamma_verdict = "UNCLEAR"

# ----- TEST 2: Persistence Distribution (SHUFFLED) -----
print("\n" + "=" * 60)
print("TEST 2: Persistence Distribution (Shuffled A=programming)")
print("=" * 60)

all_ents_shuffled = []
all_ginis_shuffled = []
for trial in range(n_trials):
    np.random.seed(trial * 42 + 13)
    idx = np.random.permutation(len(vecs_prog_real))
    vecs_s = vecs_prog_real[idx]
    pv = compute_h1_pers_values(vecs_s, topo_engine)
    all_ents_shuffled.append(entropy_of_persistence(pv))
    all_ginis_shuffled.append(gini_coefficient(pv))

avg_ent_shuffled = np.mean(all_ents_shuffled)
avg_gini_shuffled = np.mean(all_ginis_shuffled)
std_ent_shuffled = np.std(all_ents_shuffled)
std_gini_shuffled = np.std(all_ginis_shuffled)

print(f"\n{'Metric':>20} {'Real A':>12} {'Shuffled':>15}")
print("-" * 50)
print(f"{'Entropy':>20} {1.5596:>12.4f} {avg_ent_shuffled:>15.4f} +/- {std_ent_shuffled:.4f}")
print(f"{'Gini':>20} {0.4842:>12.4f} {avg_gini_shuffled:>15.4f} +/- {std_gini_shuffled:.4f}")
print(f"{'Cycles':>20} {8:>12} {np.mean([len(compute_h1_pers_values(vecs_prog_real[np.random.permutation(20)], topo_engine)) for _ in range(n_trials)]):>15.1f}")

ent_diff = abs(avg_ent_shuffled - 1.5596)
if ent_diff < 0.3:
    print(f"\n[GEOMETRIC] entropy difference = {ent_diff:.4f} -> shuffled is similar")
    ent_verdict = "GEOMETRIC"
else:
    print(f"\n[SEMANTIC] entropy difference = {ent_diff:.4f} -> shuffled is different")
    ent_verdict = "SEMANTIC"

# ----- TEST 3: A+B Super-linear Emergence (SHUFFLED) -----
print("\n" + "=" * 60)
print("TEST 3: A+B Super-linear Emergence (Shuffled)")
print("=" * 60)

# For shuffled test: mix A and B but with random correspondence
# Keep embeddings intact, but randomly pair A+B items
all_ratio_shuffled = []
for trial in range(n_trials):
    np.random.seed(trial * 42 + 19)
    
    # Randomly shuffle the combined pool
    all_vecs = np.vstack([vecs_prog_real, vecs_phys_real])
    idx = np.random.permutation(len(all_vecs))
    vecs_ab_shuffled = all_vecs[idx]
    
    pv_ab = compute_h1_pers_values(vecs_ab_shuffled, topo_engine)
    n_ab = len(pv_ab)
    
    # Real: A_only=8, B_only=12, expected linear=20, actual=31
    ratio = n_ab / max(8, 1)
    all_ratio_shuffled.append(ratio)

avg_ratio_shuffled = np.mean(all_ratio_shuffled)
std_ratio_shuffled = np.std(all_ratio_shuffled)
real_ratio = 31 / 8  # 3.875

print(f"\n{'Metric':>20} {'Real A+B':>12} {'Shuffled A+B':>18}")
print("-" * 55)
print(f"{'Betti-1 count':>20} {31:>12} {np.mean([len(compute_h1_pers_values(np.vstack([vecs_prog_real, vecs_phys_real])[np.random.permutation(40)], topo_engine)) for _ in range(n_trials)]):>18.1f}")
print(f"{'Ratio vs A-only':>20} {real_ratio:>12.3f} {avg_ratio_shuffled:>18.3f} +/- {std_ratio_shuffled:.3f}")

ratio_diff_pct = abs(avg_ratio_shuffled - real_ratio) / max(real_ratio, 0.1) * 100

if ratio_diff_pct < 30:
    print(f"\n[GEOMETRIC] super-linear emergence is similar in shuffled data")
    mix_verdict = "GEOMETRIC"
else:
    print(f"\n[SEMANTIC] super-linear emergence is REDUCED in shuffled data")
    mix_verdict = "SEMANTIC"

# ----- TEST 4: Cross-domain edges mechanism -----
print("\n" + "=" * 60)
print("TEST 4: Cross-domain Edge Mechanism (one trial)")
print("=" * 60)

# Compute pairwise cosine similarity to estimate "cross-domain edges"
# In real data: A points cluster together, B points cluster together
# Cross-domain similarity should be low
def cosine_sim(a, b):
    return np.dot(a, b)

n_A = len(vecs_prog_real)
n_B = len(vecs_phys_real)
all_vecs_ab = np.vstack([vecs_prog_real, vecs_phys_real])

# Count "edges" above a threshold (proxy for VR complex edges)
threshold = 0.5  # cosine similarity threshold
intra_A = sum(1 for i in range(n_A) for j in range(i+1, n_A) 
               if cosine_sim(vecs_prog_real[i], vecs_prog_real[j]) > threshold)
intra_B = sum(1 for i in range(n_B) for j in range(i+1, n_B) 
               if cosine_sim(vecs_phys_real[i], vecs_phys_real[j]) > threshold)
cross_AB = sum(1 for i in range(n_A) for j in range(n_B) 
               if cosine_sim(vecs_prog_real[i], vecs_phys_real[j]) > threshold)

print(f"\nEdge counts (threshold={threshold}):")
print(f"  Intra-A (programming): {intra_A}")
print(f"  Intra-B (physics):     {intra_B}")
print(f"  Cross-AB:              {cross_AB}")

# VR complex intuition: C(k,2) extra cycles from k cross-edges
expected_extra_from_cross = cross_AB * (cross_AB - 1) / 2 if cross_AB > 1 else 0
observed_extra = 31 - 8  # 23 extra cycles beyond A-only

print(f"\n  Cross-AB edges: {cross_AB}")
print(f"  C(k,2) expected extra cycles: ~{expected_extra_from_cross:.0f}")
print(f"  Observed extra cycles (A+B - A-only): {observed_extra}")

if cross_AB < 3:
    print(f"\n  [SPARSE] Cross-AB edges very sparse (cosine < {threshold})")
    print(f"  -> Super-linear emergence NOT from cross-domain SIMILARITY edges")
    print(f"  -> Likely from VR filtration's global connectivity reorganization")
else:
    print(f"\n  Cross-domain edges exist but C(k,2) can't explain 23 extra cycles")

# Try lower threshold
threshold_low = 0.3
cross_AB_low = sum(1 for i in range(n_A) for j in range(n_B) 
                   if cosine_sim(vecs_prog_real[i], vecs_phys_real[j]) > threshold_low)
print(f"\n  With lower threshold={threshold_low}: cross_AB = {cross_AB_low}")

# ============================================================
# OVERALL VERDICT
# ============================================================
print("\n" + "=" * 60)
print("OVERALL VERDICT")
print("=" * 60)

print(f"\n1. Betti-1 Growth (gamma):")
print(f"   Real={gamma_real:.3f}, Shuffled={gamma_shuffled:.3f}, Diff={gamma_diff_pct:.1f}%")
print(f"   => {gamma_verdict}")

print(f"\n2. Persistence Distribution (entropy):")
print(f"   Real=1.5596, Shuffled={avg_ent_shuffled:.4f}, Diff={ent_diff:.4f}")
print(f"   => {ent_verdict}")

print(f"\n3. A+B Super-linear Emergence (ratio):")
print(f"   Real={real_ratio:.3f}, Shuffled={avg_ratio_shuffled:.3f}, Diff={ratio_diff_pct:.1f}%")
print(f"   => {mix_verdict}")

# Overall
geo_count = sum(1 for v in [gamma_verdict, ent_verdict, mix_verdict] if v == "GEOMETRIC")
sem_count = sum(1 for v in [gamma_verdict, ent_verdict, mix_verdict] if v == "SEMANTIC")

print(f"\n" + "=" * 60)
if geo_count >= 2:
    overall = "GEOMETRIC: H1 structure is primarily GEOMETRIC (random graph behavior)"
elif sem_count >= 2:
    overall = "SEMANTIC: H1 structure encodes SEMANTIC information"
else:
    overall = f"UNCLEAR: {gamma_verdict}, {ent_verdict}, {mix_verdict}"

print(f"OVERALL: {overall}")
print(f"Geometry votes: {geo_count}/3, Semantic votes: {sem_count}/3")
print("=" * 60)

# Save
results_dir = os.path.join(PKG_DIR, 'benchmarks', 'results')
os.makedirs(results_dir, exist_ok=True)
import time
result_file = os.path.join(results_dir, f'random_baseline_p3_{int(time.time())}.json')
save_data = {
    'test': 'Random Baseline P0',
    'gamma_real': gamma_real,
    'gamma_shuffled': float(gamma_shuffled),
    'gamma_diff_pct': float(gamma_diff_pct),
    'gamma_verdict': gamma_verdict,
    'entropy_real': 1.5596,
    'entropy_shuffled': float(avg_ent_shuffled),
    'entropy_diff': float(ent_diff),
    'ent_verdict': ent_verdict,
    'ratio_real': float(real_ratio),
    'ratio_shuffled': float(avg_ratio_shuffled),
    'ratio_diff_pct': float(ratio_diff_pct),
    'mix_verdict': mix_verdict,
    'overall': overall,
    'cross_AB_edges': int(cross_AB),
    'expected_extra_cycles': float(expected_extra_from_cross),
    'observed_extra_cycles': int(observed_extra),
}
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: {result_file}")
