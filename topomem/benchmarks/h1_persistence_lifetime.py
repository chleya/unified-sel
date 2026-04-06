#!/usr/bin/env python3
"""
Priority 2: H1 Persistence Lifetime Filter

Core hypothesis: Short-persistence H1 cycles = geometric noise (fragile to embedding perturbation).
                  Long-persistence H1 cycles = semantic structure (robust to embedding perturbation).

Test: Add Gaussian noise to embeddings, measure which H1 cycles survive.
If long-pers cycles are robust -> they encode real structure.
If all cycles equally fragile -> H1 is mostly noise.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json, os, sys

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)  # topomem/
ROOT_DIR = os.path.dirname(PKG_DIR)     # unified-sel/
sys.path.insert(0, ROOT_DIR)

from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.config import EmbeddingConfig, TopologyConfig

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1, norms)
    return vecs / norms

def load_corpus(domain):
    """Load corpus items from test_corpus."""
    corpus_file = os.path.join(PKG_DIR, 'data', 'test_corpus', f'{domain}.json')
    with open(corpus_file, encoding='utf-8') as f:
        data = json.load(f)
    return [(item['content'], item.get('test_question', '')) for item in data]

def add_gaussian_noise(vecs, noise_scale=0.05):
    """Add Gaussian noise to embeddings. noise_scale = std dev as fraction of embedding norm."""
    noise = np.random.randn(*vecs.shape).astype(np.float32) * noise_scale
    noisy = vecs + noise
    return normalize(noisy)

def compute_h1_cycles(vecs, topo_engine):
    """Compute H1 cycles and return list of (birth, death, persistence)."""
    result = topo_engine.compute_persistence(vecs.astype(np.float64))
    # result is a list indexed by dimension: result[0]=H0, result[1]=H1, result[2]=H2...
    h1_pairs = result[1] if len(result) > 1 else np.array([]).reshape(0, 2)
    cycles = []
    for p in h1_pairs:
        if len(p) >= 2:
            birth, death = float(p[0]), float(p[1])
            pers = death - birth
            cycles.append({'birth': birth, 'death': death, 'persistence': pers})
    return cycles

def match_cycles_by_persistence(cycles1, cycles2, threshold=0.01):
    """
    Match cycles between two sets by similar birth/death values.
    Returns: matched, only_in_1, only_in_2
    """
    matched = []
    used1, used2 = set(), set()
    
    # Sort by persistence descending (match high-pers first)
    idx1 = sorted(range(len(cycles1)), key=lambda i: cycles1[i]['persistence'], reverse=True)
    idx2 = sorted(range(len(cycles2)), key=lambda i: cycles2[i]['persistence'], reverse=True)
    
    for i in idx1:
        c1 = cycles1[i]
        for j in idx2:
            if j in used2:
                continue
            c2 = cycles2[j]
            # Check if birth and death are similar
            if (abs(c1['birth'] - c2['birth']) < threshold and 
                abs(c1['death'] - c2['death']) < threshold):
                matched.append((c1, c2))
                used1.add(i)
                used2.add(j)
                break
    
    only1 = [cycles1[i] for i in range(len(cycles1)) if i not in used1]
    only2 = [cycles2[j] for j in range(len(cycles2)) if j not in used2]
    return matched, only1, only2

print("=" * 60)
print("Priority 2: H1 Persistence Lifetime Filter")
print("Hypothesis: Long-pers cycles = semantic (robust)")
print("=" * 60)

# Load programming corpus (domain A)
corpus = load_corpus('programming')
contents = [c[0] for c in corpus]
questions = [c[1] for c in corpus]
print(f"\nDomain A (programming): {len(contents)} items")

# Encode
emb_mgr = EmbeddingManager(EmbeddingConfig())
vecs_A = emb_mgr.encode_batch(contents)
vecs_A = normalize(vecs_A).astype(np.float64)
print(f"Embeddings shape: {vecs_A.shape}")

# Compute H1 with clean embeddings
topo_config = TopologyConfig()
topo_engine = TopologyEngine(topo_config)
clean_cycles = compute_h1_cycles(vecs_A, topo_engine)
print(f"\nClean H1 cycles: {len(clean_cycles)}")
if clean_cycles:
    pers_values = [c['persistence'] for c in clean_cycles]
    print(f"  Persistence: min={min(pers_values):.4f}, max={max(pers_values):.4f}, mean={np.mean(pers_values):.4f}")
    print(f"  Median={np.median(pers_values):.4f}")
    # Quartiles
    q1 = np.percentile(pers_values, 25)
    q3 = np.percentile(pers_values, 75)
    print(f"  Q1={q1:.4f}, Q3={q3:.4f}")

# Test perturbation levels
noise_levels = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
n_trials = 20

print(f"\n{'Noise':>8} {'Trials':>8} {'Cycles':>8} {'Survival':>10} {'Long-surv':>12} {'Short-surv':>12}")
print("-" * 62)

results_by_pers = {}

for noise in noise_levels:
    all_survived = []
    long_survived = []
    short_survived = []
    
    for trial in range(n_trials):
        np.random.seed(trial * 42)
        noisy_vecs = add_gaussian_noise(vecs_A, noise_scale=noise)
        noisy_cycles = compute_h1_cycles(noisy_vecs, topo_engine)
        
        matched, _, _ = match_cycles_by_persistence(clean_cycles, noisy_cycles, threshold=0.05)
        survival_rate = len(matched) / max(len(clean_cycles), 1)
        all_survived.append(survival_rate)
        
        # Split clean cycles by persistence quartile
        if clean_cycles and pers_values:
            q1_val = np.percentile(pers_values, 25)
            q3_val = np.percentile(pers_values, 75)
            
            long_cycles = [c for c in clean_cycles if c['persistence'] >= q3_val]
            short_cycles = [c for c in clean_cycles if c['persistence'] <= q1_val]
            
            if long_cycles:
                long_matched = [m for m in matched if m[0] in long_cycles]
                long_survived.append(len(long_matched) / max(len(long_cycles), 1))
            else:
                long_survived.append(0.0)
            
            if short_cycles:
                short_matched = [m for m in matched if m[0] in short_cycles]
                short_survived.append(len(short_matched) / max(len(short_cycles), 1))
            else:
                short_survived.append(0.0)
    
    avg_surv = np.mean(all_survived)
    avg_long = np.mean(long_survived) if long_survived else 0
    avg_short = np.mean(short_survived) if short_survived else 0
    
    results_by_pers[noise] = {
        'survival_rate': avg_surv,
        'long_pers_survival': avg_long,
        'short_pers_survival': avg_short,
    }
    
    print(f"{noise:>8.2f} {n_trials:>8} {len(clean_cycles):>8} {avg_surv:>10.2%} {avg_long:>12.2%} {avg_short:>12.2%}")

# Key finding
print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

# Compare long vs short survival rates
if results_by_pers:
    last = results_by_pers[noise_levels[-1]]
    diff = last['long_pers_survival'] - last['short_pers_survival']
    
    if diff > 0.15:
        verdict = "LONG-PERS IS MORE ROBUST -> H1 encodes semantic structure"
        symbol = "[PASS]"
    elif diff < -0.15:
        verdict = "SHORT-PERS IS MORE ROBUST -> H1 is mostly noise"
        symbol = "[FAIL]"
    else:
        verdict = f"NO DIFFERENCE (diff={diff:.2%}) -> H1 is uniformly fragile"
        symbol = "[TIE]"
    
    print(f"\n{symbol} At noise={noise_levels[-1]:.0%}:")
    print(f"  Long-persistence cycles survival: {last['long_pers_survival']:.1%}")
    print(f"  Short-persistence cycles survival: {last['short_pers_survival']:.1%}")
    print(f"  Difference: {diff:+.1%}")
    print(f"\n  {verdict}")

# Survival curve analysis
print("\nSurvival by noise level (long-persistence only):")
for noise, r in results_by_pers.items():
    bar = '#' * int(r['long_pers_survival'] * 20)
    print(f"  noise={noise:.2f}: {r['long_pers_survival']:>6.1%} {bar}")

# Save results
results_dir = os.path.join(PKG_DIR, 'benchmarks', 'results')
os.makedirs(results_dir, exist_ok=True)
import time
result_file = os.path.join(results_dir, f'persistence_lifetime_{int(time.time())}.json')

save_data = {
    'noise_levels': noise_levels,
    'n_trials': n_trials,
    'n_clean_cycles': len(clean_cycles),
    'results': {str(k): v for k, v in results_by_pers.items()},
    'clean_cycles': clean_cycles,
}
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: {result_file}")
