#!/usr/bin/env python3
"""
Priority 3: H1 Shape Metrics -- Persistence Distribution Entropy

Core hypothesis: The SHAPE of H1 persistence distribution (not individual cycle persistence)
encodes structural information beyond individual cycle robustness.

Test:
1. Persistence distribution entropy: uniform distribution = high entropy (random geometry)
   peaked distribution = low entropy (structured/functional cycles cluster)
2. Betti-1 growth curve: how H1 count grows as nodes are added
   - Linear growth = random (no topological structure)
   - Sub-linear growth = structured (shared cycles between nodes)
3. Cross-domain mixing: add B domain items, measure how A's entropy changes
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json, os, sys
from collections import Counter

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
    corpus_file = os.path.join(PKG_DIR, 'data', 'test_corpus', f'{domain}.json')
    with open(corpus_file, encoding='utf-8') as f:
        data = json.load(f)
    return [item['content'] for item in data]

def entropy_of_persistence(pers_values, n_bins=10):
    """Compute entropy of persistence distribution."""
    if len(pers_values) < 2:
        return 0.0
    hist, _ = np.histogram(pers_values, bins=n_bins, range=(0, max(pers_values) + 1e-8))
    hist = hist / max(sum(hist), 1)
    hist = hist[hist > 0]  # remove zero bins
    return -np.sum(hist * np.log(hist + 1e-12))

def gini_coefficient(values):
    """Compute Gini coefficient of persistence distribution. Higher = more unequal."""
    if len(values) < 2:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1) * sorted_vals))) / (n * cum[-1]) - (n + 1) / n

def compute_h1_pers_values(vecs, topo_engine):
    """Compute H1 persistence values for a point cloud."""
    result = topo_engine.compute_persistence(vecs.astype(np.float64))
    h1_pairs = result[1] if len(result) > 1 else np.array([]).reshape(0, 2)
    pers_values = []
    for p in h1_pairs:
        if len(p) >= 2:
            pers = max(0, float(p[1]) - float(p[0]))
            if pers > 1e-8:  # filter out zero-persistence
                pers_values.append(pers)
    return pers_values

print("=" * 60)
print("Priority 3: H1 Shape Metrics")
print("Hypothesis: Persistence distribution SHAPE = structural signal")
print("=" * 60)

# Load all three domains
prog = load_corpus('programming')
phys = load_corpus('physics')
geo = load_corpus('geography')
print(f"\nCorpora loaded: programming={len(prog)}, physics={len(phys)}, geography={len(geo)}")

# Encode all
emb_mgr = EmbeddingManager(EmbeddingConfig())
vecs_prog = normalize(emb_mgr.encode_batch(prog)).astype(np.float64)
vecs_phys = normalize(emb_mgr.encode_batch(phys)).astype(np.float64)
vecs_geo = normalize(emb_mgr.encode_batch(geo)).astype(np.float64)
print(f"Shapes: prog={vecs_prog.shape}, phys={vecs_phys.shape}, geo={vecs_geo.shape}")

topo_config = TopologyConfig()
topo_engine = TopologyEngine(topo_config)

# ============================================================
# TEST 1: Betti-1 growth curve (A only)
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: Betti-1 Growth Curve (A = programming)")
print("=" * 60)

n_points_range = list(range(5, len(vecs_prog) + 1, 2))  # 5,7,9,11,13,15,17,19
betti1_growth = []

for n in n_points_range:
    subset = vecs_prog[:n]
    pers_values = compute_h1_pers_values(subset, topo_engine)
    betti1_growth.append({
        'n': n,
        'betti1': len(pers_values),
        'mean_pers': np.mean(pers_values) if pers_values else 0.0,
    })

print(f"\n{'n':>4} {'Betti-1':>8} {'Mean Pers':>10} {'Growth':>8}")
print("-" * 35)
for r in betti1_growth:
    prev = betti1_growth[betti1_growth.index(r) - 1]['betti1'] if betti1_growth.index(r) > 0 else 0
    growth = r['betti1'] - prev
    print(f"{r['n']:>4} {r['betti1']:>8} {r['mean_pers']:>10.4f} {growth:>+8}")

# Analyze growth type
betti1_counts = [r['betti1'] for r in betti1_growth]
n_vals = [r['n'] for r in betti1_growth]

# Fit: betti1 = alpha * n^gamma (power law)
# log(betti1) = log(alpha) + gamma * log(n)
log_n = np.log(n_vals)
log_b = np.log([max(b, 0.1) for b in betti1_counts])
gamma = np.polyfit(log_n, log_b, 1)[0] if np.std(log_b) > 0 else 0

print(f"\nBetti-1 growth exponent (gamma): {gamma:.3f}")
print(f"  gamma=1.0: linear growth (random geometric graph)")
print(f"  gamma=2.0: quadratic growth (expected for VR complex)")
print(f"  gamma<1.0: sub-linear (shared cycle structure)")
print(f"\n  => ", end='')
if gamma < 0.8:
    print("SUB-LINEAR growth -> shared cycle structure EXISTS")
    growth_type = "SUBLINEAR"
elif gamma > 1.5:
    print("SUPER-LINEAR growth -> dense cycle formation")
    growth_type = "SUPERLINEAR"
else:
    print("NEAR-LINEAR growth -> mostly independent edges")
    growth_type = "LINEAR"

# ============================================================
# TEST 2: Persistence Distribution Entropy
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Persistence Distribution Entropy")
print("=" * 60)

# Compute entropy for each domain individually
domains = {
    'A (programming)': vecs_prog,
    'B (physics)': vecs_phys,
    'C (geography)': vecs_geo,
}

entropy_results = {}
for name, vecs in domains.items():
    pv = compute_h1_pers_values(vecs, topo_engine)
    if pv:
        ent = entropy_of_persistence(pv)
        gini = gini_coefficient(pv)
        entropy_results[name] = {
            'entropy': ent,
            'gini': gini,
            'n_cycles': len(pv),
            'mean_pers': np.mean(pv),
            'std_pers': np.std(pv),
            'pers_values': pv,
        }
    else:
        entropy_results[name] = {'entropy': 0, 'gini': 0, 'n_cycles': 0}

print(f"\n{'Domain':>20} {'Cycles':>8} {'Entropy':>10} {'Gini':>8} {'Mean Pers':>10}")
print("-" * 60)
for name, r in entropy_results.items():
    print(f"{name:>20} {r['n_cycles']:>8} {r['entropy']:>10.4f} {r['gini']:>8.4f} {r['mean_pers']:>10.4f}")

# ============================================================
# TEST 3: Cross-domain mixing - how does A's entropy change?
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Cross-Domain Mixing - A's Entropy Under Invasion")
print("=" * 60)

# Phase A only
pv_A = compute_h1_pers_values(vecs_prog, topo_engine)
ent_A = entropy_of_persistence(pv_A)
gini_A = gini_coefficient(pv_A)

# Phase A+B
vecs_AB = np.vstack([vecs_prog, vecs_phys])
pv_AB = compute_h1_pers_values(vecs_AB, topo_engine)
ent_AB = entropy_of_persistence(pv_AB)
gini_AB = gini_coefficient(pv_AB)

# Phase A+B+C
vecs_ABC = np.vstack([vecs_prog, vecs_phys, vecs_geo])
pv_ABC = compute_h1_pers_values(vecs_ABC, topo_engine)
ent_ABC = entropy_of_persistence(pv_ABC)
gini_ABC = gini_coefficient(pv_ABC)

print(f"\n{'Phase':>12} {'Cycles':>8} {'Entropy':>10} {'dH':>10} {'Gini':>8} {'dG':>8}")
print("-" * 62)
print(f"{'A only':>12} {len(pv_A):>8} {ent_A:>10.4f} {'--':>10} {gini_A:>8.4f} {'--':>8}")
print(f"{'A+B':>12} {len(pv_AB):>8} {ent_AB:>10.4f} {ent_AB-ent_A:>+10.4f} {gini_AB:>8.4f} {gini_AB-gini_A:>+8.4f}")
print(f"{'A+B+C':>12} {len(pv_ABC):>8} {ent_ABC:>10.4f} {ent_ABC-ent_A:>+10.4f} {gini_ABC:>8.4f} {gini_ABC-gini_A:>+8.4f}")

# ============================================================
# TEST 4: Separation - are domains topologically distinct?
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Domain Topological Separation")
print("=" * 60)

# Mix A and B in ratio, measure how many A-cycles survive
def count_A_cycles_after_invasion(vecs_A, vecs_B, mix_ratio, topo_engine):
    """Count how many A-pure cycles exist after B invasion."""
    n_A = len(vecs_A)
    n_B = int(n_A * mix_ratio)
    vecs_mixed = np.vstack([vecs_A, vecs_B[:n_B]])
    
    # Compute cycles on mixed cloud
    mixed_pv = compute_h1_pers_values(vecs_mixed, topo_engine)
    return len(mixed_pv)

pv_phys = compute_h1_pers_values(vecs_phys, topo_engine)

print(f"\n{'Mix Ratio (B/A)':>16} {'Mixed Cycles':>14} {'A+B (total)':>14} {'Cycle Ratio':>14}")
print("-" * 62)

for ratio in [0.0, 0.25, 0.5, 1.0, 2.0]:
    if ratio == 0.0:
        mixed_cycles = len(pv_A)
        total_expected = len(pv_A) + len(pv_phys)
    else:
        n_B = int(len(vecs_prog) * ratio)
        vecs_mixed = np.vstack([vecs_prog, vecs_phys[:n_B]])
        mixed_cycles = len(compute_h1_pers_values(vecs_mixed, topo_engine))
        total_expected = len(pv_A) + len(pv_phys)
    
    cycle_ratio = mixed_cycles / max(len(pv_A), 1)
    print(f"{ratio:>16.2f} {mixed_cycles:>14} {total_expected:>14} {cycle_ratio:>14.3f}")

# ============================================================
# KEY FINDINGS
# ============================================================
print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

# Finding 1: Growth type
print(f"\n1. Betti-1 Growth Exponent: {gamma:.3f} ({growth_type})")
if growth_type == "SUBLINEAR":
    print("   [NEW] Sub-linear growth confirms shared cycle structure")
    print("   -> Multiple nodes share the same H1 cycles (not independent)")
else:
    print("   [NEGATIVE] Near-linear/super growth suggests independent cycles")

# Finding 2: Entropy change
ent_change_AB = ent_AB - ent_A
ent_change_ABC = ent_ABC - ent_A
print(f"\n2. Entropy Change Under Invasion:")
print(f"   A->A+B: {ent_change_AB:+.4f}")
print(f"   A->A+B+C: {ent_change_ABC:+.4f}")

if abs(ent_change_AB) < 0.1:
    print("   [ROBUST] A's topological structure is STABLE under B invasion")
    ent_verdict = "ROBUST"
elif ent_change_AB > 0.1:
    print("   [CHAOS] A's structure became MORE random under B invasion")
    ent_verdict = "ENTROPY_RISE"
else:
    print("   [ORDER] A's structure became MORE structured under B invasion")
    ent_verdict = "ENTROPY_FALL"

# Finding 3: Cross-domain entropy comparison
all_ents = {k: v['entropy'] for k, v in entropy_results.items()}
if max(all_ents.values()) - min(all_ents.values()) < 0.3:
    print(f"\n3. Cross-domain entropy variation: SMALL ({max(all_ents.values()) - min(all_ents.values()):.4f})")
    print("   [NEGATIVE] All domains have similar persistence entropy distributions")
    print("   -> H1 entropy does NOT distinguish semantic domains")
    sep_verdict = "ENTROPY_INDISTINGUISHABLE"
else:
    print(f"\n3. Cross-domain entropy variation: LARGE")
    print("   [POTENTIAL] H1 entropy CAN distinguish domains")
    sep_verdict = "ENTROPY_DISTINGUISHABLE"

# Finding 4: Gini comparison
all_ginis = {k: v['gini'] for k, v in entropy_results.items()}
print(f"\n4. Gini Coefficient (inequality of persistence):")
for name, g in all_ginis.items():
    print(f"   {name}: {g:.4f}")

# Overall verdict
print(f"\n" + "=" * 60)
print("OVERALL VERDICT")
print("=" * 60)

if growth_type == "SUBLINEAR" and ent_verdict == "ROBUST":
    overall = "H1 STRUCTURE IS MEANINGFUL: sub-linear growth + entropy stability"
elif sep_verdict == "ENTROPY_DISTINGUISHABLE":
    overall = "H1 CAN DISTINGUISH DOMAINS via entropy"
else:
    overall = f"H1 Shape: growth={growth_type}, entropy={ent_verdict}, sep={sep_verdict}"

print(f"\n{overall}")

# Save results
results_dir = os.path.join(PKG_DIR, 'benchmarks', 'results')
os.makedirs(results_dir, exist_ok=True)
import time
result_file = os.path.join(results_dir, f'h1_shape_metrics_{int(time.time())}.json')

save_data = {
    'test': 'H1 Shape Metrics',
    'growth_exponent': gamma,
    'growth_type': growth_type,
    'entropy_results': {k: {a: float(v[a]) for a in ['entropy', 'gini', 'n_cycles', 'mean_pers', 'std_pers']} 
                        for k, v in entropy_results.items()},
    'entropy_change_AB': float(ent_change_AB),
    'entropy_change_ABC': float(ent_change_ABC),
    'entropy_verdict': ent_verdict,
    'separation_verdict': sep_verdict,
    'overall': overall,
    'betti1_growth': betti1_growth,
}
with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: {result_file}")
