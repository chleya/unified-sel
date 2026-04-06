#!/usr/bin/env python3
"""
P1: H1 Health Metrics Dashboard

Tracks H1 as an "embedding space CT scan" - detects geometric changes
in the embedding space, not semantic content.

Key metrics:
1. Betti-1 count - number of cycles = "loops" in the embedding manifold
2. Persistence entropy - distribution shape of cycle lifetimes
3. Persistence variance (Claude's key) - stability indicator
4. Top-K mean persistence - strength of structural cycles
5. H1 growth rate (gamma) - how fast structure forms as data grows
6. Wasserstein distance to baseline - topological drift
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json, time
from scipy.stats import wasserstein_distance

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

def persistence_entropy(values):
    """Shannon entropy of persistence distribution."""
    if len(values) == 0:
        return 0.0
    total = np.sum(values)
    if total == 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def gini_coefficient(values):
    """Gini coefficient of persistence distribution."""
    if len(values) <= 1:
        return 0.0
    sorted_vals = np.sort(np.abs(values))
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)

def compute_h1_metrics(vecs, topo_engine):
    """Compute H1 health metrics from normalized embedding vectors."""
    n = len(vecs)
    result = topo_engine.compute_persistence(vecs)

    # Extract H1 pairs (dimension 1)
    h1_pairs = []
    if len(result) > 1 and len(result[1]) > 0:
        for pair in result[1]:
            birth, death = float(pair[0]), float(pair[1])
            if death > birth:  # Valid cycle
                h1_pairs.append((birth, death))

    if len(h1_pairs) == 0:
        return {
            'betti1': 0, 'mean_pers': 0.0, 'persistence_entropy': 0.0,
            'persistence_variance': 0.0, 'gini': 0.0,
            'top3_mean': 0.0, 'top5_mean': 0.0,
            'min_pers': 0.0, 'max_pers': 0.0,
            'std_pers': 0.0, 'q1_pers': 0.0, 'q3_pers': 0.0,
            'h1_pairs': []
        }

    pers_values = np.array([p[1] - p[0] for p in h1_pairs])
    births = np.array([p[0] for p in h1_pairs])

    sorted_pers = np.sort(pers_values)
    return {
        'betti1': len(h1_pairs),
        'mean_pers': np.mean(pers_values),
        'persistence_entropy': persistence_entropy(pers_values),
        'persistence_variance': np.var(pers_values),
        'gini': gini_coefficient(pers_values),
        'top3_mean': np.mean(sorted_pers[-3:]) if len(sorted_pers) >= 3 else np.mean(sorted_pers),
        'top5_mean': np.mean(sorted_pers[-5:]) if len(sorted_pers) >= 5 else np.mean(sorted_pers),
        'min_pers': np.min(pers_values),
        'max_pers': np.max(pers_values),
        'std_pers': np.std(pers_values),
        'q1_pers': np.percentile(pers_values, 25),
        'q3_pers': np.percentile(pers_values, 75),
        'h1_pairs': [(float(b), float(d)) for b, d in h1_pairs]
    }

def betti1_growth_curve(all_vecs, topo_engine, max_n=25):
    """Compute Betti-1 count as a function of n points."""
    n = min(max_n, len(all_vecs))
    bettis = []
    for k in range(1, n + 1):
        # Subsample k points
        indices = np.linspace(0, len(all_vecs)-1, k, dtype=int)
        sub_vecs = all_vecs[indices]
        m = compute_h1_metrics(sub_vecs, topo_engine)
        bettis.append(m['betti1'])
    return bettis

def fit_gamma(betti_curve):
    """Fit power-law gamma to Betti-1 growth curve: B(n) ~ n^gamma."""
    # Only fit from points where Betti > 0
    xs = []
    ys = []
    for i, b in enumerate(betti_curve):
        if b > 0:
            xs.append(i + 1)
            ys.append(b)
    if len(xs) < 3:
        return 0.0
    log_x = np.log(np.array(xs, dtype=float))
    log_y = np.log(np.array(ys, dtype=float))
    gamma, _ = np.polyfit(log_x, log_y, 1)
    return gamma

def wasserstein_to_baseline(current_pairs, baseline_pairs):
    """Wasserstein distance between two H1 persistence diagrams."""
    if len(current_pairs) == 0 and len(baseline_pairs) == 0:
        return 0.0
    if len(current_pairs) == 0:
        current_pairs = [(0, 0)]
    if len(baseline_pairs) == 0:
        baseline_pairs = [(0, 0)]
    current_pers = np.array([d - b for b, d in current_pairs])
    baseline_pers = np.array([d - b for b, d in baseline_pairs])
    if len(current_pers) == 0:
        current_pers = np.array([0.0])
    if len(baseline_pers) == 0:
        baseline_pers = np.array([0.0])
    return wasserstein_distance(current_pers, baseline_pers)

print("=" * 60)
print("P1: H1 Health Metrics Dashboard")
print("=" * 60)

# Load data
print("\n[1/6] Loading corpus data...")
prog = load_corpus('programming')
phys = load_corpus('physics')
geo = load_corpus('geography')

print(f"  Programming: {len(prog)} items")
print(f"  Physics: {len(phys)} items")
print(f"  Geography: {len(geo)} items")

# Encode
print("\n[2/6] Encoding embeddings...")
emb = EmbeddingManager(EmbeddingConfig())
vecs_prog = normalize(emb.encode_batch(prog))
vecs_phys = normalize(emb.encode_batch(phys))
vecs_geo = normalize(emb.encode_batch(geo))

print(f"  Prog: {vecs_prog.shape}, range [{vecs_prog.min():.3f}, {vecs_prog.max():.3f}]")
print(f"  Phys: {vecs_phys.shape}, range [{vecs_phys.min():.3f}, {vecs_phys.max():.3f}]")
print(f"  Geo: {vecs_geo.shape}, range [{vecs_geo.min():.3f}, {vecs_geo.max():.3f}]")

# Init topology
topo = TopologyEngine(TopologyConfig())

results = {}

# ===== SCENARIO 1: Phase-by-phase accumulation (simulates forgetting test) =====
print("\n[3/6] Scenario 1: Phase accumulation (A → A+B → A+B+C)...")
phases = [
    ('A_only (prog)', vecs_prog),
    ('A+B (prog+phys)', np.vstack([vecs_prog, vecs_phys])),
    ('A+B+C (all)', np.vstack([vecs_prog, vecs_phys, vecs_geo])),
]

phase_results = {}
baseline_pairs = None

for phase_name, vecs in phases:
    print(f"\n  --- {phase_name} ({len(vecs)} nodes) ---")
    m = compute_h1_metrics(vecs, topo)

    # Betti growth curve
    growth = betti1_growth_curve(vecs, topo)
    gamma = fit_gamma(growth)

    # Wasserstein to baseline
    w_dist = wasserstein_to_baseline(m['h1_pairs'], baseline_pairs if baseline_pairs else m['h1_pairs'])

    print(f"  Betti-1: {m['betti1']} cycles")
    print(f"  Gamma: {gamma:.3f}")
    print(f"  Mean persistence: {m['mean_pers']:.4f}")
    print(f"  Persistence variance: {m['persistence_variance']:.6f}")
    print(f"  Entropy: {m['persistence_entropy']:.4f}")
    print(f"  Gini: {m['gini']:.4f}")
    print(f"  Top-3 persistence: {m['top3_mean']:.4f}")
    print(f"  Top-5 persistence: {m['top5_mean']:.4f}")
    print(f"  Wasserstein to baseline: {w_dist:.6f}")

    phase_results[phase_name] = {
        'n': len(vecs), 'betti1': m['betti1'], 'gamma': gamma,
        'mean_pers': m['mean_pers'], 'pers_variance': m['persistence_variance'],
        'entropy': m['persistence_entropy'], 'gini': m['gini'],
        'top3': m['top3_mean'], 'top5': m['top5_mean'],
        'w_dist': w_dist,
        'growth_curve': growth
    }

    if baseline_pairs is None:
        baseline_pairs = m['h1_pairs']

results['phase_accumulation'] = phase_results

# ===== SCENARIO 2: Incremental node addition (what happens as A grows) =====
print("\n[4/6] Scenario 2: Incremental growth of A (5 → 25 nodes)...")
incremental_curves = {}
for n_points in [5, 8, 10, 13, 15, 17, 19, 20]:
    if n_points > len(vecs_prog):
        continue
    vecs_n = vecs_prog[:n_points]
    m = compute_h1_metrics(vecs_n, topo)
    growth = betti1_growth_curve(vecs_prog[:n_points], topo)
    gamma = fit_gamma(growth) if any(b > 0 for b in growth) else 0.0
    incremental_curves[f'n{n_points}'] = {
        'betti1': m['betti1'],
        'gamma': gamma,
        'mean_pers': m['mean_pers'],
        'pers_variance': m['persistence_variance'],
        'entropy': m['persistence_entropy'],
        'growth_curve': growth
    }
    print(f"  n={n_points:2d}: Betti={m['betti1']}, gamma={gamma:.3f}, pers_var={m['persistence_variance']:.6f}")

results['incremental_growth'] = incremental_curves

# ===== SCENARIO 3: Domain interference test (A stability after B/C invasion) =====
print("\n[5/6] Scenario 3: Domain interference (A stability after B,C invasion)...")

# Baseline: A only
m_A = compute_h1_metrics(vecs_prog, topo)
baseline_A_pairs = m_A['h1_pairs']

# After B invasion: measure A's H1 in the combined space
m_AB = compute_h1_metrics(np.vstack([vecs_prog, vecs_phys]), topo)
m_ABC = compute_h1_metrics(np.vstack([vecs_prog, vecs_phys, vecs_geo]), topo)

# Extract A's contribution from combined space - use A-only points
m_A_in_AB = compute_h1_metrics(vecs_prog, topo)  # Same as m_A

# How much did A's H1 change?
a_stability = {
    'betti1_baseline': m_A['betti1'],
    'betti1_after_B': m_AB['betti1'],
    'betti1_after_C': m_ABC['betti1'],
    'pers_var_baseline': m_A['persistence_variance'],
    'pers_var_after_B': m_AB['persistence_variance'],
    'pers_var_after_C': m_ABC['persistence_variance'],
    'entropy_baseline': m_A['persistence_entropy'],
    'entropy_after_B': m_AB['persistence_entropy'],
    'entropy_after_C': m_ABC['persistence_entropy'],
    'w_AB': wasserstein_to_baseline(m_AB['h1_pairs'], m_A['h1_pairs']),
    'w_ABC': wasserstein_to_baseline(m_ABC['h1_pairs'], m_A['h1_pairs']),
}

print(f"  Baseline (A only): Betti={a_stability['betti1_baseline']}, pers_var={a_stability['pers_var_baseline']:.6f}")
print(f"  After B invasion:  Betti={a_stability['betti1_after_B']}, pers_var={a_stability['pers_var_after_B']:.6f}")
print(f"  After C invasion:  Betti={a_stability['betti1_after_C']}, pers_var={a_stability['pers_var_after_C']:.6f}")
print(f"  W(A→A+B):  {a_stability['w_AB']:.6f}")
print(f"  W(A→A+B+C): {a_stability['w_ABC']:.6f}")

results['domain_interference'] = a_stability

# ===== SCENARIO 4: Health verdict =====
print("\n[6/6] Health Verdict...")

verdict = {
    'geometric_stability': 'STABLE' if phase_results['A+B (prog+phys)']['w_dist'] < 1.0 else 'UNSTABLE',
    'interference_resistance': 'RESISTANT' if a_stability['betti1_after_B'] >= a_stability['betti1_baseline'] * 0.5 else 'FRAGILE',
    'structural_complexity': 'HIGH' if phase_results['A+B+C (all)']['betti1'] > 20 else 'LOW',
    'persistence_stability': 'STABLE' if abs(a_stability['pers_var_after_B'] - a_stability['pers_var_baseline']) < 0.01 else 'DRIFTING',
}

print(f"\n{'=' * 50}")
print("HEALTH VERDICT")
print(f"{'=' * 50}")
for key, val in verdict.items():
    emoji = {'STABLE': '✅', 'RESISTANT': '✅', 'HIGH': '📈', 'LOW': '📉', 'FRAGILE': '⚠️', 'UNSTABLE': '❌', 'DRIFTING': '🔄'}.get(val, '❓')
    print(f"  {emoji} {key}: {val}")

# Summary table
print(f"\n{'=' * 50}")
print("SUMMARY TABLE")
print(f"{'=' * 50}")
print(f"\n{'Phase':<25} {'Betti1':<8} {'Gamma':<8} {'Entropy':<8} {'Gini':<8} {'W_dist':<10}")
print("-" * 75)
for name, r in phase_results.items():
    print(f"{name:<25} {r['betti1']:<8} {r['gamma']:<8.3f} {r['entropy']:<8.4f} {r['gini']:<8.4f} {r['w_dist']:<10.6f}")

print(f"\n{'Incremental n':<15} {'Betti1':<8} {'Gamma':<8} {'Pers_var':<12} {'Entropy':<8}")
print("-" * 60)
for name, r in incremental_curves.items():
    print(f"{name:<15} {r['betti1']:<8} {r['gamma']:<8.3f} {r['pers_variance']:<12.6f} {r['entropy']:<8.4f}")

# Save results
results_dir = os.path.join(TOPOMEM_DIR, 'benchmarks', 'results')
os.makedirs(results_dir, exist_ok=True)
ts = int(time.time())
result_file = os.path.join(results_dir, f'h1_health_dashboard_{ts}.json')

with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved: {result_file}")
print(f"\nKey metrics for 'H1 as CT scan':")
print(f"  - Wasserstein distance A→A+B: {a_stability['w_AB']:.6f}")
print(f"  - Wasserstein distance A→A+B+C: {a_stability['w_ABC']:.6f}")
print(f"  - Persistence variance shift (A→A+B): {abs(a_stability['pers_var_after_B'] - a_stability['pers_var_baseline']):.6f}")
print(f"  - H1 stability ratio: {a_stability['betti1_after_B'] / max(a_stability['betti1_baseline'], 1):.3f}")
