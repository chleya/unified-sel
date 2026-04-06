#!/usr/bin/env python3
"""
P0: Cross-Domain Gap Detection

Core question: Is there a "semantic cliff" between programming and physics domains?

If YES (bimodal): H1 resolution is sufficient, domains are semantically isolated
If NO (unimodal): H1 is a threshold artifact, embedding space is smooth

Method: Examine the cosine similarity distribution between domain pairs.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# F:\unified-sel\topomem\benchmarks -> F:\unified-sel\topomem -> F:\unified-sel
TOPOMEM_DIR = os.path.dirname(SCRIPT_DIR)  # F:\unified-sel\topomem
PROJECT_DIR = os.path.dirname(TOPOMEM_DIR)  # F:\unified-sel
sys.path.insert(0, PROJECT_DIR)

from topomem.embedding import EmbeddingManager
from topomem.config import EmbeddingConfig

def load_corpus(domain):
    path = os.path.join(TOPOMEM_DIR, 'data', 'test_corpus', f'{domain}.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [item['content'] for item in data]

print("=" * 60)
print("P0: Cross-Domain Gap Detection")
print("=" * 60)

# Load data
prog = load_corpus('programming')
phys = load_corpus('physics')
geo = load_corpus('geography')

emb = EmbeddingManager(EmbeddingConfig())
vecs_prog = emb.encode_batch(prog)
vecs_phys = emb.encode_batch(phys)
vecs_geo = emb.encode_batch(geo)

# Normalize
vecs_prog = vecs_prog / np.linalg.norm(vecs_prog, axis=1, keepdims=True)
vecs_phys = vecs_phys / np.linalg.norm(vecs_phys, axis=1, keepdims=True)
vecs_geo = vecs_geo / np.linalg.norm(vecs_geo, axis=1, keepdims=True)

# ---- Core metric: pairwise cosine similarity ----
def cosine_matrix(a, b):
    return np.dot(a, b.T)

# All domain pairs
prog_phys = cosine_matrix(vecs_prog, vecs_phys)  # 20x20
prog_geo = cosine_matrix(vecs_prog, vecs_geo)   # 20x10
phys_geo = cosine_matrix(vecs_phys, vecs_geo)   # 20x10

# Within-domain
prog_within = []
for i in range(len(vecs_prog)):
    for j in range(i+1, len(vecs_prog)):
        prog_within.append(np.dot(vecs_prog[i], vecs_prog[j]))

phys_within = []
for i in range(len(vecs_phys)):
    for j in range(i+1, len(vecs_phys)):
        phys_within.append(np.dot(vecs_phys[i], vecs_phys[j]))

prog_within = np.array(prog_within)
phys_within = np.array(phys_within)
prog_phys_flat = prog_phys.flatten()
prog_geo_flat = prog_geo.flatten()
phys_geo_flat = phys_geo.flatten()

# ---- Basic statistics ----
print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(f"\n{'Metric':<25} {'Within-A':<12} {'Within-B':<12} {'Cross':<12}")
print("-" * 65)
print(f"{'Mean':<25} {np.mean(prog_within):.4f}     {np.mean(phys_within):.4f}     {np.mean(prog_phys_flat):.4f}")
print(f"{'Std':<25} {np.std(prog_within):.4f}     {np.std(phys_within):.4f}     {np.std(prog_phys_flat):.4f}")
print(f"{'Min':<25} {np.min(prog_within):.4f}     {np.min(phys_within):.4f}     {np.min(prog_phys_flat):.4f}")
print(f"{'Max':<25} {np.max(prog_within):.4f}     {np.max(phys_within):.4f}     {np.max(prog_phys_flat):.4f}")
print(f"{'Median':<25} {np.median(prog_within):.4f}     {np.median(phys_within):.4f}     {np.median(prog_phys_flat):.4f}")

# ---- Gap detection in 0.0-0.4 range ----
print("\n" + "=" * 60)
print("GAP ANALYSIS: Programming vs Physics")
print("=" * 60)

low = 0.0
high = 0.5
n_bins = 25
bin_width = (high - low) / n_bins

# Build histogram
bins = np.linspace(low, high, n_bins + 1)
hist_cross, edges = np.histogram(prog_phys_flat, bins=bins)
hist_within_prog, _ = np.histogram(prog_within, bins=bins)
hist_within_phys, _ = np.histogram(phys_within, bins=bins)

print(f"\nHistogram of cross-domain similarity (prog vs phys, 0.0-0.5):")
print(f"\n{'Bin range':<15} {'Cross':<10} {'Within-A':<10} {'Within-B':<10}")
print("-" * 50)
for i in range(n_bins):
    lo, hi = edges[i], edges[i+1]
    if hist_cross[i] > 0 or i < 5 or i >= n_bins - 3:
        print(f"[{lo:.2f}, {hi:.2f})  {hist_cross[i]:<10} {hist_within_prog[i]:<10} {hist_within_phys[i]:<10}")
    elif i == 5:
        print("... (middle bins with no cross-domain hits) ...")

# ---- Gap detection: find empty bins in cross-domain ----
cross_only = prog_phys_flat

# Look for gaps in 0.0 to max_cross range
max_cross = np.max(cross_only)
print(f"\nMax cross-domain similarity: {max_cross:.4f}")

# Check each 0.05-wide bin for emptiness
print(f"\nGap detection (0.0 to {max_cross:.2f}, bin_width=0.05):")
print(f"\n{'Bin':<15} {'Cross count':<15} {'Gap?'}")
print("-" * 40)

gaps = []
for i in range(int(max_cross / 0.05) + 1):
    lo = i * 0.05
    hi = (i + 1) * 0.05
    count = np.sum((cross_only >= lo) & (cross_only < hi))
    is_gap = count == 0
    if is_gap:
        gaps.append((lo, hi))
    if count > 0 or is_gap or lo < 0.2:
        print(f"[{lo:.2f}, {hi:.2f})  {count:<15} {'*** GAP ***' if is_gap else ''}")

# ---- Bimodality test ----
print("\n" + "=" * 60)
print("BIMODALITY TEST")
print("=" * 60)

# Hartigan's dip test approximation: check if distribution is unimodal
# Simple heuristic: count local maxima in histogram
hist = hist_cross[:int(max_cross / bin_width) + 1]
local_maxima = []
for i in range(1, len(hist) - 1):
    if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0:
        local_maxima.append((i * bin_width, hist[i]))

print(f"\nLocal maxima in cross-domain histogram: {local_maxima}")
print(f"Number of peaks: {len(local_maxima)}")

# Statistical test: bimodality coefficient
def bimodality_coefficient(data):
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    skew = np.mean((data - mean) ** 3) / (std ** 3 + 1e-10)
    kurt = np.mean((data - mean) ** 4) / (std ** 4 + 1e-10) - 3
    bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) + 1e-10)
    return bc

bc_cross = bimodality_coefficient(cross_only)
bc_prog = bimodality_coefficient(prog_within)
bc_phys = bimodality_coefficient(phys_within)

print(f"\nBimodality Coefficient (BC > 0.555 suggests bimodal):")
print(f"  Within-A (prog): {bc_prog:.4f}")
print(f"  Within-B (phys): {bc_phys:.4f}")
print(f"  Cross (A vs B):  {bc_cross:.4f}")

# ---- Gap conclusion ----
print("\n" + "=" * 60)
print("GAP VERDICT")
print("=" * 60)

within_a_min = np.min(prog_within)
within_b_min = np.min(phys_within)
cross_max = np.max(cross_only)

print(f"\nWithin-A (prog) min similarity: {within_a_min:.4f}")
print(f"Within-B (phys) min similarity: {within_b_min:.4f}")
print(f"Cross-domain max similarity:     {cross_max:.4f}")

# The gap = the region where within-domain and cross-domain don't overlap
gap_exists = cross_max < min(within_a_min, within_b_min)

print(f"\n{'=' * 50}")
if gap_exists:
    gap_size = min(within_a_min, within_b_min) - cross_max
    print(f"GAP EXISTS: size = {gap_size:.4f}")
    print(f"  Within-domain: [{min(within_a_min, within_b_min):.3f}, {max(np.max(prog_within), np.max(phys_within)):.3f}]")
    print(f"  Cross-domain:  [0.000, {cross_max:.3f}]")
    print(f"\n  => SEMANTIC CLIFF exists!")
    print(f"  => Domains are completely separated in embedding space")
    print(f"  => H1 resolution IS sufficient")
    print(f"  => H1 cycles reflect genuine domain structure")
    verdict = "GAP_EXISTS"
else:
    overlap = min(within_a_min, within_b_min) - cross_max
    print(f"NO GAP: overlap = {overlap:.4f}")
    print(f"  Within-domain and cross-domain overlap by {overlap:.4f}")
    print(f"\n  => SMOOTH TRANSITION between domains")
    print(f"  => VR filtration's threshold is NOT aligned with semantic boundary")
    print(f"  => H1 cycles may be a threshold artifact")
    print(f"  => Need to check: which threshold captures the semantic boundary?")
    verdict = "NO_GAP"

# ---- Find optimal threshold for semantic separation ----
print("\n" + "=" * 60)
print("OPTIMAL SEPARATION THRESHOLD")
print("=" * 60)

# Try different thresholds and compute "domain separation score"
print(f"\n{'Threshold':<12} {'Cross>A':<12} {'Cross>B':<12} {'Separation':<15}")
print("-" * 55)

best_thresh = 0.0
best_score = -999

for t in np.arange(0.05, 0.95, 0.05):
    cross_above_t = np.mean(cross_only >= t)
    # How much of within-domain is above threshold
    within_a_above = np.mean(prog_within >= t)
    within_b_above = np.mean(phys_within >= t)
    # Separation: cross is low, within is high
    separation = (within_a_above + within_b_above) / 2 - cross_above_t
    if separation > best_score:
        best_score = separation
        best_thresh = t
    if t <= 0.5 or t >= 0.7:
        print(f"{t:.2f}         {cross_above_t:.4f}       {within_a_above:.4f}       {separation:.4f}")

print(f"\nBest threshold for semantic separation: {best_thresh:.2f}")
print(f"Best separation score: {best_score:.4f}")

# Compare to VR filtration threshold
vr_thresh = 0.3
cross_above_vr = np.mean(cross_only >= vr_thresh)
within_above_vr_a = np.mean(prog_within >= vr_thresh)
within_above_vr_b = np.mean(phys_within >= vr_thresh)
vr_separation = (within_above_vr_a + within_above_vr_b) / 2 - cross_above_vr

print(f"\nVR filtration threshold ({vr_thresh:.2f}):")
print(f"  Cross above threshold: {cross_above_vr:.4f} ({cross_above_vr * 100:.1f}%)")
print(f"  Within-A above:        {within_above_vr_a:.4f} ({within_above_vr_a * 100:.1f}%)")
print(f"  Within-B above:        {within_above_vr_b:.4f} ({within_above_vr_b * 100:.1f}%)")
print(f"  Separation score:      {vr_separation:.4f}")
print(f"  Gap exists at VR threshold: {cross_above_vr == 0 and (within_above_vr_a > 0 or within_above_vr_b > 0)}")

# ---- Similarity distribution details ----
print("\n" + "=" * 60)
print("SIMILARITY DISTRIBUTION (all pairs)")
print("=" * 60)

all_sims = {
    'Within Prog (A)': sorted(prog_within),
    'Within Phys (B)': sorted(phys_within),
    'Prog vs Phys (A-B)': sorted(prog_phys_flat),
    'Prog vs Geo': sorted(prog_geo_flat),
    'Phys vs Geo': sorted(phys_geo_flat),
}

print(f"\n{'Domain pair':<22} {'Min':>8} {'Q25':>8} {'Median':>8} {'Q75':>8} {'Max':>8}")
print("-" * 70)
for name, vals in all_sims.items():
    vals = np.array(vals)
    print(f"{name:<22} {np.min(vals):8.4f} {np.percentile(vals,25):8.4f} {np.median(vals):8.4f} {np.percentile(vals,75):8.4f} {np.max(vals):8.4f}")

# ---- Overall H1 implication ----
print("\n" + "=" * 60)
print("H1 IMPLICATION")
print("=" * 60)

print(f"\nVerdict: {verdict}")
print(f"\nCross-domain max similarity: {cross_max:.4f}")
print(f"Within-domain min similarity: {min(within_a_min, within_b_min):.4f}")

if verdict == "GAP_EXISTS":
    print(f"\n>>> H1's VR threshold ({vr_thresh:.2f}) is WELL CALIBRATED")
    print(f">>> H1 cycles at threshold={vr_thresh:.2f} capture GENUINE domain boundaries")
    print(f">>> H1 is NOT a threshold artifact")
    print(f">>> H1 resolution IS sufficient for this embedding space")
    print(f"\n>>> RECOMMENDATION: H1 is worth pursuing")
    print(f">>> Focus on H1 health metrics and consolidation decisions")
else:
    gap_size_pct = (min(within_a_min, within_b_min) - cross_max) / (min(within_a_min, within_b_min)) * 100
    print(f"\n>>> H1's VR threshold ({vr_thresh:.2f}) is MISALIGNED")
    print(f">>> Gap size: {gap_size_pct:.1f}% of within-domain similarity range")
    print(f">>> H1 cycles at threshold={vr_thresh:.2f} may include spurious cross-domain connections")
    print(f">>> Optimal threshold for semantic separation: {best_thresh:.2f}")
    print(f"\n>>> RECOMMENDATION: H1 as 'CT scan' needs CAUTION")
    print(f">>> The embedding space has smooth transitions between domains")
    print(f">>> H1 captures geometric structure, not semantic boundaries")
    print(f">>> Consider using optimal threshold={best_thresh:.2f} instead of VR threshold={vr_thresh:.2f}")

# Save results
import time
results_dir = os.path.join(TOPOMEM_DIR, 'benchmarks', 'results')
os.makedirs(results_dir, exist_ok=True)
result_file = os.path.join(results_dir, f'gap_detection_{int(time.time())}.json')

save_data = {
    'verdict': verdict,
    'cross_max': float(cross_max),
    'within_a_min': float(within_a_min),
    'within_b_min': float(within_b_min),
    'vr_threshold': float(vr_thresh),
    'gap_exists_at_vr': bool(cross_above_vr == 0),
    'optimal_threshold': float(best_thresh),
    'best_separation_score': float(best_score),
    'gaps_found': [(float(lo), float(hi)) for lo, hi in gaps],
    'bimodality_coefficient': float(bc_cross),
    'all_pairs': {
        'within_prog': {k: float(v) for k, v in [
            ('min', np.min(prog_within)), ('median', np.median(prog_within)),
            ('max', np.max(prog_within)), ('std', np.std(prog_within))
        ]},
        'within_phys': {k: float(v) for k, v in [
            ('min', np.min(phys_within)), ('median', np.median(phys_within)),
            ('max', np.max(phys_within)), ('std', np.std(phys_within))
        ]},
        'prog_phys': {k: float(v) for k, v in [
            ('min', np.min(prog_phys_flat)), ('median', np.median(prog_phys_flat)),
            ('max', np.max(prog_phys_flat)), ('std', np.std(prog_phys_flat))
        ]},
    }
}

with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)

print(f"\nResults saved: {result_file}")
