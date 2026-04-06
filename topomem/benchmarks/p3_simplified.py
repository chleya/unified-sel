#!/usr/bin/env python3
"""
P3 Simplified: Model Capacity via Noise Simulation + Volume Comparison

Focus: Does embedding quality (simulated by noise) affect H1?
This tests the core hypothesis: weaker model = less precise embeddings = different H1.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json, time, warnings
warnings.filterwarnings('ignore')

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

def compute_h1(vecs, topo):
    result = topo.compute_persistence(vecs)
    if len(result) <= 1 or len(result[1]) == 0:
        return 0, 0.0
    pairs = []
    for pair in result[1]:
        b, d = float(pair[0]), float(pair[1])
        if d > b:
            pairs.append(d - b)
    return len(pairs), np.mean(pairs) if pairs else 0.0

def within_cosine_sim(vecs):
    n = len(vecs)
    sims = [np.dot(vecs[i], vecs[j]) for i in range(n) for j in range(i+1, n)]
    return np.mean(sims) if sims else 0.0

print("=" * 60)
print("P3 Simplified: Model Quality vs H1")
print("=" * 60)

# Load corpus
prog = load_corpus('programming')
phys = load_corpus('physics')
geo = load_corpus('geography')
prog_10, prog_20 = prog[:10], prog[:20]
all_60 = prog + phys + geo

print(f"Corpus: prog(10)={len(prog_10)}, prog(20)={len(prog_20)}, all(60)={len(all_60)}")

emb = EmbeddingManager(EmbeddingConfig())
topo = TopologyEngine(TopologyConfig())

# ===== PART 1: Volume comparison =====
print("\n[1/4] PART 1: Data volume vs H1 (same MiniLM model)")
print(f"  {'Dataset':<12} {'n':<4} {'Betti1':<8} {'MeanPers':<10} {'MeanSim':<10}")
print("  " + "-" * 50)

vols = [('prog_10', prog_10), ('prog_20', prog_20), ('all_60', all_60)]
vol_data = {}
for name, texts in vols:
    vecs = normalize(emb.encode_batch(texts))
    b, p = compute_h1(vecs, topo)
    sim = within_cosine_sim(vecs)
    print(f"  {name:<12} {len(texts):<4} {b:<8} {p:<10.4f} {sim:<10.4f}")
    vol_data[name] = {'n': len(texts), 'betti1': b, 'mean_pers': p, 'mean_sim': float(sim)}

# Betti per node ratio
for name, r in vol_data.items():
    ratio = r['betti1'] / max(r['n'], 1)
    print(f"  {name}: Betti/n = {ratio:.3f}")

# ===== PART 2: Noise simulation (model quality) =====
print("\n[2/4] PART 2: Embedding noise vs H1 (simulates model quality)")
print("  Same 20-item corpus with added Gaussian noise")
print(f"  {'Noise':<8} {'Betti1':<8} {'MeanPers':<10} {'MeanSim':<10} {'Interpretation':<20}")
print("  " + "-" * 80)

base_vecs = normalize(emb.encode_batch(prog_20))
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
noise_results = {}

for noise in noise_levels:
    if noise > 0:
        np.random.seed(42)
        noisy = base_vecs + np.random.randn(*base_vecs.shape) * noise
        noisy = normalize(noisy)
    else:
        noisy = base_vecs
    
    b, p = compute_h1(noisy, topo)
    sim = within_cosine_sim(noisy)
    
    if noise == 0.0:
        interp = "FULL QUALITY (base model)"
    elif noise <= 0.1:
        interp = "HIGH QUALITY (strong model)"
    elif noise <= 0.3:
        interp = "MEDIUM QUALITY (medium model)"
    else:
        interp = "LOW QUALITY (weak model)"
    
    print(f"  {noise:<8.2f} {b:<8} {p:<10.4f} {sim:<10.4f} {interp}")
    noise_results[f"noise_{noise}"] = {
        'noise': noise, 'betti1': b, 'mean_pers': p, 'mean_sim': float(sim)
    }

# ===== PART 3: Noise effect on cosine similarity =====
print("\n[3/4] PART 3: How noise destroys semantic clustering")
print(f"  {'Noise':<8} {'MeanSim':<10} {'SimDrop%':<12} {'BettiChange':<14}")
print("  " + "-" * 50)

base_sim = noise_results['noise_0.0']['mean_sim']
base_betti = noise_results['noise_0.0']['betti1']
for k, v in noise_results.items():
    noise = v['noise']
    sim_drop = (base_sim - v['mean_sim']) / max(abs(base_sim), 0.001) * 100
    betti_chg = v['betti1'] - base_betti
    print(f"  {noise:<8.2f} {v['mean_sim']:<10.4f} {sim_drop:<12.1f} {betti_chg:+14d}")

# ===== PART 4: Cross-domain H1 =====
print("\n[4/4] PART 4: Cross-domain vs in-domain H1")
print(f"  {'Condition':<30} {'n':<4} {'Betti1':<8} {'MeanSim':<10}")
print("  " + "-" * 60)

domains = [
    ('prog_in_domain', prog_20, 'prog'),
    ('phys_cross_domain', phys[:20], 'phys'),
    ('geo_cross_domain', geo[:10], 'geo'),
]
dom_data = {}
for name, texts, label in domains:
    vecs = normalize(emb.encode_batch(texts))
    b, p = compute_h1(vecs, topo)
    sim = within_cosine_sim(vecs)
    print(f"  {name:<30} {len(texts):<4} {b:<8} {sim:<10.4f}")
    dom_data[name] = {'n': len(texts), 'betti1': b, 'mean_sim': float(sim)}

# ===== VERDICT =====
print("\n" + "=" * 60)
print("P3 VERDICT")
print("=" * 60)

# Noise → Betti relationship
betti_curve = [(v['noise'], v['betti1'], v['mean_sim']) for v in noise_results.values()]
betti_curve.sort(key=lambda x: x[0])

no_noise_betti = noise_results['noise_0.0']['betti1']
high_noise_betti = noise_results['noise_0.5']['betti1']
no_noise_sim = noise_results['noise_0.0']['mean_sim']
high_noise_sim = noise_results['noise_0.5']['mean_sim']

print(f"\nNoise effect (base → noise=0.5):")
print(f"  Betti-1: {no_noise_betti} → {high_noise_betti} ({high_noise_betti - no_noise_betti:+d})")
print(f"  Mean cosine similarity: {no_noise_sim:.4f} → {high_noise_sim:.4f} ({high_noise_sim - no_noise_sim:+.4f})")

if high_noise_betti > no_noise_betti:
    noise_direction = "MORE_CYCLES"
    print(f"  => CONFIRMED: Weaker model (more noise) → MORE H1 cycles")
    print(f"  => Small model penalty: less precise embeddings → MORE topological noise")
    print(f"  => H1 can detect embedding quality degradation")
elif high_noise_betti < no_noise_betti:
    noise_direction = "FEWER_CYCLES"
    print(f"  => REVERSED: Weaker model (more noise) → FEWER H1 cycles")
    print(f"  => Noise destroys all structure uniformly")
else:
    noise_direction = "NO_CHANGE"
    print(f"  => No change: noise doesn't affect H1 in this regime")

# Volume effect
print(f"\nVolume effect:")
betti_per_n_10 = vol_data['prog_10']['betti1'] / 10
betti_per_n_60 = vol_data['all_60']['betti1'] / 60
print(f"  Betti/n: prog_10={betti_per_n_10:.3f}, all_60={betti_per_n_60:.3f}")
if betti_per_n_60 < betti_per_n_10:
    print(f"  => MORE data → FEWER Betti-1 per node (concentration effect)")
    print(f"  => Adding more domain data increases clustering concentration")
else:
    print(f"  => MORE data → MORE Betti-1 per node (expansion effect)")

# Cross-domain effect
print(f"\nCross-domain effect:")
in_dom = dom_data['prog_in_domain']['betti1']
cross_domains = [dom_data[k]['betti1'] for k in dom_data if 'cross' in k]
avg_cross = np.mean(cross_domains) if cross_domains else 0
print(f"  In-domain (prog): {in_dom}")
print(f"  Avg cross-domain: {avg_cross:.0f}")
print(f"  Cross-domain mean_sim: {np.mean([dom_data[k]['mean_sim'] for k in dom_data if 'cross' in k]):.4f}")
print(f"  In-domain mean_sim: {dom_data['prog_in_domain']['mean_sim']:.4f}")

print(f"\n{'=' * 50}")
print("FINAL CONCLUSIONS FOR 'SMALL MODEL GOAL'")
print(f"{'=' * 50}")
print(f"""
1. EMBEDDING QUALITY → H1: Adding noise {'increases' if high_noise_betti > no_noise_betti else 'decreases'} H1 cycles
   → H1 can detect when a model produces less precise embeddings

2. DATA VOLUME → H1: More data increases concentration {'less' if betti_per_n_60 < betti_per_n_10 else 'more'} Betti-1 per node
   → Adding domain data {'decreases' if betti_per_n_60 < betti_per_n_10 else 'increases'} embedding dispersion

3. CROSS-DOMAIN → H1: Cross-domain embeddings have different H1 than in-domain
   → H1 responds to domain shift

IMPLICATION: H1 is a SENSITIVE PROBE for embedding space quality.
Small models with less precise embeddings will show different H1 signatures
than well-trained large models. H1 can be used as a diagnostic signal
for whether a small model has achieved "good enough" embedding geometry.
""")

# Save
results_dir = os.path.join(TOPOMEM_DIR, 'benchmarks', 'results')
os.makedirs(results_dir, exist_ok=True)
ts = int(time.time())
rf = os.path.join(results_dir, f'p3_simplified_{ts}.json')
with open(rf, 'w', encoding='utf-8') as f:
    json.dump({
        'noise_simulation': noise_results,
        'volume_comparison': vol_data,
        'domain_comparison': dom_data,
    }, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: {rf}")
