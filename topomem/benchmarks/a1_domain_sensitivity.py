"""
A1: H2 Cross-Domain Sensitivity Test
验证 H2 对"域分离度"的敏感性

假设：
- 分离条件（域间距离大）：H2/H1 ≈ 0（无域间空洞）
- 混合条件（域间距离小）：H2/H1 >> 0（有域间空洞）
"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')

HF_CACHE = r"F:\unified-sel\topomem\data\models\hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE

sys.path.insert(0, r"F:\unified-sel")
from topomem.topology import TopologyEngine
from topomem.config import TopologyConfig
import numpy as np

D = 384  # MiniLM dimension
N_PER_DOMAIN = 10
SEED = 42
N_TRIALS = 20

rng = np.random.RandomState(SEED)

def make_synthetic_domain(center, spread=0.1, n=10):
    pts = rng.randn(n, D) * spread
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts + center
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def h2_h1_ratio(points, topo):
    dgms = topo.compute_persistence(points)
    h1 = len(dgms[1])
    h2 = len(dgms[2]) if len(dgms) > 2 else 0
    return h1, h2, h2 / max(h1, 1)

def run_trial(trial, separation):
    # Domain A: fixed center
    center_a = rng.randn(D)
    center_a = center_a / np.linalg.norm(center_a)
    
    # Domain B: separation controls mix
    offset = rng.randn(D)
    offset = offset / np.linalg.norm(offset)
    center_b = center_a + separation * offset
    center_b = center_b / np.linalg.norm(center_b)
    
    dom_a = make_synthetic_domain(center_a, spread=0.1, n=N_PER_DOMAIN)
    dom_b = make_synthetic_domain(center_b, spread=0.1, n=N_PER_DOMAIN)
    
    # Condition 1: Pure A only
    cfg = TopologyConfig(max_homology_dim=2)
    topo_a = TopologyEngine(cfg)
    h1_a, h2_a, _ = h2_h1_ratio(dom_a, topo_a)
    
    # Condition 2: A+B separated/mixed
    combined = np.vstack([dom_a, dom_b])
    topo_ab = TopologyEngine(cfg)
    h1_ab, h2_ab, ratio_ab = h2_h1_ratio(combined, topo_ab)
    
    # Compute domain separation metric (cosine between centroids)
    centroid_a = dom_a.mean(axis=0)
    centroid_b = dom_b.mean(axis=0)
    centroid_a = centroid_a / np.linalg.norm(centroid_a)
    centroid_b = centroid_b / np.linalg.norm(centroid_b)
    cos_sim = np.dot(centroid_a, centroid_b)
    
    return {
        "trial": trial,
        "separation": separation,
        "cos_sim_AB": float(cos_sim),
        "H1_A": h1_a,
        "H2_A": h2_a,
        "H1_AB": h1_ab,
        "H2_AB": h2_ab,
        "H2_H1_ratio_AB": ratio_ab,
    }

results = []

for trial in range(N_TRIALS):
    r_sep = run_trial(trial, separation=5.0)   # Separated
    r_mix = run_trial(trial, separation=0.5)    # Mixed
    results.append({"separated": r_sep, "mixed": r_mix})
    print(f"Trial {trial:2d}: sep_H2={r_sep['H2_AB']:2d} sep_ratio={r_sep['H2_H1_ratio_AB']:.3f} | mix_H2={r_mix['H2_AB']:2d} mix_ratio={r_mix['H2_H1_ratio_AB']:.3f} | delta={r_mix['H2_H1_ratio_AB']-r_sep['H2_H1_ratio_AB']:+.3f}")

# Summary statistics
sep_ratios = [r["separated"]["H2_H1_ratio_AB"] for r in results]
mix_ratios = [r["mixed"]["H2_H1_ratio_AB"] for r in results]
sep_h2 = [r["separated"]["H2_AB"] for r in results]
mix_h2 = [r["mixed"]["H2_AB"] for r in results]

print("\n" + "="*60)
print("SUMMARY: H2 Cross-Domain Sensitivity (A1)")
print("="*60)
print(f"  N trials: {N_TRIALS}")
print(f"  Points per domain: {N_PER_DOMAIN}")
print()
print(f"  SEPARATED (sep=5.0):")
print(f"    H2 count:  mean={np.mean(sep_h2):.1f}  std={np.std(sep_h2):.1f}  median={np.median(sep_h2):.0f}")
print(f"    H2/H1:     mean={np.mean(sep_ratios):.3f}  std={np.std(sep_ratios):.3f}")
print()
print(f"  MIXED (sep=0.5):")
print(f"    H2 count:  mean={np.mean(mix_h2):.1f}  std={np.std(mix_h2):.1f}  median={np.median(mix_h2):.0f}")
print(f"    H2/H1:     mean={np.mean(mix_ratios):.3f}  std={np.std(mix_ratios):.3f}")
print()

delta_ratios = [m - s for s, m in zip(sep_ratios, mix_ratios)]
delta_h2 = [m - s for s, m in zip(sep_h2, mix_h2)]
print(f"  DELTA (mixed - separated):")
print(f"    H2 count:  mean={np.mean(delta_h2):+.1f}")
print(f"    H2/H1:     mean={np.mean(delta_ratios):+.3f}")

# T-test
from scipy import stats
t_ratio = stats.ttest_rel(mix_ratios, sep_ratios)
t_h2 = stats.ttest_rel(mix_h2, sep_h2)
print(f"\n  Paired t-test:")
print(f"    H2 count:  t={t_h2.statistic:.2f}  p={t_h2.pvalue:.4f} {'***' if t_h2.pvalue<0.001 else '**' if t_h2.pvalue<0.01 else '*' if t_h2.pvalue<0.05 else ''}")
print(f"    H2/H1:     t={t_ratio.statistic:.2f}  p={t_ratio.pvalue:.4f} {'***' if t_ratio.pvalue<0.001 else '**' if t_ratio.pvalue<0.01 else '*' if t_ratio.pvalue<0.05 else ''}")

# Effect size (Cohen's d)
def cohens_d(a, b):
    diff = np.mean(a) - np.mean(b)
    pooled = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    return diff / pooled if pooled > 0 else 0

d_h2 = cohens_d(mix_h2, sep_h2)
d_ratio = cohens_d(mix_ratios, sep_ratios)
print(f"\n  Cohen's d:")
print(f"    H2 count:  d={d_h2:.2f}  {'LARGE' if abs(d_h2)>0.8 else 'medium' if abs(d_h2)>0.5 else 'small'}")
print(f"    H2/H1:     d={d_ratio:.2f}  {'LARGE' if abs(d_ratio)>0.8 else 'medium' if abs(d_ratio)>0.5 else 'small'}")

print()
verdict = "PASS" if np.mean(delta_ratios) > 0 and t_ratio.pvalue < 0.05 else "FAIL"
print(f"  VERDICT: {verdict}")
print(f"  → H2 {'IS' if verdict=='PASS' else 'is NOT'} sensitive to domain separation/mixing")

# Save results
out = {
    "experiment": "A1_H2_CrossDomain_Sensitivity",
    "n_trials": N_TRIALS,
    "n_per_domain": N_PER_DOMAIN,
    "separated": {"h2": sep_h2, "h2_h1_ratio": sep_ratios},
    "mixed": {"h2": mix_h2, "h2_h1_ratio": mix_ratios},
    "delta_h2_mean": float(np.mean(delta_h2)),
    "delta_ratio_mean": float(np.mean(delta_ratios)),
    "t_h2_pvalue": float(t_h2.pvalue),
    "t_ratio_pvalue": float(t_ratio.pvalue),
    "cohens_d_h2": float(d_h2),
    "cohens_d_ratio": float(d_ratio),
    "verdict": verdict,
    "individual_trials": results,
}

import time
ts = int(time.time())
outpath = rf"F:\unified-sel\topomem\benchmarks\results\a1_domain_sensitivity_{ts}.json"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to: {outpath}")
