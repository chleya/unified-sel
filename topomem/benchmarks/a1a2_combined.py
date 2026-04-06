"""
A1 + A2: H2 Domain Sensitivity Combined
A1: Separated vs Mixed domains → H2 count (not ratio)
A2: Fixed n, increasing domain count → H2 count monotonic increase
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
from scipy import stats

D = 384
N_TRIALS = 30

# ============================================================
# A1: Domain Separation Sensitivity (H2 count primary metric)
# ============================================================
print("="*60)
print("A1: H2 Domain Separation Sensitivity")
print("="*60)

rng_a1 = np.random.RandomState(42)

def make_domain(center, spread=0.1, n=10):
    pts = rng_a1.randn(n, D) * spread
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts + center
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def compute_h12(points, cfg):
    dgms = cfg.compute_persistence(points)
    h1 = len(dgms[1])
    h2 = len(dgms[2]) if len(dgms) > 2 else 0
    return h1, h2

sep_h2, mix_h2 = [], []
sep_h1, mix_h1 = [], []

for trial in range(N_TRIALS):
    rng_a1.seed(trial * 1000)
    
    # Domain A: random center
    c_a = rng_a1.randn(D)
    c_a = c_a / np.linalg.norm(c_a)
    
    # Domain B: far (separated) vs near (mixed)
    offset = rng_a1.randn(D)
    offset = offset / np.linalg.norm(offset)
    c_b_sep = c_a + 5.0 * offset
    c_b_sep = c_b_sep / np.linalg.norm(c_b_sep)
    c_b_mix = c_a + 0.5 * offset
    c_b_mix = c_b_mix / np.linalg.norm(c_b_mix)
    
    dom_a = make_domain(c_a, n=10)
    dom_b_sep = make_domain(c_b_sep, n=10)
    dom_b_mix = make_domain(c_b_mix, n=10)
    
    cfg_sep = TopologyConfig(max_homology_dim=2)
    topo_sep = TopologyEngine(cfg_sep)
    h1_s, h2_s = compute_h12(np.vstack([dom_a, dom_b_sep]), topo_sep)
    
    cfg_mix = TopologyConfig(max_homology_dim=2)
    topo_mix = TopologyEngine(cfg_mix)
    h1_m, h2_m = compute_h12(np.vstack([dom_a, dom_b_mix]), topo_mix)
    
    sep_h2.append(h2_s); sep_h1.append(h1_s)
    mix_h2.append(h2_m); mix_h1.append(h1_m)
    
    delta = h2_m - h2_s
    print(f"  Trial {trial:2d}: sep_H2={h2_s:2d} mix_H2={h2_m:2d} delta={delta:+2d}")

sep_h2 = np.array(sep_h2); mix_h2 = np.array(mix_h2)

t_a1 = stats.ttest_rel(mix_h2, sep_h2)
d_a1 = (np.mean(mix_h2) - np.mean(sep_h2)) / np.sqrt((np.std(mix_h2)**2 + np.std(sep_h2)**2) / 2)

print(f"\n  SEPARATED: mean={np.mean(sep_h2):.2f} std={np.std(sep_h2):.2f}")
print(f"  MIXED:     mean={np.mean(mix_h2):.2f} std={np.std(mix_h2):.2f}")
print(f"  Delta:     mean={np.mean(mix_h2-sep_h2):+.2f}")
print(f"  t-test:    t={t_a1.statistic:.2f} p={t_a1.pvalue:.4f} {'***' if t_a1.pvalue<0.001 else '**' if t_a1.pvalue<0.01 else '*' if t_a1.pvalue<0.05 else '†' if t_a1.pvalue<0.10 else ''}")
print(f"  Cohen's d: {d_a1:.2f} {'LARGE' if abs(d_a1)>0.8 else 'medium' if abs(d_a1)>0.5 else 'small'}")

a1_verdict = "SUPPORT" if t_a1.pvalue < 0.05 else "INCONCLUSIVE"
print(f"\n  A1 VERDICT: {a1_verdict}")
print(f"  → H2 count {'IS' if a1_verdict=='SUPPORT' else 'may be'} sensitive to domain separation/mixing")

# ============================================================
# A2: Domain Count Sensitivity (fixed n, varying domains)
# ============================================================
print("\n" + "="*60)
print("A2: H2 vs Domain Count (n fixed = 20)")
print("="*60)

rng_a2 = np.random.RandomState(42)
N_PER_DOMAIN_A2 = 10

def make_random_domain(n=D, spread=0.1, n_pts=10):
    pts = rng_a2.randn(n_pts, D) * spread
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    center = rng_a2.randn(D)
    center = center / np.linalg.norm(center)
    pts = pts + center
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

results_a2 = {}

for n_domains in [1, 2, 3, 4, 5]:
    h2_list = []
    for trial in range(N_TRIALS):
        rng_a2.seed(trial * 100 + n_domains)
        domains = [make_random_domain() for _ in range(n_domains)]
        combined = np.vstack(domains)
        cfg = TopologyConfig(max_homology_dim=2)
        topo = TopologyEngine(cfg)
        dgms = topo.compute_persistence(combined)
        h2 = len(dgms[2]) if len(dgms) > 2 else 0
        h2_list.append(h2)
    results_a2[n_domains] = h2_list
    print(f"  n_domains={n_domains}: mean_H2={np.mean(h2_list):.1f} std={np.std(h2_list):.1f} median={np.median(h2_list):.0f}")

# Spearman correlation: domain count vs H2
domain_counts = []
h2_values = []
for nd, h2s in results_a2.items():
    for h2 in h2s:
        domain_counts.append(nd)
        h2_values.append(h2)

r_spearman, p_spearman = stats.spearmanr(domain_counts, h2_values)
r_pearson, p_pearson = stats.pearsonr(domain_counts, h2_values)

print(f"\n  Correlation (domain_count vs H2):")
print(f"    Spearman: r={r_spearman:.3f} p={p_spearman:.4f} {'***' if p_spearman<0.001 else '**' if p_spearman<0.01 else '*' if p_spearman<0.05 else ''}")
print(f"    Pearson:  r={r_pearson:.3f} p={p_pearson:.4f}")

# Monotonic trend test (Jonckheere-Terpstra would be ideal, use Kruskal-Wallis)
groups = [results_a2[nd] for nd in [1, 2, 3, 4, 5]]
kw = stats.kruskal(*groups)
print(f"    Kruskal-Wallis: H={kw.statistic:.2f} p={kw.pvalue:.4f}")

a2_verdict = "SUPPORT" if p_spearman < 0.05 else "INCONCLUSIVE"
print(f"\n  A2 VERDICT: {a2_verdict}")
print(f"  → H2 count {'IS' if a2_verdict=='SUPPORT' else 'may NOT be'} monotonically increasing with domain count")

# ============================================================
# Overall Synthesis
# ============================================================
print("\n" + "="*60)
print("SYNTHESIS: A1 + A2")
print("="*60)
print(f"  A1 (Separation Sensitivity): {a1_verdict}")
print(f"    → Mixed domains produce {np.mean(mix_h2-sep_h2):+.1f} more H2 cycles on average")
print(f"    → Effect size: {'LARGE' if abs(d_a1)>0.8 else 'medium' if abs(d_a1)>0.5 else 'small'} (d={d_a1:.2f})")
print()
print(f"  A2 (Domain Count Sensitivity): {a2_verdict}")
print(f"    → Spearman r={r_spearman:.3f} between domain count and H2")
print()
if a1_verdict == "SUPPORT" and a2_verdict == "SUPPORT":
    print("  OVERALL: BOTH SUPPORTED")
    print("  → H2 IS a cross-domain boundary sensitivity indicator")
    print("  → Physical meaning: H2 counts topological 'cavities' formed at semantic domain boundaries")
elif a1_verdict == "INCONCLUSIVE" or a2_verdict == "INCONCLUSIVE":
    print("  OVERALL: INCONCLUSIVE - more trials needed")
else:
    print("  OVERALL: NEEDS REVIEW")

# Save
out = {
    "experiment": "A1+A2_H2_Domain_Sensitivity",
    "n_trials": N_TRIALS,
    "a1": {
        "verdict": a1_verdict,
        "sep_h2": sep_h2.tolist(), "mix_h2": mix_h2.tolist(),
        "delta_mean": float(np.mean(mix_h2 - sep_h2)),
        "t_stat": float(t_a1.statistic), "t_pvalue": float(t_a1.pvalue),
        "cohens_d": float(d_a1),
    },
    "a2": {
        "verdict": a2_verdict,
        "results_by_n_domains": {str(k): v for k, v in results_a2.items()},
        "spearman_r": float(r_spearman), "spearman_p": float(p_spearman),
        "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
    },
}

ts = int(__import__('time').time())
outpath = rf"F:\unified-sel\topomem\benchmarks\results\a1a2_combined_{ts}.json"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: {outpath}")
