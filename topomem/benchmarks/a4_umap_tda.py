"""
A4: UMAP dimensionality reduction + TDA (using F: Python venv)
Uses F:\python_env\Scripts\python.exe with NUMBA_DISABLE_JIT=1
"""
import sys, os, warnings, json, time
warnings.filterwarnings('ignore')

HF_CACHE = r"F:\unified-sel\topomem\data\models\hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE
os.environ["NUMBA_DISABLE_JIT"] = "1"

sys.path.insert(0, r"F:\unified-sel")
from topomem.topology import TopologyEngine
from topomem.config import TopologyConfig
from topomem.embedding import EmbeddingManager
import numpy as np
from scipy import stats
import umap

# ============================================================
# Part 1: Fragmentation across dimensions
# ============================================================
print("="*60)
print("Part 1: Fragmentation across dimensions (with JIT disabled)")
print("="*60)

D = 384
rng = np.random.RandomState(42)

def make_2domain(n_per=20, mix_ratio=0.0, seed=42):
    rng2 = np.random.RandomState(seed)
    c_a = rng2.randn(D); c_a = c_a / np.linalg.norm(c_a)
    offset = rng2.randn(D); offset = offset / np.linalg.norm(offset)
    sep = 5.0 - mix_ratio * 4.5
    c_b = c_a + sep * offset; c_b = c_b / np.linalg.norm(c_b)
    
    def cluster(c, n, spread=0.1):
        pts = rng2.randn(n, D) * spread
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts + c; return pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    return np.vstack([cluster(c_a, n_per), cluster(c_b, n_per)])

def tda_summary(pts, dim=384):
    if dim < 384:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=dim, metric='cosine', random_state=42)
        pts_r = reducer.fit_transform(pts)
    else:
        pts_r = pts
    
    cfg = TopologyConfig(max_homology_dim=2)
    topo = TopologyEngine(cfg)
    dgms = topo.compute_persistence(pts_r)
    
    h0 = len(dgms[0])
    h1 = len(dgms[1])
    h2 = len(dgms[2]) if len(dgms) > 2 else 0
    n = len(pts_r)
    return {"H0": h0, "H1": h1, "H2": h2, "H0/n": h0/n, "H1/m": h1/n, "H2/m": h2/n, "H2/H1": h2/max(h1,1)}

dims = [5, 10, 15, 20, 30, 384]

print("\n  SEPARATED (mix=0.0):")
sep_pts = make_2domain(n_per=20, mix_ratio=0.0)
for dim in dims:
    s = tda_summary(sep_pts, dim)
    print(f"    dim={dim:>3}: H0={s['H0']:>3} H1={s['H1']:>3} H2={s['H2']:>3}  H0/n={s['H0/n']:.3f}")

print("\n  MIXED (mix=0.8):")
mix_pts = make_2domain(n_per=20, mix_ratio=0.8)
for dim in dims:
    s = tda_summary(mix_pts, dim)
    print(f"    dim={dim:>3}: H0={s['H0']:>3} H1={s['H1']:>3} H2={s['H2']:>3}  H0/n={s['H0/n']:.3f}")

# ============================================================
# Part 2: Domain sensitivity across dimensions (N=20 trials)
# ============================================================
print("\n" + "="*60)
print("Part 2: H2 Domain Sensitivity (N=20 trials)")
print("="*60)

N = 20
for dim in [10, 15, 20, 30, 384]:
    sep_h2, mix_h2 = [], []
    for trial in range(N):
        pts_s = make_2domain(n_per=20, mix_ratio=0.0, seed=trial)
        pts_m = make_2domain(n_per=20, mix_ratio=0.8, seed=trial)
        
        s_s = tda_summary(pts_s, dim)
        s_m = tda_summary(pts_m, dim)
        sep_h2.append(s_s['H2'])
        mix_h2.append(s_m['H2'])
    
    sep_h2 = np.array(sep_h2); mix_h2 = np.array(mix_h2)
    t = stats.ttest_rel(mix_h2, sep_h2)
    d = (np.mean(mix_h2) - np.mean(sep_h2)) / np.sqrt((np.std(mix_h2)**2 + np.std(sep_h2)**2) / 2)
    sig = '***' if t.pvalue < 0.001 else '**' if t.pvalue < 0.01 else '*' if t.pvalue < 0.05 else '†' if t.pvalue < 0.10 else ''
    print(f"  dim={dim:>3}: sep={np.mean(sep_h2):.1f} mix={np.mean(mix_h2):.1f} delta={np.mean(mix_h2-sep_h2):+.1f} t={t.statistic:.2f} p={t.pvalue:.4f}{sig} d={d:.2f}")

# ============================================================
# Part 3: Real retrieval with UMAP pre-processing
# ============================================================
print("\n" + "="*60)
print("Part 3: Real Retrieval - UMAP(15D) vs 384D")
print("="*60)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def retrieval_test(dim, emb_mgr, domain_snippets, n_trials=5):
    """Retrieval on real deer-flow code snippets."""
    domain_names = list(domain_snippets.keys())
    
    def retrieve_pv(query, corpus_embs, k=5):
        sims = [cosine_sim(query, c) for c in corpus_embs]
        return np.argsort(sims)[::-1][:k]
    
    def retrieve_topo_h0(query_emb, corpus_embs, k=5):
        """Use H0 birth + cosine similarity."""
        if len(corpus_embs) < 3:
            return retrieve_pv(query_emb, corpus_embs, k)
        
        cfg = TopologyConfig(max_homology_dim=2)
        topo = TopologyEngine(cfg)
        
        if dim < 384:
            reducer = umap.UMAP(n_neighbors=min(10, len(corpus_embs)-1), min_dist=0.1, n_components=dim, metric='cosine', random_state=42)
            emb_reduced = reducer.fit_transform(corpus_embs)
            query_r = reducer.transform(query_emb.reshape(1, -1)).reshape(-1)
        else:
            emb_reduced = corpus_embs
            query_r = query_emb
        
        dgms = topo.compute_persistence(emb_reduced)
        h0_birth = dgms[0][:, 0]
        
        # Weight cosine sim by inverse H0 birth distance
        query_birth_mean = h0_birth.mean()
        sims_raw = np.array([cosine_sim(query_r, c) for c in emb_reduced])
        birth_diff = np.abs(h0_birth - query_birth_mean)
        weights = 1.0 / (birth_diff + 0.01)
        weighted_sims = sims_raw * weights
        return np.argsort(weighted_sims)[::-1][:k]
    
    r5_pv, r5_topo = [], []
    
    for trial in range(n_trials):
        rng3 = np.random.RandomState(trial)
        
        # Build corpus: all items from all domains
        all_embs, all_labels = [], []
        for di, snippets in enumerate(domain_snippets.values()):
            for s in snippets:
                try:
                    e = emb_mgr.encode(s)
                    all_embs.append(e)
                    all_labels.append(di)
                except:
                    pass
        
        if len(all_embs) < 10:
            continue
        
        all_embs = np.array(all_embs)
        all_labels = np.array(all_labels)
        
        # Leave-one-out per domain
        for qi in range(min(6, len(all_embs))):
            query_emb = all_embs[qi]
            query_label = all_labels[qi]
            corpus_idx = np.concatenate([np.arange(len(all_embs))[:qi], np.arange(len(all_embs))[qi+1:]])
            corpus_embs = all_embs[corpus_idx]
            corpus_labels = all_labels[corpus_idx]
            
            if len(np.unique(corpus_labels)) < 2:
                continue
            
            topk_pv = retrieve_pv(query_emb, corpus_embs, k=5)
            topk_topo = retrieve_topo_h0(query_emb, corpus_embs, k=5)
            
            r5_pv.append(np.mean(corpus_labels[topk_pv] == query_label))
            r5_topo.append(np.mean(corpus_labels[topk_topo] == query_label))
    
    return {"PV_R@5": np.mean(r5_pv), "Topo_R@5": np.mean(r5_topo), "n_tests": len(r5_pv)}

print("  Loading deer-flow snippets...")
emb_mgr = EmbeddingManager()

import os, random
def load_snippets(path, max_per=15, seed=42):
    rng4 = random.Random(seed)
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')]
    result = {}
    for sd in subdirs[:6]:
        sd_path = os.path.join(path, sd)
        py_files = []
        for root, dirs, files in os.walk(sd_path):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv']]
            for f in files:
                if f.endswith('.py') and len(f) > 10:
                    try:
                        sz = os.path.getsize(os.path.join(root, f))
                        if 200 < sz < 30000:
                            py_files.append(os.path.join(root, f))
                    except:
                        pass
        sampled = rng4.sample(py_files, min(len(py_files), max_per))
        snippets = []
        for fp in sampled:
            try:
                lines = open(fp, 'r', encoding='utf-8', errors='ignore').readlines()
                meaningful = [l.strip() for l in lines[:80] if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('"""')]
                if meaningful:
                    snippets.append('\n'.join(meaningful[:30]))
            except:
                pass
        if snippets:
            result[sd] = snippets
    return result

deerflow_path = r"F:\workspace-ideas\deer-flow\skills"
snippets = load_snippets(deerflow_path, max_per=15)
print(f"  Domains loaded: {list(snippets.keys())}")
print(f"  Items: {[len(v) for v in snippets.values()]}")

if len(snippets) >= 2:
    for dim in [15, 20, 30, 384]:
        r = retrieval_test(dim, emb_mgr, snippets, n_trials=3)
        print(f"  dim={dim:>3}: PV_R@5={r['PV_R@5']:.3f} Topo_R@5={r['Topo_R@5']:.3f} (n={r['n_tests']})")
else:
    print("  Not enough domains for retrieval test")

emb_mgr.unload()

ts = int(time.time())
print(f"\nDone. ts={ts}")
