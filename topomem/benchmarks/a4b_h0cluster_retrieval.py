"""
A4b: Diagnose why H0/n=1.000 always, then test real retrieval

核心问题：VR filtration 在所有维度都产生 H0/n=1.000（完全碎片化）
这不是维度问题，而是 VR filtration 的几何本质

新的策略：
1. 理解 H0 碎片化的含义：每个点在其 filtration radius 超过其他点距离之前都是孤立的
2. 关键指标不是 H0/n，而是 H0 的 birth values —— 它们编码了簇结构！
3. 用 H0 birth values 而不是 H0 count 做聚类
4. 真实检索基准（deer-flow 代码库）
"""
import sys, os, warnings, json, time
warnings.filterwarnings('ignore')

HF_CACHE = r"F:\unified-sel\topomem\data\models\hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE

sys.path.insert(0, r"F:\unified-sel")
from topomem.topology import TopologyEngine
from topomem.config import TopologyConfig
from topomem.embedding import EmbeddingManager
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy import stats

D = 384
rng = np.random.RandomState(42)

print("="*60)
print("Diagnosis: H0 birth values encode cluster structure")
print("="*60)

# Create 3 well-separated clusters (simulating 3 domains)
centers = []
for _ in range(3):
    c = rng.randn(D); c = c / np.linalg.norm(c)
    centers.append(c)

clusters = []
for ci, c in enumerate(centers):
    pts = rng.randn(20, D) * 0.1
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts + c
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    clusters.append(pts)

all_pts = np.vstack(clusters)  # 60 points, 3 clusters
true_labels = np.array([0]*20 + [1]*20 + [2]*20)

cfg = TopologyConfig(max_homology_dim=2)
topo = TopologyEngine(cfg)
dgms = topo.compute_persistence(all_pts)

h0_dgm = dgms[0]  # (60, 2) birth-death pairs
birth_values = h0_dgm[:, 0]  # birth times of each H0 component

print(f"\nH0 birth statistics:")
print(f"  n points: {len(all_pts)}")
print(f"  H0 birth values: min={birth_values.min():.4f} max={birth_values.max():.4f} std={birth_values.std():.4f}")
print(f"  Birth value histogram (5 bins): {np.histogram(birth_values, bins=5)[0]}")

# Key insight: cluster by H0 birth values (NOT using TDA single-linkage)
# Points that merge early (low birth value) are from dense clusters
# Points that merge late (high birth value) are from sparse/isolated regions
sorted_idx = np.argsort(birth_values)
print(f"\n  Birth values sorted (first 10): {birth_values[sorted_idx[:10]].round(4)}")
print(f"  Birth values sorted (last 10): {birth_values[sorted_idx[-10:]].round(4)}")

# Try clustering by H0 birth using k-means
from sklearn.cluster import KMeans
for n_clusters in [2, 3, 4, 5]:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(birth_values.reshape(-1, 1))
    ARI = adjusted_rand_score(true_labels, pred)
    NMI = normalized_mutual_info_score(true_labels, pred)
    print(f"\n  KMeans on H0 birth (k={n_clusters}): ARI={ARI:.3f} NMI={NMI:.3f}")

# Also test: cosine similarity clustering (baseline)
from sklearn.cluster import KMeans as KM2
cos_sim_matrix = all_pts @ all_pts.T
km_cos = KM2(n_clusters=3, random_state=42, n_init=10)
pred_cos = km_cos.fit_predict(cos_sim_matrix)
ARI_cos = adjusted_rand_score(true_labels, pred_cos)
NMI_cos = normalized_mutual_info_score(true_labels, pred_cos)
print(f"\n  KMeans on cosine sim (k=3): ARI={ARI_cos:.3f} NMI={NMI_cos:.3f}")

# ============================================================
# Key insight: what does H0 actually encode?
# ============================================================
print("\n" + "="*60)
print("What does H0 birth encode?")
print("="*60)

# H0 birth = the filtration radius at which this connected component first appears
# For a single point, birth = 0 (appears at radius 0)
# For a cluster of points: birth = the distance to the nearest neighbor cluster
# → H0 birth encodes INTER-cluster distances (very useful!)

# Compute pairwise distances
from scipy.spatial.distance import pdist, squareform
dist_matrix = squareform(pdist(all_pts, metric='cosine'))

# For each point, find its nearest neighbor distance
nn_dist = dist_matrix[np.arange(len(all_pts)), np.argsort(dist_matrix, axis=1)[:, 1]]

# Correlation between H0 birth and nearest neighbor distance
corr, pval = stats.pearsonr(birth_values, nn_dist)
print(f"\n  Pearson corr(H0 birth, NN distance): r={corr:.3f} p={pval:.4f}")

# Per-cluster analysis
print(f"\n  Per-cluster H0 birth means:")
for ci in range(3):
    mask = true_labels == ci
    print(f"    Cluster {ci}: mean_birth={birth_values[mask].mean():.4f} std={birth_values[mask].std():.4f}")

# ============================================================
# Real retrieval: deer-flow corpus
# ============================================================
print("\n" + "="*60)
print("Real Retrieval: deer-flow + UMAP + H0 birth clustering")
print("="*60)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def load_deerflow_snippets(max_per_dir=20, seed=42):
    """Load real Python snippets from deer-flow skills."""
    import random
    rng = random.Random(seed)
    
    skills_path = r"F:\workspace-ideas\deer-flow\skills"
    subdirs = [d for d in os.listdir(skills_path) if os.path.isdir(os.path.join(skills_path, d)) and not d.startswith('.')]
    
    domain_snippets = {}
    for sd in subdirs[:6]:
        sd_path = os.path.join(skills_path, sd)
        py_files = []
        for root, dirs, files in os.walk(sd_path):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv']]
            for f in files:
                if f.endswith('.py') and len(f) > 10:
                    try:
                        size = os.path.getsize(os.path.join(root, f))
                        if 200 < size < 30000:
                            py_files.append(os.path.join(root, f))
                    except:
                        pass
        sampled = rng.sample(py_files, min(len(py_files), max_per_dir))
        
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
            domain_snippets[sd] = snippets
    
    return domain_snippets

print("  Loading deer-flow skills subdirectories...")
domain_snippets = load_deerflow_snippets(max_per_dir=20)
domain_names = list(domain_snippets.keys())
print(f"  Found {len(domain_names)} domains: {domain_names}")
print(f"  Items per domain: {[len(sn) for sn in domain_snippets.values()]}")

# Encode
emb_mgr = EmbeddingManager()
corpus_embeddings = []
corpus_labels = []
corpus_texts = []

for di, (dname, snippets) in enumerate(domain_snippets.items()):
    for s in snippets:
        try:
            e = emb_mgr.encode(s)
            corpus_embeddings.append(e)
            corpus_labels.append(di)
            corpus_texts.append(s[:100])
        except:
            pass

emb_mgr.unload()
corpus_emb = np.array(corpus_embeddings)
corpus_labels = np.array(corpus_labels)
n = len(corpus_emb)
print(f"\n  Total corpus: {n} items, {len(domain_names)} domains")

# Compute H0 birth for corpus
cfg = TopologyConfig(max_homology_dim=2)
topo = TopologyEngine(cfg)
dgms = topo.compute_persistence(corpus_emb)
h0_birth = dgms[0][:, 0]
print(f"  H0 birth: min={h0_birth.min():.4f} max={h0_birth.max():.4f} std={h0_birth.std():.4f}")

# Cluster by H0 birth using k-means
from sklearn.cluster import KMeans
km_h0 = KMeans(n_clusters=len(domain_names), random_state=42, n_init=10)
h0_labels = km_h0.fit_predict(h0_birth.reshape(-1, 1))
ari_h0 = adjusted_rand_score(corpus_labels, h0_labels)
nmi_h0 = normalized_mutual_info_score(corpus_labels, h0_labels)
print(f"\n  KMeans on H0 birth (k={len(domain_names)}): ARI={ari_h0:.3f} NMI={nmi_h0:.3f}")

# Cosine similarity baseline
km_cos = KMeans(n_clusters=len(domain_names), random_state=42, n_init=10)
sim_matrix = corpus_emb @ corpus_emb.T
km_cos.fit(sim_matrix)
cos_labels = km_cos.predict(sim_matrix)
ari_cos = adjusted_rand_score(corpus_labels, cos_labels)
nmi_cos = normalized_mutual_info_score(corpus_labels, cos_labels)
print(f"  KMeans on cosine sim (k={len(domain_names)}): ARI={ari_cos:.3f} NMI={nmi_cos:.3f}")

# Retrieval: query by text
def retrieve_purevec(query_emb, corpus_emb, k=5):
    sims = [cosine_sim(query_emb, c) for c in corpus_emb]
    return np.argsort(sims)[::-1][:k]

def retrieve_h0birth(query_emb, corpus_emb, h0_birth, topo, k=5):
    """Use H0 birth to filter: only retrieve items with similar birth values."""
    # Find items with similar H0 birth to query
    query_birth = h0_birth.mean()  # approximate
    birth_diff = np.abs(h0_birth - query_birth)
    weights = 1.0 / (birth_diff + 0.01)
    sims = np.array([cosine_sim(query_emb, c) for c in corpus_emb])
    weighted_sims = sims * weights
    return np.argsort(weighted_sims)[::-1][:k]

print("\n  Retrieval test (leave-one-out within domain):")
# For each domain, pick 1 item as query, rest as corpus
from sklearn.metrics import ndcg_score

# Create test queries (1 item per domain)
test_results = {"PV": [], "H0": []}
for di in range(len(domain_names)):
    domain_mask = corpus_labels == di
    domain_items = np.where(domain_mask)[0]
    if len(domain_items) < 2:
        continue
    
    # Query = last item, corpus = rest of domain + other domains
    query_idx = domain_items[-1]
    query_emb = corpus_emb[query_idx]
    corpus_idx = np.concatenate([domain_items[:-1], np.where(~domain_mask)[0]])
    corpus_emb_sub = corpus_emb[corpus_idx]
    corpus_labels_sub = corpus_labels[corpus_idx]
    h0_birth_sub = h0_birth[corpus_idx]
    
    # Topo for reduced corpus
    cfg_sub = TopologyConfig(max_homology_dim=2)
    topo_sub = TopologyEngine(cfg_sub)
    dgms_sub = topo_sub.compute_persistence(corpus_emb_sub)
    h0_birth_sub = dgms_sub[0][:, 0]
    
    # PV retrieval
    topk_pv = retrieve_purevec(query_emb, corpus_emb_sub, k=5)
    correct_pv = np.sum(corpus_labels_sub[topk_pv] == corpus_labels[query_idx])
    test_results["PV"].append(correct_pv / min(5, len(domain_items[:-1])))
    
    # H0 birth retrieval
    topk_h0 = retrieve_h0birth(query_emb, corpus_emb_sub, h0_birth_sub, topo_sub, k=5)
    correct_h0 = np.sum(corpus_labels_sub[topk_h0] == corpus_labels[query_idx])
    test_results["H0"].append(correct_h0 / min(5, len(domain_items[:-1])))

print(f"  PureVec avg R@5: {np.mean(test_results['PV']):.2f}")
print(f"  H0-birth avg R@5: {np.mean(test_results['H0']):.2f}")

# ============================================================
# Save
# ============================================================
out = {
    "experiment": "A4b_H0Birth_RealRetrieval",
    "h0_birth_stats": {"min": float(h0_birth.min()), "max": float(h0_birth.max()), "std": float(h0_birth.std())},
    "clustering": {"H0_birth_ARI": ari_h0, "H0_birth_NMI": nmi_h0, "cos_ARI": ari_cos, "cos_NMI": nmi_cos},
    "retrieval": {"PV_R@5": np.mean(test_results['PV']), "H0_R@5": np.mean(test_results['H0'])},
}

ts = int(time.time())
outpath = rf"F:\unified-sel\topomem\benchmarks\results\a4b_h0birth_{ts}.json"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: {outpath}")
