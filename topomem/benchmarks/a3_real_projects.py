"""
A3: Real Codebase H2 Profile Validation
验证 H2 在真实代码库上的 profile 差异

目标：
1. 不同项目的 H2 profile 是否有差异？
2. 同一个项目内，语义内聚的模块 vs 语义混乱的模块，H2 是否有差异？
3. 当模块边界被破坏（过度耦合）时，H2 如何变化？
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
from scipy import stats

PROJECTS_DIR = r"F:\workspace-ideas"

# ============================================================
# Part 1: Real project H2 profiles
# ============================================================
print("="*60)
print("Part 1: Real Project H2 Profiles")
print("="*60)

def sample_python_files(project_path, max_files=30, seed=42):
    """Sample Python files from a project directory."""
    import random
    rng = random.Random(seed)
    py_files = []
    for root, dirs, files in os.walk(project_path):
        # Skip test/vendor/node_modules dirs
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'env', 'node_modules', 'test', 'tests', 'vendor']]
        for f in files:
            if f.endswith('.py') and len(f) > 10:  # Skip __init__.py and very short names
                full_path = os.path.join(root, f)
                try:
                    size = os.path.getsize(full_path)
                    if 200 < size < 50000:  # 200B to 50KB
                        py_files.append(full_path)
                except:
                    pass
    sampled = rng.sample(py_files, min(len(py_files), max_files))
    return sampled

def extract_code_snippets(file_paths, max_lines=50):
    """Extract meaningful code snippets (first non-empty lines)."""
    snippets = []
    for fp in file_paths:
        try:
            lines = open(fp, 'r', encoding='utf-8', errors='ignore').readlines()
            # Get first non-empty, non-comment lines
            meaningful = []
            for l in lines[:100]:
                l = l.strip()
                if l and not l.startswith('#') and not l.startswith('"""') and not l.startswith("'''"):
                    meaningful.append(l)
                    if len(meaningful) >= max_lines:
                        break
            if meaningful:
                snippets.append('\n'.join(meaningful[:max_lines]))
        except:
            pass
    return snippets

def compute_h12(embeddings, label=""):
    """Compute H1 and H2 for given embeddings."""
    pts = np.array(embeddings)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    cfg = TopologyConfig(max_homology_dim=2)
    topo = TopologyEngine(cfg)
    dgms = topo.compute_persistence(pts)
    h0 = len(dgms[0])
    h1 = len(dgms[1])
    h2 = len(dgms[2]) if len(dgms) > 2 else 0
    n = len(pts)
    h1_m = h1 / n if n > 0 else 0
    h2_m = h2 / n if n > 0 else 0
    ratio = h2 / max(h1, 1)
    return {"n": n, "H0": h0, "H1": h1, "H2": h2, "H1/m": h1_m, "H2/m": h2_m, "H2/H1": ratio}

def run_project(project_name, project_path, max_files=30):
    print(f"\n  Loading {project_name}...")
    files = sample_python_files(project_path, max_files=max_files)
    print(f"    Sampled {len(files)} files")
    
    snippets = extract_code_snippets(files)
    print(f"    Extracted {len(snippets)} snippets")
    
    if len(snippets) < 5:
        return None
    
    # Encode
    emb_mgr = EmbeddingManager()
    embeddings = []
    for s in snippets:
        try:
            e = emb_mgr.encode(s)
            embeddings.append(e)
        except:
            pass
    
    emb_mgr.unload()
    
    if len(embeddings) < 5:
        return None
    
    result = compute_h12(embeddings, project_name)
    result["project"] = project_name
    result["n_files"] = len(files)
    result["n_snippets"] = len(snippets)
    print(f"    H0={result['H0']} H1={result['H1']} H2={result['H2']} H2/m={result['H2/m']:.3f} ratio={result['H2/H1']:.3f}")
    return result

# Projects to test
projects = [
    ("deer-flow skills", r"F:\workspace-ideas\deer-flow\skills"),
    ("deer-flow backend", r"F:\workspace-ideas\deer-flow\backend"),
    ("hermes-agent", r"F:\workspace-ideas\hermes-agent"),
    ("hermes-tools", r"F:\workspace-ideas\hermes-agent\tools"),
    ("cognee", r"F:\workspace-ideas\cognee"),
]

project_results = {}
for name, path in projects:
    if os.path.exists(path):
        result = run_project(name, path, max_files=30)
        if result:
            project_results[name] = result
    else:
        print(f"  {name}: path not found ({path})")

print("\n" + "="*60)
print("Part 1 Summary: Real Project H2 Profiles")
print("="*60)
print(f"{'Project':<20} {'n':>4} {'H1':>5} {'H2':>5} {'H1/m':>6} {'H2/m':>6} {'H2/H1':>7}")
print("-"*60)
for name, r in project_results.items():
    print(f"{name:<20} {r['n']:>4} {r['H1']:>5} {r['H2']:>5} {r['H1/m']:>6.3f} {r['H2/m']:>6.3f} {r['H2/H1']:>7.3f}")

# Statistical test: are H2/m different across projects?
if len(project_results) >= 2:
    h2m_values = [r['H2/m'] for r in project_results.values()]
    h1m_values = [r['H1/m'] for r in project_results.values()]
    print(f"\n  H2/m range: {min(h2m_values):.3f} - {max(h2m_values):.3f}")
    print(f"  H1/m range: {min(h1m_values):.3f} - {max(h1m_values):.3f}")
    # Coefficient of variation
    cv_h2 = np.std(h2m_values) / np.mean(h2m_values) if np.mean(h2m_values) > 0 else 0
    cv_h1 = np.std(h1m_values) / np.mean(h1m_values) if np.mean(h1m_values) > 0 else 0
    print(f"  CV(H2/m): {cv_h2:.2f}  CV(H1/m): {cv_h1:.2f}")
    print(f"  → {'H2' if cv_h2 > cv_h1 else 'H1'} shows MORE variation across projects")

# ============================================================
# Part 2: Domain Invasion on Real Code (deer-flow)
# ============================================================
print("\n" + "="*60)
print("Part 2: Domain Invasion on Real Code (deer-flow)")
print("="*60)

def load_snippets_from_dir(path, max_per_dir=15, seed=42):
    """Load snippets from subdirectories (simulating domains)."""
    import random
    rng = random.Random(seed)
    
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    subdirs = [d for d in subdirs if not d.startswith('.') and d not in ['__pycache__', 'venv']]
    
    domain_snippets = {}
    for sd in subdirs[:8]:  # max 8 subdirs
        sd_path = os.path.join(path, sd)
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
        snippets = extract_code_snippets(sampled, max_lines=30)
        if snippets:
            domain_snippets[sd] = snippets
    
    return domain_snippets

deerflow_path = r"F:\workspace-ideas\deer-flow\skills"
if os.path.exists(deerflow_path):
    print(f"  Loading deer-flow skills subdirectories...")
    domains = load_snippets_from_dir(deerflow_path, max_per_dir=15)
    print(f"  Found {len(domains)} domains: {list(domains.keys())}")
    
    if len(domains) >= 2:
        # Part 2a: Each domain individually
        print("\n  Part 2a: Individual domain H2 profiles")
        domain_results = {}
        emb_mgr = EmbeddingManager()
        
        for sd_name, snippets in domains.items():
            embs = []
            for s in snippets:
                try:
                    e = emb_mgr.encode(s)
                    embs.append(e)
                except:
                    pass
            if len(embs) >= 5:
                r = compute_h12(embs)
                domain_results[sd_name] = r
                print(f"    {sd_name:<30} n={r['n']:>3} H2={r['H2']:>2} H2/m={r['H2/m']:.3f} H2/H1={r['H2/H1']:.3f}")
        
        emb_mgr.unload()
        
        # Part 2b: Mix domains
        if len(domain_results) >= 2:
            domain_keys = list(domain_results.keys())
            
            # Pure: first domain only
            pure_r = domain_results[domain_keys[0]]
            print(f"\n  Part 2b: Domain invasion response")
            print(f"    PURE (1 domain):         H2={pure_r['H2']:>2} H2/m={pure_r['H2/m']:.3f}")
            
            # Mix 2 domains
            combined_2 = {}
            for ki in [domain_keys[0], domain_keys[1]]:
                for s in domains[ki]:
                    k = f"combined_{ki}"
                    if k not in combined_2:
                        combined_2[k] = []
                    combined_2[k].append(s)
            
            # Actually compute mixed
            emb_mgr2 = EmbeddingManager()
            all_snippets_2 = domains[domain_keys[0]] + domains[domain_keys[1]]
            mixed_embs = []
            for s in all_snippets_2:
                try:
                    e = emb_mgr2.encode(s)
                    mixed_embs.append(e)
                except:
                    pass
            mixed_2_r = compute_h12(mixed_embs)
            print(f"    MIXED (2 domains):      H2={mixed_2_r['H2']:>2} H2/m={mixed_2_r['H2/m']:.3f}")
            
            if len(domain_keys) >= 3:
                all_snippets_3 = domains[domain_keys[0]] + domains[domain_keys[1]] + domains[domain_keys[2]]
                mixed_3_embs = []
                for s in all_snippets_3:
                    try:
                        e = emb_mgr2.encode(s)
                        mixed_3_embs.append(e)
                    except:
                        pass
                mixed_3_r = compute_h12(mixed_3_embs)
                print(f"    MIXED (3 domains):      H2={mixed_3_r['H2']:>2} H2/m={mixed_3_r['H2/m']:.3f}")
            
            emb_mgr2.unload()
            
            # Delta
            delta_h2 = mixed_2_r['H2'] - pure_r['H2']
            delta_h2m = mixed_2_r['H2/m'] - pure_r['H2/m']
            print(f"\n    Invasion delta (2 vs 1):  delta_H2={delta_h2:+2d}  delta_H2/m={delta_h2m:+.3f}")

# ============================================================
# Part 3: Over-coupling experiment (synthetic)
# ============================================================
print("\n" + "="*60)
print("Part 3: Over-Coupling Experiment (Synthetic)")
print("="*60)

rng_oc = np.random.RandomState(999)

def make_cluster(center, spread=0.1, n=10):
    pts = rng_oc.randn(n, 384) * spread
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts + center
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

# Clean: 2 well-separated domains
c_a = rng_oc.randn(384); c_a = c_a / np.linalg.norm(c_a)
c_b = c_a + 4.0 * (rng_oc.randn(384) / np.linalg.norm(rng_oc.randn(384)))
c_b = c_b / np.linalg.norm(c_b)

clean_a = make_cluster(c_a); clean_b = make_cluster(c_b)
clean_embs = np.vstack([clean_a, clean_b])
cfg_oc = TopologyConfig(max_homology_dim=2)
topo_clean = TopologyEngine(cfg_oc)
clean_r = compute_h12(clean_embs)
print(f"  CLEAN (sep=4.0):  H2={clean_r['H2']:>2} H2/m={clean_r['H2/m']:.3f} ratio={clean_r['H2/H1']:.3f}")

# Over-coupled: add cross-domain "bridge" points
# Sample some B points near A (simulating module boundary violation)
bridge = clean_a[:3] + 0.3 * rng_oc.randn(3, 384)
bridge = bridge / np.linalg.norm(bridge, axis=1, keepdims=True)
coupled_embs = np.vstack([clean_a, clean_b, bridge])
topo_coupled = TopologyEngine(cfg_oc)
coupled_r = compute_h12(coupled_embs)
print(f"  COUPLED (+3 bridge):  H2={coupled_r['H2']:>2} H2/m={coupled_r['H2/m']:.3f} ratio={coupled_r['H2/H1']:.3f}")

# Even more coupled: more bridges
bridge2 = clean_a[:5] + 0.5 * rng_oc.randn(5, 384)
bridge2 = bridge2 / np.linalg.norm(bridge2, axis=1, keepdims=True)
coupled2_embs = np.vstack([clean_a, clean_b, bridge2])
topo_coupled2 = TopologyEngine(cfg_oc)
coupled2_r = compute_h12(coupled2_embs)
print(f"  VERY COUPLED (+5):  H2={coupled2_r['H2']:>2} H2/m={coupled2_r['H2/m']:.3f} ratio={coupled2_r['H2/H1']:.3f}")

delta_coup = coupled_r['H2'] - clean_r['H2']
print(f"\n  Over-coupling delta: {delta_coup:+.1f} H2 cycles")
print(f"  → {'H2 INCREASES with over-coupling' if delta_coup > 0 else 'H2 decreases with over-coupling'}")

# ============================================================
# Save
# ============================================================
out = {
    "experiment": "A3_Real_Codebase_H2_Profile",
    "part1": project_results,
    "part3_overcoupling": {
        "clean_H2": clean_r['H2'], "clean_H2_m": clean_r['H2/m'],
        "coupled_H2": coupled_r['H2'], "coupled_H2_m": coupled_r['H2/m'],
        "very_coupled_H2": coupled2_r['H2'], "very_coupled_H2_m": coupled2_r['H2/m'],
        "delta": delta_coup,
    },
}

ts = int(time.time())
outpath = rf"F:\unified-sel\topomem\benchmarks\results\a3_real_projects_{ts}.json"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: {outpath}")
