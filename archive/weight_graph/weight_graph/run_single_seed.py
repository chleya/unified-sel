import sys, time, gc, json
sys.path.insert(0, '.')
import pickle
import torch
import torch.nn as nn
from weight_graph.config import GraphBuildConfig, AnalysisConfig
from weight_graph.analyzers import detect_communities
from weight_graph.graph_builder import GraphBuilder
from weight_graph.extractor import WeightMatrix

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
print(f"[seed={seed}] Starting...", flush=True)

# Load matrices
t0 = time.time()
with open('weight_graph/cache_matrices.pkl', 'rb') as f:
    matrices = pickle.load(f)
print(f"[seed={seed}] Loaded matrices in {time.time()-t0:.1f}s", flush=True)

# Generate random matrices
t0 = time.time()
torch.manual_seed(seed)
random_mats = []
for tmpl in matrices:
    linear = nn.Linear(tmpl.d_in, tmpl.d_out, bias=False)
    w = linear.weight.data.float().numpy()
    random_mats.append(WeightMatrix(
        name=getattr(tmpl, 'name', ''),
        layer_index=tmpl.layer_index,
        component=tmpl.component,
        weight=w,
        d_in=tmpl.d_in,
        d_out=tmpl.d_out,
    ))
print(f"[seed={seed}] Generated random matrices in {time.time()-t0:.1f}s", flush=True)
del matrices
gc.collect()

# Build graph
config = GraphBuildConfig(sparsify_method='topk', topk=32, add_residual=True)
builder = GraphBuilder(config)
t0 = time.time()
g = builder.build_full_model(random_mats)
print(f"[seed={seed}] Built graph in {time.time()-t0:.1f}s, nodes={g.num_nodes}, edges={g.num_edges}", flush=True)
del random_mats
gc.collect()

# Detect communities
analysis = AnalysisConfig(community_method='louvain')
t0 = time.time()
c = detect_communities(g, analysis)
print(f"[seed={seed}] Community detection done in {time.time()-t0:.1f}s", flush=True)
print(f"[seed={seed}] RESULT: modularity={c.modularity:.6f}, communities={c.num_communities}", flush=True)

# Save result
result = {"seed": seed, "modularity": float(c.modularity), "communities": c.num_communities}
with open(f"results/weight_graph/exp03/seed_{seed}_result.json", "w") as f:
    json.dump(result, f)
print(f"[seed={seed}] Saved result", flush=True)
