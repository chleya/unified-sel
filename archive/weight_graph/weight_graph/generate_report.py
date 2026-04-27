"""
生成 weight_graph 实验报告可视化
"""
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('results/weight_graph/report')
OUT.mkdir(parents=True, exist_ok=True)

# Load data
e01 = {}
for comp in ['gate', 'up', 'down']:
    e01[comp] = json.load(open(f'results/weight_graph/exp01/stats_mlp_{comp}.json'))

e02_stats = json.load(open('results/weight_graph/exp02/full_stats.json'))
e02_comm = json.load(open('results/weight_graph/exp02/communities.json'))
e02_pr = json.load(open('results/weight_graph/exp02/pagerank.json'))
h1 = json.load(open('results/weight_graph/exp03/h1_results.json'))
e04 = json.load(open('results/weight_graph/exp04/cross_scale_results.json'))
e05 = json.load(open('results/weight_graph/exp05/pruning_rankings.json'))

print('All data loaded')

# 1. H1: Trained vs Random Modularity
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['Trained', 'Random\n(seed=42)']
values = [h1['trained_modularity'], h1['random_mean']]
colors = ['steelblue', 'lightgray']
bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)
ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Significance threshold (0.3)')
ax.set_ylabel('Modularity')
ax.set_title('H1: Training Produces Higher Modularity\n(Trained vs Random Initialization)', fontsize=12)
ax.set_ylim(0, 1.0)
ax.legend()
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=11)
diff = values[0] - values[1]
ax.text(0.5, 0.5, f'Delta = +{diff:.4f}\nH1 SUPPORTED', ha='center', va='center', transform=ax.transAxes,
         fontsize=14, fontweight='bold', color='green',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
plt.tight_layout()
plt.savefig(OUT / 'h1_trained_vs_random.png', dpi=150)
plt.close()
print('h1_trained_vs_random.png saved')

# 2. Cross-scale: Modularity & Communities
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
models = list(e04.keys())
x = np.arange(len(models))

mods = [e04[m]['modularity'] for m in models]
axes[0].bar(x, mods, color='steelblue', edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['0.5B\n(Qwen2.5)', '1.5B\n(Qwen2.5)'])
axes[0].set_ylabel('Modularity')
axes[0].set_title('Modularity vs Model Scale')
axes[0].axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
axes[0].set_ylim(0.8, 0.95)
for i, v in enumerate(mods):
    axes[0].text(i, v + 0.003, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

comms = [e04[m]['num_communities'] for m in models]
axes[1].bar(x, comms, color='darkorange', edgecolor='white')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['0.5B', '1.5B'])
axes[1].set_ylabel('Number of Communities')
axes[1].set_title('Community Count vs Model Scale')
for i, v in enumerate(comms):
    axes[1].text(i, v + 0.2, f'{v}', ha='center', va='bottom', fontsize=10)

nodes = [e04[m]['num_nodes']/1000 for m in models]
axes[2].bar(x, nodes, color='seagreen', edgecolor='white')
axes[2].set_xticks(x)
axes[2].set_xticklabels(['0.5B', '1.5B'])
axes[2].set_ylabel('Nodes (thousands)')
axes[2].set_title('Graph Size vs Model Scale')
for i, v in enumerate(nodes):
    axes[2].text(i, v + 5, f'{v:.0f}K', ha='center', va='bottom', fontsize=10)

plt.suptitle('exp04: Cross-Scale Comparison (0.5B vs 1.5B)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT / 'exp04_cross_scale.png', dpi=150)
plt.close()
print('exp04_cross_scale.png saved')

# 3. Community Structure
sizes = list(e02_comm['community_sizes'].values())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(sizes, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('Community Size Distribution (0.5B)')
axes[0].set_xlabel('Community Size')
axes[0].set_ylabel('Count')
axes[0].set_yscale('log')

profile = e02_comm.get('profile', {})
layer_ranges = []
for cid in sorted(profile.keys(), key=int):
    layers = profile[cid].get('layers', [])
    if layers:
        layer_ranges.append((cid, min(layers), max(layers), len(layers)))
layer_ranges.sort(key=lambda x: x[1])
cid_y = {cid: i for i, (cid, _, _, _) in enumerate(layer_ranges)}
for cid, lmin, lmax, sz in layer_ranges:
    axes[1].barh(cid_y[cid], lmax - lmin + 1, left=lmin, height=0.6, color='steelblue', alpha=0.8)
    axes[1].text(lmin + (lmax-lmin)/2, cid_y[cid], f'C{cid}({sz})', ha='center', va='center', fontsize=8)
axes[1].set_yticks(list(cid_y.values()))
axes[1].set_yticklabels([f'C{cid}' for cid, _, _, _ in layer_ranges])
axes[1].set_xlabel('Layer Index')
axes[1].set_ylabel('Community')
axes[1].set_title('Community Layer Coverage (0.5B)')
axes[1].set_xlim(0, 24)

plt.suptitle(f'exp02: Community Structure -- {len(sizes)} Communities, Modularity={e02_comm["modularity"]:.4f}', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'exp02_communities.png', dpi=150)
plt.close()
print('exp02_communities.png saved')

# 4. PageRank Top Nodes
top_n = e02_pr['top_n'][:30]
names = [n for n, _ in top_n]
scores = [s for _, s in top_n]
layer_dist = e02_pr['layer_distribution']
layer_vals = [layer_dist.get(str(i), 0) for i in range(24)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_bar = ['steelblue' if 'L23' in n else 'darkorange' for n in names]
axes[0].barh(range(len(names)), scores, color=colors_bar, edgecolor='white')
axes[0].set_yticks(range(len(names)))
axes[0].set_yticklabels(names, fontsize=6)
axes[0].invert_yaxis()
axes[0].set_xlabel('PageRank Score')
axes[0].set_title('Top 30 Nodes by PageRank')

axes[1].bar(range(24), layer_vals, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('PageRank Mass')
axes[1].set_title('PageRank Mass by Layer')
axes[1].axhline(y=1/24, color='red', linestyle='--', alpha=0.5, label='Uniform')
axes[1].legend()

plt.suptitle('exp02: PageRank Analysis (0.5B)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'exp02_pagerank.png', dpi=150)
plt.close()
print('exp02_pagerank.png saved')

# 5. Community Pruning Priority
comm_pr = e05['community_ranking']
sorted_comms = sorted(comm_pr.items(), key=lambda x: float(x[1]))
cids = [x[0] for x in sorted_comms]
pr_vals = [float(x[1]) for x in sorted_comms]

fig, ax = plt.subplots(figsize=(10, 5))
colors_pr = ['red' if i < 3 else 'orange' if i < 6 else 'steelblue' for i in range(len(cids))]
ax.barh(range(len(cids)), pr_vals, color=colors_pr, edgecolor='white')
ax.set_yticks(range(len(cids)))
ax.set_yticklabels([f'Community {c}' for c in cids])
ax.set_xlabel('Total PageRank Mass')
ax.set_title('Community Pruning Priority (Bottom = Prune First)\nexp05: Lower total PageRank = weaker community = prune first', fontsize=11)
ax.invert_yaxis()
ax.text(0.02, 0.08, 'RED = prune first\nORANGE = prune second\nBLUE = preserve', transform=ax.transAxes,
        fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig(OUT / 'exp05_pruning_priority.png', dpi=150)
plt.close()
print('exp05_pruning_priority.png saved')

# 6. Single Layer Stats
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, comp in enumerate(['gate', 'up', 'down']):
    stats = e01[comp]
    shape = stats['component']
    n_in, n_out = 4864, 896 if comp != 'down' else (896, 4864)
    info = f'{comp.upper()}\n{shape}\nNodes: {stats["num_nodes"]:,}\nEdges: {stats["num_edges"]:,}\nDensity: {stats["density"]:.4f}\nAvg in-deg: {stats["avg_in_degree"]:.1f}\nAvg out-deg: {stats["avg_out_degree"]:.1f}'
    axes[idx].text(0.5, 0.5, info, ha='center', va='center', fontsize=10, transform=axes[idx].transAxes,
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[idx].set_title(f'{comp.upper()} ({shape})')
    axes[idx].axis('off')
plt.suptitle('exp01: Single Layer Statistics', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'exp01_single_layer_stats.png', dpi=150)
plt.close()
print('exp01_single_layer_stats.png saved')

# 7. Key Findings Summary Table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
findings = [
    ['H1: Trained vs Random Modularity', 'Trained = 0.8855', 'Random = 0.5617', 'Delta = +0.3238', 'H1 SUPPORTED'],
    ['exp02: 0.5B Full Model', 'Nodes: 414,720', 'Edges: 8,456,320', 'Modularity: 0.876', '14 Communities'],
    ['exp04: Cross-Scale', '0.5B: mod=0.876, 14 comm', '1.5B: mod=0.896, 17 comm', 'Larger = more modular', 'H1 EXTENDED'],
    ['exp05: Pruning Ranking', 'Neuron 490 is top hub', 'L23 = highest PR layer', 'Community 7 = weakest', 'Ranking ready'],
    ['exp02: Key Hub Neuron', 'L23_mlp_in_490', 'Top PageRank across layers', 'Multiple L16-L23 appearances', 'H4 SUPPORTED'],
    ['exp02: Community Structure', 'Communities organized by layer depth', 'Comm 0: layers 0-2', 'Comm 7: layers 11-15', 'H2 SUPPORTED'],
    ['SCC/Cycle Detection', 'Failed on 414K+ nodes', 'NetworkX too slow', 'Needs graph-tool or sampling', 'NOT COMPLETED'],
    ['exp05: Perplexity Validation', 'No GPU available', 'GGUF available but not integrated', 'Ranking data ready', 'NOT COMPLETED'],
]
table = ax.table(
    cellText=findings,
    colLabels=['Item', 'Value 1', 'Value 2', 'Value 3', 'Status'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    elif findings[row-1][-1] == 'H1 SUPPORTED' or findings[row-1][-1] == 'H1 EXTENDED' or findings[row-1][-1] == 'H2 SUPPORTED' or findings[row-1][-1] == 'H4 SUPPORTED':
        cell.set_facecolor('#C6EFCE')
    elif 'NOT COMPLETED' in findings[row-1][-1]:
        cell.set_facecolor('#FFCCCC')
    else:
        cell.set_facecolor('#D9E1F2')
ax.set_title('weight_graph: Key Findings Summary', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(OUT / 'key_findings_table.png', dpi=150, bbox_inches='tight')
plt.close()
print('key_findings_table.png saved')

print(f'\nAll report visualizations saved to {OUT}')
print(f'Files:')
for f in sorted(OUT.glob('*.png')):
    print(f'  {f.name}')
