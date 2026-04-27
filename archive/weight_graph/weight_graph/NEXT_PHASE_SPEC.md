# Weight Graph — 下阶段实验规范

**角色**：本文档是给实施者（minimax agent）的完整执行合同。  
**日期**：2026-04-10  
**前置依赖**：exp01-05 全部完成（见 REPORT.md）  
**目标**：完成 Phase 0 → Phase 2c → Phase 4，产出论文核心数据

---

## 当前资产清单（实施者必读）

### 已有代码（全部可用，不需要重写）

| 文件 | 状态 | 关键 API |
|------|------|----------|
| `weight_graph/extractor.py` | ✅ | `WeightExtractor.extract()`, `.extract_random_init(seed)`, `.get_model_info()` |
| `weight_graph/graph_builder.py` | ✅ | `GraphBuilder.build_single_layer()`, `.build_full_model()` |
| `weight_graph/analyzers.py` | ✅ | `basic_stats()`, `degree_distribution()`, `detect_communities()`, `community_profile()`, `compute_pagerank()`, `detect_cycles()` |
| `weight_graph/viz.py` | ✅ | `plot_degree_distribution()`, `plot_community_structure()`, `plot_modularity_comparison()`, `plot_pagerank_distribution()` |
| `weight_graph/config.py` | ✅ | `ExperimentConfig`, `ExtractionConfig`, `GraphBuildConfig`, `AnalysisConfig` |
| `weight_graph/utils.py` | ✅ | `sparsify_percentile()`, `sparsify_topk()`, `sparsify_sigma()`, `save_results()`, `ensure_dir()` |
| `experiments/capability/benchmark.py` | ✅ | CLI runner，支持 `--routing-monitor` 参数 |
| `core/capability_benchmark.py` | ✅ | `run_capability_benchmark()`，支持 7 种 monitor |

### 已有缓存（节省重复计算）

| 文件 | 内容 |
|------|------|
| `weight_graph/cache_matrices.pkl` | Qwen2.5-0.5B 全部 MLP 权重矩阵 |
| `weight_graph/cache_graph.pkl` | Qwen2.5-0.5B 全模型图（414K 节点 / 8.5M 边） |
| `weight_graph/cache_matrices_1b5.pkl` | Qwen2.5-1.5B 全部 MLP 权重矩阵 |
| `weight_graph/cache_graph_1b5.pkl` | Qwen2.5-1.5B 全模型图（882K 节点 / 18M 边） |

### 已有结果

| 指标 | 0.5B Trained | 0.5B Random | 1.5B Trained |
|------|-------------|-------------|--------------|
| Modularity | 0.886 | 0.562 | 0.896 |
| 社区数 | 14 | — | 17 |
| Hub neuron | #490（跨层） | — | — |
| PageRank top layer | L23 (5.25%) | — | — |

### 已知问题

1. exp03 只跑了 1 个 random seed → 无 p-value
2. SCC/环路检测在 414K 图上超时 → 需要 scipy sparse 或采样
3. exp05 只有排名，无 perplexity 验证

---

## 任务 1：exp03 补充 random seeds（P0，预计 2-3 小时）

### 目标
将 exp03 的 random baseline 从 1 个扩展到 5 个 seed，计算统计显著性。

### 文件
修改 `weight_graph/experiments/exp03_trained_vs_random.py`

### 具体步骤

```python
# 1. 加载已有 trained graph（用缓存，不要重新构建）
import pickle
with open("weight_graph/cache_graph.pkl", "rb") as f:
    trained_graph = pickle.load(f)

# 2. 对 trained graph 重新计算 modularity（用已有 API）
from weight_graph.analyzers import detect_communities
from weight_graph.config import AnalysisConfig
config = AnalysisConfig(community_method="louvain", community_resolution=1.0)
trained_comm = detect_communities(trained_graph, config)
trained_modularity = trained_comm.modularity  # 应该 ≈ 0.886

# 3. 生成 5 个 random init 模型并分别计算 modularity
from weight_graph.extractor import WeightExtractor
from weight_graph.graph_builder import GraphBuilder
from weight_graph.config import ExtractionConfig, GraphBuildConfig

extraction_config = ExtractionConfig(
    model_name="Qwen/Qwen2.5-0.5B",
    layer_types=["mlp"],
    layer_indices=None,
)
graph_config = GraphBuildConfig(
    sparsify_method="topk",
    topk=32,
    add_residual=True,
)

random_modularities = []
for seed in [42, 123, 456, 789, 1024]:
    print(f"Random seed {seed}...")
    extractor = WeightExtractor(extraction_config)
    random_matrices = extractor.extract_random_init(seed=seed)
    builder = GraphBuilder(graph_config)
    random_graph = builder.build_full_model(random_matrices)
    random_comm = detect_communities(random_graph, config)
    random_modularities.append(random_comm.modularity)
    print(f"  modularity = {random_comm.modularity:.4f}")

# 4. 统计检验
import numpy as np
from scipy import stats as sp_stats
random_mean = np.mean(random_modularities)
random_std = np.std(random_modularities)
z_score = (trained_modularity - random_mean) / max(random_std, 1e-8)
p_value = 1 - sp_stats.norm.cdf(z_score)

# 5. 保存结果到 results/weight_graph/exp03/h1_results_5seed.json
```

### 输出格式
```json
{
  "trained_modularity": 0.886,
  "random_modularities": [0.562, ...],
  "random_mean": ...,
  "random_std": ...,
  "z_score": ...,
  "p_value": ...,
  "n_random": 5,
  "seeds": [42, 123, 456, 789, 1024],
  "h1_supported": true/false
}
```

### 判断标准
- p < 0.05 → H1 统计显著，继续后续所有任务
- 0.05 < p < 0.1 → 边缘显著，继续但需在论文中标注
- p > 0.1 → 考虑止损

### 注意
- 每个 random init 需要 `AutoModel.from_config()` 创建新模型，内存占用 ≈ 2GB
- 跑完一个 seed 后 `del model` 释放内存再跑下一个
- 预计每个 seed 耗时 20-40 分钟（构建图 + Louvain 检测）

---

## 任务 2：MMLU Per-Subject Accuracy（P0，预计 3-4 小时）

### 目标
获取 Qwen2.5-0.5B 在 MMLU 57 个 subject 上的逐科准确率，作为 ground truth。

### 新文件
`weight_graph/experiments/exp06_mmlu_ground_truth.py`

### 具体步骤

```python
"""
实验 6：MMLU Per-Subject Accuracy Ground Truth

目标：获取 Qwen2.5-0.5B 在 MMLU 57 个 subject 上的逐科准确率。
这是 Phase 2c（逐层拓扑 × 任务性能关联）的前置条件。

方法：
  - 使用 lm-evaluation-harness 框架（推荐）
  - 或手动实现 4-choice multiple choice evaluation

预期耗时：3-4 小时（CPU 推理）
"""

# 方法 1（推荐）：使用 lm-evaluation-harness
# pip install lm-eval
# lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen2.5-0.5B \
#     --tasks mmlu \
#     --batch_size 8 \
#     --output_path results/weight_graph/exp06/

# 方法 2（手动实现）：
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def evaluate_mmlu_subject(model, tokenizer, subject: str) -> float:
    """
    在单个 MMLU subject 上评估 4-choice accuracy。
    
    步骤：
    1. 加载 MMLU 数据集的指定 subject
       ds = load_dataset("cais/mmlu", subject, split="test")
    2. 对每道题，构造 prompt：
       "The following is a multiple choice question about {subject}.\n\n"
       "{question}\n"
       "A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\n"
       "Answer:"
    3. 获取模型在 A/B/C/D 四个 token 上的 logit
    4. 取 argmax 作为模型答案
    5. 和 ground truth 对比，计算 accuracy
    """
    ds = load_dataset("cais/mmlu", subject, split="test")
    
    correct = 0
    total = 0
    choices = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]
    
    for item in ds:
        question = item["question"]
        options = item["choices"]  # list of 4 strings
        answer_idx = item["answer"]  # 0-3
        
        prompt = (
            f"The following is a multiple choice question about {subject.replace('_', ' ')}.\n\n"
            f"{question}\n"
        )
        for i, opt in enumerate(options):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "Answer:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # last token logits
            choice_logits = logits[choice_ids]
            pred = choice_logits.argmax().item()
        
        if pred == answer_idx:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def run():
    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    
    # MMLU 57 subjects
    # 可通过 load_dataset("cais/mmlu", "all") 获取列表
    # 或硬编码列表（见下方 MMLU_SUBJECTS）
    
    results = {}
    for subject in MMLU_SUBJECTS:
        print(f"Evaluating {subject}...", end=" ")
        acc = evaluate_mmlu_subject(model, tokenizer, subject)
        results[subject] = {"accuracy": acc, "n_samples": "..."}
        print(f"acc = {acc:.3f}")
    
    # 按 STEM / Humanities / Social / Other 分组统计
    stem_subjects = [s for s in results if s in STEM_SET]
    humanities_subjects = [s for s in results if s in HUMANITIES_SET]
    # ...
    
    # 保存到 results/weight_graph/exp06/mmlu_per_subject.json
    save_results(results, Path("results/weight_graph/exp06/mmlu_per_subject.json"))
```

### MMLU Subjects 列表

```python
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

STEM_SET = {
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
    "medical_genetics", "virology",
}

HUMANITIES_SET = {
    "formal_logic", "high_school_european_history",
    "high_school_us_history", "high_school_world_history",
    "international_law", "jurisprudence", "logical_fallacies",
    "moral_disputes", "moral_scenarios", "philosophy", "prehistory",
    "professional_law", "world_religions",
}
```

### 输出格式
```json
{
  "model": "Qwen/Qwen2.5-0.5B",
  "per_subject": {
    "abstract_algebra": {"accuracy": 0.25, "n_samples": 100},
    "anatomy": {"accuracy": 0.42, "n_samples": 135},
    ...
  },
  "by_category": {
    "STEM": {"avg_accuracy": 0.31, "subjects": 22},
    "Humanities": {"avg_accuracy": 0.38, "subjects": 13},
    "Social": {"avg_accuracy": 0.35, "subjects": 12},
    "Other": {"avg_accuracy": 0.40, "subjects": 10}
  },
  "overall_accuracy": 0.35
}
```

### 注意
- CPU 推理一道题 ≈ 0.5-1 秒，57 subjects × ~100 题 ≈ 3 小时
- 如果时间紧，先跑 10 个 subjects（STEM 5 + Humanities 5）做 sanity check
- `load_dataset("cais/mmlu", subject)` 需要网络下载，首次运行较慢
- 确保 tokenizer 的 A/B/C/D token ID 正确（打印验证一下）

---

## 任务 3：逐层拓扑向量提取（P1，预计 2-3 小时）

### 目标
对 Qwen2.5-0.5B 的每一层提取一个 ~10 维的拓扑特征向量。

### 新文件
`weight_graph/experiments/exp07_layer_topo_vectors.py`

### 具体步骤

```python
"""
实验 7：逐层拓扑向量提取

对每层 L_i 构建单层图，计算 10 维拓扑向量。
结合 exp06 的 MMLU per-subject accuracy 做回归分析。
"""

import pickle
import numpy as np
from pathlib import Path

from weight_graph.config import ExtractionConfig, GraphBuildConfig, AnalysisConfig
from weight_graph.extractor import WeightExtractor
from weight_graph.graph_builder import GraphBuilder
from weight_graph.analyzers import basic_stats, degree_distribution, detect_communities, compute_pagerank
from weight_graph.utils import ensure_dir, save_results


def extract_layer_topo_vector(matrices_for_layer, graph_config, analysis_config) -> dict:
    """
    对一层的所有 MLP 组件构建子图，计算拓扑向量。
    
    返回 ~10 维特征：
    {
        "modularity": float,        # Louvain modularity
        "num_communities": int,      # 社区数
        "density": float,            # 图密度
        "avg_in_degree": float,      # 平均入度
        "max_in_degree": int,        # 最大入度（hub 程度）
        "max_out_degree": int,       # 最大出度
        "degree_std": float,         # 度分布标准差（越大=越不均匀）
        "pagerank_entropy": float,   # PageRank 分布的 Shannon 熵
        "pagerank_gini": float,      # PageRank 分布的 Gini 系数
        "reciprocity": float,        # 双向边比例
    }
    """
    builder = GraphBuilder(graph_config)
    
    # 合并该层所有组件为一个图
    # 方法：对 gate/up/down 分别构建子图，然后合并
    from weight_graph.graph_builder import _merge_into, WeightGraph
    merged = WeightGraph()
    for matrix in matrices_for_layer:
        sub = builder.build_single_layer(matrix)
        _merge_into(merged, sub)
    
    stats = basic_stats(merged)
    dd = degree_distribution(merged)
    comm = detect_communities(merged, analysis_config)
    pr = compute_pagerank(merged, analysis_config)
    
    # PageRank entropy
    pr_values = np.array(list(pr["scores"].values()))
    pr_values = pr_values / pr_values.sum() if pr_values.sum() > 0 else pr_values
    pr_entropy = float(-np.sum(pr_values[pr_values > 0] * np.log2(pr_values[pr_values > 0])))
    
    # PageRank Gini
    pr_sorted = np.sort(pr_values)
    n = len(pr_sorted)
    if n > 0 and pr_sorted.sum() > 0:
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * pr_sorted) / (n * np.sum(pr_sorted))) - (n + 1) / n)
    else:
        gini = 0.0
    
    return {
        "modularity": comm.modularity,
        "num_communities": comm.num_communities,
        "density": stats["density"],
        "avg_in_degree": stats["avg_in_degree"],
        "max_in_degree": stats["max_in_degree"],
        "max_out_degree": stats["max_out_degree"],
        "degree_std": float(np.std(dd["in_degrees"])) if len(dd["in_degrees"]) > 0 else 0.0,
        "pagerank_entropy": pr_entropy,
        "pagerank_gini": gini,
        "reciprocity": stats["reciprocity"],
    }


def run():
    output_dir = ensure_dir(Path("results/weight_graph/exp07"))
    
    extraction_config = ExtractionConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        layer_types=["mlp"],
        layer_indices=None,
    )
    graph_config = GraphBuildConfig(sparsify_method="topk", topk=32, add_residual=False)
    analysis_config = AnalysisConfig(community_method="louvain", community_resolution=1.0)
    
    # 提取所有权重（或用缓存）
    cache_path = Path("weight_graph/cache_matrices.pkl")
    if cache_path.exists():
        import pickle
        with open(cache_path, "rb") as f:
            all_matrices = pickle.load(f)
        print(f"Loaded {len(all_matrices)} cached matrices")
    else:
        extractor = WeightExtractor(extraction_config)
        all_matrices = extractor.extract()
    
    # 按层分组
    from collections import defaultdict
    by_layer = defaultdict(list)
    for m in all_matrices:
        by_layer[m.layer_index].append(m)
    
    num_layers = max(by_layer.keys()) + 1
    print(f"Model has {num_layers} layers")
    
    # 对每层计算拓扑向量
    topo_matrix = {}  # layer_index → topo_vector dict
    for layer_idx in range(num_layers):
        print(f"Layer {layer_idx}/{num_layers-1}...", end=" ", flush=True)
        matrices = by_layer[layer_idx]
        topo = extract_layer_topo_vector(matrices, graph_config, analysis_config)
        topo_matrix[layer_idx] = topo
        print(f"mod={topo['modularity']:.3f}, pr_ent={topo['pagerank_entropy']:.1f}")
    
    save_results(topo_matrix, output_dir / "layer_topo_vectors.json")
    
    # 转为 numpy 矩阵方便后续回归
    feature_names = list(topo_matrix[0].keys())
    X = np.array([[topo_matrix[l][f] for f in feature_names] for l in range(num_layers)])
    np.save(output_dir / "topo_matrix.npy", X)
    save_results({"feature_names": feature_names, "shape": list(X.shape)}, 
                 output_dir / "topo_matrix_meta.json")
    
    print(f"\nDone. Topo matrix shape: {X.shape}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    run()
```

### 输出
- `results/weight_graph/exp07/layer_topo_vectors.json` — 每层的拓扑向量
- `results/weight_graph/exp07/topo_matrix.npy` — [24, 10] numpy 矩阵
- `results/weight_graph/exp07/topo_matrix_meta.json` — 特征名和 shape

### 注意
- 用 `add_residual=False`（单层分析不加残差）
- 每层的 Louvain 检测在 ~5K 节点图上很快（< 1 秒）
- 总耗时 ≈ 24 层 × 2 分钟 ≈ 1 小时

---

## 任务 4：逐层拓扑 × MMLU 回归分析（P1，预计 1 天）

### 前置条件
- 任务 2 完成（有 MMLU per-subject accuracy）
- 任务 3 完成（有逐层拓扑向量）

### 新文件
`weight_graph/experiments/exp08_topo_ability_regression.py`

### 核心方法

```python
"""
实验 8：逐层拓扑 × 任务性能回归分析

核心问题：哪些层的拓扑特征和哪些学科的能力相关？

方法：
  1. 加载拓扑矩阵 X [24 layers, 10 features]
  2. 加载 MMLU accuracy Y [57 subjects]
  3. 但问题是：X 是 per-layer，Y 是 per-subject，维度不匹配
  
  解决方案（三种，全部实现，对比结果）：
  
  方案 A：Layer Ablation（因果性验证）
    - 对每层 L_i，将该层权重置零 → 重跑 MMLU → 观察各 subject 的 accuracy 下降
    - 得到 ΔY_i [57] = Y_original - Y_without_layer_i
    - 然后对 X_i [10] 和 ΔY_i [57] 做 correlation
    - 优势：因果性；劣势：需要 24 次推理（每次 3 小时 → 不现实 on CPU）
    
  方案 B：Attention Rollout Proxy（近似方案）
    - 对每个 MMLU subject 的样本，记录各层的平均激活范数
    - 作为"该层对该 subject 的贡献度"的 proxy
    - 然后 correlation(topo_feature, layer_contribution)
    - 优势：可行；劣势：激活范数 ≠ 贡献度
    
  方案 C：Cross-Layer Feature Aggregation（最实用）
    - 将 24 层的拓扑向量聚合为模型级特征：
      X_model = [mean(X), std(X), max(X), min(X), X[0], X[-1]]
      → 60 维模型级特征向量
    - 对比不同模型（0.5B vs 1.5B）的 X_model 和 overall accuracy
    - 但只有 2 个数据点，统计意义有限
    
  推荐先做方案 B（最可行），然后方案 A 作为选做。
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

from weight_graph.utils import ensure_dir, save_results


def run():
    output_dir = ensure_dir(Path("results/weight_graph/exp08"))
    
    # 1. 加载拓扑矩阵
    X = np.load("results/weight_graph/exp07/topo_matrix.npy")  # [24, 10]
    with open("results/weight_graph/exp07/topo_matrix_meta.json") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    num_layers = X.shape[0]
    
    # 2. 加载 MMLU per-subject accuracy
    with open("results/weight_graph/exp06/mmlu_per_subject.json") as f:
        mmlu = json.load(f)
    subjects = sorted(mmlu["per_subject"].keys())
    Y = np.array([mmlu["per_subject"][s]["accuracy"] for s in subjects])  # [57]
    
    # 3. 方案 B：Attention Rollout Proxy
    #    这里需要对每个 MMLU subject 跑一遍模型，记录各层激活范数
    #    得到 layer_contributions[subject][layer] = avg activation norm
    
    # 实现（简化版）：
    # 对每个 subject 取前 10 道题，记录每层 hidden_states 的 L2 范数
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    
    from datasets import load_dataset
    
    layer_contributions = {}  # subject → [24] layer norms
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        norms = np.zeros(num_layers)
        n_samples = min(10, len(ds))  # 只取前 10 道题节省时间
        
        for idx in range(n_samples):
            item = ds[idx]
            prompt = f"Question about {subject}: {item['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # outputs.hidden_states: tuple of [1, seq_len, hidden_size]
                for l in range(num_layers):
                    hs = outputs.hidden_states[l + 1]  # +1 because [0] is embedding
                    norms[l] += hs.norm(dim=-1).mean().item()
        
        norms /= n_samples
        layer_contributions[subject] = norms.tolist()
        print(f"{subject}: done")
    
    # 4. Correlation 分析
    #    对每个 (topo_feature, subject)，计算 correlation(X[:, f], contributions[:, l])
    #    但这里 X 和 contributions 都是 [24] 维（按层），可以直接做 Pearson correlation
    
    C = np.array([layer_contributions[s] for s in subjects])  # [57, 24]
    
    # 对每个拓扑特征 f，计算它和各 subject 贡献度的跨层相关性
    correlation_matrix = np.zeros((len(feature_names), len(subjects)))
    pvalue_matrix = np.zeros_like(correlation_matrix)
    
    for fi, fname in enumerate(feature_names):
        topo_per_layer = X[:, fi]  # [24]
        for si, subject in enumerate(subjects):
            contrib_per_layer = C[si, :]  # [24]
            r, p = sp_stats.pearsonr(topo_per_layer, contrib_per_layer)
            correlation_matrix[fi, si] = r
            pvalue_matrix[fi, si] = p
    
    # 5. 汇总：哪些拓扑特征和哪些学科领域有强关联？
    results = {
        "feature_names": feature_names,
        "subjects": subjects,
        "correlation_matrix": correlation_matrix.tolist(),  # [10, 57]
        "pvalue_matrix": pvalue_matrix.tolist(),
        "strong_correlations": [],  # |r| > 0.5 且 p < 0.05 的
    }
    
    for fi, fname in enumerate(feature_names):
        for si, subject in enumerate(subjects):
            r = correlation_matrix[fi, si]
            p = pvalue_matrix[fi, si]
            if abs(r) > 0.5 and p < 0.05:
                results["strong_correlations"].append({
                    "feature": fname,
                    "subject": subject,
                    "r": float(r),
                    "p": float(p),
                })
    
    save_results(results, output_dir / "topo_ability_correlations.json")
    
    # 6. 可视化：correlation heatmap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=90, fontsize=6)
    ax.set_title("Topo Feature × MMLU Subject Correlation")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
    
    print(f"\nStrong correlations found: {len(results['strong_correlations'])}")
    for sc in results["strong_correlations"][:10]:
        print(f"  {sc['feature']} × {sc['subject']}: r={sc['r']:.3f}, p={sc['p']:.4f}")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    run()
```

### 注意
- `output_hidden_states=True` 会让 inference 稍慢并占更多内存
- 每个 subject 只取 10 道题（共 570 次 inference，≈ 30 分钟 on CPU）
- 如果内存不够，可以分批处理（每次 10 个 subject）

---

## 任务 5：Neuron 490 消融实验（P2，预计 2-3 小时）

### 目标
验证跨层 hub neuron #490 的因果重要性。

### 新文件
`weight_graph/experiments/exp09_neuron_ablation.py`

### 方法

```python
"""
实验 9：Hub Neuron 消融

将 hidden_dim[490] 在所有层中置零，观察模型 perplexity 和 MMLU accuracy 变化。
对照：置零一个 random neuron（如 #100）。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np


def zero_out_neuron(model, neuron_idx: int):
    """
    在所有层的 MLP 中将指定 hidden dimension 的权重置零。
    
    对每层的 gate_proj, up_proj, down_proj：
    - gate_proj.weight[:, neuron_idx] = 0  （该维度的输入权重）
    - up_proj.weight[:, neuron_idx] = 0
    - down_proj.weight[neuron_idx, :] = 0  （该维度的输出权重）
    """
    for name, param in model.named_parameters():
        if "gate_proj.weight" in name or "up_proj.weight" in name:
            param.data[:, neuron_idx] = 0.0
        elif "down_proj.weight" in name:
            param.data[neuron_idx, :] = 0.0


def evaluate_perplexity(model, tokenizer, dataset_name="wikitext", 
                        config="wikitext-2-raw-v1", max_samples=200) -> float:
    """计算 perplexity。"""
    ds = load_dataset(dataset_name, config, split="test")
    text = "\n\n".join([s for s in ds["text"] if len(s.strip()) > 0][:max_samples])
    
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
    
    return float(torch.exp(outputs.loss))


def run():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    results = {}
    
    for condition, neuron_idx in [("baseline", None), ("hub_490", 490), ("control_100", 100)]:
        print(f"\nCondition: {condition}")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()
        
        if neuron_idx is not None:
            zero_out_neuron(model, neuron_idx)
            print(f"  Zeroed out neuron {neuron_idx} across all layers")
        
        ppl = evaluate_perplexity(model, tokenizer)
        print(f"  Perplexity: {ppl:.2f}")
        
        # 选几个有代表性的 MMLU subject 测试
        test_subjects = ["abstract_algebra", "high_school_biology", "philosophy", 
                         "computer_security", "us_foreign_policy"]
        mmlu_scores = {}
        for subject in test_subjects:
            acc = evaluate_mmlu_subject(model, tokenizer, subject)
            mmlu_scores[subject] = acc
            print(f"  {subject}: {acc:.3f}")
        
        results[condition] = {
            "neuron": neuron_idx,
            "perplexity": ppl,
            "mmlu_scores": mmlu_scores,
        }
        
        del model
    
    # 计算消融影响
    baseline_ppl = results["baseline"]["perplexity"]
    hub_ppl = results["hub_490"]["perplexity"]
    ctrl_ppl = results["control_100"]["perplexity"]
    
    results["analysis"] = {
        "hub_ppl_increase": hub_ppl - baseline_ppl,
        "control_ppl_increase": ctrl_ppl - baseline_ppl,
        "hub_is_more_important": (hub_ppl - baseline_ppl) > (ctrl_ppl - baseline_ppl),
    }
    
    save_results(results, Path("results/weight_graph/exp09/neuron_ablation.json"))
```

### 判断标准
- 如果 hub_490 消融后 PPL 增加 >> control_100 消融 → **H4 因果性确认**
- 如果两者差异不大 → PageRank 只是统计相关不是因果

---

## 执行顺序

```
任务 1 (exp03 补 seeds)      ─┐
                              ├── 并行
任务 2 (MMLU ground truth)   ─┘
                              ↓
任务 3 (逐层拓扑向量)        ← 依赖任务 1 的缓存（但不依赖其结果）
                              ↓
任务 4 (回归分析)             ← 依赖任务 2 + 任务 3
                              ↓
任务 5 (neuron ablation)     ← 独立，可和任务 3 并行
```

**最短路径**：任务 1 + 2 并行（1 天）→ 任务 3（半天）→ 任务 4（1 天）→ 任务 5（半天）= **3 天**

---

## 止损规则

| 检查点 | 条件 | 动作 |
|--------|------|------|
| 任务 1 完成后 | p > 0.1 | 暂停全部，重新评估稀疏化策略 |
| 任务 2 完成后 | 0.5B MMLU overall < 0.25 | 模型太弱，换用 1.5B 做后续 |
| 任务 4 完成后 | strong_correlations < 3 | 静态拓扑无法预测能力，转向运行时激活分析 |
| 任务 5 完成后 | hub_490 消融无显著影响 | PageRank 信号是噪声，放弃 H4 |

---

## 验收标准

全部完成后，应有以下文件：

```
results/weight_graph/
├── exp03/h1_results_5seed.json       ← 5-seed 统计显著的 H1 验证
├── exp06/mmlu_per_subject.json       ← 57 subjects 的能力指纹
├── exp07/layer_topo_vectors.json     ← 24 层的拓扑向量
├── exp07/topo_matrix.npy             ← [24, 10] numpy 矩阵
├── exp08/topo_ability_correlations.json  ← 核心发现
├── exp08/correlation_heatmap.png     ← 论文可用的图
├── exp09/neuron_ablation.json        ← hub neuron 因果验证
```

且以下结论至少成立一个：
1. H1 统计显著（p < 0.05）
2. 存在 >= 3 个 topo-feature × subject 强相关（|r| > 0.5, p < 0.05）
3. Neuron 490 消融后 PPL 增加 > 2× control
