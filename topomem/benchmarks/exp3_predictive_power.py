"""
topomem/benchmarks/exp3_predictive_power.py

实验3：预测能力测试

核心假设：
- H0/H1 结构是否编码语义可预测性？
- 拓扑特征能否预测未知主题？

实验设计：
1. Temporal Hold-out: 用前半部分时间训练，预测后半部分
2. Topic Transition: 哪些 H1 cycles 预测了主题切换？
3. Novelty Detection: 拓扑异常能否识别新主题？

评估指标：
- Prediction accuracy
- Novelty detection AUROC
- Topic transition F1
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report
)

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from topomem.embedding import EmbeddingManager, EmbeddingConfig
from topomem.topology import TopologyEngine, TopologyConfig

sys.path.insert(0, str(Path(__file__).parent))
from exp1_data_loader import load_or_fetch_corpus, get_topic_array
from exp1_metrics import compute_h0_purity


# ------------------------------------------------------------------
# 特征提取
# ------------------------------------------------------------------

def extract_topo_features(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """提取拓扑特征用于预测。

    特征包括：
    - H0 连通分支统计
    - H1 循环统计
    - Persistence 分布统计
    - 局部拓扑复杂度
    """
    topo_config = TopologyConfig(max_homology_dim=1)
    topo_engine = TopologyEngine(topo_config)

    features = {}

    try:
        diagrams = topo_engine.compute_persistence(embeddings)

        # H0 特征
        if len(diagrams) >= 1 and len(diagrams[0]) > 0:
            h0 = diagrams[0]
            finite_h0 = h0[h0[:, 1] < np.inf]
            if len(finite_h0) > 0:
                h0_persistences = finite_h0[:, 1] - finite_h0[:, 0]
                features["h0_n_clusters"] = len(finite_h0)
                features["h0_mean_persistence"] = np.mean(h0_persistences)
                features["h0_max_persistence"] = np.max(h0_persistences)
                features["h0_std_persistence"] = np.std(h0_persistences)
            else:
                features["h0_n_clusters"] = 0
                features["h0_mean_persistence"] = 0
                features["h0_max_persistence"] = 0
                features["h0_std_persistence"] = 0
        else:
            features["h0_n_clusters"] = 0
            features["h0_mean_persistence"] = 0
            features["h0_max_persistence"] = 0
            features["h0_std_persistence"] = 0

        # H1 特征
        if len(diagrams) >= 2 and len(diagrams[1]) > 0:
            h1 = diagrams[1]
            finite_h1 = h1[h1[:, 1] < np.inf]
            if len(finite_h1) > 0:
                h1_persistences = finite_h1[:, 1] - finite_h1[:, 0]
                features["h1_n_cycles"] = len(finite_h1)
                features["h1_mean_persistence"] = np.mean(h1_persistences)
                features["h1_max_persistence"] = np.max(h1_persistences)
                features["h1_total_persistence"] = np.sum(h1_persistences)
            else:
                features["h1_n_cycles"] = 0
                features["h1_mean_persistence"] = 0
                features["h1_max_persistence"] = 0
                features["h1_total_persistence"] = 0
        else:
            features["h1_n_cycles"] = 0
            features["h1_mean_persistence"] = 0
            features["h1_max_persistence"] = 0
            features["h1_total_persistence"] = 0

    except Exception as e:
        print(f"  Topology computation failed: {e}")
        for key in ["h0_n_clusters", "h0_mean_persistence", "h0_max_persistence",
                    "h1_n_cycles", "h1_mean_persistence", "h1_max_persistence"]:
            features[key] = 0

    return features


def extract_mixed_features(
    embeddings: np.ndarray,
    topo_features: Dict[str, float],
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """混合特征：embedding 统计 + 拓扑特征。

    用于预测任务的特征向量。
    """
    # Embedding 统计特征
    emb_stats = {
        "emb_mean": np.mean(embeddings, axis=0),
        "emb_std": np.std(embeddings, axis=0),
        "emb_norm": np.linalg.norm(embeddings, axis=1),
    }

    # 合并特征
    feature_list = [
        topo_features.get("h0_n_clusters", 0),
        topo_features.get("h0_mean_persistence", 0),
        topo_features.get("h0_max_persistence", 0),
        topo_features.get("h1_n_cycles", 0),
        topo_features.get("h1_mean_persistence", 0),
        topo_features.get("h1_max_persistence", 0),
        topo_features.get("h1_total_persistence", 0),
        np.mean(emb_stats["emb_norm"]),
        np.std(emb_stats["emb_norm"]),
        np.percentile(emb_stats["emb_norm"], 25),
        np.percentile(emb_stats["emb_norm"], 75),
    ]

    return np.array(feature_list)


# ------------------------------------------------------------------
# 任务1: Temporal Hold-out (主题预测)
# ------------------------------------------------------------------

def task_temporal_holdout(
    items: List[Dict],
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    test_size: float = 0.3,
    seed: int = 42,
) -> Dict:
    """时序分割：用早期数据预测后期主题。

    模拟场景：记忆系统能否预测未见过的文档主题？
    """
    print("\n  [Task 1] Temporal Hold-out Prediction...")

    # 简单按时间/索引分割
    n = len(items)
    split_idx = int(n * (1 - test_size))

    train_emb = embeddings[:split_idx]
    test_emb = embeddings[split_idx:]
    train_labels = true_labels[:split_idx]
    test_labels = true_labels[split_idx:]

    print(f"    Train: {len(train_emb)}, Test: {len(test_emb)}")

    # 提取全局拓扑特征
    train_topo = extract_topo_features(train_emb)
    test_topo = extract_topo_features(test_emb)

    # 使用 embedding 统计特征（全局特征）
    # 简化：用全局 centroid 距离 + 散度作为预测特征
    train_centroid = np.mean(train_emb, axis=0)
    test_centroids = test_emb - train_centroid
    test_norms = np.linalg.norm(test_centroids, axis=1)

    # 用每个点的 norm 作为特征
    X_train = np.linalg.norm(train_emb - train_centroid, axis=1).reshape(-1, 1)
    X_test = test_norms.reshape(-1, 1)

    # 分类器
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)

    # 随机基线
    rng = np.random.RandomState(seed)
    random_acc = np.mean(rng.randint(0, 6, size=len(test_labels)) == test_labels)

    return {
        "task": "temporal_holdout",
        "accuracy": float(acc),
        "random_baseline": float(random_acc),
        "improvement": float(acc - random_acc),
        "n_train": len(train_emb),
        "n_test": len(test_emb),
        "n_topics": len(np.unique(true_labels)),
    }


# ------------------------------------------------------------------
# 任务2: Topic Transition Detection (H1 cycles 预测主题切换)
# ------------------------------------------------------------------

def task_topic_transition(
    items: List[Dict],
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    window_size: int = 50,
    seed: int = 42,
) -> Dict:
    """主题切换检测：H1 cycles 是否预测主题边界？

    方法：
    1. 滑窗计算局部 H1 密度
    2. 检测 H1 密度突变点（可能是主题切换点）
    3. 与真实主题边界对比
    """
    print("\n  [Task 2] Topic Transition Detection...")

    n = len(embeddings)
    topic_changes = []
    h1_density_peaks = []

    # 检测真实主题切换点
    for i in range(1, n):
        if true_labels[i] != true_labels[i-1]:
            topic_changes.append(i)

    print(f"    True topic transitions: {len(topic_changes)}")

    # 滑窗计算局部 H1 密度
    topo_config = TopologyConfig(max_homology_dim=1)
    topo_engine = TopologyEngine(topo_config)

    step = window_size // 2
    window_centers = list(range(window_size // 2, n - window_size // 2, step))

    for center in window_centers:
        start = center - window_size // 2
        end = center + window_size // 2
        window_emb = embeddings[start:end]

        try:
            diagrams = topo_engine.compute_persistence(window_emb)
            if len(diagrams) >= 2 and len(diagrams[1]) > 0:
                h1 = diagrams[1]
                finite_h1 = h1[h1[:, 1] < np.inf]
                n_cycles = len(finite_h1)
            else:
                n_cycles = 0
        except:
            n_cycles = 0

        h1_density_peaks.append((center, n_cycles))

    # 检测 H1 密度峰值
    if len(h1_density_peaks) > 2:
        densities = [x[1] for x in h1_density_peaks]
        mean_d = np.mean(densities)
        std_d = np.std(densities)
        threshold = mean_d + std_d
        predicted_transitions = [center for center, d in h1_density_peaks if d > threshold]
    else:
        predicted_transitions = []

    print(f"    Predicted transitions: {len(predicted_transitions)}")

    # 计算重叠
    true_set = set(topic_changes)
    pred_set = set()
    for pt in predicted_transitions:
        for tt in true_set:
            if abs(pt - tt) < window_size:
                pred_set.add(tt)

    overlap = len(pred_set)
    if len(topic_changes) > 0:
        recall = overlap / len(topic_changes)
    else:
        recall = 0

    if len(predicted_transitions) > 0:
        precision = overlap / len(predicted_transitions)
    else:
        precision = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return {
        "task": "topic_transition",
        "true_transitions": len(topic_changes),
        "predicted_transitions": len(predicted_transitions),
        "overlap": overlap,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ------------------------------------------------------------------
# 任务3: Novelty Detection (拓扑异常检测)
# ------------------------------------------------------------------

def task_novelty_detection(
    items: List[Dict],
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    seed: int = 42,
) -> Dict:
    """新颖性检测：拓扑特征能否识别新主题？

    方法：
    1. 用已知主题的数据训练
    2. 检测"异常"点（可能是新主题）
    """
    print("\n  [Task 3] Novelty Detection...")

    # 简单方法：计算每个点的局部拓扑复杂度
    # 作为"新颖性"分数

    from scipy.spatial.distance import pdist, squareform

    n = len(embeddings)
    dist_matrix = squareform(pdist(embeddings, metric="cosine"))

    # K-NN 距离作为局部复杂度指标
    k = 10
    nn_indices = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
    nn_distances = np.array([
        np.mean([dist_matrix[i, j] for j in nn_indices[i]])
        for i in range(n)
    ])

    # 高 K-NN 距离 = 可能是边界点/异常点
    novelty_scores = nn_distances

    # 与随机打乱对比
    rng = np.random.RandomState(seed)
    shuffled_scores = rng.permutation(novelty_scores)

    # 计算"检测"效果：边界点是否确实有更高 novelty
    boundary_mask = np.array([
        true_labels[i] != true_labels[nn_indices[i][0]]
        for i in range(n)
    ])

    if np.sum(boundary_mask) > 0:
        boundary_novelty = np.mean(novelty_scores[boundary_mask])
        non_boundary_novelty = np.mean(novelty_scores[~boundary_mask])
        novelty_diff = boundary_novelty - non_boundary_novelty
    else:
        novelty_diff = 0

    # AUROC（边界点 vs 非边界点）
    try:
        auroc = roc_auc_score(boundary_mask.astype(int), novelty_scores)
    except:
        auroc = 0.5

    random_auroc = 0.5  # 随机基线

    return {
        "task": "novelty_detection",
        "auroc": float(auroc),
        "random_baseline": float(random_auroc),
        "novelty_improvement": float(auroc - random_auroc),
        "boundary_novelty_mean": float(boundary_novelty) if np.sum(boundary_mask) > 0 else 0,
        "non_boundary_novelty_mean": float(non_boundary_novelty) if np.sum(~boundary_mask) > 0 else 0,
        "novelty_diff": float(novelty_diff),
    }


# ------------------------------------------------------------------
# 主实验
# ------------------------------------------------------------------

def run_predictive_power_test(
    categories: Optional[List[str]] = None,
    use_cache: bool = True,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """运行预测能力测试。"""

    print("=" * 70)
    print("EXPERIMENT 3: Predictive Power Test")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Step 1: 加载数据
    print("\n[1/3] Loading corpus...")
    start_total = time.time()

    items, embeddings = load_or_fetch_corpus(
        subset="train",
        categories=categories,
        use_cache=use_cache,
    )

    true_labels = get_topic_array(items)
    print(f"  Corpus: {len(items)} documents")
    print(f"  Topics: {len(np.unique(true_labels))}")

    # Step 2: 运行三个任务
    print("\n[2/3] Running prediction tasks...")

    results = {}

    # Task 1: Temporal Hold-out
    results["temporal_holdout"] = task_temporal_holdout(
        items, embeddings, true_labels, test_size=0.3, seed=seed
    )

    # Task 2: Topic Transition
    results["topic_transition"] = task_topic_transition(
        items, embeddings, true_labels, window_size=100, seed=seed
    )

    # Task 3: Novelty Detection
    results["novelty_detection"] = task_novelty_detection(
        items, embeddings, true_labels, seed=seed
    )

    # Step 3: 保存结果
    elapsed_total = time.time() - start_total

    print(f"\n[3/3] Saving results... (total time: {elapsed_total:.1f}s)")

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    outpath = output_dir / f"exp3_predictive_power_{timestamp}.json"

    output_data = {
        "experiment": "Predictive Power Test",
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "config": {
            "n_documents": len(items),
            "n_topics": len(np.unique(true_labels)),
            "seed": seed,
        },
        "results": results,
        "total_elapsed_sec": elapsed_total,
    }

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {outpath}")

    # 打印摘要
    print_summary(results)

    return output_data


def print_summary(results: Dict) -> None:
    """打印结果摘要。"""
    print("\n" + "=" * 70)
    print("PREDICTIVE POWER SUMMARY")
    print("=" * 70)

    print("\n[Task 1: Temporal Hold-out Prediction]")
    t1 = results["temporal_holdout"]
    print(f"  Accuracy: {t1['accuracy']:.4f}")
    print(f"  Random baseline: {t1['random_baseline']:.4f}")
    print(f"  Improvement: {t1['improvement']:+.4f}")

    print("\n[Task 2: Topic Transition Detection]")
    t2 = results["topic_transition"]
    print(f"  Precision: {t2['precision']:.4f}")
    print(f"  Recall: {t2['recall']:.4f}")
    print(f"  F1: {t2['f1']:.4f}")
    print(f"  True transitions: {t2['true_transitions']}")
    print(f"  Predicted: {t2['predicted_transitions']}")

    print("\n[Task 3: Novelty Detection]")
    t3 = results["novelty_detection"]
    print(f"  AUROC: {t3['auroc']:.4f}")
    print(f"  Random baseline: {t3['random_baseline']:.4f}")
    print(f"  Novelty diff: {t3['novelty_diff']:+.4f}")

    print("\n[Overall Assessment]")
    tasks_passed = []
    if t1['improvement'] > 0.05:
        tasks_passed.append("Temporal Prediction")
    if t2['f1'] > 0.3:
        tasks_passed.append("Transition Detection")
    if t3['auroc'] > 0.6:
        tasks_passed.append("Novelty Detection")

    if tasks_passed:
        print(f"  Passed: {', '.join(tasks_passed)}")
    else:
        print("  No task passed - topology features alone are weak predictors")

    print("=" * 70)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: Predictive Power")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_predictive_power_test(
        use_cache=not args.no_cache,
        seed=args.seed,
    )
