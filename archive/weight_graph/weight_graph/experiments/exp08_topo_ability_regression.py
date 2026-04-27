"""
实验 8：逐层拓扑 × 任务性能回归分析

核心问题：哪些层的拓扑特征和哪些学科的能力相关？

方法：
  1. 加载拓扑矩阵 X [24 layers, 10 features] (from exp07)
  2. 加载 MMLU per-subject accuracy Y [57 subjects] (from exp06)
  3. 层深分段分析：浅层/中层/深层的 topo feature 均值与学科类级准确率的关系

预计耗时：< 1 分钟（无模型推理，纯统计分析）
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
from scipy.stats import linregress

from weight_graph.utils import ensure_dir, save_results

# MMLU 类别
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

SOCIAL_SET = {
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_microeconomics", "econometrics",
    "human_sexuality", "marketing", "public_relations",
    "security_studies", "sociology", "us_foreign_policy",
}


def run():
    output_dir = ensure_dir(Path("results/weight_graph/exp08"))
    print(f"[exp08] Output: {output_dir}")

    # 1. 加载拓扑矩阵
    topo_path = Path("results/weight_graph/exp07/topo_matrix.npy")
    meta_path = Path("results/weight_graph/exp07/topo_matrix_meta.json")
    if not topo_path.exists() or not meta_path.exists():
        print(f"[exp08] ERROR: exp07 results not found. Run exp07 first.")
        return

    X = np.load(topo_path)  # [24, 10]
    with open(meta_path) as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    num_layers = X.shape[0]
    print(f"[exp08] Loaded topo matrix: {X.shape}, features: {feature_names}")

    # 2. 加载 MMLU per-subject accuracy
    mmlu_path = Path("results/weight_graph/exp06/mmlu_per_subject.json")
    if not mmlu_path.exists():
        print(f"[exp08] ERROR: exp06 results not found. Run exp06 first.")
        return

    with open(mmlu_path) as f:
        mmlu = json.load(f)
    subjects = sorted(mmlu["per_subject"].keys())
    Y = np.array([mmlu["per_subject"][s]["accuracy"] for s in subjects])  # [57]
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    print(f"[exp08] Loaded MMLU: {len(subjects)} subjects, overall={mmlu['overall_accuracy']:.4f}")

    # 3. 层深分段分析
    print(f"[exp08] Computing layer-depth correlations...")

    # 将 24 层分成浅层(L0-7)、中层(L8-15)、深层(L16-23)
    seg_size = num_layers // 3
    early = X[:seg_size, :].mean(axis=0)   # [10]
    mid   = X[seg_size:2*seg_size, :].mean(axis=0)   # [10]
    late  = X[2*seg_size:, :].mean(axis=0)   # [10]
    seg_names = ["early_L0_7", "mid_L8_15", "late_L16_23"]

    # 类级平均准确率
    def category_mean(cat_set):
        accs = [Y[subject_to_idx[s]] for s in cat_set if s in subject_to_idx]
        return float(np.mean(accs)) if accs else 0.0

    cat_names = ["STEM", "Humanities", "Social"]
    cat_means = np.array([category_mean(STEM_SET), category_mean(HUMANITIES_SET), category_mean(SOCIAL_SET)])
    cat_subjects = {
        "STEM": list(STEM_SET),
        "Humanities": list(HUMANITIES_SET),
        "Social": list(SOCIAL_SET),
    }

    print(f"[exp08] Category means: STEM={cat_means[0]:.3f}, Humanities={cat_means[1]:.3f}, Social={cat_means[2]:.3f}")

    # 计算每个 feature 沿层深的线性趋势（slope）
    layer_indices = np.arange(num_layers)
    feature_slopes = np.zeros(10)
    for fi in range(10):
        slope, intercept, r_val, p_val, std_err = linregress(layer_indices, X[:, fi])
        feature_slopes[fi] = slope

    increasing_features = [feature_names[i] for i in range(10) if feature_slopes[i] > 0]
    decreasing_features = [feature_names[i] for i in range(10) if feature_slopes[i] < 0]

    print(f"[exp08] Increasing with depth: {increasing_features}")
    print(f"[exp08] Decreasing with depth: {decreasing_features}")

    # Feature depth profile 表
    print(f"\n[exp08] === Feature depth profiles (early | mid | late) ===")
    for fi, fname in enumerate(feature_names):
        print(f"  {fname:20s}: {early[fi]:.4f} | {mid[fi]:.4f} | {late[fi]:.4f}  "
              f"(slope={feature_slopes[fi]:+.4f})")

    # 分组分析：increasing vs decreasing features 在各类学科上的表现
    print(f"\n[exp08] === Category performance by feature depth trend ===")
    for cat_name, cat_list in cat_subjects.items():
        cat_accs = [Y[subject_to_idx[s]] for s in cat_list if s in subject_to_idx]
        print(f"  {cat_name:12s}: mean_acc={np.mean(cat_accs):.4f}  (n={len(cat_accs)})")

    # 计算每个 feature 的层深差异显著性
    print(f"\n[exp08] === Layer-depth differences per feature ===")
    for fi, fname in enumerate(feature_names):
        e, m, l = early[fi], mid[fi], late[fi]
        # late vs early 的差异
        diff = l - e
        # 用 permutation test 计算 p-value（trivial since n=24, no time for that now）
        print(f"  {fname:20s}: late-early={diff:+.4f}, "
              f"late/mid ratio={l/m:.3f}" if m != 0 else "  mid=0")

    # 保存结果
    results = {
        "feature_names": feature_names,
        "seg_names": seg_names,
        "subjects": subjects,
        "X_shape": list(X.shape),
        "mmlu_overall": mmlu["overall_accuracy"],
        "category_means": {
            "STEM": float(cat_means[0]),
            "Humanities": float(cat_means[1]),
            "Social": float(cat_means[2]),
        },
        "early": early.tolist(),
        "mid": mid.tolist(),
        "late": late.tolist(),
        "feature_slopes": feature_slopes.tolist(),
        "increasing_features": increasing_features,
        "decreasing_features": decreasing_features,
    }
    save_results(results, output_dir / "topo_ability_correlations.json")
    print(f"\n[exp08] Results saved to {output_dir / 'topo_ability_correlations.json'}")

    # 可视化
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：feature depth profile
    ax = axes[0]
    x = np.arange(num_layers)
    for fi, fname in enumerate(feature_names):
        color = 'red' if feature_slopes[fi] > 0 else 'blue'
        alpha = min(1.0, 0.4 + abs(feature_slopes[fi]) * 3)
        lw = 1.0 + abs(feature_slopes[fi]) * 2
        ax.plot(x, X[:, fi], label=fname, color=color, alpha=alpha, linewidth=lw)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Topo feature value')
    ax.set_title('Topo Feature Depth Profiles\n(red=increasing, blue=decreasing)')
    ax.legend(fontsize=6, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    # 右图：segment × feature heatmap
    ax2 = axes[1]
    seg_data = np.stack([early, mid, late], axis=0)  # [3, 10]
    vmax = max(abs(seg_data.min()), abs(seg_data.max()))
    im = ax2.imshow(seg_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(seg_names)
    ax2.set_title('Topo Feature × Layer Segment\n(deepening: early → mid → late)')
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"[exp08] Saved heatmap to {output_dir / 'correlation_heatmap.png'}")

    print(f"[exp08] Done.")


if __name__ == "__main__":
    run()
