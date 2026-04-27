
"""
Phase 0: 理论分析 - 分析 toy problem 的特殊性和特征适用性
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analyze_toy_problem():
    """分析 toy problem 的特殊性"""
    print("=" * 80)
    print("Phase 0.1: 分析 toy problem 的特殊性")
    print("=" * 80)
    
    rng = np.random.default_rng(seed=42)
    n_samples = 10000
    
    # 生成任务 0 和任务 1 的数据
    X = rng.normal(0.0, 1.0, size=(n_samples, 4))
    boundary = X[:, 0] + X[:, 1]
    y_task0 = (boundary > 0.0).astype(int)
    y_task1 = (boundary < 0.0).astype(int)
    
    print(f"\n任务 0 和任务 1 的决策边界：")
    print(f"  任务 0: y = (x[0] + x[1] > 0.0)")
    print(f"  任务 1: y = (x[0] + x[1] < 0.0)")
    
    print(f"\n输入分布统计：")
    print(f"  x[0] + x[1] 均值: {np.mean(boundary):.4f}")
    print(f"  x[0] + x[1] 标准差: {np.std(boundary):.4f}")
    print(f"  x[0] + x[1] 最小值: {np.min(boundary):.4f}")
    print(f"  x[0] + x[1] 最大值: {np.max(boundary):.4f}")
    
    print(f"\n任务 0 和任务 1 的标签统计：")
    print(f"  任务 0 正样本比例: {np.mean(y_task0):.4f}")
    print(f"  任务 1 正样本比例: {np.mean(y_task1):.4f}")
    
    # 分析 x[0]+x[1] 的符号是否能作为路由信号
    print(f"\n分析 x[0]+x[1] 符号作为路由信号：")
    task0_sign_pos = np.mean(boundary > 0.0)
    task0_sign_neg = np.mean(boundary < 0.0)
    print(f"  任务 0 中 x[0]+x[1] > 0 的比例: {task0_sign_pos:.4f}")
    print(f"  任务 0 中 x[0]+x[1] < 0 的比例: {task0_sign_neg:.4f}")
    
    # 计算理论上的完美路由准确率
    print(f"\n理论上的完美路由准确率：")
    print(f"  如果用 x[0]+x[1] 符号路由：")
    print(f"    - 当 x[0]+x[1] > 0 时，用快照专家（任务 0 专家）")
    print(f"    - 当 x[0]+x[1] < 0 时，用当前模型（任务 1 专家）")
    print(f"  理论准确率应该接近 Oracle 上界 0.7863")
    
    return X, y_task0, y_task1, boundary


def analyze_feature_applicability(X, y_task0, y_task1, boundary):
    """分析任务签名特征的适用性"""
    print("\n" + "=" * 80)
    print("Phase 0.2: 分析任务签名特征的适用性")
    print("=" * 80)
    
    # 计算各种特征
    n_samples = len(X)
    
    print(f"\n分析 6 维任务签名特征：")
    print(f"  1. conflict_score: 需要多个专家才有意义（当前只有两个专家）")
    print(f"  2. conflict_peak: 需要多个专家才有意义（当前只有两个专家）")
    print(f"  3. confidence: 模型预测的最大概率")
    print(f"  4. input_abs_mean: 输入绝对值的均值")
    print(f"  5. input_nonnegative_ratio: 输入非负的比例（最可能有用！）")
    print(f"  6. input_zero_ratio: 输入为零的比例（不太可能有用）")
    
    # 计算 input_nonnegative_ratio 在任务 0 和任务 1 上的分布
    print(f"\n分析 input_nonnegative_ratio：")
    input_nonnegative_ratio_task0 = np.mean(X &gt;= 0, axis=1)
    input_nonnegative_ratio_task1 = np.mean(X &gt;= 0, axis=1)
    
    print(f"  任务 0 input_nonnegative_ratio 均值: {np.mean(input_nonnegative_ratio_task0):.4f}")
    print(f"  任务 0 input_nonnegative_ratio 标准差: {np.std(input_nonnegative_ratio_task0):.4f}")
    print(f"  任务 1 input_nonnegative_ratio 均值: {np.mean(input_nonnegative_ratio_task1):.4f}")
    print(f"  任务 1 input_nonnegative_ratio 标准差: {np.std(input_nonnegative_ratio_task1):.4f}")
    
    # 注意：任务 0 和任务 1 的输入分布是相同的！
    print(f"\n⚠️  关键发现：")
    print(f"  任务 0 和任务 1 的输入分布是**完全相同**的！")
    print(f"  区别只在于标签函数（决策边界符号相反）")
    print(f"  这意味着基于输入统计的特征（input_abs_mean, input_nonnegative_ratio 等）")
    print(f"  在任务 0 和任务 1 上的分布是**完全相同**的！")
    
    # 分析 x[0]+x[1] 的符号与任务的关系
    print(f"\n分析 x[0]+x[1] 的符号：")
    print(f"  任务 0 的正确决策依赖于 x[0]+x[1] > 0")
    print(f"  任务 1 的正确决策依赖于 x[0]+x[1] < 0")
    print(f"  这意味着：")
    print(f"    - 对于 x[0]+x[1] &gt; 0 的样本，任务 0 专家更可能正确")
    print(f"    - 对于 x[0]+x[1] &lt; 0 的样本，任务 1 专家更可能正确")
    
    return {
        "input_nonnegative_ratio_task0": input_nonnegative_ratio_task0,
        "input_nonnegative_ratio_task1": input_nonnegative_ratio_task1,
    }


def predict_minimal_validation_effect():
    """预测最小可行验证的效果"""
    print("\n" + "=" * 80)
    print("Phase 0.3: 预测最小可行验证的效果")
    print("=" * 80)
    
    print(f"\n基于理论分析的预测：")
    print(f"\n1. 如果只用 input_nonnegative_ratio 做路由：")
    print(f"   - 预期效果：**几乎没有提升**")
    print(f"   - 原因：任务 0 和任务 1 的输入分布相同，input_nonnegative_ratio 无区分能力")
    
    print(f"\n2. 如果用 x[0]+x[1] 的符号做路由：")
    print(f"   - 预期效果：**显著提升**，接近 Oracle 上界 0.7863")
    print(f"   - 原因：这是完美的路由信号，直接反映了哪个专家更可能正确")
    
    print(f"\n3. 如果用 confidence + input_abs_mean + input_nonnegative_ratio：")
    print(f"   - 预期效果：**可能有轻微提升**")
    print(f"   - 原因：confidence 可能有一些区分能力，但不如直接用 x[0]+x[1] 符号")
    
    print(f"\n关键洞察：")
    print(f"  - toy problem 的特殊性在于：两个任务的输入分布完全相同")
    print(f"  - 区别只在于标签函数（决策边界符号相反）")
    print(f"  - 这意味着：")
    print(f"    - 基于输入统计的特征无效")
    print(f"    - 需要基于模型内部状态的特征（confidence, disagreement 等）")
    print(f"    - 或者直接用 x[0]+x[1] 符号（这是作弊，但可以验证上限）")


def main():
    X, y_task0, y_task1, boundary = analyze_toy_problem()
    features = analyze_feature_applicability(X, y_task0, y_task1, boundary)
    predict_minimal_validation_effect()
    
    print("\n" + "=" * 80)
    print("Phase 0 完成！")
    print("=" * 80)
    print("\n结论：")
    print("  1. toy problem 的特殊性：两个任务输入分布相同，仅标签函数相反")
    print("  2. 基于输入统计的特征（input_nonnegative_ratio 等）无效")
    print("  3. 需要基于模型内部状态的特征（confidence, disagreement）")
    print("  4. 或者直接用 x[0]+x[1] 符号作为路由信号（验证上限）")


if __name__ == "__main__":
    main()

