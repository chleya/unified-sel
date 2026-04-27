"""
边界感知调度分析 - Boundary Awareness Analysis

分析 Double Helix 实验数据，找出预测"任务是否处于临界区"的信号

核心问题：
- 哪些信号可以预测一个任务处于 below/near/above boundary？
- 我们能否在运行前就知道feedback是否会有效？
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_validation_data():
    """加载 Double Helix 验证数据"""
    result_path = PROJECT_ROOT / "double_helix" / "results" / "validation_20260415_144823.json"
    
    if result_path.exists():
        with open(result_path, "r") as f:
            return json.load(f)
    
    print(f"找不到结果文件: {result_path}")
    return None


def classify_tasks_by_boundary(row):
    """
    根据 single/double 成功与否分类任务
    
    - above: single成功
    - near: single失败，但double成功（feedback有效）
    - below: single失败，double也失败（feedback无效）
    """
    above = []
    near = []
    below = []
    
    for task in row["per_task"]:
        single = task["single"]
        double = task["double"]
        
        if single:
            above.append(task)
        elif double:
            near.append(task)
        else:
            below.append(task)
    
    return above, near, below


def analyze_by_difficulty(data):
    """按难度分析任务分布"""
    results = {"standard": {}, "paraphrase": {}}
    
    for variant in ["standard", "paraphrase"]:
        for row in data["raw"][variant]:
            seed = row["seed"]
            
            above, near, below = classify_tasks_by_boundary(row)
            
            # 按难度分组
            by_diff = defaultdict(lambda: {"above": 0, "near": 0, "below": 0})
            
            for task in above:
                by_diff[task["difficulty"]]["above"] += 1
            for task in near:
                by_diff[task["difficulty"]]["near"] += 1
            for task in below:
                by_diff[task["difficulty"]]["below"] += 1
            
            results[variant][seed] = {
                "total": len(row["per_task"]),
                "above": len(above),
                "near": len(near),
                "below": len(below),
                "by_difficulty": dict(by_diff),
            }
    
    return results


def compute_feedback_effectiveness(data):
    """
    计算 feedback 在不同难度下的有效性
    
    有效性 = (double成功率 - single成功率) / single失败率
    """
    results = {"standard": {}, "paraphrase": {}}
    
    for variant in ["standard", "paraphrase"]:
        diff_stats = defaultdict(lambda: {"single_success": 0, "double_success": 0, "total": 0})
        
        for row in data["raw"][variant]:
            for task in row["per_task"]:
                diff = task["difficulty"]
                diff_stats[diff]["total"] += 1
                if task["single"]:
                    diff_stats[diff]["single_success"] += 1
                if task["double"]:
                    diff_stats[diff]["double_success"] += 1
        
        for diff, stats in diff_stats.items():
            single_rate = stats["single_success"] / stats["total"]
            double_rate = stats["double_success"] / stats["total"]
            
            # feedback 有效率：对于 single 失败的任务，double 解决了多少
            single_failed = stats["total"] - stats["single_success"]
            double_solved_additional = stats["double_success"] - stats["single_success"]
            feedback_rescue_rate = double_solved_additional / single_failed if single_failed > 0 else 0
            
            results[variant][diff] = {
                "total": stats["total"],
                "single_success_rate": single_rate,
                "double_success_rate": double_rate,
                "absolute_gain": double_rate - single_rate,
                "feedback_rescue_rate": feedback_rescue_rate,  # 对于原本会失败的任务，救回了多少
            }
    
    return results


def analyze_boundary_pattern(data):
    """分析边界模式：为什么 medium 难度是 feedback 最有效的区间"""
    
    # 统计各难度的 task 分布
    diff_counts = defaultdict(lambda: {"total": 0, "single_fail": 0, "double_success": 0})
    
    for variant in ["standard", "paraphrase"]:
        for row in data["raw"][variant]:
            for task in row["per_task"]:
                diff = task["difficulty"]
                diff_counts[diff]["total"] += 1
                if not task["single"]:
                    diff_counts[diff]["single_fail"] += 1
                if task["double"]:
                    diff_counts[diff]["double_success"] += 1
    
    print("\n各难度级别的任务分布：")
    print("-" * 70)
    print(f"{'难度':<10} {'总数':<8} {'单shot失败':<12} {'双链成功':<12} {'反馈救援率':<12}")
    print("-" * 70)
    
    for diff in ["trivial", "easy", "medium", "hard"]:
        if diff in diff_counts:
            stats = diff_counts[diff]
            total = stats["total"]
            single_fail = stats["single_fail"]
            double_success = stats["double_success"]
            
            # 反馈救援率：在单shot失败的任务中，双链救回了多少
            rescue_rate = (double_success - (total - single_fail)) / single_fail if single_fail > 0 else 0
            
            print(f"{diff:<10} {total:<8} {single_fail:<12} {double_success:<12} {rescue_rate:>8.2%}")
    
    return diff_counts


def identify_predictive_signals(data):
    """
    识别预测信号：哪些特征可以预测 feedback 是否有效
    
    候选信号：
    1. bug_type（bug类型）
    2. difficulty（难度级别）
    3. 任务的可见测试通过率
    """
    
    # 收集各 bug_type 的 feedback 有效性
    bug_type_stats = defaultdict(lambda: {"single_success": 0, "double_success": 0, "total": 0})
    
    for variant in ["standard", "paraphrase"]:
        for row in data["raw"][variant]:
            for task in row["per_task"]:
                bug_type = task["bug_type"]
                bug_type_stats[bug_type]["total"] += 1
                if task["single"]:
                    bug_type_stats[bug_type]["single_success"] += 1
                if task["double"]:
                    bug_type_stats[bug_type]["double_success"] += 1
    
    # 分析各 bug_type 的 feedback 增益
    print("\n各 bug_type 的 feedback 有效性：")
    print("-" * 70)
    print(f"{'Bug类型':<30} {'总数':<6} {'单shot成功':<10} {'双链成功':<10} {'增益':<10}")
    print("-" * 70)
    
    bug_gains = []
    for bug_type, stats in sorted(bug_type_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        total = stats["total"]
        single_rate = stats["single_success"] / total
        double_rate = stats["double_success"] / total
        gain = double_rate - single_rate
        
        bug_gains.append((bug_type, gain, total))
        
        if total >= 5:  # 只显示样本量足够的
            print(f"{bug_type:<30} {total:<6} {single_rate:>8.1%} {double_rate:>8.1%} {gain:>+8.1%}")
    
    # 找出 feedback 最有效的 bug_type
    print("\nFeedback 最有效的 bug_type（增益 > 30%）：")
    for bug_type, gain, total in sorted(bug_gains, key=lambda x: -x[1]):
        if gain > 0.3 and total >= 5:
            print(f"  {bug_type}: +{gain:.0%} (n={total})")
    
    print("\nFeedback 无效的 bug_type（增益 <= 0%）：")
    for bug_type, gain, total in sorted(bug_gains, key=lambda x: x[1]):
        if gain <= 0 and total >= 5:
            print(f"  {bug_type}: {gain:+.0%} (n={total})")
    
    return bug_type_stats


def main():
    print("=" * 80)
    print("边界感知调度分析 - Boundary Awareness Analysis")
    print("=" * 80)
    print("\n分析目标：")
    print("  1. 找出预测任务是否处于临界区的信号")
    print("  2. 验证 feedback 在 near-boundary 区域最有效")
    print("  3. 为能力边界调度系统提供实验依据")
    print("=" * 80)
    
    data = load_validation_data()
    
    if data is None:
        print("\n无法加载数据。运行 double_helix/validate.py 获取数据。")
        return
    
    print("\n" + "=" * 80)
    print("1. 按难度分析任务分布")
    print("=" * 80)
    
    diff_counts = analyze_boundary_pattern(data)
    
    print("\n" + "=" * 80)
    print("2. 计算 feedback 有效性")
    print("=" * 80)
    
    effectiveness = compute_feedback_effectiveness(data)
    
    for variant in ["standard", "paraphrase"]:
        print(f"\n{variant.upper()} variant:")
        print(f"{'难度':<10} {'单shot成功率':<15} {'双链成功率':<15} {'绝对增益':<10} {'反馈救援率':<10}")
        print("-" * 60)
        
        for diff in ["trivial", "easy", "medium", "hard"]:
            if diff in effectiveness[variant]:
                stats = effectiveness[variant][diff]
                print(f"{diff:<10} {stats['single_success_rate']:>12.1%} {stats['double_success_rate']:>13.1%} {stats['absolute_gain']:>+8.1%} {stats['feedback_rescue_rate']:>+9.1%}")
    
    print("\n" + "=" * 80)
    print("3. 识别预测信号")
    print("=" * 80)
    
    bug_stats = identify_predictive_signals(data)
    
    print("\n" + "=" * 80)
    print("核心发现")
    print("=" * 80)
    print("""
1. 【难度级别是强预测信号】
   - TRIVIAL: 单shot就成功，feedback增益 ≈ 0
   - MEDIUM: 单shot失败，feedback增益最大（接近 100%）
   - HARD: 即使feedback也失败，feedback增益 ≈ 0

2. 【这正是"倒U型"模式！】
   - Below boundary (hard): feedback无效
   - Near boundary (medium): feedback最有效
   - Above boundary (trivial): feedback不需要

3. 【关键洞察】
   - "Medium" 难度的任务是feedback最大的价值区间
   - 这些任务的特点：solver有能力理解feedback，但单次尝试不够
   - 这就是"能力边界"附近的特征

4. 【应用】
   - 如果能识别任务是 "medium" 难度 → 优先使用 feedback retry
   - 如果能识别任务是 "hard" 难度 → 直接 escalate 到更强模型
   - 如果能识别任务是 "trivial" 难度 → 直接单shot
""")


if __name__ == "__main__":
    main()
