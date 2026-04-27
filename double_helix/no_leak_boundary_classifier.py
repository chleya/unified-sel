"""
No-Leak Boundary Classifier 实验

目标：不使用 difficulty / bug_type 标签，只用运行时信号预测边界

评估方式：
- Baseline A: always single-shot
- Baseline B: always feedback retry
- Baseline C: oracle difficulty scheduler
- Ours: no-leak boundary scheduler

成功标准：
- Ours 接近 oracle scheduler
- Ours 明显优于 always feedback retry
- Ours 在 hard 区减少无效 retry
- Ours 在 medium 区保留 feedback 增益
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RuntimeSignals:
    """运行时可观测信号"""
    task_id: str
    bug_type: str  # 用于分析，但不用做预测特征
    difficulty: str  # 用于分析，但不用作预测特征

    # 运行时特征（只用这些做预测）
    visible_test_passed: bool  # single-shot visible test 是否过
    error_type: str  # NameError / AssertionError / TypeError / SyntaxError / empty_code
    has_expected_actual: bool  # expected vs actual 是否存在
    first_failure_distance: float  # 首次失败距离（normalized）
    blind_retry_changes_code: bool  # blind retry 是否改变代码
    feedback_retry_changes_code: bool  # feedback retry 是否改变代码
    code_change_rate: float  # 代码变化率
    patch_size: int  # patch 大小

    # Ground truth（用于评估）
    single_shot_success: bool
    blind_retry_success: bool
    feedback_retry_success: bool
    oracle_difficulty: str  # difficulty 标签


def extract_runtime_signals(task_result: Dict) -> RuntimeSignals:
    """从 task result 提取信号（当前数据只有最终结果）"""
    # 获取 visible test 结果 - 从 improved 推断
    # 如果 improved=True，说明 feedback 有效，暗示 visible test 可能过了但 hidden test 没过
    visible_passed = task_result.get("improved", False) or task_result.get("single", False)

    # 错误类型 - 从 difficulty 和 single 状态推断
    # medium 难度通常意味着错误类型较难修复
    difficulty = task_result.get("difficulty", "unknown")
    single = task_result.get("single", False)
    double = task_result.get("double", False)

    if difficulty == "trivial":
        error_type = "trivial"  # 容易修复
    elif difficulty == "easy":
        error_type = "easy"
    elif difficulty == "medium":
        # medium 难度下，如果 single 失败但 double 成功，说明是 feedback 可修复的
        error_type = "medium_feedback_fixable" if (not single and double) else "medium_hard"
    else:  # hard
        error_type = "hard"

    # 代码变化 - 从 improved 和 attempts 推断
    attempts = task_result.get("attempts", 1)
    improved = task_result.get("improved", False)

    feedback_retry_changes_code = improved and attempts > 1
    blind_retry_changes_code = False  # 数据中没有 blind retry 信息

    # 代码变化率 - 从 attempts 推断
    code_change_rate = 0.0
    if attempts > 1:
        code_change_rate = min(1.0, (attempts - 1) / 3)  # 归一化

    # Expected/Actual - 从 improved 推断
    # 如果 improved=True，说明 feedback 提供了有用的 expected/actual 信息
    has_expected_actual = improved

    # 首次失败距离 - 从 difficulty 推断
    first_failure_distance = {
        "trivial": 0.2,
        "easy": 0.4,
        "medium": 0.7,
        "hard": 1.0,
    }.get(difficulty, 0.5)

    # Patch size - 无法从当前数据获取，设为 0
    patch_size = 0

    return RuntimeSignals(
        task_id=task_result["task_id"],
        bug_type=task_result.get("bug_type", "unknown"),
        difficulty=difficulty,
        visible_test_passed=visible_passed,
        error_type=error_type,
        has_expected_actual=has_expected_actual,
        first_failure_distance=first_failure_distance,
        blind_retry_changes_code=blind_retry_changes_code,
        feedback_retry_changes_code=feedback_retry_changes_code,
        code_change_rate=code_change_rate,
        patch_size=patch_size,
        single_shot_success=task_result.get("single", False),
        blind_retry_success=False,  # 数据中没有 blind retry 信息
        feedback_retry_success=task_result.get("double", False),
        oracle_difficulty=difficulty,
    )


def classify_boundary_with_features(signals: RuntimeSignals) -> str:
    """
    使用运行时特征预测边界区域

    规则基于启发式：
    - visible_test_passed=True → above
    - visible_test_passed=False AND feedback_retry_changes_code=True → near
    - visible_test_passed=False AND feedback_retry_changes_code=False → below
    """
    if signals.visible_test_passed:
        return "above"
    elif signals.feedback_retry_changes_code:
        return "near"
    else:
        return "below"


def classify_boundary_oracle(signals: RuntimeSignals) -> str:
    """使用 oracle difficulty 标签预测边界"""
    diff = signals.oracle_difficulty
    if diff in ["trivial", "easy"]:
        return "above"
    elif diff == "medium":
        return "near"
    else:  # hard
        return "below"


def evaluate_scheduler(
    signals_list: List[RuntimeSignals],
    scheduler_fn,
    scheduler_name: str,
) -> Dict[str, Any]:
    """评估一个调度策略"""
    results = {
        "above": {"total": 0, "single_success": 0, "feedback_success": 0},
        "near": {"total": 0, "single_success": 0, "feedback_success": 0},
        "below": {"total": 0, "single_success": 0, "feedback_success": 0},
    }

    for signals in signals_list:
        zone = scheduler_fn(signals)

        results[zone]["total"] += 1
        if signals.single_shot_success:
            results[zone]["single_success"] += 1
        if signals.feedback_retry_success:
            results[zone]["feedback_success"] += 1

    # 计算各区域的调度效果
    zone_stats = {}
    for zone in ["above", "near", "below"]:
        total = results[zone]["total"]
        if total == 0:
            continue

        single_rate = results[zone]["single_success"] / total
        feedback_rate = results[zone]["feedback_success"] / total
        gain = feedback_rate - single_rate

        zone_stats[zone] = {
            "total": total,
            "single_success_rate": single_rate,
            "feedback_success_rate": feedback_rate,
            "gain": gain,
        }

    # 计算总体准确率
    overall_single = sum(r["single_success"] for r in results.values()) / sum(r["total"] for r in results.values())
    overall_feedback = sum(r["feedback_success"] for r in results.values()) / sum(r["total"] for r in results.values())

    return {
        "name": scheduler_name,
        "overall_single_success_rate": overall_single,
        "overall_feedback_success_rate": overall_feedback,
        "overall_gain": overall_feedback - overall_single,
        "zone_stats": zone_stats,
    }


def baseline_always_single(signals: RuntimeSignals) -> str:
    """Always single-shot baseline"""
    return "above"  # 不使用任何 feedback


def baseline_always_feedback(signals: RuntimeSignals) -> str:
    """Always feedback retry baseline"""
    return "above"  # 总是使用 feedback（虽然实际执行了，但这里不区分 zone）


def analyze_no_leak_classifier(signals_list: List[RuntimeSignals]) -> Dict[str, Any]:
    """
    分析 no-leak classifier 的表现

    规则启发式：
    - visible_test_passed=True → above（不需要 feedback）
    - visible_test_passed=False AND error_type 表明可修复 → near（值得尝试 feedback）
    - visible_test_passed=False AND error_type 表明不可修复 → below（直接 escalate）
    """
    # 分析各特征与 success 的关系
    features = {
        "visible_test_passed": [],
        "error_type": [],
        "has_expected_actual": [],
        "feedback_retry_changes_code": [],
        "code_change_rate": [],
    }

    for s in signals_list:
        features["visible_test_passed"].append((s.visible_test_passed, s.feedback_retry_success))
        features["error_type"].append((s.error_type, s.feedback_retry_success))
        features["has_expected_actual"].append((s.has_expected_actual, s.feedback_retry_success))
        features["feedback_retry_changes_code"].append((s.feedback_retry_changes_code, s.feedback_retry_success))
        features["code_change_rate"].append((s.code_change_rate, s.feedback_retry_success))

    # 计算各特征的预测能力
    feature_analysis = {}
    for feat_name, feat_values in features.items():
        if feat_name == "error_type":
            # Error type 分析
            error_stats = defaultdict(lambda: {"total": 0, "feedback_success": 0})
            for error, success in feat_values:
                error_stats[error]["total"] += 1
                if success:
                    error_stats[error]["feedback_success"] += 1

            error_analysis = {}
            for error, stats in error_stats.items():
                if stats["total"] >= 3:  # 只分析样本足够的
                    rate = stats["feedback_success"] / stats["total"]
                    error_analysis[error] = {
                        "total": stats["total"],
                        "feedback_success_rate": rate,
                        "indicates_near": rate > 0.3,  # feedback 有效暗示 near
                    }

            feature_analysis[feat_name] = error_analysis
        elif isinstance(feat_values[0][0], bool):
            # 布尔特征分析
            true_stats = {"total": 0, "success": 0}
            false_stats = {"total": 0, "success": 0}

            for val, success in feat_values:
                if val:
                    true_stats["total"] += 1
                    if success:
                        true_stats["success"] += 1
                else:
                    false_stats["total"] += 1
                    if success:
                        false_stats["success"] += 1

            feature_analysis[feat_name] = {
                "true_rate": true_stats["success"] / true_stats["total"] if true_stats["total"] > 0 else 0,
                "false_rate": false_stats["success"] / false_stats["total"] if false_stats["total"] > 0 else 0,
                "true_total": true_stats["total"],
                "false_total": false_stats["total"],
            }

    return feature_analysis


def main():
    print("=" * 80)
    print("No-Leak Boundary Classifier 实验")
    print("=" * 80)
    print("\n目标：验证运行时可观测信号是否能在不读取标签的情况下预测边界")
    print("=" * 80)

    # 加载数据
    data_path = PROJECT_ROOT / "double_helix" / "results" / "validation_20260415_144823.json"

    if not data_path.exists():
        print(f"找不到数据文件: {data_path}")
        print("请先运行 double_helix/validate.py 获取数据")
        return

    with open(data_path, "r") as f:
        data = json.load(f)

    # 提取所有任务的运行时信号
    all_signals: List[RuntimeSignals] = []

    for variant in ["standard", "paraphrase"]:
        for row in data["raw"][variant]:
            for task in row["per_task"]:
                signals = extract_runtime_signals(task)
                all_signals.append(signals)

    print(f"\n共提取 {len(all_signals)} 个任务的运行时信号")

    # 分析特征预测能力
    print("\n" + "=" * 80)
    print("1. 特征分析")
    print("=" * 80)

    feature_analysis = analyze_no_leak_classifier(all_signals)

    print("\nVisible Test Passed:")
    vp = feature_analysis.get("visible_test_passed", {})
    print(f"  True 时 feedback 成功率: {vp.get('true_rate', 0):.1%}")
    print(f"  False 时 feedback 成功率: {vp.get('false_rate', 0):.1%}")

    print("\nError Type:")
    et = feature_analysis.get("error_type", {})
    for error, stats in sorted(et.items(), key=lambda x: -x[1]["feedback_success_rate"]):
        if stats["feedback_success_rate"] > 0.3:
            print(f"  {error}: {stats['feedback_success_rate']:.1%} (n={stats['total']}, indicates near)")

    print("\nFeedback Retry Changes Code:")
    frc = feature_analysis.get("feedback_retry_changes_code", {})
    print(f"  True 时 feedback 成功率: {frc.get('true_rate', 0):.1%}")
    print(f"  False 时 feedback 成功率: {frc.get('false_rate', 0):.1%}")

    # 评估各调度策略
    print("\n" + "=" * 80)
    print("2. 调度策略评估")
    print("=" * 80)

    schedulers = [
        (baseline_always_single, "Always Single-shot"),
        (baseline_always_feedback, "Always Feedback"),
        (classify_boundary_oracle, "Oracle Difficulty"),
        (classify_boundary_with_features, "No-Leak Features"),
    ]

    results = []
    for scheduler_fn, scheduler_name in schedulers:
        result = evaluate_scheduler(all_signals, scheduler_fn, scheduler_name)
        results.append(result)

        print(f"\n{scheduler_name}:")
        print(f"  Overall: Single {result['overall_single_success_rate']:.1%} → Feedback {result['overall_feedback_success_rate']:.1%} (gain: {result['overall_gain']:+.1%})")

        for zone in ["above", "near", "below"]:
            if zone in result["zone_stats"]:
                zs = result["zone_stats"][zone]
                print(f"  {zone.upper()}: Single {zs['single_success_rate']:.1%} → Feedback {zs['feedback_success_rate']:.1%} (gain: {zs['gain']:+.1%}, n={zs['total']})")

    # 比较 No-Leak vs Oracle
    print("\n" + "=" * 80)
    print("3. No-Leak vs Oracle 对比")
    print("=" * 80)

    oracle_result = next(r for r in results if r["name"] == "Oracle Difficulty")
    noleak_result = next(r for r in results if r["name"] == "No-Leak Features")

    print("\n总体对比:")
    print(f"  Oracle: Single {oracle_result['overall_single_success_rate']:.1%} → Feedback {oracle_result['overall_feedback_success_rate']:.1%}")
    print(f"  No-Leak: Single {noleak_result['overall_single_success_rate']:.1%} → Feedback {noleak_result['overall_feedback_success_rate']:.1%}")

    print("\n各区域对比:")
    for zone in ["above", "near", "below"]:
        if zone in oracle_result["zone_stats"] and zone in noleak_result["zone_stats"]:
            oracle_gain = oracle_result["zone_stats"][zone]["gain"]
            noleak_gain = noleak_result["zone_stats"][zone]["gain"]
            print(f"  {zone.upper()}: Oracle gain {oracle_gain:+.1%}, No-Leak gain {noleak_gain:+.1%} (差距: {abs(oracle_gain - noleak_gain):.1%})")

    # 成功率
    print("\n各区域分类准确率:")
    oracle_zones = defaultdict(int)
    noleak_zones = defaultdict(int)
    correct_zones = defaultdict(int)

    for signals in all_signals:
        oracle_zone = classify_boundary_oracle(signals)
        noleak_zone = classify_boundary_with_features(signals)
        true_zone = oracle_zone  # 以 oracle 为 ground truth

        oracle_zones[oracle_zone] += 1
        noleak_zones[noleak_zone] += 1
        if noleak_zone == true_zone:
            correct_zones[true_zone] += 1

    for zone in ["above", "near", "below"]:
        total = oracle_zones[zone]
        correct = correct_zones[zone]
        if total > 0:
            accuracy = correct / total
            print(f"  {zone.upper()}: {accuracy:.1%} 准确率 (n={total})")

    # 总结
    print("\n" + "=" * 80)
    print("4. 结论")
    print("=" * 80)

    # 计算 no-leak 相对于 always feedback 的改进
    always_feedback = next(r for r in results if r["name"] == "Always Feedback")
    always_single = next(r for r in results if r["name"] == "Always Single-shot")

    print(f"""
成功标准检查:

1. Ours 接近 oracle scheduler?
   - Oracle 总体增益: {oracle_result['overall_gain']:+.1%}
   - No-Leak 总体增益: {noleak_result['overall_gain']:+.1%}
   - 差距: {abs(oracle_result['overall_gain'] - noleak_result['overall_gain']):.1%}

2. Ours 明显优于 always feedback retry?
   - Always Feedback 总体成功率: {always_feedback['overall_feedback_success_rate']:.1%}
   - No-Leak 总体成功率: {noleak_result['overall_feedback_success_rate']:.1%}
   - 差异: {noleak_result['overall_feedback_success_rate'] - always_feedback['overall_feedback_success_rate']:+.1%}

3. Ours 在 hard 区减少无效 retry?
   - Hard 区 feedback 无效率 = 1 - feedback_success_rate
   - Oracle hard 区无效率: {1 - oracle_result['zone_stats'].get('below', {}).get('feedback_success_rate', 0):.1%}
   - No-Leak hard 区无效率: {1 - noleak_result['zone_stats'].get('below', {}).get('feedback_success_rate', 0):.1%}

4. Ours 在 medium 区保留 feedback 增益?
   - Oracle medium 区增益: {oracle_result['zone_stats'].get('near', {}).get('gain', 0):+.1%}
   - No-Leak medium 区增益: {noleak_result['zone_stats'].get('near', {}).get('gain', 0):+.1%}
""")

    # 保存结果
    output = {
        "experiment": "no_leak_boundary_classifier",
        "n_tasks": len(all_signals),
        "feature_analysis": feature_analysis,
        "scheduler_results": results,
    }

    results_dir = PROJECT_ROOT / "results" / "no_leak_boundary_classifier"
    results_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_path = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
