"""
Structural Bayesian Field - Validation Experiment

目标：验证结构贝叶斯场能否：
1. 维护信念权重分布
2. 根据反馈更新权重
3. 正确决定行动（连接到 Double Helix 的能力边界调度）

实验设计：
- 加载 Double Helix 的 traces
- 用结构贝叶斯场处理
- 对比：SBF 的 action 决定 vs Double Helix 的真实 boundary
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Import SBF core
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from structural_bayesian_field.core import (
    Belief,
    Observation,
    StructureField,
    Action,
)


def load_double_helix_traces() -> List[Dict[str, Any]]:
    """
    加载 Double Helix 的 traces（来自 Phase D）
    """
    traces_dir = PROJECT_ROOT / "results" / "runtime_trace_boundary_experiment"

    if not traces_dir.exists():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    # Find the latest trace file
    trace_files = sorted(traces_dir.glob("experiment_*.json"), reverse=True)

    if not trace_files:
        raise FileNotFoundError(f"No trace files found in {traces_dir}")

    latest_trace = trace_files[0]
    print(f"Loading traces from: {latest_trace}")

    with open(latest_trace, "r") as f:
        data = json.load(f)

    return data["traces"]


def extract_verifier_signals(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 Double Helix trace 中提取 verifier 信号
    """
    return {
        "first_error_type": trace_data.get("first_error_type", "pass"),
        "first_patch_size": trace_data.get("first_patch_size", 0),
        "first_error_message_len": trace_data.get("first_error_message_len", 0),
        "first_visible_pass": trace_data.get("first_visible_pass", False),
        "first_hidden_pass": trace_data.get("first_hidden_pass", False),
    }


def run_sbf_validation():
    """
    运行结构贝叶斯场验证实验
    """
    print("=" * 80)
    print("Structural Bayesian Field - Validation Experiment")
    print("=" * 80)

    # 1. 加载 Double Helix 的 traces
    traces = load_double_helix_traces()
    print(f"\nLoaded {len(traces)} traces from Double Helix")

    # 2. 创建结构贝叶斯场
    sbf = StructureField(field_id="sbf_validation")

    # 初始化一些信念结构（模拟多个结构/adapter）
    for i in range(5):
        belief = Belief(
            structure_id=f"struct_{i}",
            belief_weight=0.2,
            utility=0.5,
            effectiveness_score=0.5,
        )
        sbf.add_belief(belief)

    print(f"\nInitialized SBF with {len(sbf.beliefs)} beliefs")

    # 3. 处理每个 trace
    results = []
    action_counts = defaultdict(int)
    boundary_match_counts = defaultdict(int)

    for trace_idx, trace_data in enumerate(traces):
        # 提取 verifier 信号
        verifier_signals = extract_verifier_signals(trace_data)

        # 确定反馈信号（feedback_success）
        feedback_signal = 1.0 if trace_data.get("feedback_success", False) else 0.0

        # 确定真实 boundary label
        true_boundary = trace_data.get("boundary_label", "unknown")

        # 创建观测
        observation = Observation(
            observation_id=f"obs_{trace_idx}",
            input_data=None,
            feedback_signal=feedback_signal,
            verifier_signals=verifier_signals,
            boundary_label=true_boundary,
        )

        # SBF 决定行动
        action = sbf.decide_action(observation)
        action_counts[action.value] += 1

        # 更新信念权重
        sbf.update_belief_weights(observation)

        # 将 SBF action 映射到 boundary label 进行对比
        predicted_boundary = None
        if action == Action.ACCEPT:
            predicted_boundary = "above"
        elif action == Action.VERIFY:
            predicted_boundary = "near"
        elif action == Action.ESCALATE:
            predicted_boundary = "below"

        # 记录结果
        match = predicted_boundary == true_boundary if predicted_boundary else False
        if predicted_boundary:
            key = f"{true_boundary}_predicted_{predicted_boundary}"
            boundary_match_counts[key] += 1

        results.append({
            "trace_idx": trace_idx,
            "true_boundary": true_boundary,
            "predicted_boundary": predicted_boundary,
            "action": action.value,
            "match": match,
            "sbf_stats": sbf.get_field_stats(),
        })

        if (trace_idx + 1) % 50 == 0:
            print(f"\nProcessed {trace_idx + 1}/{len(traces)} traces")
            print(f"  SBF stats: {sbf.get_field_stats()}")

    # 4. 分析结果
    print("\n" + "=" * 80)
    print("Validation Results")
    print("=" * 80)

    # Action 分布
    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count} ({count/len(traces):.1%})")

    # Boundary 匹配情况
    print(f"\nBoundary match details:")
    for key, count in sorted(boundary_match_counts.items()):
        print(f"  {key}: {count}")

    # 计算总体准确率
    total_matches = sum(1 for r in results if r["match"])
    accuracy = total_matches / len(results) if results else 0
    print(f"\nOverall accuracy: {accuracy:.1%}")

    # 按 boundary 分别计算准确率
    boundary_accuracies = defaultdict(list)
    for r in results:
        if r["predicted_boundary"]:
            boundary_accuracies[r["true_boundary"]].append(r["match"])

    print(f"\nAccuracy by boundary:")
    for boundary, matches in sorted(boundary_accuracies.items()):
        acc = sum(matches) / len(matches) if matches else 0
        print(f"  {boundary}: {acc:.1%} (n={len(matches)})")

    # 保存结果
    results_dir = PROJECT_ROOT / "results" / "structural_bayesian_field"
    results_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_path = results_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "experiment": "structural_bayesian_field_validation",
        "n_traces": len(traces),
        "action_counts": dict(action_counts),
        "boundary_match_counts": dict(boundary_match_counts),
        "overall_accuracy": accuracy,
        "boundary_accuracies": {
            b: sum(m) / len(m) if m else 0
            for b, m in boundary_accuracies.items()
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Validation Complete")
    print("=" * 80)


if __name__ == "__main__":
    run_sbf_validation()
