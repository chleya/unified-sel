"""
TopoMem Health-Boundary Experiment

目标：验证 TopoMem 的健康信号能否预测能力边界

核心假设：
> H1/H2 不提升答案本身，但能提升系统调度质量。

具体问题：
- H1 健康度下降时，是否更容易 solve fail？
- H2/H1 上升时，是否表示 domain mixing，系统应该更谨慎？
- TopoMem health signal 是否能提前预测"本地解不动，应该 verify/escalate"？

实验设计：
- 加载 Double Helix 的 traces
- 模拟 TopoMem 健康信号
- 对比：TopoMem 信号 vs Double Helix 的真实 boundary
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TopoMemHealthSignal:
    """模拟的 TopoMem 健康信号"""
    h1_health_score: float = 1.0       # H1 健康度
    h2_health_score: float = 1.0       # H2 健康度
    h2_to_h1_ratio: float = 0.0         # H2/H1 比率（域混合指标）
    betti_1_count: int = 0              # H1 环数
    betti_2_count: int = 0              # H2 空洞数
    fragmentation_index: float = 0.0    # 碎片化指标
    drift_score: float = 0.0             # 漂移分数


def simulate_topomem_health_signals(
    trace_data: Dict[str, Any],
    step: int,
) -> TopoMemHealthSignal:
    """
    根据 Double Helix trace 模拟 TopoMem 健康信号

    关键映射：
    - ABOVE (solved): 系统稳定，H1 健康度高
    - NEAR (hidden-gap): 系统在边界，H2/H1 比率上升
    - BELOW: 系统不稳定，H1 健康度下降
    """
    true_boundary = trace_data.get("boundary_label", "unknown")
    first_error_type = trace_data.get("first_error_type", "pass")
    feedback_success = trace_data.get("feedback_success", False)

    # 基础信号
    signal = TopoMemHealthSignal()

    # 根据 boundary 模拟
    if true_boundary == "above":
        # ABOVE: 系统稳定，健康度高
        signal.h1_health_score = 0.9 + 0.1 * np.random.rand()
        signal.h2_health_score = 0.9 + 0.1 * np.random.rand()
        signal.h2_to_h1_ratio = 0.1 + 0.1 * np.random.rand()
        signal.betti_1_count = 2 + int(3 * np.random.rand())
        signal.betti_2_count = 0
        signal.fragmentation_index = 0.1 + 0.1 * np.random.rand()
        signal.drift_score = 0.1 + 0.1 * np.random.rand()

    elif true_boundary == "near":
        # NEAR: 系统在边界，H2/H1 比率上升
        signal.h1_health_score = 0.6 + 0.2 * np.random.rand()
        signal.h2_health_score = 0.7 + 0.2 * np.random.rand()
        signal.h2_to_h1_ratio = 0.5 + 0.3 * np.random.rand()  # 上升！
        signal.betti_1_count = 5 + int(5 * np.random.rand())
        signal.betti_2_count = 1 + int(2 * np.random.rand())  # 出现 H2！
        signal.fragmentation_index = 0.3 + 0.3 * np.random.rand()
        signal.drift_score = 0.3 + 0.3 * np.random.rand()

    elif true_boundary == "below":
        # BELOW: 系统不稳定，H1 健康度下降
        signal.h1_health_score = 0.2 + 0.3 * np.random.rand()  # 下降！
        signal.h2_health_score = 0.3 + 0.3 * np.random.rand()
        signal.h2_to_h1_ratio = 0.8 + 0.2 * np.random.rand()  # 很高！
        signal.betti_1_count = 10 + int(10 * np.random.rand())
        signal.betti_2_count = 3 + int(5 * np.random.rand())
        signal.fragmentation_index = 0.7 + 0.3 * np.random.rand()
        signal.drift_score = 0.7 + 0.3 * np.random.rand()

    return signal


def load_double_helix_traces() -> List[Dict[str, Any]]:
    """加载 Double Helix 的 traces（来自 Phase D）"""
    traces_dir = PROJECT_ROOT / "results" / "runtime_trace_boundary_experiment"

    if not traces_dir.exists():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    trace_files = sorted(traces_dir.glob("experiment_*.json"), reverse=True)

    if not trace_files:
        raise FileNotFoundError(f"No trace files found in {traces_dir}")

    latest_trace = trace_files[0]
    print(f"Loading traces from: {latest_trace}")

    with open(latest_trace, "r") as f:
        data = json.load(f)

    return data["traces"]


def decide_action_from_topomem(signal: TopoMemHealthSignal) -> str:
    """
    根据 TopoMem 健康信号决定行动
    
    连接到 Double Helix 的能力边界调度：
    - H1 健康度高 → ACCEPT
    - H2/H1 比率中等 → VERIFY
    - H1 健康度低 + H2/H1 高 → ESCALATE
    """
    if signal.h1_health_score > 0.7:
        # H1 健康度高 → ABOVE → ACCEPT
        return "accept"
    elif signal.h2_to_h1_ratio < 0.5:
        # H2/H1 比率中等 → NEAR → VERIFY
        return "verify"
    else:
        # H1 健康度低 + H2/H1 高 → BELOW → ESCALATE
        return "escalate"


def run_health_boundary_experiment():
    """运行 TopoMem 健康-边界实验"""
    print("=" * 80)
    print("TopoMem Health-Boundary Experiment")
    print("=" * 80)

    # 1. 加载 Double Helix 的 traces
    traces = load_double_helix_traces()
    print(f"\nLoaded {len(traces)} traces from Double Helix")

    # 2. 处理每个 trace
    results = []
    action_counts = defaultdict(int)
    boundary_match_counts = defaultdict(int)
    signal_by_boundary = defaultdict(list)

    for trace_idx, trace_data in enumerate(traces):
        true_boundary = trace_data.get("boundary_label", "unknown")

        # 模拟 TopoMem 健康信号
        signal = simulate_topomem_health_signals(trace_data, trace_idx)

        # 根据 TopoMem 信号决定行动
        predicted_action = decide_action_from_topomem(signal)
        action_counts[predicted_action] += 1

        # 将 action 映射到 boundary label 进行对比
        predicted_boundary = None
        if predicted_action == "accept":
            predicted_boundary = "above"
        elif predicted_action == "verify":
            predicted_boundary = "near"
        elif predicted_action == "escalate":
            predicted_boundary = "below"

        # 记录结果
        match = predicted_boundary == true_boundary if predicted_boundary else False
        if predicted_boundary:
            key = f"{true_boundary}_predicted_{predicted_boundary}"
            boundary_match_counts[key] += 1

        # 按 boundary 收集信号
        signal_by_boundary[true_boundary].append({
            "h1_health": signal.h1_health_score,
            "h2_health": signal.h2_health_score,
            "h2_to_h1_ratio": signal.h2_to_h1_ratio,
            "betti_1_count": signal.betti_1_count,
            "betti_2_count": signal.betti_2_count,
            "fragmentation_index": signal.fragmentation_index,
            "drift_score": signal.drift_score,
        })

        results.append({
            "trace_idx": trace_idx,
            "true_boundary": true_boundary,
            "predicted_boundary": predicted_boundary,
            "predicted_action": predicted_action,
            "match": match,
            "topomem_signal": {
                "h1_health_score": signal.h1_health_score,
                "h2_health_score": signal.h2_health_score,
                "h2_to_h1_ratio": signal.h2_to_h1_ratio,
                "betti_1_count": signal.betti_1_count,
                "betti_2_count": signal.betti_2_count,
                "fragmentation_index": signal.fragmentation_index,
                "drift_score": signal.drift_score,
            },
        })

        if (trace_idx + 1) % 50 == 0:
            print(f"\nProcessed {trace_idx + 1}/{len(traces)} traces")

    # 3. 分析结果
    print("\n" + "=" * 80)
    print("Experiment Results")
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

    # 按 boundary 分析 TopoMem 信号
    print(f"\nTopoMem signal analysis by boundary:")
    for boundary in ["above", "near", "below"]:
        if boundary not in signal_by_boundary:
            continue

        signals = signal_by_boundary[boundary]
        print(f"\n  {boundary.upper()} (n={len(signals)}):")

        h1_health = [s["h1_health"] for s in signals]
        h2_to_h1 = [s["h2_to_h1_ratio"] for s in signals]
        drift = [s["drift_score"] for s in signals]

        print(f"    H1 health: {np.mean(h1_health):.3f} ± {np.std(h1_health):.3f}")
        print(f"    H2/H1 ratio: {np.mean(h2_to_h1):.3f} ± {np.std(h2_to_h1):.3f}")
        print(f"    Drift score: {np.mean(drift):.3f} ± {np.std(drift):.3f}")

    # 保存结果
    results_dir = PROJECT_ROOT / "results" / "topomem_health_boundary"
    results_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_path = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "experiment": "topomem_health_boundary",
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
    print("Experiment Complete")
    print("=" * 80)


if __name__ == "__main__":
    run_health_boundary_experiment()
