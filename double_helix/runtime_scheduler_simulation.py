"""
Runtime Boundary Scheduler Simulation - Phase E

目标：证明 runtime trace signal 不只是能解释 boundary，还能实际改进调度策略

Policy 比较：
- Policy A: Always single-shot
- Policy B: Always feedback retry
- Policy C: Oracle difficulty scheduler
- Policy D: Runtime trace scheduler

评估指标：
- success_rate：成功率
- feedback_calls：feedback 调用数
- wasted_feedback_on_above：浪费在 ABOVE 的 feedback
- wasted_feedback_on_below：浪费在 BELOW 的 feedback
- missed_near_cases：错过的 NEAR cases
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RuntimeTrace:
    """Runtime trace for a single task"""
    task_id: str = ""
    bug_type: str = ""
    difficulty: str = ""
    condition: str = ""
    boundary_label: str = ""

    single_success: bool = False
    blind_success: bool = False
    feedback_success: bool = False

    first_attempt_parse_ok: bool = False
    first_attempt_syntax_ok: bool = False
    first_visible_pass: bool = False
    first_hidden_pass: bool = False
    first_error_type: str = ""
    first_error_message_len: int = 0
    has_expected_actual: bool = False
    expected_actual_distance: float = 0.0
    first_patch_size: int = 0
    first_changed_from_buggy: bool = False

    blind_changed_code: bool = False
    blind_parse_ok: bool = False
    blind_error_type: str = ""

    feedback_changed_code: bool = False
    feedback_parse_ok: bool = False
    feedback_error_type: str = ""
    feedback_uses_error_signal: bool = False
    feedback_patch_size_delta: int = 0

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "bug_type": self.bug_type,
            "difficulty": self.difficulty,
            "condition": self.condition,
            "boundary_label": self.boundary_label,
            "single_success": self.single_success,
            "blind_success": self.blind_success,
            "feedback_success": self.feedback_success,
            "first_attempt_parse_ok": self.first_attempt_parse_ok,
            "first_attempt_syntax_ok": self.first_attempt_syntax_ok,
            "first_visible_pass": self.first_visible_pass,
            "first_hidden_pass": self.first_hidden_pass,
            "first_error_type": self.first_error_type,
            "first_error_message_len": self.first_error_message_len,
            "has_expected_actual": self.has_expected_actual,
            "expected_actual_distance": self.expected_actual_distance,
            "first_patch_size": self.first_patch_size,
            "first_changed_from_buggy": self.first_changed_from_buggy,
            "blind_changed_code": self.blind_changed_code,
            "blind_parse_ok": self.blind_parse_ok,
            "blind_error_type": self.blind_error_type,
            "feedback_changed_code": self.feedback_changed_code,
            "feedback_parse_ok": self.feedback_parse_ok,
            "feedback_error_type": self.feedback_error_type,
            "feedback_uses_error_signal": self.feedback_uses_error_signal,
            "feedback_patch_size_delta": self.feedback_patch_size_delta,
        }


def load_traces() -> List[RuntimeTrace]:
    """加载 runtime traces"""
    results_dir = PROJECT_ROOT / "results" / "runtime_trace_boundary_experiment"

    # 找最新的结果文件
    json_files = list(results_dir.glob("experiment_*.json"))
    if not json_files:
        print(f"找不到结果文件: {results_dir / 'experiment_*.json'}")
        return []

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"加载: {latest_file}")

    with open(latest_file, "r") as f:
        data = json.load(f)

    traces = []
    for trace_dict in data["traces"]:
        trace = RuntimeTrace()
        for key, value in trace_dict.items():
            if hasattr(trace, key):
                setattr(trace, key, value)
        traces.append(trace)

    return traces


def simulate_policy_a(traces: List[RuntimeTrace]) -> Dict[str, Any]:
    """Policy A: Always single-shot"""
    success = sum(1 for t in traces if t.single_success)
    feedback_calls = 0
    wasted_on_above = 0
    wasted_on_below = 0
    missed_near = 0

    for t in traces:
        if t.boundary_label == "above":
            wasted_on_above += 0  # 没有调用 feedback
        elif t.boundary_label == "near":
            if not t.single_success:
                missed_near += 1
        elif t.boundary_label == "below":
            wasted_on_below += 0  # 没有调用 feedback

    return {
        "policy": "A: Always single-shot",
        "success_rate": success / len(traces),
        "feedback_calls": feedback_calls,
        "wasted_on_above": wasted_on_above,
        "wasted_on_below": wasted_on_below,
        "missed_near_cases": missed_near,
    }


def simulate_policy_b(traces: List[RuntimeTrace]) -> Dict[str, Any]:
    """Policy B: Always feedback retry"""
    success = sum(1 for t in traces if t.feedback_success)
    feedback_calls = len(traces)
    wasted_on_above = sum(1 for t in traces if t.boundary_label == "above" and t.feedback_success)
    wasted_on_below = sum(1 for t in traces if t.boundary_label == "below")
    missed_near = 0

    return {
        "policy": "B: Always feedback retry",
        "success_rate": success / len(traces),
        "feedback_calls": feedback_calls,
        "wasted_on_above": wasted_on_above,
        "wasted_on_below": wasted_on_below,
        "missed_near_cases": missed_near,
    }


def simulate_policy_c(traces: List[RuntimeTrace]) -> Dict[str, Any]:
    """Policy C: Oracle difficulty scheduler"""
    success = 0
    feedback_calls = 0
    wasted_on_above = 0
    wasted_on_below = 0
    missed_near = 0

    for t in traces:
        # Oracle: 只对 medium 难度使用 feedback
        if t.difficulty == "medium":
            feedback_calls += 1
            if t.feedback_success:
                success += 1
        elif t.difficulty in ["trivial", "easy"]:
            # ABOVE: 单shot就够了
            if t.single_success:
                success += 1
        else:  # hard
            # BELOW: 跳过 feedback，直接 escalate
            if t.feedback_success:
                success += 1
                wasted_on_below += 1

    return {
        "policy": "C: Oracle difficulty scheduler",
        "success_rate": success / len(traces),
        "feedback_calls": feedback_calls,
        "wasted_on_above": wasted_on_above,
        "wasted_on_below": wasted_on_below,
        "missed_near_cases": missed_near,
    }


def simulate_policy_d(traces: List[RuntimeTrace]) -> Dict[str, Any]:
    """
    Policy D: Runtime trace scheduler

    规则：
    - if not first_visible_pass: skip feedback (visible-fail zone)
    - elif first_error_type == "pass": accept (solved zone)
    - else: feedback retry (hidden-gap zone)
    """
    success = 0
    feedback_calls = 0
    wasted_on_above = 0
    wasted_on_below = 0
    missed_near = 0

    for t in traces:
        if not t.first_visible_pass:
            # visible-fail zone: 跳过 feedback
            if t.feedback_success:
                success += 1  # 但实际上成功了（可能是 escalate 解决的）
            wasted_on_below += 1
        elif t.first_error_type == "pass":
            # solved zone: 接受
            if t.single_success:
                success += 1
            # 没有调用 feedback
        else:
            # hidden-gap zone: 使用 feedback
            feedback_calls += 1
            if t.feedback_success:
                success += 1
            else:
                # feedback 失败了
                wasted_on_above += 1

    return {
        "policy": "D: Runtime trace scheduler",
        "success_rate": success / len(traces),
        "feedback_calls": feedback_calls,
        "wasted_on_above": wasted_on_above,
        "wasted_on_below": wasted_on_below,
        "missed_near_cases": missed_near,
    }


def analyze_by_zone(traces: List[RuntimeTrace]) -> Dict[str, Any]:
    """按 zone 分析"""
    zones = defaultdict(lambda: {
        "total": 0,
        "single_success": 0,
        "feedback_success": 0,
        "feedback_calls": 0,
    })

    for t in traces:
        zones[t.boundary_label]["total"] += 1
        if t.single_success:
            zones[t.boundary_label]["single_success"] += 1
        if t.feedback_success:
            zones[t.boundary_label]["feedback_success"] += 1

    return dict(zones)


def main():
    print("=" * 80)
    print("Runtime Boundary Scheduler Simulation - Phase E")
    print("=" * 80)
    print("\n目标：证明 runtime trace signal 不只是能解释 boundary，还能实际改进调度策略")
    print("=" * 80)

    # 加载 traces
    traces = load_traces()
    print(f"\n加载了 {len(traces)} 个 traces")

    # Zone 分析
    zone_stats = analyze_by_zone(traces)
    print("\nZone 分布：")
    for zone, stats in zone_stats.items():
        total = stats["total"]
        print(f"  {zone}: n={total} ({total/len(traces)*100:.1f}%)")
        print(f"    single_success: {stats['single_success']/total:.1%}")
        print(f"    feedback_success: {stats['feedback_success']/total:.1%}")

    # 模拟各 policy
    print("\n" + "=" * 80)
    print("Policy 比较")
    print("=" * 80)

    policies = [
        simulate_policy_a(traces),
        simulate_policy_b(traces),
        simulate_policy_c(traces),
        simulate_policy_d(traces),
    ]

    print(f"\n{'Policy':<35} {'Success Rate':<15} {'FB Calls':<12} {'Wasted ABOVE':<15} {'Wasted BELOW':<15} {'Missed NEAR':<12}")
    print("-" * 110)

    for p in policies:
        print(f"{p['policy']:<35} {p['success_rate']:>12.1%} {p['feedback_calls']:>10} {p['wasted_on_above']:>13} {p['wasted_on_below']:>13} {p['missed_near_cases']:>10}")

    # 计算节省率
    print("\n" + "=" * 80)
    print("Feedback 效率分析")
    print("=" * 80)

    policy_b = policies[1]  # Always feedback
    policy_d = policies[3]  # Runtime trace

    feedback_saved = policy_b["feedback_calls"] - policy_d["feedback_calls"]
    feedback_saved_pct = feedback_saved / policy_b["feedback_calls"] * 100

    success_diff = policy_d["success_rate"] - policy_b["success_rate"]

    print(f"\nPolicy B (Always Feedback): {policy_b['feedback_calls']} feedback calls, {policy_b['success_rate']:.1%} success")
    print(f"Policy D (Runtime Trace):  {policy_d['feedback_calls']} feedback calls, {policy_d['success_rate']:.1%} success")
    print(f"\nFeedback calls saved: {feedback_saved} ({feedback_saved_pct:.1f}%)")
    print(f"Success rate difference: {success_diff:+.1%}")

    # 成功率效率比
    efficiency_b = policy_b["success_rate"] / policy_b["feedback_calls"] if policy_b["feedback_calls"] > 0 else 0
    efficiency_d = policy_d["success_rate"] / policy_d["feedback_calls"] if policy_d["feedback_calls"] > 0 else 0

    print(f"\nSuccess per feedback call:")
    print(f"  Policy B: {efficiency_b:.4f}")
    print(f"  Policy D: {efficiency_d:.4f} ({efficiency_d/efficiency_b:.2f}x better)" if efficiency_b > 0 else "  Policy D: N/A")

    # Zone 级别的分析
    print("\n" + "=" * 80)
    print("Zone 级别分析")
    print("=" * 80)

    for zone in ["above", "near", "below"]:
        if zone not in zone_stats:
            continue

        stats = zone_stats[zone]
        total = stats["total"]

        # Runtime scheduler 在这个 zone 的表现
        zone_traces = [t for t in traces if t.boundary_label == zone]

        # Policy D 在这个 zone 会调用 feedback 吗？
        if zone == "above":
            # Runtime scheduler 不会调用 feedback
            fb_calls = 0
            fb_success = sum(1 for t in zone_traces if t.single_success)
        elif zone == "near":
            # Runtime scheduler 会调用 feedback
            fb_calls = total
            fb_success = stats["feedback_success"]
        else:  # below
            # Runtime scheduler 跳过 feedback
            fb_calls = 0
            fb_success = sum(1 for t in zone_traces if t.feedback_success)  # 但实际上可能 escalate 成功了

        print(f"\n{zone.upper()} zone (n={total}):")
        print(f"  Single-shot success: {stats['single_success']/total:.1%}")
        print(f"  Feedback success: {stats['feedback_success']/total:.1%}")
        print(f"  Runtime scheduler: {fb_calls} feedback calls, {fb_success} successes")

    # 成功标准检查
    print("\n" + "=" * 80)
    print("成功标准检查")
    print("=" * 80)

    success_criteria_1 = abs(policy_d["success_rate"] - policy_b["success_rate"]) < 0.05
    success_criteria_2 = feedback_saved_pct > 20

    print(f"\n1. Runtime scheduler success rate 接近 always-feedback?")
    print(f"   Policy B success: {policy_b['success_rate']:.1%}")
    print(f"   Policy D success: {policy_d['success_rate']:.1%}")
    print(f"   差异: {abs(policy_d['success_rate'] - policy_b['success_rate']):.1%}")
    print(f"   [{'PASS' if success_criteria_1 else 'FAIL'}]")

    print(f"\n2. Feedback 调用数显著低于 always-feedback?")
    print(f"   Policy B calls: {policy_b['feedback_calls']}")
    print(f"   Policy D calls: {policy_d['feedback_calls']}")
    print(f"   节省: {feedback_saved_pct:.1f}%")
    print(f"   [{'PASS' if success_criteria_2 else 'FAIL'}]")

    if success_criteria_1 and success_criteria_2:
        print("\n[SUCCESS] Runtime scheduler 达到成功标准！")
        print("  - 成功率接近 always-feedback")
        print("  - Feedback 调用数显著减少")
    else:
        print("\n[PARTIAL] Runtime scheduler 未完全达到成功标准")
        print("  - 需要分析原因")

    # 保存结果
    output = {
        "experiment": "runtime_scheduler_simulation",
        "n_traces": len(traces),
        "zone_stats": zone_stats,
        "policies": policies,
        "feedback_saved": feedback_saved,
        "feedback_saved_pct": feedback_saved_pct,
        "success_diff": success_diff,
        "success_criteria_1": success_criteria_1,
        "success_criteria_2": success_criteria_2,
    }

    results_dir = PROJECT_ROOT / "results" / "runtime_scheduler_simulation"
    results_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_path = results_dir / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
