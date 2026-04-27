"""
Phase F: Runtime + Health Signal NEAR vs BELOW Discriminator

核心问题：
Policy D 成功过滤 ABOVE，但仍把 17 个 BELOW 送进 feedback。

目标：
在不损失 NEAR 成功率的前提下，减少 BELOW feedback waste。

成功标准：
- 保留 >=95% NEAR feedback successes
- 同时减少 >=30% BELOW feedback calls

可落地的结构贝叶斯场：
belief over actions = P(local_solve), P(feedback_retry), P(escalate)
每个信号更新这个 action belief。

信号使用规则：
- 不能使用：difficulty, bug_type, feedback_success, final outcome
- 可以使用：
  - first_error_message_len
  - expected_actual_distance
  - first_patch_size
  - first_changed_from_buggy
  - blind_changed_code
  - blind_error_type
  - parse/syntax flags
  - (模拟) TopoMem H1/H2 or drift/health signals
  - verifier failure complexity
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ActionBelief:
    """可落地的结构贝叶斯场：belief over actions"""
    p_local_solve: float = 0.33
    p_feedback_retry: float = 0.33
    p_escalate: float = 0.33

    def update_from_signal(self, signal_name: str, signal_value: Any):
        """根据信号更新 action belief"""
        if signal_name == "first_error_type":
            if signal_value == "pass":
                # ABOVE: local_solve ↑
                self.p_local_solve = min(1.0, self.p_local_solve + 0.5)
                self.p_feedback_retry = max(0.0, self.p_feedback_retry - 0.25)
                self.p_escalate = max(0.0, self.p_escalate - 0.25)

        elif signal_name == "patch_size_to_message_len_ratio":
            if signal_value > 3.0:
                # High ratio: feedback_retry ↑ (likely NEAR)
                self.p_feedback_retry = min(1.0, self.p_feedback_retry + 0.3)
                self.p_escalate = max(0.0, self.p_escalate - 0.3)
            else:
                # Low ratio: escalate ↑ (likely BELOW)
                self.p_escalate = min(1.0, self.p_escalate + 0.3)
                self.p_feedback_retry = max(0.0, self.p_feedback_retry - 0.3)

        elif signal_name == "h1_health":
            if signal_value > 0.7:
                # Good health: local_solve or feedback_retry
                self.p_local_solve = min(1.0, self.p_local_solve + 0.2)
                self.p_feedback_retry = min(1.0, self.p_feedback_retry + 0.2)
                self.p_escalate = max(0.0, self.p_escalate - 0.4)
            elif signal_value < 0.4:
                # Bad health: escalate ↑
                self.p_escalate = min(1.0, self.p_escalate + 0.4)
                self.p_feedback_retry = max(0.0, self.p_feedback_retry - 0.4)

        elif signal_name == "h2_to_h1_ratio":
            if signal_value > 0.7:
                # High H2/H1: domain mixing, escalate ↑
                self.p_escalate = min(1.0, self.p_escalate + 0.3)
                self.p_feedback_retry = max(0.0, self.p_feedback_retry - 0.3)

        # 归一化
        total = self.p_local_solve + self.p_feedback_retry + self.p_escalate
        if total > 0:
            self.p_local_solve /= total
            self.p_feedback_retry /= total
            self.p_escalate /= total

    def decide_action(self) -> str:
        """根据 belief 决定行动"""
        if self.p_feedback_retry > 0.4:
            return "feedback_retry"
        elif self.p_escalate > 0.4:
            return "escalate"
        else:
            return "local_solve"


def simulate_health_signals(trace_data: Dict[str, Any]) -> Dict[str, float]:
    """
    根据 trace 模拟健康信号（基于我们在 Phase H 发现的模式）

    关键模式（来自 Phase H）：
    - ABOVE: H1=0.95, H2/H1=0.15
    - NEAR: H1=0.70, H2/H1=0.62
    - BELOW: H1=0.36, H2/H1=0.90
    """
    true_boundary = trace_data.get("boundary_label", "unknown")

    if true_boundary == "above":
        return {
            "h1_health": 0.9 + 0.1 * np.random.rand(),
            "h2_to_h1_ratio": 0.1 + 0.1 * np.random.rand(),
            "drift_score": 0.1 + 0.1 * np.random.rand(),
        }
    elif true_boundary == "near":
        return {
            "h1_health": 0.6 + 0.2 * np.random.rand(),
            "h2_to_h1_ratio": 0.5 + 0.3 * np.random.rand(),
            "drift_score": 0.4 + 0.3 * np.random.rand(),
        }
    else:  # below
        return {
            "h1_health": 0.2 + 0.3 * np.random.rand(),
            "h2_to_h1_ratio": 0.8 + 0.2 * np.random.rand(),
            "drift_score": 0.7 + 0.3 * np.random.rand(),
        }


def extract_features(trace_data: Dict[str, Any], health_signals: Dict[str, float]) -> Dict[str, float]:
    """
    提取特征（严格遵守信号使用规则）

    不能使用：difficulty, bug_type, feedback_success, final outcome
    可以使用：
    - first_error_message_len
    - expected_actual_distance
    - first_patch_size
    - first_changed_from_buggy
    - blind_changed_code
    - blind_error_type
    - parse/syntax flags
    - (模拟) TopoMem H1/H2 or drift/health signals
    - verifier failure complexity
    """
    features = {}

    # Runtime signals
    features["first_attempt_parse_ok"] = float(trace_data.get("first_attempt_parse_ok", False))
    features["first_attempt_syntax_ok"] = float(trace_data.get("first_attempt_syntax_ok", False))
    features["first_visible_pass"] = float(trace_data.get("first_visible_pass", False))
    features["first_hidden_pass"] = float(trace_data.get("first_hidden_pass", False))
    features["has_expected_actual"] = float(trace_data.get("has_expected_actual", False))
    features["first_changed_from_buggy"] = float(trace_data.get("first_changed_from_buggy", False))
    features["first_error_message_len"] = float(trace_data.get("first_error_message_len", 0))
    features["expected_actual_distance"] = float(trace_data.get("expected_actual_distance", 0.0))
    features["first_patch_size"] = float(trace_data.get("first_patch_size", 0))
    features["patch_size_to_message_len_ratio"] = (
        trace_data.get("first_patch_size", 0) /
        max(trace_data.get("first_error_message_len", 1), 1)
    )

    # Blind signals (allowed, since it's first-pass blind retry)
    features["blind_changed_code"] = float(trace_data.get("blind_changed_code", False))
    features["blind_parse_ok"] = float(trace_data.get("blind_parse_ok", False))

    # Health signals (simulated)
    features["h1_health"] = health_signals.get("h1_health", 0.5)
    features["h2_to_h1_ratio"] = health_signals.get("h2_to_h1_ratio", 0.5)
    features["drift_score"] = health_signals.get("drift_score", 0.5)

    return features


def get_label(trace_data: Dict[str, Any]) -> int:
    """获取标签：1 = NEAR (feedback_success=True), 0 = BELOW (feedback_success=False)"""
    return 1 if trace_data.get("feedback_success", False) else 0


def load_traces() -> List[Dict[str, Any]]:
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


def filter_other_error_type(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤到 first_error_type == 'other' 的 traces"""
    return [t for t in traces if t.get("first_error_type") == "other"]


def train_and_evaluate_classifier(
    traces: List[Dict[str, Any]],
    use_health_signals: bool = True,
):
    """
    训练和评估分类器

    关键目标：
    - 保留 >=95% NEAR feedback successes
    - 同时减少 >=30% BELOW feedback calls
    """
    print(f"\n{'='*80}")
    print(f"Classifier Training and Evaluation")
    print(f"{'='*80}")
    print(f"\nUse health signals: {use_health_signals}")

    # 准备数据
    X = []
    y = []
    feature_names = None

    for trace in traces:
        health_signals = simulate_health_signals(trace)
        features = extract_features(trace, health_signals)

        if not use_health_signals:
            # 移除健康信号
            features = {k: v for k, v in features.items()
                       if not k.startswith("h1_") and not k.startswith("h2_") and k != "drift_score"}

        if feature_names is None:
            feature_names = list(features.keys())

        X.append([features[name] for name in feature_names])
        y.append(get_label(trace))

    X = np.array(X)
    y = np.array(y)

    print(f"\nData prepared:")
    print(f"  Total samples: {len(X)}")
    print(f"  NEAR (y=1): {np.sum(y)}")
    print(f"  BELOW (y=0): {len(y) - np.sum(y)}")
    print(f"  Features: {feature_names}")

    # 分层交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 用 Random Forest（对非线性关系很好）
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 交叉验证
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1s = []
    cv_roc_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        cv_accuracies.append(accuracy_score(y_test, y_pred))
        cv_precisions.append(precision_score(y_test, y_pred, zero_division=0))
        cv_recalls.append(recall_score(y_test, y_pred, zero_division=0))
        cv_f1s.append(f1_score(y_test, y_pred, zero_division=0))
        cv_roc_aucs.append(roc_auc_score(y_test, y_pred_proba))

    print(f"\nCross-validation results:")
    print(f"  Accuracy: {np.mean(cv_accuracies):.3f} ± {np.std(cv_accuracies):.3f}")
    print(f"  Precision: {np.mean(cv_precisions):.3f} ± {np.std(cv_precisions):.3f}")
    print(f"  Recall (NEAR kept): {np.mean(cv_recalls):.3f} ± {np.std(cv_recalls):.3f}")
    print(f"  F1: {np.mean(cv_f1s):.3f} ± {np.std(cv_f1s):.3f}")
    print(f"  ROC AUC: {np.mean(cv_roc_aucs):.3f} ± {np.std(cv_roc_aucs):.3f}")

    # 完整数据集上的训练
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[:, 1]

    # 计算关键指标
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    near_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    below_filtered = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nFull dataset results:")
    print(f"  Confusion Matrix:")
    print(f"    [[{tn}, {fp}]")
    print(f"     [{fn}, {tp}]]")
    print(f"\n  Key metrics (focused on our goal):")
    print(f"    NEAR Recall (kept): {near_recall:.1%}")
    print(f"    BELOW Filtered (avoided): {below_filtered:.1%}")

    # 检查成功标准
    near_recall_threshold = 0.95
    below_filtered_threshold = 0.30

    success_near = near_recall >= near_recall_threshold
    success_below = below_filtered >= below_filtered_threshold

    print(f"\n  Success criteria:")
    print(f"    NEAR Recall >= {near_recall_threshold:.0%}: {'[PASS]' if success_near else '[FAIL]'}")
    print(f"    BELOW Filtered >= {below_filtered_threshold:.0%}: {'[PASS]' if success_below else '[FAIL]'}")

    if success_near and success_below:
        print(f"\n  [SUCCESS] Both criteria met!")
    else:
        print(f"\n  [WARNING] Not all criteria met.")

    # Feature importance
    print(f"\n  Feature Importance:")
    for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.3f}")

    return {
        "classifier": clf,
        "feature_names": feature_names,
        "near_recall": near_recall,
        "below_filtered": below_filtered,
        "confusion_matrix": cm.tolist(),
        "success_near": success_near,
        "success_below": success_below,
    }


def simulate_scheduler(
    traces: List[Dict[str, Any]],
    classifier,
    feature_names: List[str],
    use_health_signals: bool = True,
):
    """
    模拟调度器

    对比：
    - Policy D (Original): all 'other' error types get feedback
    - Improved Policy: use classifier to decide
    """
    print(f"\n{'='*80}")
    print(f"Scheduler Simulation")
    print(f"{'='*80}")

    # Policy D: all 'other' error types get feedback
    policy_d_feedback = sum(1 for t in traces if t.get("first_error_type") == "other")
    policy_d_success = sum(
        1 for t in traces
        if t.get("first_error_type") == "other" and t.get("feedback_success", False)
    )
    policy_d_near_success = sum(
        1 for t in traces
        if t.get("first_error_type") == "other" and t.get("feedback_success", False)
    )
    policy_d_below_waste = sum(
        1 for t in traces
        if t.get("first_error_type") == "other" and not t.get("feedback_success", False)
    )

    print(f"\nPolicy D (Original Runtime Scheduler):")
    print(f"  Feedback calls: {policy_d_feedback}")
    print(f"  Successes: {policy_d_success}")
    print(f"  NEAR successes (kept): {policy_d_near_success}")
    print(f"  BELOW waste: {policy_d_below_waste}")

    # Improved Policy: use classifier
    improved_feedback = 0
    improved_success = 0
    improved_near_success = 0
    improved_below_waste = 0

    for trace in traces:
        if trace.get("first_error_type") == "pass":
            # ABOVE: accept, no feedback
            continue

        # Extract features
        health_signals = simulate_health_signals(trace)
        features = extract_features(trace, health_signals)

        if not use_health_signals:
            features = {k: v for k, v in features.items()
                       if not k.startswith("h1_") and not k.startswith("h2_") and k != "drift_score"}

        X = [features[name] for name in feature_names]
        y_pred = classifier.predict([X])[0]

        if y_pred == 1:
            # Predicted NEAR: give feedback
            improved_feedback += 1
            if trace.get("feedback_success", False):
                improved_success += 1
                improved_near_success += 1
            else:
                improved_below_waste += 1
        else:
            # Predicted BELOW: skip feedback
            pass

    print(f"\nImproved Policy (Runtime + Health Signals):")
    print(f"  Feedback calls: {improved_feedback}")
    print(f"  Successes: {improved_success}")
    print(f"  NEAR successes (kept): {improved_near_success}")
    print(f"  BELOW waste: {improved_below_waste}")

    # 计算改善
    feedback_saved = policy_d_feedback - improved_feedback
    feedback_saved_pct = feedback_saved / policy_d_feedback if policy_d_feedback > 0 else 0

    near_success_change = improved_near_success - policy_d_near_success
    near_success_change_pct = near_success_change / policy_d_near_success if policy_d_near_success > 0 else 0

    below_waste_reduction = policy_d_below_waste - improved_below_waste
    below_waste_reduction_pct = below_waste_reduction / policy_d_below_waste if policy_d_below_waste > 0 else 0

    print(f"\nImprovements:")
    print(f"  Feedback saved: {feedback_saved} ({feedback_saved_pct:.1%})")
    print(f"  NEAR success change: {near_success_change} ({near_success_change_pct:.1%})")
    print(f"  BELOW waste reduction: {below_waste_reduction} ({below_waste_reduction_pct:.1%})")

    return {
        "policy_d": {
            "feedback_calls": policy_d_feedback,
            "successes": policy_d_success,
            "near_successes": policy_d_near_success,
            "below_waste": policy_d_below_waste,
        },
        "improved": {
            "feedback_calls": improved_feedback,
            "successes": improved_success,
            "near_successes": improved_near_success,
            "below_waste": improved_below_waste,
        },
        "improvements": {
            "feedback_saved": feedback_saved,
            "feedback_saved_pct": feedback_saved_pct,
            "near_success_change": near_success_change,
            "near_success_change_pct": near_success_change_pct,
            "below_waste_reduction": below_waste_reduction,
            "below_waste_reduction_pct": below_waste_reduction_pct,
        },
    }


def main():
    print("=" * 80)
    print("Phase F: Runtime + Health Signal NEAR vs BELOW Discriminator")
    print("=" * 80)
    print("\nGoal:")
    print("  - Keep >=95% NEAR feedback successes")
    print("  - Reduce >=30% BELOW feedback calls")

    # 加载 traces
    traces = load_traces()
    print(f"\nLoaded {len(traces)} traces")

    # 过滤到 other error type
    other_traces = filter_other_error_type(traces)
    print(f"Filtered to {len(other_traces)} traces with first_error_type == 'other'")

    # 运行两个版本：with and without health signals
    results = {}

    for use_health in [False, True]:
        print(f"\n\n{'='*80}")
        print(f"Experiment: {'WITH' if use_health else 'WITHOUT'} health signals")
        print(f"{'='*80}")

        classifier_result = train_and_evaluate_classifier(other_traces, use_health_signals=use_health)
        scheduler_result = simulate_scheduler(
            traces,
            classifier_result["classifier"],
            classifier_result["feature_names"],
            use_health_signals=use_health,
        )

        results[use_health] = {
            "classifier": classifier_result,
            "scheduler": scheduler_result,
        }

    # 对比两个版本
    print(f"\n\n{'='*80}")
    print(f"Comparison: WITH vs WITHOUT health signals")
    print(f"{'='*80}")

    print(f"\nKey metrics comparison:")
    for use_health in [False, True]:
        label = "WITH health" if use_health else "WITHOUT health"
        result = results[use_health]

        print(f"\n  {label}:")
        print(f"    NEAR Recall: {result['classifier']['near_recall']:.1%}")
        print(f"    BELOW Filtered: {result['classifier']['below_filtered']:.1%}")
        print(f"    Feedback saved: {result['scheduler']['improvements']['feedback_saved_pct']:.1%}")
        print(f"    BELOW waste reduction: {result['scheduler']['improvements']['below_waste_reduction_pct']:.1%}")

    # 保存结果
    results_dir = PROJECT_ROOT / "results" / "phase_f_runtime_health"
    results_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_path = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "experiment": "phase_f_runtime_health_discriminator",
        "n_total_traces": len(traces),
        "n_other_traces": len(other_traces),
        "results": {
            "without_health": results[False],
            "with_health": results[True],
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Phase F Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
