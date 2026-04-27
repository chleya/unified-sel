"""
Phase G（严格版）: GroupKFold by bug_type

目标：用 GroupKFold 按 bug_type 分组，确保同一 bug_type 不会同时进入训练和测试
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def extract_features(trace_data: Dict[str, Any]) -> Dict[str, float]:
    """
    提取特征（严格遵守信号使用规则）
    - 禁止使用：difficulty, bug_type, single_success, blind_*, feedback_*
    """
    features = {}

    # Runtime signals ONLY
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

    # BLIND SIGNALS ARE NOT ALLOWED!
    # features["blind_changed_code"] = float(trace_data.get("blind_changed_code", False))
    # features["blind_parse_ok"] = float(trace_data.get("blind_parse_ok", False))

    return features


def get_label(trace_data: Dict[str, Any]) -> int:
    """获取标签：1 = NEAR (feedback_success=True), 0 = BELOW (feedback_success=False)"""
    return 1 if trace_data.get("feedback_success", False) else 0


def prepare_data(traces: List[Dict[str, Any]]):
    """准备数据，同时提取 groups（按 bug_type 分组）"""
    X = []
    y = []
    groups = []
    feature_names = None

    for trace in traces:
        features = extract_features(trace)

        if feature_names is None:
            feature_names = list(features.keys())

        X.append([features[name] for name in feature_names])
        y.append(get_label(trace))
        groups.append(trace.get("bug_type", "unknown"))

    return np.array(X), np.array(y), np.array(groups), feature_names


def evaluate_classifier(X_train, y_train, X_test, y_test, feature_names=None):
    """评估分类器"""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    near_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    below_filtered = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "classifier": clf,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": near_recall,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "near_recall": near_recall,
        "below_filtered": below_filtered,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def group_kfold_validation(traces: List[Dict[str, Any]]):
    """
    GroupKFold by bug_type

    确保同一 bug_type 不会同时进入训练和测试
    """
    print(f"\n{'='*80}")
    print(f"GroupKFold by bug_type")
    print(f"{'='*80}")

    X, y, groups, feature_names = prepare_data(traces)

    unique_groups = list(set(groups))
    print(f"\nUnique bug_types: {sorted(unique_groups)}")
    print(f"Number of groups: {len(unique_groups)}")

    # GroupKFold
    gkf = GroupKFold(n_splits=min(5, len(unique_groups)))

    all_near_recalls = []
    all_below_filtereds = []
    all_accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train bug_types: {sorted(list(set(groups_train)))}")
        print(f"  Test bug_types: {sorted(list(set(groups_test)))}")
        print(f"  Train: n={len(X_train)}, NEAR={sum(y_train)}, BELOW={len(y_train)-sum(y_train)}")
        print(f"  Test: n={len(X_test)}, NEAR={sum(y_test)}, BELOW={len(y_test)-sum(y_test)}")

        # 检查测试集多样性
        n_near_test = sum(y_test)
        n_below_test = len(y_test) - n_near_test

        if n_near_test == 0 or n_below_test == 0:
            print(f"  [SKIP] Test set has no diversity (NEAR={n_near_test}, BELOW={n_below_test})")
            continue

        result = evaluate_classifier(X_train, y_train, X_test, y_test, feature_names)
        print(f"  Near Recall: {result['near_recall']:.1%}")
        print(f"  Below Filtered: {result['below_filtered']:.1%}")
        print(f"  Confusion: {result['confusion_matrix']}")

        all_near_recalls.append(result["near_recall"])
        all_below_filtereds.append(result["below_filtered"])
        all_accuracies.append(result["accuracy"])

    print(f"\nSummary:")
    if len(all_near_recalls) == 0:
        print(f"  [WARNING] No valid test folds with diversity")
        return None

    print(f"  Accuracy: {np.mean(all_accuracies):.1%} ± {np.std(all_accuracies):.1%}")
    print(f"  Near Recall: {np.mean(all_near_recalls):.1%} ± {np.std(all_near_recalls):.1%}")
    print(f"  Below Filtered: {np.mean(all_below_filtereds):.1%} ± {np.std(all_below_filtereds):.1%}")

    # 检查成功标准
    near_recall_threshold = 0.90
    below_filtered_threshold = 0.50

    success_near = np.mean(all_near_recalls) >= near_recall_threshold
    success_below = np.mean(all_below_filtereds) >= below_filtered_threshold

    print(f"\n  Success criteria:")
    print(f"    Near Recall >= {near_recall_threshold:.0%}: {'[PASS]' if success_near else '[FAIL]'}")
    print(f"    Below Filtered >= {below_filtered_threshold:.0%}: {'[PASS]' if success_below else '[FAIL]'}")

    return {
        "all_near_recalls": all_near_recalls,
        "all_below_filtereds": all_below_filtereds,
        "all_accuracies": all_accuracies,
        "mean_near_recall": np.mean(all_near_recalls),
        "mean_below_filtered": np.mean(all_below_filtereds),
        "mean_accuracy": np.mean(all_accuracies),
        "success_near": success_near,
        "success_below": success_below,
    }


def main():
    print("=" * 80)
    print("Phase G（严格版）: GroupKFold by bug_type")
    print("=" * 80)

    # 加载 traces
    traces = load_traces()
    print(f"\nLoaded {len(traces)} traces")

    # 过滤到 other error type
    other_traces = filter_other_error_type(traces)
    print(f"Filtered to {len(other_traces)} traces with first_error_type == 'other'")

    # 运行 GroupKFold
    results = group_kfold_validation(other_traces)

    # 保存结果
    if results:
        results_dir = PROJECT_ROOT / "results" / "phase_g_strict_group_kfold"
        results_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        output_path = results_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Phase G（严格版）Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
