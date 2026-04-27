"""
Phase G: Anti-Overfit Validation for NEAR/BELOW Discriminator

目标：验证 patch_size_to_message_len_ratio 是否是真 runtime boundary signal，
还是 synthetic code-repair 数据集特化信号。

验证路线：
1. Leave-bug-type-out validation
2. Seed holdout validation
3. Variant holdout validation (if variants exist)
4. Feature ablation
5. Permutation sanity check
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
    """提取特征（严格遵守信号使用规则）"""
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

    return features


def get_label(trace_data: Dict[str, Any]) -> int:
    """获取标签：1 = NEAR (feedback_success=True), 0 = BELOW (feedback_success=False)"""
    return 1 if trace_data.get("feedback_success", False) else 0


def prepare_data(traces: List[Dict[str, Any]]):
    """准备数据"""
    X = []
    y = []
    feature_names = None
    metadata = []

    for trace in traces:
        features = extract_features(trace)

        if feature_names is None:
            feature_names = list(features.keys())

        X.append([features[name] for name in feature_names])
        y.append(get_label(trace))
        metadata.append({
            "bug_type": trace.get("bug_type", ""),
            "seed": trace.get("condition", ""),
            "difficulty": trace.get("difficulty", ""),
        })

    return np.array(X), np.array(y), feature_names, metadata


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


def leave_bug_type_out_validation(traces: List[Dict[str, Any]]):
    """
    Leave-bug-type-out validation

    训练/定阈值时排除某些 bug_type，测试在未见过的 bug_type 上是否仍有效。
    """
    print(f"\n{'='*80}")
    print(f"1. Leave-Bug-Type-Out Validation")
    print(f"{'='*80}")

    X, y, feature_names, metadata = prepare_data(traces)

    bug_types = list(set(m["bug_type"] for m in metadata))
    print(f"\nBug types in dataset: {sorted(bug_types)}")
    print(f"Number of bug types: {len(bug_types)}")

    all_near_recalls = []
    all_below_filtereds = []

    for held_out_bug_type in bug_types:
        print(f"\nHolding out bug_type: {held_out_bug_type}")

        # 分割数据
        train_mask = np.array([m["bug_type"] != held_out_bug_type for m in metadata])
        test_mask = np.array([m["bug_type"] == held_out_bug_type for m in metadata])

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"  Train: n={len(X_train)}, NEAR={sum(y_train)}, BELOW={len(y_train)-sum(y_train)}")
        print(f"  Test: n={len(X_test)}, NEAR={sum(y_test)}, BELOW={len(y_test)-sum(y_test)}")

        if len(X_test) == 0:
            print(f"  [SKIP] Test set is empty")
            continue

        n_near = sum(y_test)
        n_below = len(y_test) - n_near

        if n_near == 0 or n_below == 0:
            print(f"  [SKIP] Test set has no diversity (NEAR={n_near}, BELOW={n_below})")
            continue

        result = evaluate_classifier(X_train, y_train, X_test, y_test, feature_names)
        print(f"  Near Recall: {result['near_recall']:.1%}")
        print(f"  Below Filtered: {result['below_filtered']:.1%}")
        print(f"  Confusion: {result['confusion_matrix']}")

        all_near_recalls.append(result["near_recall"])
        all_below_filtereds.append(result["below_filtered"])

    print(f"\nSummary:")
    if len(all_near_recalls) == 0:
        print(f"  [WARNING] No valid test sets with diversity")
        mean_near_recall = np.nan
        mean_below_filtered = np.nan
        success_near = False
        success_below = False
    else:
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

        mean_near_recall = np.mean(all_near_recalls)
        mean_below_filtered = np.mean(all_below_filtereds)

    return {
        "all_near_recalls": all_near_recalls,
        "all_below_filtereds": all_below_filtereds,
        "mean_near_recall": mean_near_recall,
        "mean_below_filtered": mean_below_filtered,
        "success_near": success_near,
        "success_below": success_below,
    }


def seed_holdout_validation(traces: List[Dict[str, Any]]):
    """
    Seed holdout validation

    用部分 seed 选阈值，在未见 seed 上测试。
    这能排除 seed-specific artifact。
    """
    print(f"\n{'='*80}")
    print(f"2. Seed Holdout Validation")
    print(f"{'='*80}")

    X, y, feature_names, metadata = prepare_data(traces)

    seeds = list(set(m["seed"] for m in metadata))
    print(f"\nSeeds in dataset: {sorted(seeds)}")

    all_near_recalls = []
    all_below_filtereds = []

    for held_out_seed in seeds:
        print(f"\nHolding out seed: {held_out_seed}")

        # 分割数据
        train_mask = np.array([m["seed"] != held_out_seed for m in metadata])
        test_mask = np.array([m["seed"] == held_out_seed for m in metadata])

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"  Train: n={len(X_train)}, NEAR={sum(y_train)}, BELOW={len(y_train)-sum(y_train)}")
        print(f"  Test: n={len(X_test)}, NEAR={sum(y_test)}, BELOW={len(y_test)-sum(y_test)}")

        result = evaluate_classifier(X_train, y_train, X_test, y_test, feature_names)
        print(f"  Near Recall: {result['near_recall']:.1%}")
        print(f"  Below Filtered: {result['below_filtered']:.1%}")
        print(f"  Confusion: {result['confusion_matrix']}")

        all_near_recalls.append(result["near_recall"])
        all_below_filtereds.append(result["below_filtered"])

    print(f"\nSummary:")
    print(f"  Near Recall: {np.mean(all_near_recalls):.1%} ± {np.std(all_near_recalls):.1%}")
    print(f"  Below Filtered: {np.mean(all_below_filtereds):.1%} ± {np.std(all_below_filtereds):.1%}")

    return {
        "all_near_recalls": all_near_recalls,
        "all_below_filtereds": all_below_filtereds,
        "mean_near_recall": np.mean(all_near_recalls),
        "mean_below_filtered": np.mean(all_below_filtereds),
    }


def feature_ablation(traces: List[Dict[str, Any]]):
    """
    Feature ablation

    检查只用分母、只用分子、ratio 三者表现：
    - first_patch_size only
    - first_error_message_len only
    - patch_size_to_message_len_ratio

    如果单独某一项也完美，说明可能是更简单的泄漏。
    """
    print(f"\n{'='*80}")
    print(f"3. Feature Ablation")
    print(f"{'='*80}")

    X, y, feature_names, metadata = prepare_data(traces)

    # 找到关键特征的索引
    patch_size_idx = feature_names.index("first_patch_size")
    message_len_idx = feature_names.index("first_error_message_len")
    ratio_idx = feature_names.index("patch_size_to_message_len_ratio")

    feature_sets = {
        "all_features": list(range(len(feature_names))),
        "patch_size_only": [patch_size_idx],
        "message_len_only": [message_len_idx],
        "ratio_only": [ratio_idx],
    }

    results = {}

    for name, indices in feature_sets.items():
        print(f"\nFeature set: {name}")

        X_subset = X[:, indices]

        # 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        near_recalls = []
        below_filtereds = []

        for train_idx, test_idx in cv.split(X_subset, y):
            X_train, X_test = X_subset[train_idx], X_subset[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            result = evaluate_classifier(X_train, y_train, X_test, y_test)
            near_recalls.append(result["near_recall"])
            below_filtereds.append(result["below_filtered"])

        print(f"  Near Recall: {np.mean(near_recalls):.1%} ± {np.std(near_recalls):.1%}")
        print(f"  Below Filtered: {np.mean(below_filtereds):.1%} ± {np.std(below_filtereds):.1%}")

        results[name] = {
            "near_recall": np.mean(near_recalls),
            "below_filtered": np.mean(below_filtereds),
        }

    return results


def permutation_sanity_check(traces: List[Dict[str, Any]]):
    """
    Permutation sanity check

    打乱 boundary_label 后分类器必须接近随机。
    如果打乱后仍高，说明评估代码有泄漏。
    """
    print(f"\n{'='*80}")
    print(f"4. Permutation Sanity Check")
    print(f"{'='*80}")

    X, y, feature_names, metadata = prepare_data(traces)

    print(f"\nOriginal data:")
    print(f"  NEAR: {sum(y)}, BELOW: {len(y)-sum(y)}")

    # 5 次打乱
    n_permutations = 5
    all_accuracies = []

    for i in range(n_permutations):
        y_permuted = np.random.permutation(y)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + i)
        fold_accuracies = []

        for train_idx, test_idx in cv.split(X, y_permuted):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_permuted[train_idx], y_permuted[test_idx]

            result = evaluate_classifier(X_train, y_train, X_test, y_test)
            fold_accuracies.append(result["accuracy"])

        mean_acc = np.mean(fold_accuracies)
        all_accuracies.append(mean_acc)
        print(f"  Permutation {i+1}: Accuracy = {mean_acc:.1%}")

    print(f"\nSummary:")
    print(f"  Mean accuracy: {np.mean(all_accuracies):.1%} ± {np.std(all_accuracies):.1%}")

    # 检查是否接近随机
    random_baseline = max(sum(y), len(y)-sum(y)) / len(y)
    print(f"  Random baseline: {random_baseline:.1%}")

    within_random_range = abs(np.mean(all_accuracies) - random_baseline) < 0.10
    print(f"  Within random range: {'[PASS]' if within_random_range else '[FAIL]'}")

    return {
        "permutation_accuracies": all_accuracies,
        "mean_accuracy": np.mean(all_accuracies),
        "random_baseline": random_baseline,
        "within_random_range": within_random_range,
    }


def main():
    print("=" * 80)
    print("Phase G: Anti-Overfit Validation for NEAR/BELOW Discriminator")
    print("=" * 80)

    # 加载 traces
    traces = load_traces()
    print(f"\nLoaded {len(traces)} traces")

    # 过滤到 other error type
    other_traces = filter_other_error_type(traces)
    print(f"Filtered to {len(other_traces)} traces with first_error_type == 'other'")

    # 运行所有验证
    results = {}

    results["leave_bug_type_out"] = leave_bug_type_out_validation(other_traces)
    results["seed_holdout"] = seed_holdout_validation(other_traces)
    results["feature_ablation"] = feature_ablation(other_traces)
    results["permutation_sanity"] = permutation_sanity_check(other_traces)

    # 保存结果
    results_dir = PROJECT_ROOT / "results" / "phase_g_anti_overfit"
    results_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_path = results_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # 总体总结
    print(f"\n{'='*80}")
    print(f"Overall Summary")
    print(f"{'='*80}")

    print(f"\nLeave-Bug-Type-Out:")
    print(f"  Near Recall: {results['leave_bug_type_out']['mean_near_recall']:.1%}")
    print(f"  Below Filtered: {results['leave_bug_type_out']['mean_below_filtered']:.1%}")
    print(f"  Success: {'[PASS]' if results['leave_bug_type_out']['success_near'] and results['leave_bug_type_out']['success_below'] else '[FAIL]'}")

    print(f"\nSeed Holdout:")
    print(f"  Near Recall: {results['seed_holdout']['mean_near_recall']:.1%}")
    print(f"  Below Filtered: {results['seed_holdout']['mean_below_filtered']:.1%}")

    print(f"\nFeature Ablation:")
    for name, res in results["feature_ablation"].items():
        print(f"  {name}: Near={res['near_recall']:.1%}, Below={res['below_filtered']:.1%}")

    print(f"\nPermutation Sanity Check:")
    print(f"  Mean accuracy: {results['permutation_sanity']['mean_accuracy']:.1%}")
    print(f"  Within random: {'[PASS]' if results['permutation_sanity']['within_random_range'] else '[FAIL]'}")

    print("\n" + "=" * 80)
    print("Phase G Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
