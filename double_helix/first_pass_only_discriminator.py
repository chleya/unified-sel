"""
Phase G: First-pass Only NEAR vs BELOW Discriminator

目标：只用 first-pass 信号（feedback 之前的信号）来区分 NEAR 和 BELOW

First-pass 信号定义：
- 所有在 single_attempt 之后、feedback_retry 之前收集的信号
- 不包括任何 feedback_* 开头的信号
- 不包括 blind_* 信号（因为 blind retry 也需要运行，不算 first-pass）

关键原则：
1. NO CHEATING - 不能使用任何需要运行 feedback 才能获得的信号
2. NO LEAKAGE - 不能使用需要运行 blind retry 才能获得的信号
3. ONLY FIRST-PASS - 只用 single_attempt 后的 verifier 信号
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

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


def load_traces() -> List[RuntimeTrace]:
    """Load traces from runtime_trace_boundary_experiment.py output"""
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

    traces = []
    for trace_data in data["traces"]:
        trace = RuntimeTrace(**trace_data)
        traces.append(trace)

    return traces


def filter_other_error_type(traces: List[RuntimeTrace]) -> List[RuntimeTrace]:
    """Filter to only first_error_type == 'other'"""
    return [t for t in traces if t.first_error_type == "other"]


def extract_first_pass_features(trace: RuntimeTrace) -> Dict[str, float]:
    """
    Extract ONLY first-pass features (NO feedback, NO blind)
    
    First-pass = after single_attempt, before any retry/feedback
    """
    features = {}

    # First-pass only features
    features["first_attempt_parse_ok"] = float(trace.first_attempt_parse_ok)
    features["first_attempt_syntax_ok"] = float(trace.first_attempt_syntax_ok)
    features["first_visible_pass"] = float(trace.first_visible_pass)
    features["first_hidden_pass"] = float(trace.first_hidden_pass)
    features["has_expected_actual"] = float(trace.has_expected_actual)
    features["first_changed_from_buggy"] = float(trace.first_changed_from_buggy)

    # Numeric features from first-pass
    features["first_error_message_len"] = float(trace.first_error_message_len)
    features["expected_actual_distance"] = float(trace.expected_actual_distance)
    features["first_patch_size"] = float(trace.first_patch_size)

    # Derived features from first-pass only
    features["patch_size_to_message_len_ratio"] = (
        trace.first_patch_size / max(trace.first_error_message_len, 1)
    )

    # Explicitly NOT including:
    # - blind_* (requires running blind retry)
    # - feedback_* (requires running feedback)
    # - single_success (we're trying to predict if feedback helps, not if single works)

    return features


def get_label(trace: RuntimeTrace) -> int:
    """Get binary label: 1 = NEAR (feedback helps), 0 = BELOW (feedback doesn't help)"""
    return 1 if trace.feedback_success else 0


def analyze_first_pass_feature_distributions(traces: List[RuntimeTrace]):
    """Analyze first-pass feature distributions by zone"""
    near_traces = [t for t in traces if t.feedback_success]
    below_traces = [t for t in traces if not t.feedback_success]

    print(f"\n=== First-pass Feature Distribution Analysis ===")
    print(f"NEAR samples (feedback_success=True): {len(near_traces)}")
    print(f"BELOW samples (feedback_success=False): {len(below_traces)}")

    if not near_traces or not below_traces:
        print("  [WARNING] Not enough samples in one or both zones")
        return

    feature_names = list(extract_first_pass_features(traces[0]).keys())

    for feature_name in feature_names:
        near_values = [extract_first_pass_features(t)[feature_name] for t in near_traces]
        below_values = [extract_first_pass_features(t)[feature_name] for t in below_traces]

        near_mean = np.mean(near_values)
        near_std = np.std(near_values)
        below_mean = np.mean(below_values)
        below_std = np.std(below_values)

        diff = near_mean - below_mean
        pooled_std = np.sqrt((near_std**2 + below_std**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0

        print(f"\n  {feature_name}:")
        print(f"    NEAR: {near_mean:.3f} ± {near_std:.3f}")
        print(f"    BELOW: {below_mean:.3f} ± {below_std:.3f}")
        print(f"    Difference: {diff:.3f} (Cohen's d: {cohens_d:.3f})")


def train_and_evaluate_first_pass_classifier(traces: List[RuntimeTrace]):
    """Train and evaluate a classifier using ONLY first-pass features"""
    print(f"\n=== First-pass Only Classifier Evaluation ===")

    X = []
    y = []
    feature_names = None

    for trace in traces:
        features = extract_first_pass_features(trace)
        if feature_names is None:
            feature_names = list(features.keys())
        X.append([features[name] for name in feature_names])
        y.append(get_label(trace))

    X = np.array(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        print("  [WARNING] Only one class present, cannot train classifier")
        return None, None

    print(f"  Features used (first-pass only):")
    for i, name in enumerate(feature_names):
        print(f"    {i+1}. {name}")

    # Try multiple classifiers with stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ]

    best_clf = None
    best_score = 0
    best_name = ""
    best_results = None

    for name, clf in classifiers:
        print(f"\n  {name}:")

        # Cross-validation
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        cv_roc_auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        print(f"    CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"    CV ROC AUC: {cv_roc_auc.mean():.3f} ± {cv_roc_auc.std():.3f}")

        # Fit on full data for feature importance and detailed metrics
        clf.fit(X, y)

        # Predictions
        y_pred = clf.predict(X)
        y_pred_proba = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else None

        # Metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else None

        print(f"    Train Accuracy: {acc:.3f}")
        print(f"    Precision: {prec:.3f}")
        print(f"    Recall: {rec:.3f}")
        print(f"    F1: {f1:.3f}")
        if roc_auc is not None:
            print(f"    ROC AUC: {roc_auc:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"    Confusion Matrix:")
        print(f"      [[{cm[0][0]}, {cm[0][1]}]")
        print(f"       [{cm[1][0]}, {cm[1][1]}]]")

        # Classification report
        print(f"    Classification Report:")
        print(classification_report(y, y_pred, target_names=["BELOW", "NEAR"], zero_division=0))

        # Feature importance
        if hasattr(clf, "feature_importances_"):
            print(f"\n    Feature Importance:")
            for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
                print(f"      {name}: {imp:.3f}")
        elif hasattr(clf, "coef_"):
            print(f"\n    Coefficients:")
            for name, coef in sorted(zip(feature_names, clf.coef_[0]), key=lambda x: -abs(x[1])):
                print(f"      {name}: {coef:.3f}")

        # Track best
        mean_score = cv_scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_clf = clf
            best_name = name
            best_results = {
                "cv_accuracy": cv_scores.mean(),
                "cv_accuracy_std": cv_scores.std(),
                "cv_roc_auc": cv_roc_auc.mean(),
                "cv_roc_auc_std": cv_roc_auc.std(),
                "train_accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc,
                "confusion_matrix": cm.tolist(),
            }

    print(f"\n  Best classifier: {best_name} (CV Accuracy: {best_score:.3f})")

    return best_clf, feature_names, best_results


def simulate_improved_scheduler_first_pass(
    traces: List[RuntimeTrace],
    classifier,
    feature_names
):
    """Simulate scheduler with NEAR vs BELOW discrimination using ONLY first-pass features"""
    print(f"\n=== Improved Scheduler Simulation (First-pass Only) ===")

    if classifier is None:
        print("  [SKIP] No classifier available")
        return

    # Extract first-pass features for all traces
    X = []
    for trace in traces:
        features = extract_first_pass_features(trace)
        X.append([features[name] for name in feature_names])
    X = np.array(X)

    # Get predictions
    y_pred = classifier.predict(X)

    # Simulate scheduling
    total = len(traces)
    feedback_calls = 0
    successes = 0
    wasted_feedback_on_below = 0
    missed_near_cases = 0

    for i, trace in enumerate(traces):
        if trace.first_error_type == "pass":
            # ABOVE zone - accept
            successes += 1 if trace.single_success else 0
        else:
            # other error type - use first-pass classifier
            predicted_near = bool(y_pred[i])
            actual_near = bool(trace.feedback_success)

            if predicted_near:
                # Predicted NEAR - give feedback
                feedback_calls += 1
                successes += 1 if trace.feedback_success else 0

                if not actual_near:
                    wasted_feedback_on_below += 1
            else:
                # Predicted BELOW - skip feedback
                if actual_near:
                    missed_near_cases += 1

    success_rate = successes / total if total > 0 else 0
    print(f"  Improved Scheduler (First-pass Only):")
    print(f"    Success rate: {success_rate:.1%}")
    print(f"    Feedback calls: {feedback_calls}")
    print(f"    Wasted feedback on BELOW: {wasted_feedback_on_below}")
    print(f"    Missed NEAR cases: {missed_near_cases}")
    print(f"    Feedback saved: {total - feedback_calls} ({(total - feedback_calls) / total:.1%})")

    # Compare with Policy D (original runtime scheduler)
    print(f"\n  Policy D (Original Runtime Scheduler):")
    policy_d_feedback = sum(1 for t in traces if t.first_error_type != "pass")
    policy_d_success = sum(
        1 if (t.first_error_type == "pass" and t.single_success) or
             (t.first_error_type != "pass" and t.feedback_success)
        else 0
        for t in traces
    )
    policy_d_success_rate = policy_d_success / total if total > 0 else 0
    print(f"    Success rate: {policy_d_success_rate:.1%}")
    print(f"    Feedback calls: {policy_d_feedback}")

    # Compare with Oracle (Policy C)
    print(f"\n  Policy C (Oracle Difficulty):")
    oracle_feedback = sum(1 for t in traces if t.feedback_success)
    oracle_success = sum(
        1 if (t.first_error_type == "pass" and t.single_success) or
             (t.feedback_success)
        else 0
        for t in traces
    )
    oracle_success_rate = oracle_success / total if total > 0 else 0
    print(f"    Success rate: {oracle_success_rate:.1%}")
    print(f"    Feedback calls: {oracle_feedback}")

    return {
        "improved": {
            "success_rate": success_rate,
            "feedback_calls": feedback_calls,
            "wasted_feedback_on_below": wasted_feedback_on_below,
            "missed_near_cases": missed_near_cases,
        },
        "policy_d": {
            "success_rate": policy_d_success_rate,
            "feedback_calls": policy_d_feedback,
        },
        "policy_c": {
            "success_rate": oracle_success_rate,
            "feedback_calls": oracle_feedback,
        },
    }


def main():
    print("=" * 80)
    print("Phase G: First-pass Only NEAR vs BELOW Discriminator")
    print("=" * 80)
    print("\n原则：")
    print("  1. NO CHEATING - 不使用任何需要运行 feedback 才能获得的信号")
    print("  2. NO LEAKAGE - 不使用任何需要运行 blind retry 才能获得的信号")
    print("  3. ONLY FIRST-PASS - 只用 single_attempt 后的 verifier 信号")

    # Load traces
    traces = load_traces()
    print(f"\nLoaded {len(traces)} traces")

    # Filter to other error type
    other_traces = filter_other_error_type(traces)
    print(f"\nFiltered to {len(other_traces)} traces with first_error_type == 'other'")

    if len(other_traces) < 10:
        print("\n[WARNING] Not enough 'other' error type traces for meaningful analysis")
        return

    # Analyze first-pass feature distributions
    analyze_first_pass_feature_distributions(other_traces)

    # Train and evaluate first-pass only classifier
    classifier, feature_names, best_results = train_and_evaluate_first_pass_classifier(other_traces)

    # Simulate improved scheduler
    scheduler_results = simulate_improved_scheduler_first_pass(traces, classifier, feature_names)

    # Save results
    output = {
        "experiment": "first_pass_only_discriminator",
        "n_total_traces": len(traces),
        "n_other_traces": len(other_traces),
        "principles": [
            "NO CHEATING - no feedback_* signals",
            "NO LEAKAGE - no blind_* signals",
            "ONLY FIRST-PASS - only single_attempt verifier signals",
        ],
        "features_used": feature_names,
        "best_classifier_results": best_results,
        "scheduler_results": scheduler_results,
    }

    results_dir = PROJECT_ROOT / "results" / "first_pass_only_discriminator"
    results_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_path = results_dir / f"discriminator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 80)
    print("Phase G Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
