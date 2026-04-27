"""
Phase F: NEAR vs BELOW Discriminator

目标：在 first_error_type == "other" 的样本中，找到能区分 NEAR 和 BELOW 的信号

候选信号：
- error message content
- expected_actual_distance
- patch size
- feedback_changed_code
- feedback_uses_error_signal
- first_patch_size
- buggy_code vs first_attempt edit distance
- visible/hidden failure pattern

特别关注：为什么 first_visible_pass 全是 True？如果 visible test 设计太弱，
当前 "hidden-gap" 其实混入了 BELOW。这会影响 boundary 定义。
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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


def extract_features(trace: RuntimeTrace) -> Dict[str, float]:
    """Extract candidate features from a trace"""
    features = {}

    # Basic binary features
    features["first_attempt_parse_ok"] = float(trace.first_attempt_parse_ok)
    features["first_attempt_syntax_ok"] = float(trace.first_attempt_syntax_ok)
    features["first_visible_pass"] = float(trace.first_visible_pass)
    features["first_hidden_pass"] = float(trace.first_hidden_pass)
    features["has_expected_actual"] = float(trace.has_expected_actual)
    features["first_changed_from_buggy"] = float(trace.first_changed_from_buggy)
    features["blind_changed_code"] = float(trace.blind_changed_code)
    features["feedback_changed_code"] = float(trace.feedback_changed_code)
    features["feedback_uses_error_signal"] = float(trace.feedback_uses_error_signal)

    # Numeric features
    features["first_error_message_len"] = float(trace.first_error_message_len)
    features["expected_actual_distance"] = float(trace.expected_actual_distance)
    features["first_patch_size"] = float(trace.first_patch_size)

    # Derived features
    features["patch_size_to_message_len_ratio"] = (
        trace.first_patch_size / max(trace.first_error_message_len, 1)
    )

    return features


def get_label(trace: RuntimeTrace) -> int:
    """Get binary label: 1 = NEAR, 0 = BELOW"""
    # NEAR = feedback_success is True
    # BELOW = feedback_success is False
    return 1 if trace.feedback_success else 0


def analyze_feature_distributions(traces: List[RuntimeTrace]):
    """Analyze feature distributions by zone"""
    near_traces = [t for t in traces if t.feedback_success]
    below_traces = [t for t in traces if not t.feedback_success]

    print(f"\n=== Feature Distribution Analysis ===")
    print(f"NEAR samples (feedback_success=True): {len(near_traces)}")
    print(f"BELOW samples (feedback_success=False): {len(below_traces)}")

    if not near_traces or not below_traces:
        print("  [WARNING] Not enough samples in one or both zones")
        return

    feature_names = list(extract_features(traces[0]).keys())

    for feature_name in feature_names:
        near_values = [extract_features(t)[feature_name] for t in near_traces]
        below_values = [extract_features(t)[feature_name] for t in below_traces]

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


def train_and_evaluate_classifier(traces: List[RuntimeTrace]):
    """Train and evaluate a classifier to distinguish NEAR vs BELOW"""
    print(f"\n=== Classifier Evaluation ===")

    X = []
    y = []
    feature_names = None

    for trace in traces:
        features = extract_features(trace)
        if feature_names is None:
            feature_names = list(features.keys())
        X.append([features[name] for name in feature_names])
        y.append(get_label(trace))

    X = np.array(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        print("  [WARNING] Only one class present, cannot train classifier")
        return None, None

    # Try multiple classifiers
    classifiers = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ]

    best_clf = None
    best_score = 0
    best_name = ""

    for name, clf in classifiers:
        print(f"\n  {name}:")

        # Cross-validation
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        print(f"    CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Fit on full data for feature importance
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
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_clf = clf
            best_name = name

    print(f"\n  Best classifier: {best_name} (CV Accuracy: {best_score:.3f})")

    return best_clf, feature_names


def analyze_visible_pass_issue(traces: List[RuntimeTrace]):
    """Analyze why first_visible_pass is always True"""
    print(f"\n=== Visible Pass Issue Analysis ===")

    all_visible_pass = all(t.first_visible_pass for t in traces)
    print(f"  first_visible_pass == True for all traces: {all_visible_pass}")

    if all_visible_pass:
        print(f"\n  [WARNING] Visible test seems too weak!")
        print(f"  This means 'hidden-gap' zone may actually include BELOW samples.")
        print(f"  Need to strengthen visible tests or redefine boundary zones.")

        # Analyze error types
        error_types = defaultdict(int)
        for t in traces:
            error_types[t.first_error_type] += 1

        print(f"\n  First error type distribution:")
        for et, count in sorted(error_types.items()):
            print(f"    {et}: {count}")


def simulate_improved_scheduler(traces: List[RuntimeTrace], classifier, feature_names):
    """Simulate scheduler with NEAR vs BELOW discrimination"""
    print(f"\n=== Improved Scheduler Simulation ===")

    if classifier is None:
        print("  [SKIP] No classifier available")
        return

    # Extract features for all traces
    X = []
    for trace in traces:
        features = extract_features(trace)
        X.append([features[name] for name in feature_names])
    X = np.array(X)

    # Get predictions
    y_pred = classifier.predict(X)

    # Simulate scheduling
    total = len(traces)
    feedback_calls = 0
    successes = 0

    for i, trace in enumerate(traces):
        if trace.first_error_type == "pass":
            # ABOVE zone - accept
            successes += 1 if trace.single_success else 0
        else:
            # other error type - use classifier
            predicted_near = bool(y_pred[i])
            if predicted_near:
                # Predicted NEAR - give feedback
                feedback_calls += 1
                successes += 1 if trace.feedback_success else 0
            else:
                # Predicted BELOW - skip feedback (or escalate)
                # For now, just skip and count as failure
                pass

    success_rate = successes / total if total > 0 else 0
    print(f"  Improved Scheduler:")
    print(f"    Success rate: {success_rate:.1%}")
    print(f"    Feedback calls: {feedback_calls}")
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


def main():
    print("=" * 80)
    print("Phase F: NEAR vs BELOW Discriminator")
    print("=" * 80)

    # Load traces
    traces = load_traces()
    print(f"\nLoaded {len(traces)} traces")

    # Analyze visible pass issue first
    analyze_visible_pass_issue(traces)

    # Filter to other error type
    other_traces = filter_other_error_type(traces)
    print(f"\nFiltered to {len(other_traces)} traces with first_error_type == 'other'")

    if len(other_traces) < 10:
        print("\n[WARNING] Not enough 'other' error type traces for meaningful analysis")
        return

    # Analyze feature distributions
    analyze_feature_distributions(other_traces)

    # Train and evaluate classifier
    classifier, feature_names = train_and_evaluate_classifier(other_traces)

    # Simulate improved scheduler
    simulate_improved_scheduler(traces, classifier, feature_names)

    # Save results
    output = {
        "experiment": "near_below_discriminator",
        "n_total_traces": len(traces),
        "n_other_traces": len(other_traces),
        "visible_pass_all_true": all(t.first_visible_pass for t in traces),
    }

    results_dir = PROJECT_ROOT / "results" / "near_below_discriminator"
    results_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_path = results_dir / f"discriminator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 80)
    print("Phase F Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
