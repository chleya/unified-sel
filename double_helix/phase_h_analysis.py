"""
Phase H Analysis: Cross-solver Validation

目标：
1. 对比 SearchLocalSolver 和 HeuristicLocalSolver 的 boundary 分布
2. 测试跨 solver 迁移学习：训练在 A，测试在 B
3. 验证 first-pass 信号的跨 solver 泛化能力
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
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
    solver_name: str = ""
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


def load_traces_from_file(file_path: Path) -> List[RuntimeTrace]:
    """Load traces from a JSON file"""
    with open(file_path, "r") as f:
        data = json.load(f)

    traces = []
    for trace_data in data.get("traces", []):
        trace = RuntimeTrace(**trace_data)
        traces.append(trace)

    return traces


def extract_first_pass_features(trace: RuntimeTrace) -> Dict[str, float]:
    """Extract ONLY first-pass features (NO feedback, NO blind)"""
    features = {}

    features["first_attempt_parse_ok"] = float(trace.first_attempt_parse_ok)
    features["first_attempt_syntax_ok"] = float(trace.first_attempt_syntax_ok)
    features["first_visible_pass"] = float(trace.first_visible_pass)
    features["first_hidden_pass"] = float(trace.first_hidden_pass)
    features["has_expected_actual"] = float(trace.has_expected_actual)
    features["first_changed_from_buggy"] = float(trace.first_changed_from_buggy)

    features["first_error_message_len"] = float(trace.first_error_message_len)
    features["expected_actual_distance"] = float(trace.expected_actual_distance)
    features["first_patch_size"] = float(trace.first_patch_size)

    features["patch_size_to_message_len_ratio"] = (
        trace.first_patch_size / max(trace.first_error_message_len, 1)
    )

    return features


def get_label(trace: RuntimeTrace) -> int:
    """Get binary label: 1 = NEAR (feedback helps), 0 = BELOW (feedback doesn't help)"""
    return 1 if trace.feedback_success else 0


def analyze_boundary_distribution(traces: List[RuntimeTrace], solver_name: str):
    """Analyze boundary distribution for a solver"""
    print(f"\n{'='*80}")
    print(f"Boundary Distribution for {solver_name}")
    print(f"{'='*80}")

    boundary_counts = defaultdict(int)
    for trace in traces:
        boundary_counts[trace.boundary_label] += 1

    total = len(traces)
    print(f"\nTotal traces: {total}")
    for label, count in sorted(boundary_counts.items()):
        print(f"  {label}: {count} ({count/total:.1%})")

    # Filter to other error type
    other_traces = [t for t in traces if t.first_error_type == "other"]
    print(f"\nTraces with first_error_type == 'other': {len(other_traces)}")

    if len(other_traces) > 0:
        near_traces = [t for t in other_traces if t.feedback_success]
        below_traces = [t for t in other_traces if not t.feedback_success]
        print(f"  NEAR (feedback_success=True): {len(near_traces)}")
        print(f"  BELOW (feedback_success=False): {len(below_traces)}")


def cross_solver_transfer_learning(
    traces_source: List[RuntimeTrace],
    traces_target: List[RuntimeTrace],
    source_solver: str,
    target_solver: str,
):
    """Train on source solver, test on target solver"""
    print(f"\n{'='*80}")
    print(f"Cross-solver Transfer Learning: {source_solver} → {target_solver}")
    print(f"{'='*80}")

    # Filter to other error type
    source_other = [t for t in traces_source if t.first_error_type == "other"]
    target_other = [t for t in traces_target if t.first_error_type == "other"]

    print(f"\nSource ({source_solver}): {len(source_other)} 'other' traces")
    print(f"Target ({target_solver}): {len(target_other)} 'other' traces")

    if len(source_other) < 10 or len(target_other) < 10:
        print("  [WARNING] Not enough traces for meaningful transfer learning")
        return None

    # Extract features
    feature_names = list(extract_first_pass_features(source_other[0]).keys())

    X_source = []
    y_source = []
    for trace in source_other:
        features = extract_first_pass_features(trace)
        X_source.append([features[name] for name in feature_names])
        y_source.append(get_label(trace))

    X_target = []
    y_target = []
    for trace in target_other:
        features = extract_first_pass_features(trace)
        X_target.append([features[name] for name in feature_names])
        y_target.append(get_label(trace))

    X_source = np.array(X_source)
    y_source = np.array(y_source)
    X_target = np.array(X_target)
    y_target = np.array(y_target)

    # Train on source
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_source, y_source)

    # Test on target
    y_pred = clf.predict(X_target)
    y_pred_proba = clf.predict_proba(X_target)[:, 1]

    # Metrics
    acc = accuracy_score(y_target, y_pred)
    prec = precision_score(y_target, y_pred, zero_division=0)
    rec = recall_score(y_target, y_pred, zero_division=0)
    f1 = f1_score(y_target, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_target, y_pred_proba)

    print(f"\nTransfer learning performance:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall: {rec:.3f}")
    print(f"  F1: {f1:.3f}")
    print(f"  ROC AUC: {roc_auc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_target, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    [[{cm[0][0]}, {cm[0][1]}]")
    print(f"     [{cm[1][0]}, {cm[1][1]}]]")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_target, y_pred, target_names=["BELOW", "NEAR"], zero_division=0))

    # Feature importance
    print(f"\n  Feature Importance (from source model):")
    for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.3f}")

    return {
        "source_solver": source_solver,
        "target_solver": target_solver,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }


def compare_feature_distributions(
    traces_search: List[RuntimeTrace],
    traces_heuristic: List[RuntimeTrace],
):
    """Compare feature distributions between solvers"""
    print(f"\n{'='*80}")
    print(f"Feature Distribution Comparison")
    print(f"{'='*80}")

    # Filter to other error type and NEAR
    search_near = [t for t in traces_search if t.first_error_type == "other" and t.feedback_success]
    heuristic_near = [t for t in traces_heuristic if t.first_error_type == "other" and t.feedback_success]

    print(f"\nNEAR samples:")
    print(f"  SearchLocalSolver: {len(search_near)}")
    print(f"  HeuristicLocalSolver: {len(heuristic_near)}")

    if len(search_near) < 5 or len(heuristic_near) < 5:
        print("  [WARNING] Not enough NEAR samples for comparison")
        return

    feature_names = list(extract_first_pass_features(search_near[0]).keys())

    print(f"\nFeature comparison (NEAR zone only):")
    for feature_name in feature_names:
        search_values = [extract_first_pass_features(t)[feature_name] for t in search_near]
        heuristic_values = [extract_first_pass_features(t)[feature_name] for t in heuristic_near]

        search_mean = np.mean(search_values)
        search_std = np.std(search_values)
        heuristic_mean = np.mean(heuristic_values)
        heuristic_std = np.std(heuristic_values)

        diff = search_mean - heuristic_mean
        pooled_std = np.sqrt((search_std**2 + heuristic_std**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0

        print(f"\n  {feature_name}:")
        print(f"    Search: {search_mean:.3f} ± {search_std:.3f}")
        print(f"    Heuristic: {heuristic_mean:.3f} ± {heuristic_std:.3f}")
        print(f"    Difference: {diff:.3f} (Cohen's d: {cohens_d:.3f})")


def main():
    print("=" * 80)
    print("Phase H Analysis: Cross-solver Validation")
    print("=" * 80)

    # Find trace files
    results_dir = PROJECT_ROOT / "results"

    # SearchLocalSolver traces (from Phase D)
    search_trace_dir = results_dir / "runtime_trace_boundary_experiment"
    search_trace_files = sorted(search_trace_dir.glob("experiment_*.json"), reverse=True)

    # HeuristicLocalSolver traces (from Phase H)
    heuristic_trace_dir = results_dir / "phase_h_heuristic_traces"
    heuristic_trace_files = sorted(heuristic_trace_dir.glob("traces_HeuristicLocalSolver_*.json"), reverse=True)

    if not search_trace_files:
        print("ERROR: No SearchLocalSolver trace files found!")
        return

    if not heuristic_trace_files:
        print("ERROR: No HeuristicLocalSolver trace files found!")
        return

    search_trace_file = search_trace_files[0]
    heuristic_trace_file = heuristic_trace_files[0]

    print(f"\nLoading SearchLocalSolver traces from: {search_trace_file}")
    traces_search = load_traces_from_file(search_trace_file)

    print(f"Loading HeuristicLocalSolver traces from: {heuristic_trace_file}")
    traces_heuristic = load_traces_from_file(heuristic_trace_file)

    # Analyze boundary distributions
    analyze_boundary_distribution(traces_search, "SearchLocalSolver")
    analyze_boundary_distribution(traces_heuristic, "HeuristicLocalSolver")

    # Compare feature distributions
    compare_feature_distributions(traces_search, traces_heuristic)

    # Cross-solver transfer learning
    transfer_results = []

    # Search → Heuristic
    result = cross_solver_transfer_learning(
        traces_search, traces_heuristic,
        "SearchLocalSolver", "HeuristicLocalSolver"
    )
    if result:
        transfer_results.append(result)

    # Heuristic → Search
    result = cross_solver_transfer_learning(
        traces_heuristic, traces_search,
        "HeuristicLocalSolver", "SearchLocalSolver"
    )
    if result:
        transfer_results.append(result)

    # Save results
    if transfer_results:
        output_dir = PROJECT_ROOT / "results" / "phase_h_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        output_path = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_path, "w") as f:
            json.dump({"transfer_results": transfer_results}, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Phase H Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
