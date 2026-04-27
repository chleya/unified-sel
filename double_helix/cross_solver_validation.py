"""
Phase H: Cross-solver Validation

目标：验证 first-pass boundary classifier 在不同 solver 上的泛化能力

Solvers to test:
1. SearchLocalSolver (current, already validated)
2. HeuristicLocalSolver (new)
3. OracleSolver (oracle, for reference)

实验设计：
- 对每个 solver 收集 250 traces (5 seeds × 50 tasks)
- 训练在 SearchLocalSolver 上，测试在 HeuristicLocalSolver 上
- 或者交叉验证：训练在 solver A，测试在 solver B

关键问题：
- patch_size_to_message_len_ratio 这个信号是否在不同 solver 上都有效？
- 还是说这个信号是 SearchLocalSolver 特有的？
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

# Import from double_helix
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    SearchLocalSolver,
    HeuristicLocalSolver,
    OracleSolver,
    generate_code_tasks,
    _run_code_task,
    _task_verifier_tests,
    _task_search_tests,
)


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


def _try_once_with_trace(
    task: BenchmarkTask,
    solver,
    error_feedback: str = "",
) -> Tuple[Dict[str, Any], str, bool, bool, str]:
    """Run one attempt and collect trace data"""
    fn_name = task.metadata.get("function_name", "solve")
    visible_tests = _task_search_tests(task)
    buggy_code = task.metadata.get("buggy_code", "")

    if error_feedback and hasattr(solver, "revise"):
        first = solver.solve(task)
        attempt = solver.revise(task, first, error_feedback)
    else:
        attempt = solver.solve(task)

    code = attempt.answer

    # Collect trace data
    trace_data = {
        "first_attempt_parse_ok": attempt.parse_ok,
        "first_attempt_syntax_ok": attempt.syntax_ok,
        "first_error_message_len": len(attempt.error_message) if attempt.error_message else 0,
        "first_patch_size": len(code) if code else 0,
        "first_changed_from_buggy": code != buggy_code if buggy_code else False,
    }

    if not code or not code.strip():
        trace_data["first_visible_pass"] = False
        trace_data["first_hidden_pass"] = False
        trace_data["first_error_type"] = "empty_code"
        trace_data["has_expected_actual"] = False
        trace_data["expected_actual_distance"] = 1.0
        return trace_data, code, False, False, "empty_code"

    vis_pass, vis_msg = _run_code_task(fn_name, code, visible_tests)
    trace_data["first_visible_pass"] = vis_pass

    if not vis_pass:
        trace_data["first_hidden_pass"] = False
        trace_data["first_error_type"] = "visible_fail"
        trace_data["has_expected_actual"] = False
        trace_data["expected_actual_distance"] = 1.0
        return trace_data, code, False, False, f"visible_fail:{vis_msg}"

    all_tests = _task_verifier_tests(task)
    all_pass, all_msg = _run_code_task(fn_name, code, all_tests)
    trace_data["first_hidden_pass"] = all_pass

    if all_pass:
        trace_data["first_error_type"] = "pass"
        trace_data["has_expected_actual"] = False
        trace_data["expected_actual_distance"] = 1.0
    else:
        trace_data["first_error_type"] = "other"
        trace_data["has_expected_actual"] = "expected" in all_msg.lower() and "actual" in all_msg.lower()
        trace_data["expected_actual_distance"] = 0.2 if trace_data["has_expected_actual"] else 0.0

    return trace_data, code, vis_pass, all_pass, "pass" if all_pass else f"hidden_fail:{all_msg}"


def collect_traces_for_solver(
    solver_name: str,
    num_seeds: int = 5,
    num_tasks_per_seed: int = 50,
) -> List[RuntimeTrace]:
    """Collect runtime traces for a specific solver"""
    print(f"\n{'='*80}")
    print(f"Collecting traces for solver: {solver_name}")
    print(f"{'='*80}")

    traces = []

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx * 100
        print(f"\nRunning seed {seed} ({seed_idx+1}/{num_seeds})...")

        # Generate tasks
        all_tasks = generate_code_tasks(100, seed=0, variant="standard")
        rng = np.random.default_rng(seed)
        if len(all_tasks) > num_tasks_per_seed:
            indices = rng.choice(len(all_tasks), size=num_tasks_per_seed, replace=False)
            tasks = [all_tasks[i] for i in sorted(indices)]
        else:
            tasks = all_tasks

        # Create solver
        if solver_name == "SearchLocalSolver":
            solver = SearchLocalSolver()
        elif solver_name == "HeuristicLocalSolver":
            solver = HeuristicLocalSolver()
        elif solver_name == "OracleSolver":
            solver = OracleSolver()
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        for task_idx, task in enumerate(tasks):
            if (task_idx + 1) % 10 == 0:
                print(f"  Seed {seed}: {task_idx+1}/{len(tasks)} tasks done")

            # Single attempt (first-pass)
            single_trace_data, single_code, single_vis, single_all, single_fb = _try_once_with_trace(
                task, solver, error_feedback=""
            )

            # Blind retry
            blind_trace_data, blind_code, blind_vis, blind_all, blind_fb = _try_once_with_trace(
                task, solver, error_feedback=""
            )
            blind_changed_code = blind_code != single_code

            # Feedback retry
            fb_trace_data, fb_code, fb_vis, fb_all, fb_fb = _try_once_with_trace(
                task, solver, error_feedback=single_fb
            )
            feedback_changed_code = fb_code != single_code
            feedback_uses_error_signal = single_fb and single_fb != "pass"
            feedback_patch_size_delta = len(fb_code) - len(single_code) if fb_code and single_code else 0

            # Determine boundary label
            if single_all:
                boundary_label = "above"
            elif fb_all:
                boundary_label = "near"
            else:
                boundary_label = "below"

            # Create trace
            trace = RuntimeTrace(
                task_id=task.task_id,
                bug_type=task.metadata.get("bug_type", ""),
                difficulty=task.metadata.get("difficulty", ""),
                condition=f"seed_{seed}",
                solver_name=solver_name,
                boundary_label=boundary_label,

                single_success=single_all,
                blind_success=blind_all,
                feedback_success=fb_all,

                first_attempt_parse_ok=single_trace_data["first_attempt_parse_ok"],
                first_attempt_syntax_ok=single_trace_data["first_attempt_syntax_ok"],
                first_visible_pass=single_trace_data["first_visible_pass"],
                first_hidden_pass=single_trace_data["first_hidden_pass"],
                first_error_type=single_trace_data["first_error_type"],
                first_error_message_len=single_trace_data["first_error_message_len"],
                has_expected_actual=single_trace_data["has_expected_actual"],
                expected_actual_distance=single_trace_data["expected_actual_distance"],
                first_patch_size=single_trace_data["first_patch_size"],
                first_changed_from_buggy=single_trace_data["first_changed_from_buggy"],

                blind_changed_code=blind_changed_code,
                blind_parse_ok=blind_trace_data["first_attempt_parse_ok"],
                blind_error_type=blind_trace_data["first_error_type"],

                feedback_changed_code=feedback_changed_code,
                feedback_parse_ok=fb_trace_data["first_attempt_parse_ok"],
                feedback_error_type=fb_trace_data["first_error_type"],
                feedback_uses_error_signal=feedback_uses_error_signal,
                feedback_patch_size_delta=feedback_patch_size_delta,
            )

            traces.append(trace)

    print(f"\nCollected {len(traces)} traces for {solver_name}")
    return traces


def extract_first_pass_features(trace: RuntimeTrace) -> Dict[str, float]:
    """Extract ONLY first-pass features (NO feedback, NO blind)"""
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

    return features


def get_label(trace: RuntimeTrace) -> int:
    """Get binary label: 1 = NEAR (feedback helps), 0 = BELOW (feedback doesn't help)"""
    return 1 if trace.feedback_success else 0


def analyze_solver_traces(traces: List[RuntimeTrace], solver_name: str):
    """Analyze traces for a specific solver"""
    print(f"\n{'='*80}")
    print(f"Analysis for solver: {solver_name}")
    print(f"{'='*80}")

    # Boundary distribution
    boundary_counts = defaultdict(int)
    for trace in traces:
        boundary_counts[trace.boundary_label] += 1

    print(f"\nBoundary distribution:")
    total = len(traces)
    for label, count in sorted(boundary_counts.items()):
        print(f"  {label}: {count} ({count/total:.1%})")

    # Filter to other error type
    other_traces = [t for t in traces if t.first_error_type == "other"]
    print(f"\nFiltered to {len(other_traces)} traces with first_error_type == 'other'")

    if len(other_traces) < 10:
        print("  [WARNING] Not enough 'other' error type traces for meaningful analysis")
        return

    # Feature distributions
    near_traces = [t for t in other_traces if t.feedback_success]
    below_traces = [t for t in other_traces if not t.feedback_success]

    print(f"\nFeature distribution by zone:")
    print(f"  NEAR (feedback_success=True): {len(near_traces)}")
    print(f"  BELOW (feedback_success=False): {len(below_traces)}")

    if not near_traces or not below_traces:
        print("  [WARNING] Not enough samples in one or both zones")
        return

    feature_names = list(extract_first_pass_features(other_traces[0]).keys())

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
        return

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


def main():
    print("=" * 80)
    print("Phase H: Cross-solver Validation")
    print("=" * 80)

    # Solvers to test
    solvers = ["SearchLocalSolver", "HeuristicLocalSolver"]

    # Collect traces for each solver
    all_traces = {}
    for solver_name in solvers:
        traces = collect_traces_for_solver(solver_name, num_seeds=5, num_tasks_per_seed=50)
        all_traces[solver_name] = traces

        # Analyze each solver's traces
        analyze_solver_traces(traces, solver_name)

        # Save traces
        results_dir = PROJECT_ROOT / "results" / "cross_solver_validation"
        results_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        output_path = results_dir / f"traces_{solver_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        traces_data = {
            "solver": solver_name,
            "n_traces": len(traces),
            "traces": [
                {k: v for k, v in t.__dict__.items() if not k.startswith("_")}
                for t in traces
            ],
        }

        with open(output_path, "w") as f:
            json.dump(traces_data, f, indent=2)

        print(f"\nTraces saved to: {output_path}")

    # Cross-solver transfer learning
    print(f"\n{'='*80}")
    print("Cross-solver Transfer Learning Experiments")
    print(f"{'='*80}")

    transfer_results = []

    # SearchLocalSolver → HeuristicLocalSolver
    if "SearchLocalSolver" in all_traces and "HeuristicLocalSolver" in all_traces:
        result = cross_solver_transfer_learning(
            all_traces["SearchLocalSolver"],
            all_traces["HeuristicLocalSolver"],
            "SearchLocalSolver",
            "HeuristicLocalSolver",
        )
        if result:
            transfer_results.append(result)

    # HeuristicLocalSolver → SearchLocalSolver
    if "HeuristicLocalSolver" in all_traces and "SearchLocalSolver" in all_traces:
        result = cross_solver_transfer_learning(
            all_traces["HeuristicLocalSolver"],
            all_traces["SearchLocalSolver"],
            "HeuristicLocalSolver",
            "SearchLocalSolver",
        )
        if result:
            transfer_results.append(result)

    # Save transfer results
    if transfer_results:
        output_path = results_dir / f"transfer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump({"transfer_results": transfer_results}, f, indent=2)
        print(f"\nTransfer results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Phase H Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
