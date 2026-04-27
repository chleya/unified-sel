"""
Honest Boundary Experiment - 修复所有审核问题后的干净实验

核心修复：
1. patch_size 不再是 bug_type 指纹：每个任务有参数化变体，代码长度不同
2. 同一 bug_type 同时有 NEAR 和 BELOW：多种 bug 引入方式
3. 严格特征审计：禁止使用 bug_type 相关特征（patch_size, message_len）
4. GroupKFold by bug_type：避免模板记忆
5. 真实的 blind_success：不硬编码
6. 不使用 simulate_health_signals：移除标签泄漏

特征白名单（只允许真正的运行时信号）：
- first_attempt_parse_ok: 是否产生了非空代码
- first_visible_pass: 是否通过了可见测试
- first_error_type: 错误类型（NameError/AssertionError/other/pass）
- has_expected_actual: 错误消息是否包含 expected/actual

特征黑名单（禁止使用）：
- first_patch_size: bug_type 指纹
- first_error_message_len: 和 bug_type 相关
- patch_size_to_message_len_ratio: bug_type 指纹
- expected_actual_distance: 启发式估算，和 message_len 相关
- first_hidden_pass: 过滤后是常量
- first_changed_from_buggy: 过滤后是常量
- first_attempt_syntax_ok: SearchLocalSolver 保证语法正确
- boundary_label: 标签
- difficulty: 标签
- bug_type: 标签
- single_success: 标签
- blind_*: 不是 first-pass 信号
- feedback_*: 不是 first-pass 信号
"""

import json
import sys
import random
import numpy as np
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    SearchLocalSolver,
    _run_code_task,
    _task_search_tests,
    _task_verifier_tests,
    generate_code_tasks,
)

ALLOWED_FEATURES = [
    "first_attempt_parse_ok",
    "first_visible_pass",
    "first_error_type_other",
    "first_error_type_AssertionError",
    "first_error_type_NameError",
    "first_error_type_TypeError",
    "first_error_type_pass",
    "has_expected_actual",
]

FORBIDDEN_FEATURES = [
    "first_patch_size",
    "first_error_message_len",
    "patch_size_to_message_len_ratio",
    "expected_actual_distance",
    "first_hidden_pass",
    "first_changed_from_buggy",
    "first_attempt_syntax_ok",
    "boundary_label",
    "difficulty",
    "bug_type",
    "single_success",
    "blind_changed_code",
    "blind_parse_ok",
    "blind_error_type",
    "blind_success",
    "feedback_changed_code",
    "feedback_parse_ok",
    "feedback_error_type",
    "feedback_uses_error_signal",
    "feedback_patch_size_delta",
    "feedback_success",
]


@dataclass
class HonestTrace:
    task_id: str = ""
    bug_type: str = ""
    boundary_label: str = ""

    single_success: bool = False
    blind_success: bool = False
    feedback_success: bool = False

    first_attempt_parse_ok: bool = False
    first_visible_pass: bool = False
    first_error_type: str = ""
    has_expected_actual: bool = False

    first_patch_size: int = 0
    first_error_message_len: int = 0
    first_hidden_pass: bool = False
    first_changed_from_buggy: bool = False
    first_attempt_syntax_ok: bool = False
    expected_actual_distance: float = 0.0

    blind_changed_code: bool = False
    blind_parse_ok: bool = False
    blind_error_type: str = ""

    feedback_changed_code: bool = False
    feedback_parse_ok: bool = False
    feedback_error_type: str = ""
    feedback_uses_error_signal: bool = False
    feedback_patch_size_delta: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_error_type(error_msg: str) -> str:
    if not error_msg or error_msg == "pass":
        return "pass"
    error_msg = str(error_msg).lower()
    for et in ["nameerror", "assertionerror", "typeerror", "syntaxerror",
               "attributeerror", "indexerror", "keyerror", "valueerror",
               "zerodivisionerror", "overflowerror"]:
        if et in error_msg:
            return et.replace("error", "Error")
    if "empty" in error_msg or "no code" in error_msg:
        return "empty"
    return "other"


def check_has_expected_actual(error_message: str) -> bool:
    if not error_message:
        return False
    el = error_message.lower()
    return ("expected" in el and "actual" in el) or \
           ("expected" in el and "got" in el)


def collect_honest_trace(
    task: BenchmarkTask,
    solver: SearchLocalSolver,
) -> HonestTrace:
    trace = HonestTrace()
    trace.task_id = task.metadata.get("task_id", "unknown")
    trace.bug_type = task.metadata.get("bug_type", "unknown")

    fn_name = task.metadata.get("function_name", "solve")

    # First attempt
    attempt1 = solver.solve(task)
    code1 = attempt1.answer if attempt1 else ""

    parse_ok1 = bool(code1 and code1.strip())
    trace.first_attempt_parse_ok = parse_ok1
    trace.first_attempt_syntax_ok = parse_ok1
    trace.first_patch_size = len(code1)

    if not parse_ok1:
        trace.first_error_type = "empty"
        trace.first_visible_pass = False
        trace.first_hidden_pass = False
    else:
        vis_tests = _task_search_tests(task)
        vis_pass, vis_msg = _run_code_task(fn_name, code1, vis_tests)
        trace.first_visible_pass = vis_pass

        if not vis_pass:
            trace.first_error_type = parse_error_type(vis_msg)
            trace.first_error_message_len = len(vis_msg) if vis_msg else 0
            trace.has_expected_actual = check_has_expected_actual(vis_msg)
            trace.first_hidden_pass = False
        else:
            all_tests = _task_verifier_tests(task)
            all_pass, all_msg = _run_code_task(fn_name, code1, all_tests)
            trace.first_hidden_pass = all_pass
            if all_pass:
                trace.first_error_type = "pass"
            else:
                trace.first_error_type = parse_error_type(all_msg)
                trace.first_error_message_len = len(all_msg) if all_msg else 0
                trace.has_expected_actual = check_has_expected_actual(all_msg)

    trace.first_changed_from_buggy = (code1 != task.metadata.get("buggy_code", ""))

    # Single success
    trace.single_success = trace.first_hidden_pass

    # Blind retry (only if not already solved)
    if not trace.single_success:
        attempt2 = solver.solve(task)
        code2 = attempt2.answer if attempt2 else ""
        parse_ok2 = bool(code2 and code2.strip())
        trace.blind_changed_code = (code2 != code1) if code1 and code2 else False
        trace.blind_parse_ok = parse_ok2

        if parse_ok2:
            all_tests = _task_verifier_tests(task)
            blind_pass, blind_msg = _run_code_task(fn_name, code2, all_tests)
            trace.blind_success = blind_pass
            trace.blind_error_type = "pass" if blind_pass else parse_error_type(blind_msg)
        else:
            trace.blind_success = False
            trace.blind_error_type = "empty"
    else:
        trace.blind_success = True
        trace.blind_changed_code = False
        trace.blind_parse_ok = True
        trace.blind_error_type = "pass"

    # Feedback retry (only if blind failed)
    if not trace.blind_success and trace.first_error_message_len > 0:
        err_msg = ""
        if not trace.first_visible_pass:
            vis_tests = _task_search_tests(task)
            _, vis_msg = _run_code_task(fn_name, code1, vis_tests)
            err_msg = vis_msg if vis_msg else ""
        elif not trace.first_hidden_pass:
            all_tests = _task_verifier_tests(task)
            _, all_msg = _run_code_task(fn_name, code1, all_tests)
            err_msg = all_msg if all_msg else ""

        if hasattr(solver, "revise") and err_msg:
            attempt3 = solver.revise(task, attempt1, err_msg)
        else:
            attempt3 = solver.solve(task)

        code3 = attempt3.answer if attempt3 else ""
        parse_ok3 = bool(code3 and code3.strip())
        trace.feedback_changed_code = (code3 != code1) if code1 and code3 else False
        trace.feedback_parse_ok = parse_ok3
        trace.feedback_uses_error_signal = bool(err_msg)
        trace.feedback_patch_size_delta = len(code3) - len(code1) if code1 and code3 else 0

        if parse_ok3:
            all_tests = _task_verifier_tests(task)
            fb_pass, fb_msg = _run_code_task(fn_name, code3, all_tests)
            trace.feedback_success = fb_pass
            trace.feedback_error_type = "pass" if fb_pass else parse_error_type(fb_msg)
        else:
            trace.feedback_success = False
            trace.feedback_error_type = "empty"
    elif trace.blind_success:
        trace.feedback_success = True
        trace.feedback_changed_code = False
        trace.feedback_parse_ok = True
        trace.feedback_error_type = "pass"
        trace.feedback_uses_error_signal = False
        trace.feedback_patch_size_delta = 0
    else:
        trace.feedback_success = False

    # Boundary label (post-hoc, NOT used as feature)
    if trace.single_success:
        trace.boundary_label = "above"
    elif trace.feedback_success:
        trace.boundary_label = "near"
    else:
        trace.boundary_label = "below"

    return trace


def extract_allowed_features(trace: HonestTrace) -> Dict[str, float]:
    features = {}

    features["first_attempt_parse_ok"] = float(trace.first_attempt_parse_ok)
    features["first_visible_pass"] = float(trace.first_visible_pass)
    features["has_expected_actual"] = float(trace.has_expected_actual)

    error_type = trace.first_error_type
    for et in ["other", "AssertionError", "NameError", "TypeError", "pass"]:
        features[f"first_error_type_{et}"] = float(error_type == et)

    return features


def get_label(trace: HonestTrace) -> int:
    return 1 if trace.boundary_label == "near" else 0


def run_honest_experiment(
    num_tasks: int = 250,
    num_seeds: int = 5,
    solver_seed: int = 42,
) -> List[HonestTrace]:
    """收集诚实的 traces"""
    solver = SearchLocalSolver()
    all_traces = []

    seeds = [42, 123, 456, 789, 1024][:num_seeds]

    for seed in seeds:
        tasks = generate_code_tasks(num_tasks, seed=seed, variant="standard")
        rng = random.Random(seed + solver_seed)

        for task in tasks:
            trace = collect_honest_trace(task, solver)
            all_traces.append(trace)

    return all_traces


def analyze_patch_size_diversity(traces: List[HonestTrace]):
    """分析 patch_size 是否仍然是 bug_type 指纹"""
    print("\n" + "=" * 80)
    print("Patch Size Diversity Analysis")
    print("=" * 80)

    bug_type_patches = defaultdict(list)
    for t in traces:
        bug_type_patches[t.bug_type].append(t.first_patch_size)

    constant_count = 0
    varies_count = 0
    for bt, sizes in sorted(bug_type_patches.items()):
        unique = len(set(sizes))
        if unique == 1:
            constant_count += 1
            print(f"  {bt}: {sizes[0]} <-- CONSTANT (still a fingerprint!)")
        else:
            varies_count += 1
            print(f"  {bt}: min={min(sizes)}, max={max(sizes)}, unique={unique} <-- VARIES")

    print(f"\nSummary: {constant_count} constant, {varies_count} varies")
    if constant_count > 0:
        print("  [WARNING] patch_size is still a bug_type fingerprint for some types!")
        print("  patch_size MUST be excluded from features.")
    return constant_count == 0


def analyze_boundary_per_bug_type(traces: List[HonestTrace]):
    """分析同一 bug_type 是否同时有 NEAR 和 BELOW"""
    print("\n" + "=" * 80)
    print("Boundary Distribution per Bug Type")
    print("=" * 80)

    bug_type_boundaries = defaultdict(lambda: defaultdict(int))
    for t in traces:
        bug_type_boundaries[t.bug_type][t.boundary_label] += 1

    mixed_count = 0
    pure_count = 0
    for bt, boundaries in sorted(bug_type_boundaries.items()):
        has_near = boundaries.get("near", 0) > 0
        has_below = boundaries.get("below", 0) > 0
        if has_near and has_below:
            mixed_count += 1
            print(f"  {bt}: near={boundaries['near']}, below={boundaries['below']} <-- MIXED (good!)")
        else:
            pure_count += 1
            labels = ", ".join(f"{k}={v}" for k, v in sorted(boundaries.items()))
            print(f"  {bt}: {labels} <-- PURE (bug_type = boundary fingerprint!)")

    print(f"\nSummary: {mixed_count} mixed, {pure_count} pure")
    if pure_count > 0:
        print("  [WARNING] Some bug_types only have one boundary label!")
        print("  This means bug_type is still a boundary fingerprint.")
    return pure_count == 0


def run_group_kfold_validation(
    traces: List[HonestTrace],
    filter_to_hidden_gap: bool = True,
):
    """GroupKFold by bug_type, only using allowed features"""
    print("\n" + "=" * 80)
    print("GroupKFold Validation (Strict Feature Audit)")
    print("=" * 80)

    if filter_to_hidden_gap:
        filtered = [t for t in traces
                    if t.first_visible_pass and not t.first_hidden_pass
                    and t.boundary_label in ("near", "below")]
        print(f"\nFiltered to hidden-gap (NEAR + BELOW): {len(filtered)} traces")
    else:
        filtered = [t for t in traces if t.boundary_label in ("near", "below")]
        print(f"\nUsing all NEAR + BELOW: {len(filtered)} traces")

    if len(filtered) < 20:
        print("  [SKIP] Not enough data for validation")
        return None

    X = []
    y = []
    groups = []
    feature_names = None

    for t in filtered:
        features = extract_allowed_features(t)
        if feature_names is None:
            feature_names = list(features.keys())
        X.append([features[name] for name in feature_names])
        y.append(get_label(t))
        groups.append(t.bug_type)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    print(f"\nFeatures used ({len(feature_names)}):")
    for fn in feature_names:
        print(f"  - {fn}")

    print(f"\nFeatures EXCLUDED (forbidden):")
    for fn in FORBIDDEN_FEATURES[:10]:
        print(f"  - {fn}")
    print(f"  ... and {len(FORBIDDEN_FEATURES) - 10} more")

    unique_groups = sorted(list(set(groups)))
    print(f"\nUnique bug_types: {len(unique_groups)}")
    print(f"Label distribution: NEAR={sum(y)}, BELOW={len(y)-sum(y)}")

    n_near = sum(y)
    n_below = len(y) - n_near
    if n_near == 0 or n_below == 0:
        print("  [SKIP] No label diversity")
        return None

    n_splits = min(5, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    all_near_recalls = []
    all_below_filtereds = []
    all_accuracies = []
    all_roc_aucs = []
    fold_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups[test_idx]

        n_near_test = sum(y_test)
        n_below_test = len(y_test) - n_near_test

        print(f"\nFold {fold_idx + 1}:")
        print(f"  Test bug_types: {sorted(list(set(groups_test)))}")
        print(f"  Test: NEAR={n_near_test}, BELOW={n_below_test}")

        if n_near_test == 0 or n_below_test == 0:
            print(f"  [SKIP] No diversity in test set")
            continue

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        near_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        below_filtered = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = accuracy_score(y_test, y_pred)

        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = 0.5

        print(f"  Near Recall: {near_recall:.1%}")
        print(f"  Below Filtered: {below_filtered:.1%}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  ROC AUC: {roc_auc:.3f}")
        print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        all_near_recalls.append(near_recall)
        all_below_filtereds.append(below_filtered)
        all_accuracies.append(accuracy)
        all_roc_aucs.append(roc_auc)
        fold_details.append({
            "fold": fold_idx + 1,
            "test_bug_types": sorted(list(set(groups_test))),
            "near_recall": near_recall,
            "below_filtered": below_filtered,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "confusion_matrix": [int(tp), int(fp), int(fn), int(tn)],
        })

    if len(all_near_recalls) == 0:
        print("\n[FAIL] No valid folds with diversity")
        return None

    print(f"\n{'='*60}")
    print(f"SUMMARY (GroupKFold, {len(all_near_recalls)} valid folds)")
    print(f"{'='*60}")
    print(f"  Near Recall:   {np.mean(all_near_recalls):.1%} +/- {np.std(all_near_recalls):.1%}")
    print(f"  Below Filtered: {np.mean(all_below_filtereds):.1%} +/- {np.std(all_below_filtereds):.1%}")
    print(f"  Accuracy:      {np.mean(all_accuracies):.1%} +/- {np.std(all_accuracies):.1%}")
    print(f"  ROC AUC:       {np.mean(all_roc_aucs):.3f} +/- {np.std(all_roc_aucs):.3f}")

    success_near = np.mean(all_near_recalls) >= 0.90
    success_below = np.mean(all_below_filtereds) >= 0.50

    print(f"\n  Success Criteria:")
    print(f"    Near Recall >= 90%: {'[PASS]' if success_near else '[FAIL]'}")
    print(f"    Below Filtered >= 50%: {'[PASS]' if success_below else '[FAIL]'}")

    return {
        "mean_near_recall": float(np.mean(all_near_recalls)),
        "mean_below_filtered": float(np.mean(all_below_filtereds)),
        "mean_accuracy": float(np.mean(all_accuracies)),
        "mean_roc_auc": float(np.mean(all_roc_aucs)),
        "std_near_recall": float(np.std(all_near_recalls)),
        "std_below_filtered": float(np.std(all_below_filtereds)),
        "success_near": success_near,
        "success_below": success_below,
        "n_valid_folds": len(all_near_recalls),
        "fold_details": fold_details,
        "features_used": feature_names,
        "features_excluded": FORBIDDEN_FEATURES,
    }


def run_feature_importance_audit(traces: List[HonestTrace]):
    """特征重要性审计：检查哪些特征在分类中起作用"""
    print("\n" + "=" * 80)
    print("Feature Importance Audit")
    print("=" * 80)

    filtered = [t for t in traces
                if t.first_visible_pass and not t.first_hidden_pass
                and t.boundary_label in ("near", "below")]

    if len(filtered) < 20:
        print("  [SKIP] Not enough data")
        return

    X = []
    y = []
    feature_names = None

    for t in filtered:
        features = extract_allowed_features(t)
        if feature_names is None:
            feature_names = list(features.keys())
        X.append([features[name] for name in feature_names])
        y.append(get_label(t))

    X = np.array(X)
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\nFeature Importances:")
    for idx in sorted_idx:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

    # Single feature ablation
    print("\nSingle Feature Ablation:")
    for i, fn in enumerate(feature_names):
        X_single = X[:, i:i+1]
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X_single, y, cv=min(5, len(filtered) // 5), scoring="accuracy"
        )
        print(f"  {fn} alone: accuracy = {scores.mean():.3f} +/- {scores.std():.3f}")

    return {
        "feature_importances": {feature_names[i]: float(importances[i]) for i in range(len(feature_names))},
    }


def main():
    print("=" * 80)
    print("HONEST Boundary Experiment")
    print("Fixes: no patch_size, no label leakage, GroupKFold, real blind_success")
    print("=" * 80)

    # Step 1: Collect traces
    print("\n[Step 1] Collecting honest traces...")
    traces = run_honest_experiment(num_tasks=250, num_seeds=5)

    boundary_dist = Counter(t.boundary_label for t in traces)
    print(f"\nTotal traces: {len(traces)}")
    print(f"Boundary distribution: {dict(boundary_dist)}")

    # Step 2: Analyze patch_size diversity
    print("\n[Step 2] Analyzing patch_size diversity...")
    patch_size_ok = analyze_patch_size_diversity(traces)

    # Step 3: Analyze boundary per bug_type
    print("\n[Step 3] Analyzing boundary distribution per bug_type...")
    boundary_ok = analyze_boundary_per_bug_type(traces)

    # Step 4: GroupKFold validation
    print("\n[Step 4] Running GroupKFold validation...")
    results = run_group_kfold_validation(traces, filter_to_hidden_gap=True)

    # Step 5: Feature importance audit
    print("\n[Step 5] Running feature importance audit...")
    audit = run_feature_importance_audit(traces)

    # Step 6: Save results
    results_dir = PROJECT_ROOT / "results" / "honest_boundary_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_traces": len(traces),
        "boundary_distribution": dict(boundary_dist),
        "patch_size_is_fingerprint": not patch_size_ok,
        "boundary_is_pure_per_bug_type": not boundary_ok,
        "group_kfold_results": results,
        "feature_audit": audit,
        "allowed_features": ALLOWED_FEATURES,
        "forbidden_features": FORBIDDEN_FEATURES,
    }

    output_path = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Step 7: Print final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if results:
        if results["success_near"] and results["success_below"]:
            print("  [PASS] Honest experiment passes all criteria!")
            print(f"  Near Recall: {results['mean_near_recall']:.1%}")
            print(f"  Below Filtered: {results['mean_below_filtered']:.1%}")
        elif results["success_near"]:
            print("  [PARTIAL] Can detect NEAR but cannot reliably filter BELOW")
            print(f"  Near Recall: {results['mean_near_recall']:.1%}")
            print(f"  Below Filtered: {results['mean_below_filtered']:.1%}")
        else:
            print("  [FAIL] Cannot reliably detect NEAR with allowed features")
            print(f"  Near Recall: {results['mean_near_recall']:.1%}")
            print(f"  Below Filtered: {results['mean_below_filtered']:.1%}")
    else:
        print("  [FAIL] Not enough data for validation")

    if not patch_size_ok:
        print("\n  [WARNING] patch_size is still a bug_type fingerprint!")
        print("  This confirms it MUST be excluded from features.")

    if not boundary_ok:
        print("\n  [WARNING] Some bug_types only have one boundary label!")
        print("  This means bug_type is still a boundary fingerprint.")
        print("  GroupKFold is essential to avoid this artifact.")


if __name__ == "__main__":
    main()
