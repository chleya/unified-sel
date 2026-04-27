"""
Comprehensive Boundary Experiment - 三方向修复

方向1: 更丰富的运行时信号
  - visible_test_pass_ratio: 通过的可见测试比例
  - error_mentions_value: 错误消息是否提到具体值
  - error_has_diff_info: 错误消息是否包含 diff 信息
  - n_visible_tests: 可见测试数量
  - n_hidden_tests: 隐藏测试数量

方向2: 随机化 solver
  - RandomizedSearchSolver: 在候选列表中随机选择，而非总是选第一个
  - 引入温度参数控制随机性

方向3: 同一 bug_type 同时有 NEAR 和 BELOW
  - 为每个 bug_type 创建多种 hidden test 配置
  - 有些 hidden test 容易（NEAR），有些难（BELOW）
"""

import json
import sys
import random
import ast
import numpy as np
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence
from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    SolverAttempt,
    SearchLocalSolver,
    _run_code_task,
    _task_search_tests,
    _task_verifier_tests,
    generate_code_tasks,
    _candidate_code_variants,
)


class RandomizedSearchSolver(SearchLocalSolver):
    """随机化搜索 solver：在通过可见测试的候选中随机选择"""

    def __init__(self, temperature: float = 0.5, seed: int = 42):
        self.temperature = temperature
        self.rng = random.Random(seed)

    def _search_code_fix(self, task: BenchmarkTask, aggressive: bool = False) -> SolverAttempt:
        candidates = _candidate_code_variants(task, aggressive=aggressive)
        visible_passing = []

        for candidate_code, note in candidates:
            try:
                passed, feedback = _run_code_task(
                    function_name=task.metadata["function_name"],
                    code=candidate_code,
                    tests=_task_search_tests(task),
                )
            except Exception:
                passed = False
                feedback = "error"

            if passed:
                visible_passing.append((candidate_code, note, feedback))

        if not visible_passing:
            return SolverAttempt(
                answer=task.metadata["buggy_code"],
                confidence=0.18,
                notes="search_failed",
                metadata={},
            )

        if self.temperature <= 0:
            idx = 0
        elif self.temperature >= 1:
            idx = self.rng.randint(0, len(visible_passing) - 1)
        else:
            weights = [np.exp(-i * self.temperature) for i in range(len(visible_passing))]
            total = sum(weights)
            weights = [w / total for w in weights]
            idx = self.rng.choices(range(len(visible_passing)), weights=weights, k=1)[0]

        chosen_code, chosen_note, chosen_feedback = visible_passing[idx]
        return SolverAttempt(
            answer=chosen_code,
            confidence=0.94 if idx == 0 else 0.7,
            notes=f"randomized_search:{chosen_note}",
            metadata={
                "mode": "code",
                "solver_kind": "randomized_search",
                "temperature": self.temperature,
                "selected_candidate_note": chosen_note,
                "n_visible_passing": len(visible_passing),
                "selected_rank": idx + 1,
            },
        )


def create_diverse_task_variants(base_tasks: List[BenchmarkTask], rng: random.Random) -> List[BenchmarkTask]:
    """为每个任务创建多种 hidden test 配置，使同一 bug_type 可以有不同 boundary"""
    diverse_tasks = []

    for task in base_tasks:
        bt = task.metadata.get("bug_type", "unknown")
        original_hidden = list(task.metadata.get("hidden_tests", []))
        original_visible = list(task.metadata.get("visible_tests", []))

        if not original_hidden:
            diverse_tasks.append(task)
            continue

        # Variant 1: Original (keep as-is)
        diverse_tasks.append(task)

        # Variant 2: Easier hidden tests (subset) -> more likely NEAR
        if len(original_hidden) >= 2:
            n_keep = max(1, len(original_hidden) - 1)
            easier_hidden = original_hidden[:n_keep]
            easier_task = BenchmarkTask(
                task_id=f"{task.task_id}_easy",
                family=task.family,
                prompt=task.prompt,
                expected_answer=task.expected_answer,
                metadata={
                    **task.metadata,
                    "task_id": f"{task.metadata.get('task_id', '')}_easy",
                    "hidden_tests": easier_hidden,
                    "variant": "easier_hidden",
                },
            )
            diverse_tasks.append(easier_task)

        # Variant 3: Harder hidden tests (add edge cases) -> more likely BELOW
        harder_hidden = list(original_hidden)
        fn_name = task.metadata.get("function_name", "solve")

        if bt.startswith("count_"):
            harder_hidden.append({"args": [[]], "expected": 0})
        elif bt.startswith("normalize_"):
            harder_hidden.append({"args": [""], "expected": ""})
        elif bt == "reverse_words":
            harder_hidden.append({"args": [""], "expected": ""})
        elif bt == "running_max":
            harder_hidden.append({"args": [[-1]], "expected": -1})
        elif bt == "dedupe_sorted":
            harder_hidden.append({"args": [[]], "expected": []})
        elif bt == "last_element":
            harder_hidden.append({"args": [[1]], "expected": 1})
        elif bt == "wrong_comparison":
            harder_hidden.append({"args": [0], "expected": "zero"})

        if len(harder_hidden) > len(original_hidden):
            harder_task = BenchmarkTask(
                task_id=f"{task.task_id}_hard",
                family=task.family,
                prompt=task.prompt,
                expected_answer=task.expected_answer,
                metadata={
                    **task.metadata,
                    "task_id": f"{task.metadata.get('task_id', '')}_hard",
                    "hidden_tests": harder_hidden,
                    "variant": "harder_hidden",
                },
            )
            diverse_tasks.append(harder_task)

    return diverse_tasks


@dataclass
class RichTrace:
    task_id: str = ""
    bug_type: str = ""
    boundary_label: str = ""
    variant: str = ""

    single_success: bool = False
    blind_success: bool = False
    feedback_success: bool = False

    first_attempt_parse_ok: bool = False
    first_visible_pass: bool = False
    first_hidden_pass: bool = False
    first_error_type: str = ""
    first_error_message_len: int = 0
    first_patch_size: int = 0
    has_expected_actual: bool = False
    first_changed_from_buggy: bool = False

    n_visible_tests: int = 0
    n_hidden_tests: int = 0
    n_total_tests: int = 0
    visible_test_pass_ratio: float = 0.0
    error_mentions_value: bool = False
    error_has_diff_info: bool = False

    solver_selected_rank: int = 0
    solver_n_visible_passing: int = 0

    blind_changed_code: bool = False
    blind_success_real: bool = False

    feedback_changed_code: bool = False
    feedback_success_real: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_error_type(error_msg: str) -> str:
    if not error_msg or error_msg == "pass":
        return "pass"
    el = error_msg.lower()
    for et in ["nameerror", "assertionerror", "typeerror", "syntaxerror",
               "attributeerror", "indexerror", "keyerror", "valueerror"]:
        if et in el:
            return et.replace("error", "Error")
    if "empty" in el or "no code" in el:
        return "empty"
    return "other"


def check_error_mentions_value(error_msg: str) -> bool:
    if not error_msg:
        return False
    import re
    return bool(re.search(r'\d+', error_msg))


def check_error_has_diff_info(error_msg: str) -> bool:
    if not error_msg:
        return False
    el = error_msg.lower()
    return ("expected" in el and ("got" in el or "actual" in el))


def collect_rich_trace(
    task: BenchmarkTask,
    solver: RandomizedSearchSolver,
) -> RichTrace:
    trace = RichTrace()
    trace.task_id = task.metadata.get("task_id", "unknown")
    trace.bug_type = task.metadata.get("bug_type", "unknown")
    trace.variant = task.metadata.get("variant", "standard")

    fn_name = task.metadata.get("function_name", "solve")
    visible_tests = list(task.metadata.get("visible_tests", []))
    hidden_tests = list(task.metadata.get("hidden_tests", []))
    all_tests = visible_tests + hidden_tests

    trace.n_visible_tests = len(visible_tests)
    trace.n_hidden_tests = len(hidden_tests)
    trace.n_total_tests = len(all_tests)

    # First attempt
    attempt1 = solver.solve(task)
    code1 = attempt1.answer if attempt1 else ""
    trace.first_attempt_parse_ok = bool(code1 and code1.strip())
    trace.first_patch_size = len(code1)
    trace.first_changed_from_buggy = (code1 != task.metadata.get("buggy_code", ""))

    if attempt1 and attempt1.metadata:
        trace.solver_selected_rank = attempt1.metadata.get("selected_rank", 0)
        trace.solver_n_visible_passing = attempt1.metadata.get("n_visible_passing", 0)

    if not trace.first_attempt_parse_ok:
        trace.first_error_type = "empty"
        trace.first_visible_pass = False
        trace.first_hidden_pass = False
        trace.visible_test_pass_ratio = 0.0
    else:
        vis_pass, vis_msg = _run_code_task(fn_name, code1, visible_tests)
        trace.first_visible_pass = vis_pass

        if vis_pass:
            n_vis_passed = len(visible_tests)
            trace.visible_test_pass_ratio = 1.0
        else:
            n_vis_passed = 0
            trace.visible_test_pass_ratio = 0.0
            trace.first_error_type = parse_error_type(vis_msg)
            trace.first_error_message_len = len(vis_msg) if vis_msg else 0
            trace.has_expected_actual = ("expected" in (vis_msg or "").lower() and "got" in (vis_msg or "").lower())
            trace.error_mentions_value = check_error_mentions_value(vis_msg)
            trace.error_has_diff_info = check_error_has_diff_info(vis_msg)
            trace.first_hidden_pass = False

        if vis_pass:
            all_pass, all_msg = _run_code_task(fn_name, code1, all_tests)
            trace.first_hidden_pass = all_pass
            if all_pass:
                trace.first_error_type = "pass"
                trace.visible_test_pass_ratio = 1.0
            else:
                trace.first_error_type = parse_error_type(all_msg)
                trace.first_error_message_len = len(all_msg) if all_msg else 0
                trace.has_expected_actual = ("expected" in (all_msg or "").lower() and "got" in (all_msg or "").lower())
                trace.error_mentions_value = check_error_mentions_value(all_msg)
                trace.error_has_diff_info = check_error_has_diff_info(all_msg)

                # Compute partial pass ratio
                n_passed = 0
                for test in all_tests:
                    try:
                        args = test["args"]
                        expected = test["expected"]
                        ns = {}
                        exec(code1, {"__builtins__": {"range": range, "len": len, "sum": sum, "abs": abs, "int": int, "str": str, "bool": bool, "list": list, "min": min, "max": max, "sorted": sorted, "enumerate": enumerate, "reversed": reversed, "isinstance": isinstance, "type": type, "True": True, "False": False, "None": None}}, ns)
                        fn = ns.get(fn_name)
                        if fn and fn(*args) == expected:
                            n_passed += 1
                    except Exception:
                        pass
                trace.visible_test_pass_ratio = n_passed / max(len(all_tests), 1)

    trace.single_success = trace.first_hidden_pass

    # Blind retry
    if not trace.single_success:
        attempt2 = solver.solve(task)
        code2 = attempt2.answer if attempt2 else ""
        trace.blind_changed_code = (code2 != code1) if code1 and code2 else False
        if bool(code2 and code2.strip()):
            all_pass2, _ = _run_code_task(fn_name, code2, all_tests)
            trace.blind_success_real = all_pass2
        else:
            trace.blind_success_real = False
    else:
        trace.blind_success_real = True

    # Feedback retry
    if not trace.blind_success_real:
        err_msg = ""
        if not trace.first_visible_pass:
            _, vis_msg = _run_code_task(fn_name, code1, visible_tests)
            err_msg = vis_msg or ""
        elif not trace.first_hidden_pass:
            _, all_msg = _run_code_task(fn_name, code1, all_tests)
            err_msg = all_msg or ""

        if hasattr(solver, "revise") and err_msg:
            attempt3 = solver.revise(task, attempt1, err_msg)
        else:
            attempt3 = solver.solve(task)

        code3 = attempt3.answer if attempt3 else ""
        trace.feedback_changed_code = (code3 != code1) if code1 and code3 else False
        if bool(code3 and code3.strip()):
            all_pass3, _ = _run_code_task(fn_name, code3, all_tests)
            trace.feedback_success_real = all_pass3
        else:
            trace.feedback_success_real = False
    else:
        trace.feedback_success_real = True

    # Boundary label
    if trace.single_success:
        trace.boundary_label = "above"
    elif trace.feedback_success_real:
        trace.boundary_label = "near"
    else:
        trace.boundary_label = "below"

    return trace


def extract_rich_features(trace: RichTrace) -> Dict[str, float]:
    """提取丰富的特征（排除 bug_type 指纹）"""
    features = {}

    features["first_attempt_parse_ok"] = float(trace.first_attempt_parse_ok)
    features["first_visible_pass"] = float(trace.first_visible_pass)
    features["has_expected_actual"] = float(trace.has_expected_actual)
    features["error_mentions_value"] = float(trace.error_mentions_value)
    features["error_has_diff_info"] = float(trace.error_has_diff_info)

    et = trace.first_error_type
    for t in ["other", "AssertionError", "NameError", "TypeError", "pass"]:
        features[f"error_type_{t}"] = float(et == t)

    features["n_visible_tests"] = float(trace.n_visible_tests)
    features["n_hidden_tests"] = float(trace.n_hidden_tests)
    features["n_total_tests"] = float(trace.n_total_tests)
    features["visible_test_pass_ratio"] = trace.visible_test_pass_ratio

    features["solver_selected_rank"] = float(trace.solver_selected_rank)
    features["solver_n_visible_passing"] = float(trace.solver_n_visible_passing)

    features["first_error_message_len_norm"] = trace.first_error_message_len / max(trace.n_total_tests * 30, 1)
    features["first_patch_size_norm"] = trace.first_patch_size / 300.0

    return features


def main():
    print("=" * 80)
    print("COMPREHENSIVE Boundary Experiment")
    print("Three directions: richer signals + randomized solver + diverse tasks")
    print("=" * 80)

    # Step 1: Generate diverse tasks
    print("\n[Step 1] Generating diverse tasks...")
    rng = random.Random(42)
    base_tasks = generate_code_tasks(100, seed=42, variant="standard")
    diverse_tasks = create_diverse_task_variants(base_tasks, rng)
    print(f"  Base tasks: {len(base_tasks)}")
    print(f"  Diverse tasks: {len(diverse_tasks)}")

    # Step 2: Collect traces with randomized solver
    print("\n[Step 2] Collecting traces with randomized solver...")
    all_traces = []
    temperatures = [0.0, 0.3, 0.7, 1.0]
    seeds = [42, 123, 456]

    for temp in temperatures:
        for seed in seeds:
            solver = RandomizedSearchSolver(temperature=temp, seed=seed)
            for task in diverse_tasks:
                trace = collect_rich_trace(task, solver)
                all_traces.append(trace)

    print(f"  Total traces: {len(all_traces)}")
    boundary_dist = Counter(t.boundary_label for t in all_traces)
    print(f"  Boundary distribution: {dict(boundary_dist)}")

    # Step 3: Analyze feature variation
    print("\n[Step 3] Analyzing feature variation within bug_type...")
    feature_names = list(extract_rich_features(all_traces[0]).keys())

    for feat in feature_names:
        bug_type_vals = defaultdict(list)
        for t in all_traces:
            f = extract_rich_features(t)
            bug_type_vals[t.bug_type].append(f[feat])

        varies = sum(1 for vals in bug_type_vals.values() if len(set(vals)) > 1)
        constant = sum(1 for vals in bug_type_vals.values() if len(set(vals)) == 1)
        status = "VARIES" if varies > 0 else "CONSTANT"
        print(f"  {feat}: {varies} vary, {constant} constant -- {status}")

    # Step 4: Analyze boundary per bug_type
    print("\n[Step 4] Analyzing boundary distribution per bug_type...")
    bug_type_boundaries = defaultdict(lambda: defaultdict(int))
    for t in all_traces:
        bug_type_boundaries[t.bug_type][t.boundary_label] += 1

    mixed = 0
    pure = 0
    for bt, boundaries in sorted(bug_type_boundaries.items()):
        has_near = boundaries.get("near", 0) > 0
        has_below = boundaries.get("below", 0) > 0
        if has_near and has_below:
            mixed += 1
            print(f"  {bt}: near={boundaries['near']}, below={boundaries['below']} -- MIXED")
        else:
            pure += 1

    print(f"  Summary: {mixed} MIXED, {pure} PURE")

    # Step 5: GroupKFold validation
    print("\n[Step 5] GroupKFold validation (NEAR vs BELOW)...")

    hidden_gap = [t for t in all_traces
                  if t.first_visible_pass and not t.first_hidden_pass
                  and t.boundary_label in ("near", "below")]
    print(f"  Hidden-gap traces: {len(hidden_gap)}")

    if len(hidden_gap) >= 20:
        X = []
        y = []
        groups = []
        fnames = None

        for t in hidden_gap:
            features = extract_rich_features(t)
            if fnames is None:
                fnames = list(features.keys())
            X.append([features[n] for n in fnames])
            y.append(1 if t.boundary_label == "near" else 0)
            groups.append(t.bug_type)

        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)

        n_near = sum(y)
        n_below = len(y) - n_near
        print(f"  NEAR={n_near}, BELOW={n_below}")

        unique_groups = sorted(list(set(groups)))
        n_splits = min(5, len(unique_groups))
        gkf = GroupKFold(n_splits=n_splits)

        all_near_recalls = []
        all_below_filtereds = []
        all_accuracies = []
        all_roc_aucs = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_test = groups[test_idx]

            n_near_test = sum(y_test)
            n_below_test = len(y_test) - n_near_test

            print(f"\n  Fold {fold_idx + 1}:")
            print(f"    Test bug_types: {sorted(list(set(groups_test)))}")
            print(f"    NEAR={n_near_test}, BELOW={n_below_test}")

            if n_near_test == 0 or n_below_test == 0:
                print(f"    [SKIP] No diversity")
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

            print(f"    Near Recall: {near_recall:.1%}")
            print(f"    Below Filtered: {below_filtered:.1%}")
            print(f"    Accuracy: {accuracy:.1%}")
            print(f"    ROC AUC: {roc_auc:.3f}")
            print(f"    Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

            all_near_recalls.append(near_recall)
            all_below_filtereds.append(below_filtered)
            all_accuracies.append(accuracy)
            all_roc_aucs.append(roc_auc)

        if all_near_recalls:
            print(f"\n  SUMMARY ({len(all_near_recalls)} valid folds):")
            print(f"    Near Recall: {np.mean(all_near_recalls):.1%} +/- {np.std(all_near_recalls):.1%}")
            print(f"    Below Filtered: {np.mean(all_below_filtereds):.1%} +/- {np.std(all_below_filtereds):.1%}")
            print(f"    Accuracy: {np.mean(all_accuracies):.1%} +/- {np.std(all_accuracies):.1%}")
            print(f"    ROC AUC: {np.mean(all_roc_aucs):.3f} +/- {np.std(all_roc_aucs):.3f}")

            success_near = np.mean(all_near_recalls) >= 0.90
            success_below = np.mean(all_below_filtereds) >= 0.50
            print(f"\n  Success Criteria:")
            print(f"    Near Recall >= 90%: {'[PASS]' if success_near else '[FAIL]'}")
            print(f"    Below Filtered >= 50%: {'[PASS]' if success_below else '[FAIL]'}")

            # Feature importance
            clf_full = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_full.fit(X, y)
            importances = clf_full.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            print(f"\n  Feature Importances:")
            for idx in sorted_idx[:10]:
                print(f"    {fnames[idx]}: {importances[idx]:.4f}")

            results = {
                "mean_near_recall": float(np.mean(all_near_recalls)),
                "mean_below_filtered": float(np.mean(all_below_filtereds)),
                "mean_accuracy": float(np.mean(all_accuracies)),
                "mean_roc_auc": float(np.mean(all_roc_aucs)),
                "success_near": success_near,
                "success_below": success_below,
                "feature_importances": {fnames[i]: float(importances[i]) for i in range(len(fnames))},
            }
        else:
            print("  [FAIL] No valid folds")
            results = None
    else:
        print("  [SKIP] Not enough data")
        results = None

    # Step 6: Save results
    results_dir = PROJECT_ROOT / "results" / "comprehensive_boundary_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_traces": len(all_traces),
        "boundary_distribution": dict(boundary_dist),
        "n_mixed_bug_types": mixed,
        "n_pure_bug_types": pure,
        "group_kfold_results": results,
        "temperatures_used": temperatures,
        "seeds_used": seeds,
    }

    output_path = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if results:
        if results["success_near"] and results["success_below"]:
            print("  [PASS] Comprehensive experiment passes!")
        elif results["success_near"]:
            print("  [PARTIAL] Can detect NEAR but cannot filter BELOW")
        else:
            print("  [FAIL] Cannot reliably detect NEAR")

        print(f"  Near Recall: {results['mean_near_recall']:.1%}")
        print(f"  Below Filtered: {results['mean_below_filtered']:.1%}")
        print(f"  ROC AUC: {results['mean_roc_auc']:.3f}")
    else:
        print("  [FAIL] Not enough data")


if __name__ == "__main__":
    main()
