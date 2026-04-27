"""
Optimized Boundary Experiment v2

关键优化：
1. 排除所有 bug_type 代理特征（solver_n_visible_passing, first_patch_size_norm 等）
2. 阈值优化：不使用默认 0.5，而是搜索最优阈值
3. 创建更多 MIXED bug_type：为每个 bug_type 创建 3 种难度的 hidden test
4. 只使用在 bug_type 内有变化的特征
"""

import json
import sys
import random
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask, SolverAttempt, SearchLocalSolver,
    _run_code_task, _task_search_tests, _task_verifier_tests,
    generate_code_tasks, _candidate_code_variants,
)


class RandomizedSearchSolver(SearchLocalSolver):
    def __init__(self, temperature: float = 0.5, seed: int = 42):
        self.temperature = temperature
        self.rng = random.Random(seed)

    def _search_code_fix(self, task, aggressive=False):
        candidates = _candidate_code_variants(task, aggressive=aggressive)
        visible_passing = []
        for candidate_code, note in candidates:
            try:
                passed, feedback = _run_code_task(
                    task.metadata["function_name"], candidate_code,
                    _task_search_tests(task))
            except Exception:
                passed, feedback = False, "error"
            if passed:
                visible_passing.append((candidate_code, note, feedback))

        if not visible_passing:
            return SolverAttempt(task.metadata["buggy_code"], 0.18, "search_failed", {})

        if self.temperature <= 0:
            idx = 0
        elif self.temperature >= 1:
            idx = self.rng.randint(0, len(visible_passing) - 1)
        else:
            weights = [np.exp(-i * self.temperature) for i in range(len(visible_passing))]
            total = sum(weights)
            weights = [w / total for w in weights]
            idx = self.rng.choices(range(len(visible_passing)), weights=weights, k=1)[0]

        chosen_code, chosen_note, _ = visible_passing[idx]
        return SolverAttempt(chosen_code, 0.94 if idx == 0 else 0.7,
                           f"randomized_search:{chosen_note}",
                           {"mode": "code", "solver_kind": "randomized_search",
                            "temperature": self.temperature,
                            "selected_rank": idx + 1,
                            "n_visible_passing": len(visible_passing)})


def create_aggressive_variants(base_tasks, rng):
    """为每个任务创建多种 hidden test 配置"""
    diverse_tasks = []
    for task in base_tasks:
        bt = task.metadata.get("bug_type", "unknown")
        original_hidden = list(task.metadata.get("hidden_tests", []))
        fn_name = task.metadata.get("function_name", "solve")

        if not original_hidden:
            diverse_tasks.append(task)
            continue

        # Variant 1: Original
        diverse_tasks.append(task)

        # Variant 2: Easier hidden (subset) -> more likely NEAR
        if len(original_hidden) >= 2:
            n_keep = max(1, len(original_hidden) - 1)
            easier_hidden = original_hidden[:n_keep]
            diverse_tasks.append(BenchmarkTask(
                task_id=f"{task.task_id}_easy", family=task.family,
                prompt=task.prompt, expected_answer=task.expected_answer,
                metadata={**task.metadata, "task_id": f"{task.metadata.get('task_id','')}_easy",
                         "hidden_tests": easier_hidden, "variant": "easier"}))

        # Variant 3: Harder hidden (add edge cases) -> more likely BELOW
        harder_hidden = list(original_hidden)
        edge_cases = {
            "count_": [{"args": [[]], "expected": 0}],
            "normalize_": [{"args": [""], "expected": ""}, {"args": ["a"], "expected": "a"}],
            "reverse_": [{"args": [""], "expected": ""}, {"args": ["a"], "expected": "a"}],
            "running_": [{"args": [[-1]], "expected": -1}, {"args": [[0]], "expected": 0}],
            "dedupe_": [{"args": [[]], "expected": []}, {"args": [[1]], "expected": [1]}],
            "last_": [{"args": [[1]], "expected": 1}, {"args": [[0]], "expected": 0}],
            "wrong_": [{"args": [0], "expected": "zero"}, {"args": [-1], "expected": "negative"}],
        }

        for prefix, cases in edge_cases.items():
            if bt.startswith(prefix):
                harder_hidden.extend(cases)
                break

        if len(harder_hidden) > len(original_hidden):
            diverse_tasks.append(BenchmarkTask(
                task_id=f"{task.task_id}_hard", family=task.family,
                prompt=task.prompt, expected_answer=task.expected_answer,
                metadata={**task.metadata, "task_id": f"{task.metadata.get('task_id','')}_hard",
                         "hidden_tests": harder_hidden, "variant": "harder"}))

    return diverse_tasks


def collect_trace(task, solver):
    fn_name = task.metadata["function_name"]
    visible_tests = list(task.metadata.get("visible_tests", []))
    hidden_tests = list(task.metadata.get("hidden_tests", []))
    all_tests = visible_tests + hidden_tests

    result = {
        "task_id": task.metadata.get("task_id", ""),
        "bug_type": task.metadata.get("bug_type", ""),
        "variant": task.metadata.get("variant", "standard"),
        "n_visible_tests": len(visible_tests),
        "n_hidden_tests": len(hidden_tests),
        "n_total_tests": len(all_tests),
    }

    attempt1 = solver.solve(task)
    code1 = attempt1.answer if attempt1 else ""
    result["first_attempt_parse_ok"] = bool(code1 and code1.strip())
    result["first_patch_size"] = len(code1)
    result["first_changed_from_buggy"] = code1 != task.metadata.get("buggy_code", "")

    if attempt1 and attempt1.metadata:
        result["solver_selected_rank"] = attempt1.metadata.get("selected_rank", 0)
        result["solver_n_visible_passing"] = attempt1.metadata.get("n_visible_passing", 0)

    if not result["first_attempt_parse_ok"]:
        result["first_visible_pass"] = False
        result["first_hidden_pass"] = False
        result["first_error_type"] = "empty"
        result["first_error_message_len"] = 0
        result["visible_test_pass_ratio"] = 0.0
        result["error_mentions_value"] = False
        result["error_has_diff_info"] = False
        result["has_expected_actual"] = False
    else:
        vis_pass, vis_msg = _run_code_task(fn_name, code1, visible_tests)
        result["first_visible_pass"] = vis_pass

        if vis_pass:
            result["visible_test_pass_ratio"] = 1.0
            all_pass, all_msg = _run_code_task(fn_name, code1, all_tests)
            result["first_hidden_pass"] = all_pass
            if all_pass:
                result["first_error_type"] = "pass"
                result["first_error_message_len"] = 0
                result["error_mentions_value"] = False
                result["error_has_diff_info"] = False
                result["has_expected_actual"] = False
            else:
                result["first_error_type"] = _parse_error(all_msg)
                result["first_error_message_len"] = len(all_msg) if all_msg else 0
                el = (all_msg or "").lower()
                result["has_expected_actual"] = "expected" in el and ("got" in el or "actual" in el)
                result["error_mentions_value"] = bool(__import__('re').search(r'\d+', all_msg or ""))
                result["error_has_diff_info"] = result["has_expected_actual"]

                n_passed = 0
                for test in all_tests:
                    try:
                        ns = {}
                        exec(code1, {"__builtins__": __builtins__}, ns)
                        fn = ns.get(fn_name)
                        if fn and fn(*test["args"]) == test["expected"]:
                            n_passed += 1
                    except Exception:
                        pass
                result["visible_test_pass_ratio"] = n_passed / max(len(all_tests), 1)
        else:
            result["first_hidden_pass"] = False
            result["first_error_type"] = _parse_error(vis_msg)
            result["first_error_message_len"] = len(vis_msg) if vis_msg else 0
            el = (vis_msg or "").lower()
            result["has_expected_actual"] = "expected" in el and ("got" in el or "actual" in el)
            result["error_mentions_value"] = bool(__import__('re').search(r'\d+', vis_msg or ""))
            result["error_has_diff_info"] = result["has_expected_actual"]
            result["visible_test_pass_ratio"] = 0.0

    result["single_success"] = result["first_hidden_pass"]

    # Blind retry
    if not result["single_success"]:
        attempt2 = solver.solve(task)
        code2 = attempt2.answer if attempt2 else ""
        result["blind_changed_code"] = code2 != code1 if code1 and code2 else False
        if bool(code2 and code2.strip()):
            bp, _ = _run_code_task(fn_name, code2, all_tests)
            result["blind_success"] = bp
        else:
            result["blind_success"] = False
    else:
        result["blind_success"] = True
        result["blind_changed_code"] = False

    # Feedback retry
    if not result["blind_success"]:
        err_msg = ""
        if not result["first_visible_pass"]:
            _, vm = _run_code_task(fn_name, code1, visible_tests)
            err_msg = vm or ""
        elif not result["first_hidden_pass"]:
            _, am = _run_code_task(fn_name, code1, all_tests)
            err_msg = am or ""

        if hasattr(solver, "revise") and err_msg:
            attempt3 = solver.revise(task, attempt1, err_msg)
        else:
            attempt3 = solver.solve(task)

        code3 = attempt3.answer if attempt3 else ""
        result["feedback_changed_code"] = code3 != code1 if code1 and code3 else False
        if bool(code3 and code3.strip()):
            fp, _ = _run_code_task(fn_name, code3, all_tests)
            result["feedback_success"] = fp
        else:
            result["feedback_success"] = False
    else:
        result["feedback_success"] = True
        result["feedback_changed_code"] = False

    if result["single_success"]:
        result["boundary_label"] = "above"
    elif result["feedback_success"]:
        result["boundary_label"] = "near"
    else:
        result["boundary_label"] = "below"

    return result


def _parse_error(msg):
    if not msg or msg == "pass":
        return "pass"
    el = msg.lower()
    for et in ["nameerror", "assertionerror", "typeerror", "syntaxerror",
               "attributeerror", "indexerror", "keyerror", "valueerror"]:
        if et in el:
            return et.replace("error", "Error")
    return "other"


def extract_clean_features(trace):
    """只提取真正的运行时信号，排除所有 artifact 和指纹"""
    features = {}

    features["first_visible_pass"] = float(trace["first_visible_pass"])
    features["has_expected_actual"] = float(trace["has_expected_actual"])
    features["error_mentions_value"] = float(trace["error_mentions_value"])
    features["error_has_diff_info"] = float(trace["error_has_diff_info"])

    et = trace["first_error_type"]
    for t in ["other", "AssertionError", "NameError", "TypeError", "pass"]:
        features[f"error_type_{t}"] = float(et == t)

    features["visible_test_pass_ratio"] = trace["visible_test_pass_ratio"]

    features["first_error_message_len_norm"] = trace["first_error_message_len"] / max(trace["n_total_tests"] * 30, 1)

    return features


def main():
    print("=" * 80)
    print("OPTIMIZED Boundary Experiment v2")
    print("=" * 80)

    # Step 1: Generate diverse tasks
    print("\n[Step 1] Generating diverse tasks...")
    rng = random.Random(42)
    base_tasks = generate_code_tasks(100, seed=42, variant="standard")
    diverse_tasks = create_aggressive_variants(base_tasks, rng)
    print(f"  Base: {len(base_tasks)}, Diverse: {len(diverse_tasks)}")

    # Step 2: Collect traces
    print("\n[Step 2] Collecting traces...")
    all_traces = []
    for temp in [0.0, 0.3, 0.7, 1.0]:
        for seed in [42, 123, 456]:
            solver = RandomizedSearchSolver(temperature=temp, seed=seed)
            for task in diverse_tasks:
                t = collect_trace(task, solver)
                all_traces.append(t)

    print(f"  Total: {len(all_traces)}")
    bd = Counter(t["boundary_label"] for t in all_traces)
    print(f"  Boundary: {dict(bd)}")

    # Step 3: Check MIXED bug_types
    print("\n[Step 3] Boundary per bug_type...")
    bt_bounds = defaultdict(lambda: defaultdict(int))
    for t in all_traces:
        bt_bounds[t["bug_type"]][t["boundary_label"]] += 1

    mixed = sum(1 for b in bt_bounds.values()
                if b.get("near", 0) > 0 and b.get("below", 0) > 0)
    print(f"  MIXED: {mixed}, PURE: {len(bt_bounds) - mixed}")

    for bt, b in sorted(bt_bounds.items()):
        if b.get("near", 0) > 0 and b.get("below", 0) > 0:
            print(f"    {bt}: near={b['near']}, below={b['below']}")

    # Step 4: GroupKFold with threshold optimization
    print("\n[Step 4] GroupKFold with threshold optimization...")

    hidden_gap = [t for t in all_traces
                  if t["first_visible_pass"] and not t["first_hidden_pass"]
                  and t["boundary_label"] in ("near", "below")]
    print(f"  Hidden-gap: {len(hidden_gap)}")

    if len(hidden_gap) < 20:
        print("  [SKIP] Not enough data")
        return

    X = []
    y = []
    groups = []
    fnames = None

    for t in hidden_gap:
        features = extract_clean_features(t)
        if fnames is None:
            fnames = list(features.keys())
        X.append([features[n] for n in fnames])
        y.append(1 if t["boundary_label"] == "near" else 0)
        groups.append(t["bug_type"])

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    n_near = sum(y)
    n_below = len(y) - n_near
    print(f"  NEAR={n_near}, BELOW={n_below}")
    print(f"  Features: {fnames}")

    unique_groups = sorted(list(set(groups)))
    n_splits = min(5, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    all_near_recalls = []
    all_below_filtereds = []
    all_accuracies = []
    all_roc_aucs = []
    all_optimal_thresholds = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups[test_idx]

        n_near_test = sum(y_test)
        n_below_test = len(y_test) - n_near_test

        print(f"\n  Fold {fold_idx + 1}:")
        print(f"    Test: NEAR={n_near_test}, BELOW={n_below_test}")

        if n_near_test == 0 or n_below_test == 0:
            print(f"    [SKIP] No diversity")
            continue

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_train_proba = clf.predict_proba(X_train)[:, 1]

        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = 0.5

        # Threshold optimization on TRAINING SET (not test set!)
        best_threshold = 0.5
        best_score = 0
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred_t = (y_train_proba >= thresh).astype(int)
            cm = confusion_matrix(y_train, y_pred_t)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                nr = tp / (tp + fn) if (tp + fn) > 0 else 0
                bf = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (nr + bf) / 2
                if score > best_score:
                    best_score = score
                    best_threshold = thresh

        # Apply train-chosen threshold to TEST SET
        y_pred_opt = (y_pred_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_opt)
        tn, fp, fn, tp = cm.ravel()

        near_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        below_filtered = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = accuracy_score(y_test, y_pred_opt)

        # Also report default threshold=0.5 for reference
        y_pred_default = (y_pred_proba >= 0.5).astype(int)
        cm_default = confusion_matrix(y_test, y_pred_default)
        if cm_default.shape == (2, 2):
            tn_d, fp_d, fn_d, tp_d = cm_default.ravel()
            near_recall_default = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
            below_filtered_default = tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0
        else:
            near_recall_default = 0
            below_filtered_default = 0

        print(f"    Train-chosen threshold: {best_threshold:.2f}")
        print(f"    [Train-thresh] Near Recall: {near_recall:.1%}, Below Filtered: {below_filtered:.1%}")
        print(f"    [Default=0.50] Near Recall: {near_recall_default:.1%}, Below Filtered: {below_filtered_default:.1%}")
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    ROC AUC: {roc_auc:.3f}")
        print(f"    Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        all_near_recalls.append(near_recall)
        all_below_filtereds.append(below_filtered)
        all_accuracies.append(accuracy)
        all_roc_aucs.append(roc_auc)
        all_optimal_thresholds.append(best_threshold)

    if all_near_recalls:
        print(f"\n  SUMMARY ({len(all_near_recalls)} valid folds):")
        print(f"    Near Recall: {np.mean(all_near_recalls):.1%} +/- {np.std(all_near_recalls):.1%}")
        print(f"    Below Filtered: {np.mean(all_below_filtereds):.1%} +/- {np.std(all_below_filtereds):.1%}")
        print(f"    Accuracy: {np.mean(all_accuracies):.1%}")
        print(f"    ROC AUC: {np.mean(all_roc_aucs):.3f}")
        print(f"    Optimal Thresholds: {all_optimal_thresholds}")

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
        for idx in sorted_idx:
            print(f"    {fnames[idx]}: {importances[idx]:.4f}")

        results = {
            "mean_near_recall": float(np.mean(all_near_recalls)),
            "mean_below_filtered": float(np.mean(all_below_filtereds)),
            "mean_roc_auc": float(np.mean(all_roc_aucs)),
            "optimal_thresholds": [float(t) for t in all_optimal_thresholds],
            "success_near": success_near,
            "success_below": success_below,
            "feature_importances": {fnames[i]: float(importances[i]) for i in range(len(fnames))},
        }
    else:
        print("  [FAIL] No valid folds")
        results = None

    # Save
    results_dir = PROJECT_ROOT / "results" / "optimized_boundary_v2"
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_traces": len(all_traces),
        "boundary_distribution": dict(bd),
        "n_mixed_bug_types": mixed,
        "results": results,
    }
    output_path = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    if results:
        if results["success_near"] and results["success_below"]:
            print("  [PASS] Both criteria met!")
        elif results["success_near"]:
            print("  [PARTIAL] NEAR detected, BELOW not filtered")
        else:
            print("  [FAIL]")
        print(f"  Near Recall: {results['mean_near_recall']:.1%}")
        print(f"  Below Filtered: {results['mean_below_filtered']:.1%}")
        print(f"  ROC AUC: {results['mean_roc_auc']:.3f}")


if __name__ == "__main__":
    main()
