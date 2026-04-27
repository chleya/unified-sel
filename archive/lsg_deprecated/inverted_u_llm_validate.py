"""
Inverted-U LLM Validation

Critical experiment: Does the inverted-U pattern (boundary-local amplification)
hold for a real LLM (Qwen2.5-0.5B-Instruct)?

If YES: Paper is publishable, Capability Router has real validation
If NO: Negative result, need to reframe the finding

Design:
  - 20 code tasks, 3 boundary zones (ABOVE/NEAR/BELOW)
  - Single-shot vs feedback-retry for each zone
  - Qwen2.5-0.5B-Instruct as solver
  - 3 seeds for statistical robustness

Usage:
    python experiments/capability/inverted_u_llm_validate.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HF_CACHE = str(Path(__file__).resolve().parents[2] / "topomem" / "data" / "models" / "hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

from core.capability_benchmark import (
    BenchmarkTask,
    build_task_suite,
    _task_verifier_tests,
    _task_search_tests,
)

SEEDS = [7, 42, 123]
MAX_RETRIES = 3


def classify_zone(task, solver_code):
    visible_tests = _task_search_tests(task)
    hidden_tests = _task_verifier_tests(task)

    vis_pass = all(
        _run_test(solver_code, task.metadata.get("function_name", "solve"), t)
        for t in visible_tests
    )
    hid_pass = all(
        _run_test(solver_code, task.metadata.get("function_name", "solve"), t)
        for t in hidden_tests
    )

    if vis_pass and hid_pass:
        return "ABOVE"
    elif vis_pass and not hid_pass:
        return "NEAR"
    else:
        return "BELOW"


def _run_test(code, fn_name, test):
    try:
        ns = {}
        exec(code, ns)
        fn = ns.get(fn_name)
        if fn is None:
            return False
        result = fn(*test["args"])
        return result == test["expected"]
    except Exception:
        return False


def verify_code(code, task):
    all_tests = _task_search_tests(task) + _task_verifier_tests(task)
    fn_name = task.metadata.get("function_name", "solve")
    return all(_run_test(code, fn_name, t) for t in all_tests)


def get_error_feedback(code, task):
    visible_tests = _task_search_tests(task)
    fn_name = task.metadata.get("function_name", "solve")
    failures = []
    for t in visible_tests:
        if not _run_test(code, fn_name, t):
            failures.append(f"solve({', '.join(str(a) for a in t['args'])}) != {t['expected']}")
    if failures:
        return f"Failed tests: {'; '.join(failures[:3])}"
    hidden_tests = _task_verifier_tests(task)
    for t in hidden_tests:
        if not _run_test(code, fn_name, t):
            return "Visible tests pass but hidden tests fail. Check edge cases."
    return "Unknown error"


def main():
    print("=" * 60)
    print("Inverted-U LLM Validation (Qwen2.5-0.5B-Instruct)")
    print("=" * 60)

    print("\n[1/4] Loading Qwen2.5-0.5B-Instruct...")
    from double_helix.llm_validate import QwenSolver
    solver = QwenSolver()
    print("  Model loaded.")

    all_results = {}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n[2/4] Seed {seed} ({seed_idx+1}/{len(SEEDS)})...")
        tasks = build_task_suite("code", 20, seed=seed)

        seed_results = defaultdict(lambda: {"single_shot": [], "feedback_retry": []})

        for i, task in enumerate(tasks):
            task_id = task.task_id
            fn_name = task.metadata.get("function_name", "solve")
            print(f"  Task {i+1}/20: {task_id} ({task.metadata.get('bug_type', '?')})...", end=" ", flush=True)

            code_single = solver.solve(task, temperature=0.3)
            single_success = verify_code(code_single, task)
            zone = classify_zone(task, code_single)

            seed_results[zone]["single_shot"].append(single_success)

            retry_success = single_success
            if not single_success:
                current_code = code_single
                for attempt in range(MAX_RETRIES):
                    feedback = get_error_feedback(current_code, task)
                    temp = 0.3 + 0.2 * attempt
                    current_code = solver.solve(task, error_feedback=feedback, temperature=temp)
                    if verify_code(current_code, task):
                        retry_success = True
                        break

            seed_results[zone]["feedback_retry"].append(retry_success)
            print(f"zone={zone}, single={single_success}, retry={retry_success}")

        all_results[seed] = dict(seed_results)

    print("\n[3/4] Analysis...")

    zone_summary = defaultdict(lambda: {"single_shot": [], "feedback_retry": []})
    for seed, seed_data in all_results.items():
        for zone, data in seed_data.items():
            zone_summary[zone]["single_shot"].extend(data["single_shot"])
            zone_summary[zone]["feedback_retry"].extend(data["feedback_retry"])

    print(f"\n  {'Zone':<10} {'N':>4} {'Single':>8} {'Retry':>8} {'Gain':>8}")
    print(f"  {'-'*42}")

    inverted_u_holds = True
    for zone in ["ABOVE", "NEAR", "BELOW"]:
        data = zone_summary.get(zone, {"single_shot": [], "feedback_retry": []})
        n = len(data["single_shot"])
        if n == 0:
            print(f"  {zone:<10} {'0':>4} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        single_rate = np.mean(data["single_shot"])
        retry_rate = np.mean(data["feedback_retry"])
        gain = retry_rate - single_rate
        print(f"  {zone:<10} {n:>4} {single_rate:>8.3f} {retry_rate:>8.3f} {gain:>+8.3f}")

        if zone == "ABOVE" and gain > 0.05:
            inverted_u_holds = False
        if zone == "NEAR" and gain < 0.05:
            inverted_u_holds = False
        if zone == "BELOW" and gain > 0.1:
            inverted_u_holds = False

    print(f"\n  Inverted-U pattern holds: {'YES' if inverted_u_holds else 'NO'}")

    from scipy import stats as sp_stats
    near_single = zone_summary.get("NEAR", {}).get("single_shot", [])
    near_retry = zone_summary.get("NEAR", {}).get("feedback_retry", [])
    if len(near_single) >= 5:
        t_stat, p_value = sp_stats.ttest_rel(near_retry, near_single)
        print(f"  NEAR zone paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    else:
        p_value = None
        print(f"  NEAR zone: insufficient data for t-test (n={len(near_single)})")

    verdict = "INCONCLUSIVE"
    if inverted_u_holds and p_value is not None and p_value < 0.05:
        verdict = "CONFIRMED - inverted-U holds with real LLM"
    elif inverted_u_holds:
        verdict = "PROMISING - pattern holds but not statistically significant"
    else:
        verdict = "REJECTED - inverted-U does not hold with real LLM"

    print(f"\n  Verdict: {verdict}")

    output = {
        "schema_version": "capbench.result.v1",
        "metadata": {
            "data_source": "verified_execution",
            "cost_model": "abstract_units_v1",
            "oracle_assumption": False,
            "verifier_policy": "inverted_u_llm_validate",
            "benchmark_suite": "code-20-llm",
            "task_count": 20,
            "seeds": SEEDS,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "experiment": "inverted_u_llm_validate",
        "hypothesis": "Inverted-U pattern holds for Qwen2.5-0.5B-Instruct",
        "model": "Qwen2.5-0.5B-Instruct",
        "per_seed_results": {str(s): {z: {"single_shot": d["single_shot"], "feedback_retry": d["feedback_retry"]} for z, d in r.items()} for s, r in all_results.items()},
        "zone_summary": {z: {"n": len(d["single_shot"]), "single_rate": float(np.mean(d["single_shot"])), "retry_rate": float(np.mean(d["feedback_retry"])), "gain": float(np.mean(d["feedback_retry"]) - np.mean(d["single_shot"]))} for z, d in zone_summary.items()},
        "analysis": {
            "inverted_u_holds": inverted_u_holds,
            "near_p_value": p_value,
            "verdict": verdict,
        },
    }

    out_path = Path("results/inverted_u_llm")
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"inverted_u_llm_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[4/4] Results saved to: {fname}")


if __name__ == "__main__":
    main()
