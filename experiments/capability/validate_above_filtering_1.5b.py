"""
Validate ABOVE-zone filtering with Qwen2.5-1.5B.
Compare with 0.5B results to see if NEAR zone emerges.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    BenchmarkVerifier,
    OracleSolver,
)
from core.llm_solver import LlamaCppSolver


def load_code20_tasks(path: str = "data/capability_boundary_bench/code-20.eval.jsonl") -> list[BenchmarkTask]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks.append(BenchmarkTask(
                task_id=d["task_id"],
                family="code",
                prompt=d["prompt"],
                expected_answer=d.get("fixed_code", ""),
                metadata={
                    "function_name": d.get("function_name", "solve"),
                    "bug_type": d.get("bug_type", "unknown"),
                    "difficulty": d.get("difficulty", ""),
                    "visible_tests": d.get("visible_tests", []),
                    "hidden_tests": d.get("hidden_tests", []),
                    "buggy_code": d.get("buggy_code", ""),
                }
            ))
    return tasks


def run_validation(tasks: list[BenchmarkTask], solver) -> list[Dict[str, Any]]:
    verifier = BenchmarkVerifier()
    results = []

    for task in tasks:
        t0 = time.time()

        single_shot = solver.solve(task)
        single_ver = verifier.verify(task, single_shot)

        if single_ver.passed:
            zone = "ABOVE"
            feedback_gain = 0
            feedback_success = True
        else:
            feedback = single_ver.feedback
            retry_attempt = solver.revise(task, single_shot, feedback)
            retry_ver = verifier.verify(task, retry_attempt)

            if retry_ver.passed:
                zone = "NEAR"
                feedback_gain = 1
                feedback_success = True
            else:
                zone = "BELOW"
                feedback_gain = 0
                feedback_success = False

        elapsed = time.time() - t0

        results.append({
            "task_id": task.task_id,
            "bug_type": task.metadata.get("bug_type", ""),
            "difficulty": task.metadata.get("difficulty", ""),
            "zone": zone,
            "single_shot_correct": single_ver.passed,
            "feedback_success": feedback_success,
            "feedback_gain": feedback_gain,
            "confidence": round(single_shot.confidence, 3),
            "latency_s": round(elapsed, 2),
        })

    return results


def print_summary(results: list[Dict[str, Any]]):
    total = len(results)
    above = [r for r in results if r["zone"] == "ABOVE"]
    near = [r for r in results if r["zone"] == "NEAR"]
    below = [r for r in results if r["zone"] == "BELOW"]

    print(f"\n{'='*60}")
    print("ABOVE-ZONE FILTERING VALIDATION (Real LLM: Qwen2.5-1.5B)")
    print(f"{'='*60}")
    print(f"\nTotal tasks: {total}")
    print(f"  ABOVE (single-shot success):     {len(above)} ({len(above)/total*100:.1f}%)")
    print(f"  NEAR (feedback rescued):         {len(near)} ({len(near)/total*100:.1f}%)")
    print(f"  BELOW (feedback failed):         {len(below)} ({len(below)/total*100:.1f}%)")

    print(f"\n--- ABOVE-zone analysis ---")
    print(f"  Feedback would be WASTED on {len(above)}/{total} tasks")
    print(f"  ABOVE-filtering saves {len(above)} feedback calls")

    print(f"\n--- NEAR-zone analysis ---")
    if near:
        print(f"  Feedback rescued {len(near)}/{len(near)+len(below)} failed tasks")
        print(f"  NEAR-zone feedback gain: {len(near)/(len(near)+len(below))*100:.1f}%")
    else:
        print(f"  No NEAR-zone tasks found")

    print(f"\n--- BELOW-zone analysis ---")
    if below:
        print(f"  Feedback failed on {len(below)} tasks")

    print(f"\n--- Inverted-U pattern check ---")
    if len(near) > 0:
        print(f"  Pattern: ABOVE={len(above)}, NEAR={len(near)}, BELOW={len(below)}")
        print("  Inverted-U: [PASS] PRESENT" if len(near) > max(len(above), len(below)) else "  Inverted-U: [WEAK] NEAR not dominant")
    else:
        print(f"  Pattern: ABOVE={len(above)}, NEAR={len(near)}, BELOW={len(below)}")
        print("  Inverted-U: [ABSENT] model is either correct or completely wrong")

    print(f"\n--- Confidence calibration ---")
    above_conf = [r["confidence"] for r in above]
    near_conf = [r["confidence"] for r in near]
    below_conf = [r["confidence"] for r in below]
    if above_conf:
        print(f"  ABOVE avg confidence: {sum(above_conf)/len(above_conf):.3f}")
    if near_conf:
        print(f"  NEAR avg confidence:  {sum(near_conf)/len(near_conf):.3f}")
    if below_conf:
        print(f"  BELOW avg confidence: {sum(below_conf)/len(below_conf):.3f}")


def main():
    print("=" * 60)
    print("ABOVE-Zone Filtering Validation")
    print("Model: Qwen2.5-1.5B-Instruct (Q4_K_M)")
    print("=" * 60)

    print("\n[1/3] Loading code-20 tasks...")
    tasks = load_code20_tasks()
    print(f"  Loaded {len(tasks)} tasks")

    print("\n[2/3] Connecting to LLM solver...")
    solver = LlamaCppSolver(base_url="http://127.0.0.1:8082")
    print("  Connected to llama.cpp server on port 8082")

    print("\n[3/3] Running validation (single-shot + feedback retry)...")
    print("  This will take ~10-20 minutes...")
    results = run_validation(tasks, solver)

    print_summary(results)

    out_path = Path("results/capability_benchmark/above_filtering_validation_1.5b.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen2.5-1.5B-Instruct-Q4_K_M",
            "server": "llama.cpp port 8082 (Vulkan GPU)",
            "task_count": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
