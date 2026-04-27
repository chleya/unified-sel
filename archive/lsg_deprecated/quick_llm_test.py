"""
Quick LLM benchmark test - 3 tasks only for fast validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import (
    BenchmarkTask,
    BenchmarkVerifier,
    SolverAttempt,
)
from core.llm_solver import LlamaCppSolver


def quick_test():
    """Quick test with 3 code tasks using Qwen2.5-0.5B via llama.cpp."""
    
    print("=" * 60)
    print("Quick LLM Benchmark Test (3 tasks)")
    print("Model: Qwen2.5-0.5B-Instruct via llama.cpp server")
    print("=" * 60)
    
    # Step 1: Connect to llama.cpp server
    print("\n[1/4] Connecting to llama.cpp server...")
    solver = LlamaCppSolver(base_url="http://127.0.0.1:8081")
    
    # Step 2: Create test tasks
    print("\n[2/4] Creating test tasks...")
    tasks = [
        BenchmarkTask(
            task_id="test_add",
            family="code",
            prompt="def add(a, b):\n    return a - b",
            expected_answer="def add(a, b):\n    return a + b",
            metadata={"function_name": "add", "bug_type": "wrong_sign"}
        ),
        BenchmarkTask(
            task_id="test_multiply",
            family="code",
            prompt="def multiply(a, b):\n    return a + b",
            expected_answer="def multiply(a, b):\n    return a * b",
            metadata={"function_name": "multiply", "bug_type": "wrong_sign"}
        ),
        BenchmarkTask(
            task_id="test_is_even",
            family="code",
            prompt="def is_even(n):\n    return n % 2 == 1",
            expected_answer="def is_even(n):\n    return n % 2 == 0",
            metadata={"function_name": "is_even", "bug_type": "wrong_sign"}
        ),
    ]
    
    # Step 3: Run Qwen solver on each task
    print("\n[3/4] Running Qwen2.5-1.5B on 3 tasks...")
    results = []
    verifier = BenchmarkVerifier()
    
    for task in tasks:
        print(f"\n  Task: {task.task_id}")
        print(f"  Prompt: {task.prompt[:50]}...")
        
        attempt = solver.solve(task)
        
        print(f"  Answer: {attempt.answer[:80]}...")
        print(f"  Confidence: {attempt.confidence:.2f}")
        print(f"  Notes: {attempt.notes}")
        
        # Verify
        vr = verifier.verify(task, attempt)
        print(f"  Correct: {vr.passed} ({vr.feedback})")
        
        results.append({
            "task_id": task.task_id,
            "confidence": attempt.confidence,
            "correct": vr.passed,
            "feedback": vr.feedback,
            "answer_preview": attempt.answer[:100],
        })
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("[4/4] Summary")
    print("=" * 60)
    
    correct_count = sum(1 for r in results if r["correct"])
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    
    print(f"\nTasks: {len(results)}")
    print(f"Correct: {correct_count}/{len(results)} ({correct_count/len(results)*100:.0f}%)")
    print(f"Avg Confidence: {avg_confidence:.2f}")
    
    for r in results:
        status = "OK" if r["correct"] else "FAIL"
        print(f"  [{status}] {r['task_id']}: conf={r['confidence']:.2f}, fb={r['feedback']}")
    
    return results


if __name__ == "__main__":
    quick_test()
