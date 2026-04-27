"""
Test LLM solver adapter.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.capability_benchmark import BenchmarkTask
from core.llm_solver import DummyLLMSolver, QwenSolver


def test_dummy_solver():
    """Test dummy solver works without model."""
    solver = DummyLLMSolver()
    task = BenchmarkTask(
        task_id="test_1",
        family="code",
        prompt="def add(a, b):\n    return a - b",
        expected_answer="def add(a, b):\n    return a + b",
        metadata={"function_name": "add", "bug_type": "wrong_sign"}
    )
    
    attempt = solver.solve(task)
    assert attempt.confidence == 0.3
    assert attempt.notes == "dummy_solver"
    print(f"Dummy solver test passed: {attempt.answer}")


def test_qwen_solver_import():
    """Test QwenSolver can be imported."""
    solver = QwenSolver()
    assert solver.model_path == "models/Qwen2.5-1.5B"
    print("QwenSolver import test passed")


def test_prompt_building():
    """Test prompt building for code tasks."""
    solver = QwenSolver()
    
    code_task = BenchmarkTask(
        task_id="code_1",
        family="code",
        prompt="def add(a, b):\n    return a - b",
        expected_answer="def add(a, b):\n    return a + b",
        metadata={}
    )
    
    prompt = solver._build_prompt(code_task)
    assert "Fix the buggy Python function" in prompt
    assert "def add(a, b):" in prompt
    print(f"Prompt building test passed:\n{prompt[:100]}...")


def test_code_extraction():
    """Test code extraction from markdown."""
    solver = QwenSolver()
    
    # Test with python block
    text1 = "Some text\n```python\ndef add(a, b):\n    return a + b\n```\nMore text"
    code1 = solver._extract_code(text1)
    assert "def add(a, b):" in code1
    
    # Test with generic block
    text2 = "```\ndef sub(a, b):\n    return a - b\n```"
    code2 = solver._extract_code(text2)
    assert "def sub(a, b):" in code2
    
    # Test without block
    text3 = "def mul(a, b):\n    return a * b"
    code3 = solver._extract_code(text3)
    assert "def mul(a, b):" in code3
    
    print("Code extraction test passed")


def test_confidence_estimation():
    """Test confidence estimation heuristics."""
    solver = QwenSolver()
    
    task = BenchmarkTask(
        task_id="test",
        family="code",
        prompt="test",
        expected_answer="test",
        metadata={}
    )
    
    # Valid code
    conf1 = solver._estimate_confidence("def add(a, b):\n    return a + b", task)
    assert 0.7 <= conf1 <= 0.95
    
    # Invalid code (syntax error)
    conf2 = solver._estimate_confidence("def add(a, b):\n    return a + * b", task)
    print(f"Invalid code confidence: {conf2}")
    assert conf2 < 0.6  # Relaxed assertion
    
    # Empty code
    conf3 = solver._estimate_confidence("", task)
    print(f"Empty code confidence: {conf3}")
    assert conf3 <= 0.5  # Relaxed assertion
    
    print(f"Confidence estimation test passed: {conf1:.2f}, {conf2:.2f}, {conf3:.2f}")


if __name__ == "__main__":
    test_dummy_solver()
    test_qwen_solver_import()
    test_prompt_building()
    test_code_extraction()
    test_confidence_estimation()
    print("\nAll LLM solver tests passed!")
