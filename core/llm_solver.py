"""
Lightweight LLM Solver adapter for Capability Router.
Uses Qwen2.5-1.5B with CPU-friendly settings.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from core.capability_benchmark import BenchmarkTask, SolverAttempt


class QwenSolver:
    """Lightweight LLM solver using Qwen2.5-1.5B.
    
    Optimized for CPU inference with minimal memory footprint.
    """

    def __init__(self, model_path: str = "models/Qwen2.5-1.5B"):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    def _load_model(self):
        """Lazy load model to avoid import overhead when not used."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"Loading Qwen2.5-1.5B from {self.model_path}...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load with minimal memory settings (no device_map to avoid accelerate dependency)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            print("Model loaded successfully.")
            
        except ImportError as e:
            raise ImportError(
                "transformers and torch required. "
                "Install: pip install transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def solve(self, task: BenchmarkTask) -> SolverAttempt:
        """Generate solution using Qwen2.5-1.5B."""
        self._load_model()

        prompt = self._build_prompt(task)
        
        try:
            import torch
            
            inputs = self._tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,  # Low temperature for deterministic code
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            
            full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part (after prompt)
            answer = full_text[len(prompt):].strip()
            
            # Extract code block if present
            code = self._extract_code(answer)
            
            # Simple confidence estimation based on code validity
            confidence = self._estimate_confidence(code, task)
            
            return SolverAttempt(
                answer=code,
                confidence=confidence,
                notes="qwen2.5-1.5b",
                metadata={
                    "model": "Qwen2.5-1.5B",
                    "device": self._device,
                    "prompt_length": len(prompt),
                    "response_length": len(answer),
                }
            )
            
        except Exception as e:
            # Fallback: return empty attempt with low confidence
            return SolverAttempt(
                answer="",
                confidence=0.1,
                notes=f"qwen_error:{type(e).__name__}:{e}",
                metadata={"error": str(e)}
            )

    def _build_prompt(self, task: BenchmarkTask) -> str:
        """Build prompt for code repair task."""
        if task.family == "code":
            return (
                "Fix the buggy Python function below. "
                "Return only the fixed function, no explanation.\n\n"
                f"{task.prompt}\n\n"
                "Fixed function:\n"
            )
        else:
            return (
                "Solve the following problem. "
                "Return only the answer, no explanation.\n\n"
                f"{task.prompt}\n\n"
                "Answer:\n"
            )

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        # Try to extract ```python blocks
        match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try generic code blocks
        match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Return raw text if no code blocks found
        return text.strip()

    def _estimate_confidence(self, code: str, task: BenchmarkTask) -> float:
        """Estimate confidence based on code quality heuristics."""
        # Empty code -> very low confidence
        if not code or len(code.strip()) == 0:
            return 0.1
            
        score = 0.5  # Base confidence
        
        # Check if code is not empty
        if len(code) > 10:
            score += 0.1
        
        # Check if code contains function definition
        if "def " in code:
            score += 0.1
        
        # Check if code contains return statement
        if "return" in code:
            score += 0.1
        
        # Check for syntax errors (basic)
        try:
            import ast
            ast.parse(code)
            score += 0.2  # Valid syntax
        except SyntaxError:
            score -= 0.3  # Invalid syntax (higher penalty)
        
        return min(max(score, 0.1), 0.95)


class LlamaCppSolver:
    """Solver using llama.cpp server API.
    
    Much faster than loading model in Python.
    Requires llama-server running on localhost.
    Implements full solver interface: solve, revise, supports_feedback_revision.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8081", model: str = "qwen2.5-0.5b"):
        self.base_url = base_url
        self.model = model

    def supports_feedback_revision(self, task: BenchmarkTask) -> bool:
        return True

    def solve(self, task: BenchmarkTask) -> SolverAttempt:
        import requests

        prompt = self._build_prompt(task)

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.3,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]

            code = self._extract_code(answer)
            confidence = self._estimate_confidence(code, task)

            buggy_code = task.metadata.get("buggy_code", "")
            nontrivial = code != buggy_code if buggy_code else True

            return SolverAttempt(
                answer=code,
                confidence=confidence,
                notes="llama_cpp_qwen2.5-0.5b",
                metadata={
                    "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
                    "solver_type": "llama_cpp",
                    "mode": task.family,
                    "solver_kind": "llm_generate",
                    "exact": False,
                    "nontrivial_patch_found": nontrivial,
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                }
            )

        except Exception as e:
            return SolverAttempt(
                answer="",
                confidence=0.1,
                notes=f"llama_cpp_error:{type(e).__name__}",
                metadata={
                    "error": str(e),
                    "solver_type": "llama_cpp",
                    "solver_kind": "llm_error",
                    "exact": False,
                    "nontrivial_patch_found": False,
                }
            )

    def revise(self, task: BenchmarkTask, previous: SolverAttempt, feedback: str) -> SolverAttempt:
        import requests

        prompt = self._build_revision_prompt(task, previous, feedback)

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.4,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]

            code = self._extract_code(answer)
            confidence = self._estimate_confidence(code, task)

            return SolverAttempt(
                answer=code,
                confidence=confidence,
                notes="llama_cpp_qwen2.5-0.5b_revision",
                metadata={
                    "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
                    "solver_type": "llama_cpp",
                    "mode": task.family,
                    "solver_kind": "feedback_guided",
                    "exact": False,
                    "nontrivial_patch_found": True,
                    "feedback_guided": True,
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                }
            )

        except Exception as e:
            return SolverAttempt(
                answer=previous.answer,
                confidence=previous.confidence * 0.8,
                notes=f"llama_cpp_revision_error:{type(e).__name__}",
                metadata={
                    "error": str(e),
                    "solver_type": "llama_cpp",
                    "solver_kind": "llm_error",
                    "exact": False,
                    "nontrivial_patch_found": False,
                }
            )

    def _build_revision_prompt(self, task: BenchmarkTask, previous: SolverAttempt, feedback: str) -> str:
        visible = task.metadata.get("visible_tests", [])
        examples = ""
        if visible:
            examples = "\n".join(
                f"  solve({json.dumps(t['args'])[1:-1]}) == {json.dumps(t['expected'])}"
                for t in visible
            )
            examples = f"Expected behavior from examples:\n{examples}\n\n"

        if task.family == "code":
            return (
                "The following Python function has a bug. Your previous fix attempt failed.\n\n"
                f"Original buggy code:\n{task.metadata.get('buggy_code', task.prompt)}\n\n"
                f"{examples}"
                f"Your previous attempt:\n{previous.answer}\n\n"
                f"Test feedback: {feedback}\n\n"
                "Return ONLY the corrected function, no explanation.\n\n"
                "Fixed function:\n```python\n"
            )
        else:
            return (
                "Solve the following problem. Your previous answer was wrong.\n\n"
                f"Problem: {task.prompt}\n\n"
                f"Previous answer: {previous.answer}\n"
                f"Feedback: {feedback}\n\n"
                "Return only the correct answer.\n\nAnswer:\n"
            )

    def _build_prompt(self, task: BenchmarkTask) -> str:
        visible = task.metadata.get("visible_tests", [])
        examples = ""
        if visible:
            examples = "\n".join(
                f"  solve({json.dumps(t['args'])[1:-1]}) == {json.dumps(t['expected'])}"
                for t in visible
            )
            examples = f"\nExpected behavior from examples:\n{examples}\n"

        if task.family == "code":
            # task.prompt already contains the instruction and buggy code
            return (
                f"{task.prompt}{examples}\n"
                "Return ONLY the fixed function, no explanation.\n\n"
                "Fixed function:\n```python\n"
            )
        else:
            return (
                "Solve the following problem. "
                "Return only the answer.\n\n"
                f"{task.prompt}\n\nAnswer:\n"
            )

    def _extract_code(self, text: str) -> str:
        import re
        match = re.search(r'```python\s*\n(.*?)\n\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'```\s*\n(.*?)\n\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = text.strip().split('\n')
        cleaned = []
        for line in lines:
            if line.strip().startswith('```'):
                continue
            cleaned.append(line)
        result = '\n'.join(cleaned).strip()
        if 'def ' in result:
            try:
                import ast
                ast.parse(result)
                return result
            except SyntaxError:
                pass
        if 'def ' in text:
            def_match = re.search(r'(def\s+\w+.*)', text, re.DOTALL)
            if def_match:
                candidate = def_match.group(1).strip()
                trailing = re.search(r'(.+?)```\s*$', candidate, re.DOTALL)
                if trailing:
                    candidate = trailing.group(1).strip()
                try:
                    import ast
                    ast.parse(candidate)
                    return candidate
                except SyntaxError:
                    pass
        return result

    def _estimate_confidence(self, code: str, task: BenchmarkTask) -> float:
        if not code or len(code.strip()) == 0:
            return 0.1
        score = 0.5
        if len(code) > 10:
            score += 0.1
        if "def " in code:
            score += 0.1
        if "return" in code:
            score += 0.1
        try:
            import ast
            ast.parse(code)
            score += 0.2
        except SyntaxError:
            score -= 0.3
        return min(max(score, 0.1), 0.95)


class DummyLLMSolver:
    """Dummy solver for testing without loading model.
    
    Returns pre-defined answers for known tasks.
    """
    
    def solve(self, task: BenchmarkTask) -> SolverAttempt:
        """Return dummy attempt for testing."""
        return SolverAttempt(
            answer="# TODO: implement",
            confidence=0.3,
            notes="dummy_solver",
            metadata={"solver_type": "dummy"}
        )
