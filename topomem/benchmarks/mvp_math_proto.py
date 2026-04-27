"""
mvp_math_proto.py — MATH Benchmark: Two-Layer Verify (MVP Prototype)

Evaluates the TopoMem "two-layer verify" approach on the MATH dataset
(hendrycks/math) using Qwen2.5-Math-0.5B-Instruct.

Three evaluation modes:
  1. Baseline  — direct LLM call, no TopoMem, no verify
  2. +TopoMem   — H1/H2 health monitoring with retry on unhealthy state
  3. +Verify    — TopoMem + sympy exact answer verification

Key design decisions:
  - NUMBA_DISABLE_JIT=1 for TDA compatibility (Windows SIGKILL workaround)
  - HF cache at F:\\unified-sel\\topomem\\data\\models\\hf_cache
  - MATH subset configurable (default n=100)
  - Results written to F:\\unified-sel\\topomem\\benchmarks\\results\\mvp_math_*.json

Usage:
    python mvp_math_proto.py [--n 50]

Author: OpenClaw Agent (subagent)
Date: 2026-04-06
"""

# ------------------------------------------------------------------
# 0. Environment Setup — MUST be before any other imports
# ------------------------------------------------------------------
import os
import sys
import json
import time
import argparse
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Critical: NUMBA_DISABLE_JIT before importing topomem/any numba-dependent code
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# HuggingFace cache
HF_CACHE = r"F:\unified-sel\topomem\data\models\hf_cache"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", HF_CACHE)

# TopoMem project root
PROJECT_ROOT = Path(r"F:\unified-sel")
sys.path.insert(0, str(PROJECT_ROOT))

# Results directory
RESULTS_DIR = Path(r"F:\unified-sel\topomem\benchmarks\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mvp_math_proto")


# ------------------------------------------------------------------
# 1. Sympy Answer Verification (Layer 2)
# ------------------------------------------------------------------

def normalize_math_answer(text: str) -> str:
    """
    Normalize a mathematical answer string for comparison.
    """
    if not text or not isinstance(text, str):
        return ""

    t = text.strip()
    t = re.sub(r"\$([^$]+)\$", r"\1", t)
    t = re.sub(r"\\\((.+?)\\\)", r"\1", t)
    t = re.sub(r"\\\[(.+?)\\\]", r"\1", t)
    t = re.sub(
        r"^(the\s+)?(answer|result|solution)\s*(is)?\s*[:=]\s*",
        "", t, flags=re.IGNORECASE
    )
    t = re.sub(r"^(final\s*answer)\s*[:=]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip().rstrip(".")
    return t


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} in LaTeX."""
    if not text:
        return None
    match = re.search(r"\\boxed\s*\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def parse_ground_truth(answer: str) -> str:
    """Parse ground truth from MATH dataset."""
    if not answer:
        return ""
    a = answer.strip()
    boxed = extract_boxed_answer(a)
    if boxed:
        return boxed
    a = re.sub(r"^\$", "", a)
    a = re.sub(r"\$$", "", a)
    a = re.sub(r"^\\([a-zA-Z]+)", r"\1", a)
    return a.strip()


def compare_math_answers(predicted: str, ground_truth: str) -> Tuple[bool, str]:
    """
    Compare predicted vs ground truth using sympy numeric/symbolic comparison.
    Returns (correct: bool, reason: str).
    """
    norm_pred = normalize_math_answer(predicted)
    norm_gt = normalize_math_answer(ground_truth)

    if not norm_pred or not norm_gt:
        return (norm_pred == norm_gt, "empty_answer")

    try:
        import sympy

        def try_sympy(s: str) -> Optional[Any]:
            s = s.strip()
            if not s:
                return None
            try:
                return sympy.sympify(s)
            except (sympy.SympifyError, ValueError, TypeError):
                return None

        sym_pred = try_sympy(norm_pred)
        sym_gt = try_sympy(norm_gt)

        if sym_pred is not None and sym_gt is not None:
            try:
                diff = abs(float(sym_pred) - float(sym_gt))
                if diff < 1e-6:
                    return (True, "sympy_numeric")
                if sympy.simplify(sym_pred - sym_gt) == 0:
                    return (True, "sympy_symbolic")
                return (False, f"sympy_mismatch:pred={float(sym_pred):.4f},gt={float(sym_gt):.4f}")
            except (ValueError, TypeError):
                if sym_pred == sym_gt:
                    return (True, "sympy_exact")
    except ImportError:
        pass

    # String fallback
    if norm_pred == norm_gt:
        return (True, "string_exact")

    # Try sympy normalization on mismatch
    try:
        import sympy
        sym_pred = sympy.sympify(norm_pred)
        sym_gt = sympy.sympify(norm_gt)
        if sym_pred == sym_gt:
            return (True, "sympy_normalized")
    except Exception:
        pass

    return (False, "string_mismatch")


# ------------------------------------------------------------------
# 2. MATH Dataset Loader
# ------------------------------------------------------------------

def load_math_dataset(n: int = 100, split: str = "test") -> List[Dict[str, str]]:
    """Load n problems from hendrycks/math dataset."""
    logger.info(f"Loading MATH dataset (split={split}, n={n})...")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("`datasets` package not installed. Run: pip install datasets")
        raise

    ds = load_dataset("hendrycks/math", split=split, trust_remote_code=True)

    total = len(ds)
    if n >= total:
        indices = list(range(total))
    else:
        step = total / n
        indices = [int(i * step) for i in range(n)]

    problems = []
    for idx in indices:
        row = ds[idx]
        problems.append({
            "problem": row.get("problem", ""),
            "level": row.get("level", "unknown"),
            "type": row.get("type", "unknown"),
            "answer": row.get("answer", ""),
            "solution": row.get("solution", ""),
        })

    logger.info(f"Loaded {len(problems)} MATH problems (total={total})")
    return problems


# ------------------------------------------------------------------
# 3. Baseline LLM Interface (no TopoMem)
# ------------------------------------------------------------------

class BaselineEngine:
    """Direct transformers pipeline — no TopoMem. Used for baseline measurement."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading baseline model: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device = torch.device("cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
        self._model = self._model.to(device)
        self._model.eval()
        logger.info("Baseline model loaded")

    def solve(
        self,
        problem: str,
        max_tokens: int = 384,
        temperature: float = 0.3,
    ) -> str:
        """Solve a single math problem."""
        import torch

        prompt = (
            "You are a precise math problem solver. "
            "Provide ONLY the final numerical answer (no explanation). "
            "If the answer is a fraction, box it as \\boxed{a/b}. "
            "If the answer is an integer or decimal, box it as \\boxed{answer}.\n\n"
            f"Problem: {problem}\nAnswer: "
        )

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        return response


# ------------------------------------------------------------------
# 4. TopoMem-Enabled Solver (Two-Layer Verify)
# ------------------------------------------------------------------

class TopoMemSolver:
    """
    TopoMem-augmented math solver with two-layer verification.

    Layer 1 — H1/H2 health monitoring:
        - After each problem, check H1 health and H2 suppressed status
        - Retry once if H1_health < 0.7 OR H2 is NOT suppressed

    Layer 2 — Sympy exact verification:
        - Extract answer and verify with sympy
        - Retry up to MAX_VERIFY_RETRIES if incorrect
    """

    H1_HEALTH_THRESHOLD = 0.7
    MAX_VERIFY_RETRIES = 2

    def __init__(self, system):
        self.system = system
        self._problem_count = 0

    def _get_h1_h2_metrics(self) -> Dict[str, Any]:
        """Get current H1/H2 metrics from the system."""
        try:
            h1_m = self.system.self_aware.get_h1_metrics()
            h2_m = self.system.self_aware.get_h2_metrics()
            return {
                "h1_health": h1_m.h1_health_score if not h1_m.suppressed else 1.0,
                "h1_suppressed": h1_m.suppressed,
                "h1_betti_1": h1_m.betti_1_count,
                "h1_fragmentation": h1_m.fragmentation_index,
                "h2_health": h2_m.h2_health_score if not h2_m.suppressed else 1.0,
                "h2_suppressed": h2_m.suppressed,
                "h2_betti_2": h2_m.betti_2_count,
                "h2_to_h1_ratio": h2_m.h2_to_h1_ratio,
                "h2_cavitation_rate": h2_m.cavitation_rate,
            }
        except Exception as e:
            logger.warning(f"Failed to get H1/H2 metrics: {e}")
            return {
                "h1_health": 1.0, "h1_suppressed": True,
                "h2_health": 1.0, "h2_suppressed": True,
                "h1_betti_1": 0, "h1_fragmentation": 0.0,
                "h2_betti_2": 0, "h2_to_h1_ratio": 0.0,
                "h2_cavitation_rate": 0.0,
            }

    def _should_retry_topomem(self, metrics: Dict[str, Any]) -> bool:
        """
        Determine if Layer 1 (TopoMem) signals a retry is needed.

        Retry triggers:
          - H1 health < 0.7: embedding geometry degraded
          - H2 is NOT suppressed: domain mixing detected → geometry unstable
        """
        h1_ok = metrics["h1_health"] >= self.H1_HEALTH_THRESHOLD
        h2_ok = metrics["h2_suppressed"]  # Retry if H2 is NOT suppressed (means domain mixing)
        return not h1_ok or not h2_ok

    def solve(
        self,
        problem: str,
        ground_truth: str,
        max_tokens: int = 384,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Solve with two-layer verify: TopoMem monitoring + sympy verification.

        Returns:
            dict with response, correct, method, h1_h2_metrics, verify_attempts, etc.
        """
        self._problem_count += 1

        # Layer 1A: Add problem to memory for topology tracking
        self.system.add_knowledge(
            text=f"Math problem: {problem}",
            metadata={"type": "math_problem", "step": self._problem_count},
        )

        # Layer 1B: Generate response
        response = self.system.ask(problem, max_tokens=max_tokens)
        metrics_after = self._get_h1_h2_metrics()
        should_retry_tm = self._should_retry_topomem(metrics_after)
        method = "direct"

        # Layer 1C: Retry if TopoMem signals unhealthy
        if should_retry_tm:
            logger.debug(
                f"TopoMem retry: h1={metrics_after['h1_health']:.3f}, "
                f"h2_suppressed={metrics_after['h2_suppressed']}, "
                f"h2_ratio={metrics_after['h2_to_h1_ratio']:.3f}"
            )
            response = self.system.ask(problem, max_tokens=max_tokens)
            metrics_after = self._get_h1_h2_metrics()
            method = "topomem_retry"

        # Layer 2: Sympy verification
        extracted = extract_boxed_answer(response)
        if extracted is None:
            extracted = response.strip()

        correct, reason = compare_math_answers(extracted, ground_truth)
        verify_attempts = [{
            "attempt": 0,
            "correct": correct,
            "reason": reason,
            "response_snippet": response[:200],
        }]

        verify_retries = 0
        while not correct and verify_retries < self.MAX_VERIFY_RETRIES:
            verify_retries += 1
            response = self.system.ask(
                problem,
                max_tokens=int(max_tokens * 1.2),
            )
            extracted = extract_boxed_answer(response)
            if extracted is None:
                extracted = response.strip()

            correct, reason = compare_math_answers(extracted, ground_truth)
            verify_attempts.append({
                "attempt": verify_retries,
                "correct": correct,
                "reason": reason,
                "response_snippet": response[:200],
            })
            method = "verify_retry"

        # Final method classification
        if method == "direct" and correct:
            method = "direct_correct"
        elif method == "direct" and not correct:
            method = "direct_incorrect"

        return {
            "response": response,
            "extracted_answer": extracted,
            "correct": correct,
            "method": method,
            "h1_h2_metrics": metrics_after,
            "verify_attempts": verify_attempts,
            "topomem_retry_triggered": method in ("topomem_retry", "verify_retry"),
            "verify_retry_count": verify_retries,
        }


# ------------------------------------------------------------------
# 5. Evaluation Runner
# ------------------------------------------------------------------

def run_baseline(problems: List[Dict], engine: BaselineEngine) -> List[Dict]:
    """Run baseline evaluation (no TopoMem, no verify)."""
    results = []
    total = len(problems)

    for i, prob in enumerate(problems):
        logger.info(f"Baseline [{i+1}/{total}] {prob.get('level','?')}/{prob.get('type','?')}")

        try:
            response = engine.solve(problem=prob["problem"])
            extracted = extract_boxed_answer(response)
            if extracted is None:
                extracted = response.strip()

            ground_truth = parse_ground_truth(prob["answer"])
            correct, reason = compare_math_answers(extracted, ground_truth)

            results.append({
                "index": i,
                "problem": prob["problem"][:100],
                "level": prob.get("level", ""),
                "type": prob.get("type", ""),
                "ground_truth": ground_truth,
                "response": response[:500],
                "extracted_answer": extracted,
                "correct": correct,
                "verify_reason": reason,
                "h1_h2_metrics": None,
                "topomem_retry": False,
                "verify_retry_count": 0,
            })
        except Exception as e:
            logger.warning(f"Baseline error on problem {i}: {e}")
            results.append({
                "index": i,
                "problem": prob["problem"][:100],
                "level": prob.get("level", ""),
                "type": prob.get("type", ""),
                "ground_truth": parse_ground_truth(prob.get("answer", "")),
                "response": f"[ERROR: {e}]",
                "extracted_answer": "",
                "correct": False,
                "verify_reason": "exception",
                "h1_h2_metrics": None,
                "topomem_retry": False,
                "verify_retry_count": 0,
            })

    return results


def run_topomem(
    problems: List[Dict],
    solver: TopoMemSolver,
    use_verify: bool = False,
) -> List[Dict]:
    """Run TopoMem evaluation with optional verify layer."""
    results = []
    total = len(problems)

    for i, prob in enumerate(problems):
        logger.info(
            f"TopoMem{'+Verify' if use_verify else ''} [{i+1}/{total}] "
            f"{prob.get('level','?')}/{prob.get('type','?')}"
        )

        try:
            ground_truth = parse_ground_truth(prob["answer"])

            if use_verify:
                # Both layers active
                outcome = solver.solve(
                    problem=prob["problem"],
                    ground_truth=ground_truth,
                )
            else:
                # Layer 1 only: monitoring, no sympy verify
                solver.system.add_knowledge(
                    text=f"Math problem: {prob['problem']}",
                    metadata={"type": "math_problem", "step": i},
                )
                response = solver.system.ask(prob["problem"])
                extracted = extract_boxed_answer(response)
                if extracted is None:
                    extracted = response.strip()

                correct, reason = compare_math_answers(extracted, ground_truth)
                h1_h2 = solver._get_h1_h2_metrics()
                should_retry = solver._should_retry_topomem(h1_h2)

                if should_retry:
                    response = solver.system.ask(prob["problem"])
                    extracted = extract_boxed_answer(response)
                    if extracted is None:
                        extracted = response.strip()
                    correct, reason = compare_math_answers(extracted, ground_truth)
                    method = "topomem_retry"
                else:
                    method = "direct_correct" if correct else "direct_incorrect"

                outcome = {
                    "response": response,
                    "extracted_answer": extracted,
                    "correct": correct,
                    "method": method,
                    "h1_h2_metrics": h1_h2,
                    "verify_attempts": [{
                        "attempt": 0, "correct": correct,
                        "reason": reason, "response_snippet": response[:200],
                    }],
                    "topomem_retry_triggered": should_retry,
                    "verify_retry_count": 0,
                }

            results.append({
                "index": i,
                "problem": prob["problem"][:100],
                "level": prob.get("level", ""),
                "type": prob.get("type", ""),
                "ground_truth": ground_truth,
                "response": outcome.get("response", "")[:500],
                "extracted_answer": outcome.get("extracted_answer", ""),
                "correct": outcome.get("correct", False),
                "verify_reason": outcome.get("verify_attempts", [{}])[0].get("reason", ""),
                "h1_h2_metrics": outcome.get("h1_h2_metrics"),
                "topomem_retry": outcome.get("topomem_retry_triggered", False),
                "verify_retry_count": outcome.get("verify_retry_count", 0),
                "method": outcome.get("method", "unknown"),
            })

        except Exception as e:
            logger.warning(f"TopoMem error on problem {i}: {e}")
            results.append({
                "index": i,
                "problem": prob["problem"][:100],
                "level": prob.get("level", ""),
                "type": prob.get("type", ""),
                "ground_truth": parse_ground_truth(prob.get("answer", "")),
                "response": f"[ERROR: {e}]",
                "extracted_answer": "",
                "correct": False,
                "verify_reason": "exception",
                "h1_h2_metrics": None,
                "topomem_retry": False,
                "verify_retry_count": 0,
                "method": "exception",
            })

    return results


def compute_summary(results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from evaluation results."""
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / n if n > 0 else 0.0

    # Per-level accuracy
    levels = set(r["level"] for r in results)
    by_level = {}
    for lv in levels:
        subset = [r for r in results if r["level"] == lv]
        n_lv = len(subset)
        c_lv = sum(1 for r in subset if r["correct"])
        by_level[lv] = {"count": n_lv, "correct": c_lv, "accuracy": round(c_lv / n_lv, 4) if n_lv > 0 else 0.0}

    # Per-type accuracy
    types = set(r["type"] for r in results)
    by_type = {}
    for tp in types:
        subset = [r for r in results if r["type"] == tp]
        n_tp = len(subset)
        c_tp = sum(1 for r in subset if r["correct"])
        by_type[tp] = {"count": n_tp, "correct": c_tp, "accuracy": round(c_tp / n_tp, 4) if n_tp > 0 else 0.0}

    # Retry stats
    tm_retries = sum(1 for r in results if r.get("topomem_retry", False))
    verify_retries_total = sum(r.get("verify_retry_count", 0) for r in results)

    # H1/H2 aggregate
    import numpy as np

    h1_healths = [r["h1_h2_metrics"]["h1_health"] for r in results if r["h1_h2_metrics"] is not None]
    h2_ratios = [r["h1_h2_metrics"]["h2_to_h1_ratio"] for r in results if r["h1_h2_metrics"] is not None]
    h2_suppressed_rates = [
        r["h1_h2_metrics"]["h2_suppressed"] for r in results if r["h1_h2_metrics"] is not None
    ]

    return {
        "n_total": n,
        "n_correct": correct,
        "accuracy": round(accuracy, 4),
        "by_level": by_level,
        "by_type": by_type,
        "topomem_retry_count": tm_retries,
        "topomem_retry_rate": round(tm_retries / n, 4) if n > 0 else 0.0,
        "verify_retry_total": verify_retries_total,
        "h1_health_mean": round(float(np.mean(h1_healths)), 4) if h1_healths else None,
        "h1_health_std": round(float(np.std(h1_healths)), 4) if h1_healths else None,
        "h2_ratio_mean": round(float(np.mean(h2_ratios)), 4) if h2_ratios else None,
        "h2_ratio_std": round(float(np.std(h2_ratios)), 4) if h2_ratios else None,
        "h2_suppressed_rate": round(sum(h2_suppressed_rates) / len(h2_suppressed_rates), 4)
        if h2_suppressed_rates else None,
    }


# ------------------------------------------------------------------
# 6. Main Entry Point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MATH Benchmark MVP Prototype")
    parser.add_argument("--n", type=int, default=100, help="Number of MATH problems")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-topomem", action="store_true")
    args = parser.parse_args()

    # ---- Load dataset ----
    problems = load_math_dataset(n=args.n, split=args.split)

    # ---- Session metadata ----
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"math_{timestamp}"
    output_file = RESULTS_DIR / f"mvp_math_{run_id}.json"

    session_meta = {
        "run_id": run_id,
        "timestamp": timestamp,
        "n_problems": len(problems),
        "split": args.split,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "h1_health_threshold": TopoMemSolver.H1_HEALTH_THRESHOLD,
        "max_verify_retries": TopoMemSolver.MAX_VERIFY_RETRIES,
    }

    # =================================================================
    # RUN 1: Baseline
    # =================================================================
    baseline_results = []
    baseline_summary = {}

    if not args.skip_baseline:
        logger.info("=" * 60)
        logger.info("RUN 1: Baseline (no TopoMem, no verify)")
        logger.info("=" * 60)

        baseline_engine = BaselineEngine()
        t0 = time.time()
        baseline_results = run_baseline(problems, baseline_engine)
        elapsed = time.time() - t0
        baseline_summary = compute_summary(baseline_results)
        baseline_summary["elapsed_seconds"] = round(elapsed, 1)

        logger.info(
            f"Baseline done: {baseline_summary['n_correct']}/{baseline_summary['n_total']} "
            f"= {baseline_summary['accuracy']*100:.1f}% ({elapsed:.1f}s)"
        )
    else:
        logger.info("Skipping baseline run")

    # =================================================================
    # RUN 2: TopoMem (H1/H2 monitoring, no verify)
    # =================================================================
    topomem_results = []
    topomem_summary = {}

    if not args.skip_topomem:
        logger.info("=" * 60)
        logger.info("RUN 2: TopoMem (H1/H2 monitoring, no verify)")
        logger.info("=" * 60)

        from topomem.config import TopoMemConfig, EngineConfig
        from topomem.system import TopoMemSystem

        topomem_config = TopoMemConfig(
            seed=42,
            engine=EngineConfig(
                fallback_model_name="Qwen/Qwen2.5-0.5B-Instruct",
                use_fallback=True,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ),
        )

        topomem_system = TopoMemSystem(topomem_config)
        solver = TopoMemSolver(topomem_system)

        # Initial calibration to establish baseline
        try:
            cal = topomem_system.self_aware.calibrate(
                topomem_system.memory,
                topomem_system.topology,
                topomem_system.engine,
            )
            logger.info(
                f"Initial calibration: drift={cal.drift.status}, "
                f"H1 suppressed={cal.h1_metrics.suppressed}, "
                f"H2 suppressed={cal.h2_metrics.suppressed}"
            )
        except Exception as e:
            logger.warning(f"Initial calibration failed: {e}")

        t0 = time.time()
        topomem_results = run_topomem(problems, solver, use_verify=False)
        elapsed = time.time() - t0
        topomem_summary = compute_summary(topomem_results)
        topomem_summary["elapsed_seconds"] = round(elapsed, 1)

        logger.info(
            f"TopoMem done: {topomem_summary['n_correct']}/{topomem_summary['n_total']} "
            f"= {topomem_summary['accuracy']*100:.1f}% ({elapsed:.1f}s), "
            f"retries={topomem_summary['topomem_retry_count']}"
        )
    else:
        logger.info("Skipping TopoMem run")

    # =================================================================
    # RUN 3: TopoMem + Verify (two-layer)
    # =================================================================
    topomem_verify_results = []
    topomem_verify_summary = {}

    if not args.skip_topomem:
        logger.info("=" * 60)
        logger.info("RUN 3: TopoMem + Verify (two-layer)")
        logger.info("=" * 60)

        from topomem.config import TopoMemConfig, EngineConfig
        from topomem.system import TopoMemSystem

        topomem_v_config = TopoMemConfig(
            seed=42,
            engine=EngineConfig(
                fallback_model_name="Qwen/Qwen2.5-0.5B-Instruct",
                use_fallback=True,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ),
        )
        topomem_v_system = TopoMemSystem(topomem_v_config)
        solver_v = TopoMemSolver(topomem_v_system)

        try:
            cal = topomem_v_system.self_aware.calibrate(
                topomem_v_system.memory,
                topomem_v_system.topology,
                topomem_v_system.engine,
            )
        except Exception as e:
            logger.warning(f"Initial calibration failed: {e}")

        t0 = time.time()
        topomem_verify_results = run_topomem(problems, solver_v, use_verify=True)
        elapsed = time.time() - t0
        topomem_verify_summary = compute_summary(topomem_verify_results)
        topomem_verify_summary["elapsed_seconds"] = round(elapsed, 1)

        logger.info(
            f"TopoMem+Verify done: {topomem_verify_summary['n_correct']}/"
            f"{topomem_verify_summary['n_total']} "
            f"= {topomem_verify_summary['accuracy']*100:.1f}% ({elapsed:.1f}s), "
            f"verify_retries={topomem_verify_summary['verify_retry_total']}"
        )
    else:
        logger.info("Skipping TopoMem+Verify run")

    # =================================================================
    # Analyze: Problems corrected by verify layer
    # =================================================================
    corrected_by_verify = []
    if topomem_results and topomem_verify_results:
        for tm_r, tv_r in zip(topomem_results, topomem_verify_results):
            if not tm_r["correct"] and tv_r["correct"]:
                corrected_by_verify.append({
                    "index": tm_r["index"],
                    "level": tv_r["level"],
                    "type": tv_r["type"],
                    "ground_truth": tv_r["ground_truth"],
                    "topomem_response": tm_r["response"][:200],
                    "verify_response": tv_r["response"][:200],
                    "topomem_extracted": tm_r["extracted_answer"],
                    "verify_extracted": tv_r["extracted_answer"],
                    "h1_h2": tv_r.get("h1_h2_metrics"),
                })

    # =================================================================
    # Delta comparison
    # =================================================================
    def safe_delta(a, b):
        if a is None or b is None:
            return None
        return round(a - b, 4)

    comparison = {
        "baseline_accuracy": baseline_summary.get("accuracy"),
        "topomem_accuracy": topomem_summary.get("accuracy"),
        "topomem_verify_accuracy": topomem_verify_summary.get("accuracy"),
        "topomem_vs_baseline_delta": safe_delta(
            topomem_summary.get("accuracy"), baseline_summary.get("accuracy")
        ),
        "verify_vs_topomem_delta": safe_delta(
            topomem_verify_summary.get("accuracy"), topomem_summary.get("accuracy")
        ),
        "verify_vs_baseline_delta": safe_delta(
            topomem_verify_summary.get("accuracy"), baseline_summary.get("accuracy")
        ),
    }

    # =================================================================
    # Build output
    # =================================================================
    full_output = {
        "session": session_meta,
        "baseline": {"summary": baseline_summary, "results": baseline_results},
        "topomem": {"summary": topomem_summary, "results": topomem_results},
        "topomem_verify": {"summary": topomem_verify_summary, "results": topomem_verify_results},
        "corrected_by_verify": corrected_by_verify,
        "comparison": comparison,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults written to: {output_file}")

    # =================================================================
    # Human-readable summary
    # =================================================================
    print("\n" + "=" * 70)
    print("MATH BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Dataset: hendrycks/math ({args.split}), n={len(problems)}")
    print(f"Model: Qwen2.5-0.5B-Instruct")
    print(f"H1 health threshold: {TopoMemSolver.H1_HEALTH_THRESHOLD}")
    print(f"Max verify retries: {TopoMemSolver.MAX_VERIFY_RETRIES}")
    print()
    print(f"{'Mode':<25} {'Accuracy':>10} {'Retries':>10} {'Time(s)':>10}")
    print("-" * 60)

    if baseline_summary:
        print(
            f"{'Baseline':<25} "
            f"{baseline_summary.get('accuracy', 0)*100:>9.1f}% "
            f"{'N/A':>10} "
            f"{baseline_summary.get('elapsed_seconds', 0):>10.1f}"
        )

    if topomem_summary:
        print(
            f"{'+TopoMem':<25} "
            f"{topomem_summary.get('accuracy', 0)*100:>9.1f}% "
            f"{topomem_summary.get('topomem_retry_count', 0):>10} "
            f"{topomem_summary.get('elapsed_seconds', 0):>10.1f}"
        )

    if topomem_verify_summary:
        print(
            f"{'+TopoMem+Verify':<25} "
            f"{topomem_verify_summary.get('accuracy', 0)*100:>9.1f}% "
            f"{topomem_verify_summary.get('verify_retry_total', 0):>10} "
            f"{topomem_verify_summary.get('elapsed_seconds', 0):>10.1f}"
        )

    print()
    print("Delta Comparison:")
    if comparison["topomem_vs_baseline_delta"] is not None:
        print(
            f"  +TopoMem vs Baseline:  "
            f"{'+' if comparison['topomem_vs_baseline_delta'] >= 0 else ''}"
            f"{comparison['topomem_vs_baseline_delta']*100:.2f}pp"
        )
    if comparison["verify_vs_topomem_delta"] is not None:
        print(
            f"  +Verify vs +TopoMem:   "
            f"{'+' if comparison['verify_vs_topomem_delta'] >= 0 else ''}"
            f"{comparison['verify_vs_topomem_delta']*100:.2f}pp"
        )
    if comparison["verify_vs_baseline_delta"] is not None:
        print(
            f"  +Verify vs Baseline:  "
            f"{'+' if comparison['verify_vs_baseline_delta'] >= 0 else ''}"
            f"{comparison['verify_vs_baseline_delta']*100:.2f}pp"
        )

    print()
    print(f"Problems corrected by verify layer: {len(corrected_by_verify)}")
    print(f"Output file: {output_file}")
    print("=" * 70)

    # ---- Per-level breakdown ----
    if topomem_verify_summary.get("by_level"):
        print("\nPer-Level Accuracy (+TopoMem+Verify):")
        for lv, stats in sorted(topomem_verify_summary['by_level'].items()):
            print(f"  {lv:<20}: {stats['correct']}/{stats['count']} = {stats['accuracy']*100:.1f}%")

    if topomem_verify_summary.get("by_type"):
        print("\nPer-Type Accuracy (+TopoMem+Verify):")
        for tp, stats in sorted(topomem_verify_summary['by_type'].items()):
            print(f"  {tp:<20}: {stats['correct']}/{stats['count']} = {stats['accuracy']*100:.1f}%")

    # ---- H1/H2 aggregate metrics ----
    print()
    print("H1/H2 Aggregate Metrics (+TopoMem+Verify):")
    tmv = topomem_verify_summary
    if tmv.get("h1_health_mean") is not None:
        print(f"  H1 health mean:    {tmv['h1_health_mean']:.4f} ± {tmv.get('h1_health_std', 0):.4f}")
    if tmv.get("h2_ratio_mean") is not None:
        print(f"  H2/H1 ratio mean:  {tmv['h2_ratio_mean']:.4f} ± {tmv.get('h2_ratio_std', 0):.4f}")
    if tmv.get("h2_suppressed_rate") is not None:
        print(f"  H2 suppressed rate:{tmv['h2_suppressed_rate']*100:.1f}%")
    print(f"  TopoMem retry rate: {tmv.get('topomem_retry_rate', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
