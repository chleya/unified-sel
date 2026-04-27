"""
Export capability benchmark suites to JSONL format.

Two export modes control field leakage:

  public  — safe for public distribution; strips hidden_tests, fixed_code,
            expected_route, and hidden_tests_note.
  eval    — includes all fields including hidden_tests and fixed_code;
            for internal evaluation only.

Usage:
    python experiments/capability/export_bench.py --suite code --num-tasks 20 --seed 7 --mode public --output data/capability_boundary_bench/code-20.public.jsonl
    python experiments/capability/export_bench.py --suite code --num-tasks 20 --seed 7 --mode eval --output data/capability_boundary_bench/code-20.eval.jsonl
    python experiments/capability/export_bench.py --suite mixed --num-tasks 40 --seed 7 --mode public --output data/capability_boundary_bench/mixed-40.public.jsonl
    python experiments/capability/export_bench.py --suite mixed --num-tasks 40 --seed 7 --mode eval --output data/capability_boundary_bench/mixed-40.eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import build_task_suite


_PUBLIC_EXCLUDED_FIELDS = frozenset([
    "hidden_tests",
    "hidden_tests_note",
    "fixed_code",
    "expected_route",
])


def _strip_public_fields(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _PUBLIC_EXCLUDED_FIELDS}


def export_task(task, mode: str = "eval") -> dict:
    visible_tests = task.metadata.get("visible_tests", [])
    hidden_tests = task.metadata.get("hidden_tests", [])

    record = {
        "task_id": task.task_id,
        "family": task.family,
        "prompt": task.prompt,
        "function_name": task.metadata.get("function_name", ""),
        "bug_type": task.metadata.get("bug_type", ""),
        "difficulty": task.metadata.get("difficulty", ""),
        "buggy_code": task.metadata.get("buggy_code", ""),
        "fixed_code": task.expected_answer,
        "visible_tests": visible_tests,
        "hidden_tests": hidden_tests,
        "hidden_tests_note": "EVALUATION ONLY - must not be used for routing or solver input",
        "expected_route": _infer_expected_route(task),
        "ambiguity_signals": _detect_ambiguity_signals(task),
    }

    if task.family == "reasoning":
        record["expression"] = task.metadata.get("expression", "")
        record["ops"] = task.metadata.get("ops", [])
        record.pop("buggy_code", None)
        record.pop("fixed_code", None)
        record.pop("bug_type", None)
        record.pop("function_name", None)
        record.pop("ambiguity_signals", None)

    if mode == "public":
        record = _strip_public_fields(record)

    return record


def _infer_expected_route(task) -> str:
    if task.family == "reasoning":
        return "accept_or_verify"

    difficulty = task.metadata.get("difficulty", "")
    if difficulty == "trivial":
        return "accept"
    if difficulty in ("easy", "medium"):
        return "verify"
    if difficulty == "hard":
        return "escalate"
    return "verify"


def _detect_ambiguity_signals(task) -> list[str]:
    if task.family != "code":
        return []

    signals = []
    bug_type = task.metadata.get("bug_type", "")
    visible = task.metadata.get("visible_tests", [])
    hidden = task.metadata.get("hidden_tests", [])

    if not visible and hidden:
        signals.append("no_visible_tests")

    if bug_type in (
        "count_palindrome_words",
        "count_adjacent_repeat_words",
        "count_words_with_vowel",
    ):
        signals.append("word_ambiguity")

    if bug_type in (
        "count_positive",
        "count_negative",
        "count_gt_two",
        "count_even",
        "count_nonzero",
        "count_prime",
        "count_multiple_of_three",
        "count_abs_gt_two",
    ):
        signals.append("threshold_ambiguity")

    if bug_type in ("and_instead_of_or", "or_instead_of_and", "missing_not"):
        signals.append("logic_ambiguity")

    if bug_type in (
        "normalize_spaces",
        "normalize_commas",
        "normalize_pipes",
    ):
        signals.append("normalization_ambiguity")

    if bug_type in ("dedupe_sorted",):
        signals.append("deduplication_ambiguity")

    if bug_type in ("factorial_seed",):
        signals.append("seed_ambiguity")

    if len(visible) < 3 and len(hidden) >= 2:
        signals.append("visible_pass_hidden_fail_risk")

    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Export capability benchmark to JSONL")
    parser.add_argument("--suite", required=True, choices=["reasoning", "code", "mixed"])
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--suite-variant", default="standard", choices=["standard", "paraphrase"])
    parser.add_argument("--mode", required=True, choices=["public", "eval"],
                        help="public: strips hidden_tests/fixed_code/expected_route; eval: includes all fields")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    tasks = build_task_suite(args.suite, args.num_tasks, args.seed, args.suite_variant)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [export_task(t, mode=args.mode) for t in tasks]

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    code_count = sum(1 for r in records if r["family"] == "code")
    reasoning_count = sum(1 for r in records if r["family"] == "reasoning")
    bug_types = sorted(set(r.get("bug_type", "") for r in records if r.get("bug_type")))
    difficulties = sorted(set(r.get("difficulty", "") for r in records if r.get("difficulty")))
    ambiguity_count = sum(1 for r in records if r.get("ambiguity_signals"))

    print(f"Exported {len(records)} tasks to {output_path}")
    print(f"  Mode: {args.mode}")
    print(f"  Code: {code_count} | Reasoning: {reasoning_count}")
    if bug_types:
        print(f"  Bug types ({len(bug_types)}): {', '.join(bug_types)}")
    if difficulties:
        print(f"  Difficulties: {', '.join(difficulties)}")
    print(f"  Tasks with ambiguity signals: {ambiguity_count}")
    if args.mode == "public":
        print(f"  Stripped fields: {', '.join(sorted(_PUBLIC_EXCLUDED_FIELDS))}")
    else:
        print(f"  Hidden tests marked: EVALUATION ONLY")


if __name__ == "__main__":
    main()
