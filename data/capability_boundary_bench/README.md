# CapabilityBoundaryBench

Benchmark suites for evaluating LLM capability routing strategies.

## Files

| File | Mode | Contents |
|------|------|----------|
| `code-20.public.jsonl` | public | 20 code tasks, no answer leakage |
| `code-20.eval.jsonl` | eval | 20 code tasks, includes hidden tests + fixed code |
| `code-20.jsonl` | legacy | Same as eval, retained for backward compatibility |
| `mixed-40.public.jsonl` | public | 20 code + 20 reasoning tasks, no answer leakage |
| `mixed-40.eval.jsonl` | eval | 40 mixed tasks, includes hidden tests + fixed code |
| `mixed-40.jsonl` | legacy | Same as eval, retained for backward compatibility |

## Leakage Rules

### Public files (`.public.jsonl`)

Safe for external distribution. These files **exclude**:

- `hidden_tests` - test cases the solver must not see before submission
- `fixed_code` - the correct implementation (directly reveals the answer)
- `expected_route` - the ground-truth routing decision (accept/verify/escalate)
- `hidden_tests_note` - evaluation-only marker

Fields retained in public files:

- `task_id`, `family`, `prompt`, `buggy_code`, `visible_tests`
- `bug_type`, `difficulty`, `function_name` (code tasks)
- `ambiguity_signals` (code tasks)
- `expression`, `ops` (reasoning tasks)

### Eval files (`.eval.jsonl`)

For internal evaluation only. Include all fields, including `hidden_tests` and `fixed_code`.

**Do not distribute eval files externally.** If a solver has access to `hidden_tests` or `fixed_code`, the benchmark result is invalid.

### Legacy files (`.jsonl` without mode suffix)

Identical to eval files. Retained for backward compatibility with existing scripts. Prefer using `.public.jsonl` or `.eval.jsonl` with explicit mode.

## Regenerating

```bash
python experiments/capability/export_bench.py --suite code --num-tasks 20 --seed 7 --mode public --output data/capability_boundary_bench/code-20.public.jsonl
python experiments/capability/export_bench.py --suite code --num-tasks 20 --seed 7 --mode eval --output data/capability_boundary_bench/code-20.eval.jsonl
python experiments/capability/export_bench.py --suite mixed --num-tasks 40 --seed 7 --mode public --output data/capability_boundary_bench/mixed-40.public.jsonl
python experiments/capability/export_bench.py --suite mixed --num-tasks 40 --seed 7 --mode eval --output data/capability_boundary_bench/mixed-40.eval.jsonl
```

## Task Schema

### Code task (public)

```json
{
  "task_id": "code_0",
  "family": "code",
  "prompt": "Fix the buggy Python function...",
  "function_name": "solve",
  "bug_type": "add_one",
  "difficulty": "trivial",
  "buggy_code": "def solve(x):\n    return x\n",
  "visible_tests": [{"args": [4], "expected": 5}],
  "ambiguity_signals": ["visible_pass_hidden_fail_risk"]
}
```

### Code task (eval, additional fields)

```json
{
  ...same as public...,
  "fixed_code": "def solve(x):\n    return x + 1\n",
  "hidden_tests": [{"args": [0], "expected": 1}, {"args": [9], "expected": 10}],
  "hidden_tests_note": "EVALUATION ONLY - must not be used for routing or solver input",
  "expected_route": "accept"
}
```

### Reasoning task

```json
{
  "task_id": "reasoning_0",
  "family": "reasoning",
  "prompt": "Compute the value of: 6 * 7 * 9 + 6",
  "difficulty": "",
  "visible_tests": [],
  "expression": "6 * 7 * 9 + 6",
  "ops": ["*", "*", "+"]
}
```

Reasoning tasks have no `hidden_tests`, `fixed_code`, or `ambiguity_signals`.

## Caveats

- `bug_type` is retained in public files. It reveals the *category* of the bug but not the exact fix. If this is too much information for your evaluation, strip it before distribution.
- `visible_tests` include expected outputs. This is intentional: the solver should see these to guide its repair attempt.
- The oracle assumption applies: `expected_route` in eval files is derived from `difficulty`, which is a heuristic, not a validated ground truth.
