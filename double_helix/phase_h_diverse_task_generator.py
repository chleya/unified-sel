"""
Phase H: Diversity Stress Test - Diverse Task Generator

目标：
- 新增 6 个任务族：string_transform, list_filter, numeric_loop, nested_condition, stateful_accumulator, edge_case_guard
- 每个任务族都要有 ABOVE/NEAR/BELOW examples
- 重点：不要让某个 bug_type 永远只对应一个 boundary label

新增任务族设计：
1. string_transform: 字符串处理（uppercase, lowercase, capitalize, trim, replace）
2. list_filter: 列表过滤（filter_even, filter_gt, filter_contains）
3. numeric_loop: 数值循环（sum_range, product_range, count_divisible）
4. nested_condition: 嵌套条件（is_prime, is_leap, is_valid_date）
5. stateful_accumulator: 有状态累加器（running_sum, running_max, running_mean）
6. edge_case_guard: 边界情况保护（handle_none, handle_empty, handle_overflow）
"""

import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DiverseTaskDef:
    """一个多样化任务定义"""
    task_id: str
    family: str
    bug_type: str
    difficulty: str
    function_name: str
    buggy_code: str
    fixed_code: str
    visible_tests: List[Dict[str, Any]]
    hidden_tests: List[Dict[str, Any]]
    # 这个任务在 SearchLocalSolver 上的预期 boundary（用于数据生成）
    expected_boundary: str = ""


def generate_string_transform_tasks() -> List[DiverseTaskDef]:
    """生成 string_transform 任务族"""
    tasks = []

    # 任务 1: uppercase (trivial, expected ABOVE)
    tasks.append(DiverseTaskDef(
        task_id="string_uppercase_trivial",
        family="string_transform",
        bug_type="uppercase_wrong",
        difficulty="trivial",
        function_name="solve",
        buggy_code="def solve(s):\n    return s + 'UP'\n",
        fixed_code="def solve(s):\n    return s.upper()\n",
        visible_tests=[{"args": ["hello"], "expected": "HELLO"}],
        hidden_tests=[{"args": ["world"], "expected": "WORLD"}, {"args": ["test"], "expected": "TEST"}],
        expected_boundary="above",
    ))

    # 任务 2: capitalize (medium, expected NEAR)
    tasks.append(DiverseTaskDef(
        task_id="string_capitalize_medium",
        family="string_transform",
        bug_type="capitalize_partial",
        difficulty="medium",
        function_name="solve",
        buggy_code="def solve(s):\n    return s[0].upper() + s[1:]\n",
        fixed_code="def solve(s):\n    return s.capitalize()\n",
        visible_tests=[{"args": ["hello world"], "expected": "Hello world"}],
        hidden_tests=[{"args": ["foo bar"], "expected": "Foo bar"}, {"args": ["test"], "expected": "Test"}],
        expected_boundary="near",
    ))

    # 任务 3: trim hard (hard, expected BELOW)
    tasks.append(DiverseTaskDef(
        task_id="string_trim_hard",
        family="string_transform",
        bug_type="trim_recursive",
        difficulty="hard",
        function_name="solve",
        buggy_code="def solve(s):\n    while s and (s[0] == ' ' or s[-1] == ' '):\n        s = s[1:-1]\n    return s\n",
        fixed_code="def solve(s):\n    return s.strip()\n",
        visible_tests=[{"args": ["  hello  "], "expected": "hello"}],
        hidden_tests=[{"args": ["   foo   "], "expected": "foo"}, {"args": ["   "], "expected": ""}],
        expected_boundary="below",
    ))

    # 任务 4: replace (trivial, expected ABOVE)
    tasks.append(DiverseTaskDef(
        task_id="string_replace_trivial",
        family="string_transform",
        bug_type="replace_wrong",
        difficulty="trivial",
        function_name="solve",
        buggy_code="def solve(s, old, new):\n    return s + new\n",
        fixed_code="def solve(s, old, new):\n    return s.replace(old, new)\n",
        visible_tests=[{"args": ["hello", "l", "x"], "expected": "hexxo"}],
        hidden_tests=[{"args": ["world", "o", "a"], "expected": "warld"}, {"args": ["test", "t", "b"], "expected": "besb"}],
        expected_boundary="above",
    ))

    # 任务 5: lowercase (medium, expected NEAR)
    tasks.append(DiverseTaskDef(
        task_id="string_lowercase_medium",
        family="string_transform",
        bug_type="lowercase_partial",
        difficulty="medium",
        function_name="solve",
        buggy_code="def solve(s):\n    return ''.join(c.lower() if c.isupper() else c for c in s)\n",
        fixed_code="def solve(s):\n    return s.lower()\n",
        visible_tests=[{"args": ["HELLO WORLD"], "expected": "hello world"}],
        hidden_tests=[{"args": ["FOO BAR"], "expected": "foo bar"}, {"args": ["TEST"], "expected": "test"}],
        expected_boundary="near",
    ))

    # 任务 6: complex transform (hard, expected BELOW)
    tasks.append(DiverseTaskDef(
        task_id="string_complex_hard",
        family="string_transform",
        bug_type="complex_transform",
        difficulty="hard",
        function_name="solve",
        buggy_code="def solve(s):\n    result = []\n    for c in s:\n        if c in 'aeiou':\n            result.append(c.upper())\n        else:\n            result.append(c.lower())\n    return ''.join(result)\n",
        fixed_code="def solve(s):\n    return s.swapcase()\n",
        visible_tests=[{"args": ["Hello World"], "expected": "hELLO wORLD"}],
        hidden_tests=[{"args": ["Foo Bar"], "expected": "fOO bAR"}, {"args": ["TEST"], "expected": "test"}],
        expected_boundary="below",
    ))

    return tasks


def generate_list_filter_tasks() -> List[DiverseTaskDef]:
    """生成 list_filter 任务族"""
    tasks = []

    # 任务 1: filter even (trivial, expected ABOVE)
    tasks.append(DiverseTaskDef(
        task_id="list_filter_even_trivial",
        family="list_filter",
        bug_type="filter_odd",
        difficulty="trivial",
        function_name="solve",
        buggy_code="def solve(nums):\n    return [x for x in nums if x % 2 == 1]\n",
        fixed_code="def solve(nums):\n    return [x for x in nums if x % 2 == 0]\n",
        visible_tests=[{"args": [[1, 2, 3, 4]], "expected": [2, 4]}],
        hidden_tests=[{"args": [[5, 6, 7]], "expected": [6]}, {"args": [[8, 9]], "expected": [8]}],
        expected_boundary="above",
    ))

    # 任务 2: filter gt (medium, expected NEAR)
    tasks.append(DiverseTaskDef(
        task_id="list_filter_gt_medium",
        family="list_filter",
        bug_type="filter_ge",
        difficulty="medium",
        function_name="solve",
        buggy_code="def solve(nums, threshold):\n    return [x for x in nums if x >= threshold]\n",
        fixed_code="def solve(nums, threshold):\n    return [x for x in nums if x > threshold]\n",
        visible_tests=[{"args": [[1, 3, 5, 7], 4], "expected": [5, 7]}],
        hidden_tests=[{"args": [[2, 4, 6], 3], "expected": [4, 6]}, {"args": [[10], 5], "expected": [10]}],
        expected_boundary="near",
    ))

    # 任务 3: filter contains (hard, expected BELOW)
    tasks.append(DiverseTaskDef(
        task_id="list_filter_contains_hard",
        family="list_filter",
        bug_type="filter_starts_with",
        difficulty="hard",
        function_name="solve",
        buggy_code="def solve(strings, substr):\n    return [s for s in strings if s.startswith(substr)]\n",
        fixed_code="def solve(strings, substr):\n    return [s for s in strings if substr in s]\n",
        visible_tests=[{"args": [["apple", "banana", "grape"], "ap"], "expected": ["apple"]}],
        hidden_tests=[{"args": [["foo", "bar", "foobar"], "bar"], "expected": ["bar", "foobar"]}, {"args": [["test"], "es"], "expected": ["test"]}],
        expected_boundary="below",
    ))

    # 任务 4: filter length (trivial, expected ABOVE)
    tasks.append(DiverseTaskDef(
        task_id="list_filter_length_trivial",
        family="list_filter",
        bug_type="filter_short",
        difficulty="trivial",
        function_name="solve",
        buggy_code="def solve(strings, min_len):\n    return [s for s in strings if len(s) < min_len]\n",
        fixed_code="def solve(strings, min_len):\n    return [s for s in strings if len(s) >= min_len]\n",
        visible_tests=[{"args": [["a", "ab", "abc"], 2], "expected": ["ab", "abc"]}],
        hidden_tests=[{"args": [["x", "xy"], 1], "expected": ["x", "xy"]}, {"args": [["long"], 3], "expected": ["long"]}],
        expected_boundary="above",
    ))

    # 任务 5: filter unique (medium, expected NEAR)
    tasks.append(DiverseTaskDef(
        task_id="list_filter_unique_medium",
        family="list_filter",
        bug_type="filter_duplicates",
        difficulty="medium",
        function_name="solve",
        buggy_code="def solve(nums):\n    seen = set()\n    result = []\n    for x in nums:\n        if x not in seen:\n            seen.add(x)\n        else:\n            result.append(x)\n    return result\n",
        fixed_code="def solve(nums):\n    seen = set()\n    result = []\n    for x in nums:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result\n",
        visible_tests=[{"args": [[1, 2, 2, 3, 3, 3]], "expected": [1, 2, 3]}],
        hidden_tests=[{"args": [[5, 5, 5]], "expected": [5]}, {"args": [[1, 2]], "expected": [1, 2]}],
        expected_boundary="near",
    ))

    # 任务 6: filter nested (hard, expected BELOW)
    tasks.append(DiverseTaskDef(
        task_id="list_filter_nested_hard",
        family="list_filter",
        bug_type="filter_flat",
        difficulty="hard",
        function_name="solve",
        buggy_code="def solve(lists, min_len):\n    result = []\n    for sublist in lists:\n        if len(sublist) > min_len:\n            result.extend(sublist)\n    return result\n",
        fixed_code="def solve(lists, min_len):\n    return [sublist for sublist in lists if len(sublist) > min_len]\n",
        visible_tests=[{"args": [[[1], [2, 3], [4, 5, 6]], 1], "expected": [[2, 3], [4, 5, 6]]}],
        hidden_tests=[{"args": [[[7], [8, 9]], 0], "expected": [[7], [8, 9]]}, {"args": [[[10]], 2], "expected": []}],
        expected_boundary="below",
    ))

    return tasks


def generate_diverse_tasks(num_tasks_per_family: int = 20, seed: int = 42) -> List[DiverseTaskDef]:
    """生成多样化任务"""
    random.seed(seed)

    tasks = []

    # 生成各个任务族
    tasks.extend(generate_string_transform_tasks())
    tasks.extend(generate_list_filter_tasks())

    # TODO: 添加更多任务族

    # 每个任务可以有多个 seed 变体
    expanded_tasks = []
    for task in tasks:
        for s in [42, 123, 456, 789, 1024]:
            expanded_task = DiverseTaskDef(
                task_id=f"{task.task_id}_seed_{s}",
                family=task.family,
                bug_type=task.bug_type,
                difficulty=task.difficulty,
                function_name=task.function_name,
                buggy_code=task.buggy_code,
                fixed_code=task.fixed_code,
                visible_tests=task.visible_tests,
                hidden_tests=task.hidden_tests,
                expected_boundary=task.expected_boundary,
            )
            expanded_tasks.append(expanded_task)

    return expanded_tasks


def main():
    print("=" * 80)
    print("Phase H: Diverse Task Generator")
    print("=" * 80)

    # 生成任务
    tasks = generate_diverse_tasks()
    print(f"\nGenerated {len(tasks)} diverse tasks")

    # 统计
    family_counts = defaultdict(int)
    boundary_counts = defaultdict(int)
    bug_type_boundary_map = defaultdict(set)

    for task in tasks:
        family_counts[task.family] += 1
        boundary_counts[task.expected_boundary] += 1
        bug_type_boundary_map[task.bug_type].add(task.expected_boundary)

    print(f"\nTask family distribution:")
    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count}")

    print(f"\nBoundary distribution:")
    for boundary, count in sorted(boundary_counts.items()):
        print(f"  {boundary}: {count}")

    print(f"\nBug type boundary mapping (checking for diversity):")
    all_diverse = True
    for bug_type, boundaries in sorted(bug_type_boundary_map.items()):
        print(f"  {bug_type}: {sorted(boundaries)}")
        if len(boundaries) == 1:
            print(f"    [WARNING] Only one boundary!")
            all_diverse = False

    if all_diverse:
        print(f"\n[SUCCESS] All bug types have multiple boundaries!")
    else:
        print(f"\n[WARNING] Some bug types only have one boundary!")

    # 保存任务
    output_dir = PROJECT_ROOT / "results" / "phase_h_diverse_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    output_path = output_dir / f"tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "experiment": "phase_h_diverse_tasks",
        "num_tasks": len(tasks),
        "tasks": [asdict(t) for t in tasks],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nTasks saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Diverse Task Generator Complete")
    print("=" * 80)


if __name__ == "__main__":
    from collections import defaultdict
    main()
