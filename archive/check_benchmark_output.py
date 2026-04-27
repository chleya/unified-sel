#!/usr/bin/env python
"""快速检查基准测试输出结构"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.capability_benchmark import run_capability_benchmark

print("=== 检查基准测试输出结构 ===")

result = run_capability_benchmark(
    suite="code",
    protocol="monitor_gate",
    num_tasks=5,
    seed=7,
    local_solver_name="search",
    routing_monitor_name="semantic",
    routing_signal_threshold=0.5,
    escalation_signal_threshold=0.9,
)

print(f"\n协议: {result['protocol']}")
print(f"监控器: {result['routing_monitor_name']}")
print(f"\nSummary 字段: {list(result['summary'].keys())}")
print(f"\n单个任务结果字段: {list(result['results'][0].keys())}")

print("\n=== 完整summary ===")
import json
print(json.dumps(result['summary'], indent=2, ensure_ascii=False))
