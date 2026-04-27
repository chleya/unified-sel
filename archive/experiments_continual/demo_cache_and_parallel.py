"""
demo_cache_and_parallel.py — 新机制快速入门演示

演示：
1. 种子级缓存机制的使用
2. 多进程并行优化的使用
3. 缓存如何避免重复运行
"""

from __future__ import annotations

import time
from typing import Dict, List

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_utils import run_seed_with_cache, run_multiple_seeds_with_cache, config_to_hash_prefix
from core.experiment_config import NoBoundaryConfig


def dummy_seed_fn(seed: int) -> Dict:
    """模拟一个耗时的单种子实验"""
    print(f"[RUN] 开始运行种子 {seed}...")
    time.sleep(1.0)  # 模拟1秒的计算时间
    accuracy = 0.5 + seed * 0.02
    result = {
        "seed": seed,
        "accuracy": accuracy,
        "forgetting": 0.1 - seed * 0.01,
        "run_time": 1.0
    }
    print(f"[DONE] 种子 {seed} 完成，准确率 {accuracy:.3f}")
    return result


def demo_seed_cache():
    """演示种子级缓存"""
    print("\n" + "=" * 60)
    print("演示1: 种子级缓存机制")
    print("=" * 60)
    
    experiment_name = "demo_cache_test"
    
    # 第一次运行 - 没有缓存，需要运行
    print("\n--- 第一次运行 ---")
    result1, cached1 = run_seed_with_cache(
        dummy_seed_fn,
        seed=42,
        experiment_name=experiment_name,
        force_rerun=False
    )
    print(f"结果来自缓存: {cached1}")
    
    # 第二次运行 - 应该从缓存加载
    print("\n--- 第二次运行 ---")
    result2, cached2 = run_seed_with_cache(
        dummy_seed_fn,
        seed=42,
        experiment_name=experiment_name,
        force_rerun=False
    )
    print(f"结果来自缓存: {cached2}")
    
    # 验证结果一致
    assert result1 == result2
    print("\n[OK] 种子级缓存工作正常！")


def demo_multiple_seeds_with_cache_and_parallel():
    """演示带缓存的多种子并行"""
    print("\n" + "=" * 60)
    print("演示2: 带缓存的多种子并行运行")
    print("=" * 60)
    
    experiment_name = "demo_parallel_test"
    seeds = [7, 8, 9, 10, 11]  # 5个种子，和真实实验一样
    
    # 配置哈希
    config = NoBoundaryConfig()
    cache_prefix = config_to_hash_prefix(config, exclude_fields=["seeds"])
    print(f"\n配置哈希前缀: {cache_prefix}")
    
    # 第一次运行 - 并行运行5个种子
    print("\n--- 第一次并行运行 ---")
    start_time = time.time()
    results1, num_cached1 = run_multiple_seeds_with_cache(
        dummy_seed_fn,
        seeds=seeds,
        experiment_name=experiment_name,
        cache_prefix=cache_prefix,
        num_workers=5  # 6核CPU，跑5个并行
    )
    elapsed1 = time.time() - start_time
    
    print(f"\n结果统计:")
    print(f"  总耗时: {elapsed1:.2f}秒")
    print(f"  来自缓存: {num_cached1}个")
    print(f"  运行的: {len(results1) - num_cached1}个")
    
    # 第二次运行 - 应该全部来自缓存
    print("\n--- 第二次并行运行 ---")
    start_time = time.time()
    results2, num_cached2 = run_multiple_seeds_with_cache(
        dummy_seed_fn,
        seeds=seeds,
        experiment_name=experiment_name,
        cache_prefix=cache_prefix,
        num_workers=5
    )
    elapsed2 = time.time() - start_time
    
    print(f"\n结果统计:")
    print(f"  总耗时: {elapsed2:.2f}秒")
    print(f"  来自缓存: {num_cached2}个")
    print(f"  运行的: {len(results2) - num_cached2}个")
    
    print("\n[OK] 带缓存的多种子并行工作正常！")
    print(f"   第一次: 5个种子并行，耗时 ~1秒")
    print(f"   第二次: 全部来自缓存，耗时 ~0秒")


def demo_config_hash():
    """演示配置哈希前缀"""
    print("\n" + "=" * 60)
    print("演示3: 配置哈希前缀")
    print("=" * 60)
    
    config1 = NoBoundaryConfig()
    config2 = NoBoundaryConfig()
    config2.pool.max_structures = 20  # 修改一个参数
    
    hash1 = config_to_hash_prefix(config1)
    hash2 = config_to_hash_prefix(config2)
    
    print(f"\n默认配置哈希: {hash1}")
    print(f"修改后配置哈希: {hash2}")
    print(f"配置不同，哈希不同: {hash1 != hash2}")
    
    hash1_with_exclude = config_to_hash_prefix(config1, exclude_fields=["seeds"])
    hash2_with_exclude = config_to_hash_prefix(config2, exclude_fields=["seeds", "pool"])
    print(f"\n排除部分字段后的哈希: {hash1_with_exclude}")
    print(f"排除更多字段后的哈希: {hash2_with_exclude}")
    
    print("\n[OK] 配置哈希工作正常！")


def main():
    print("\n" + "=" * 60)
    print("新机制快速入门演示")
    print("Unified-SEL 实验缓存 + 并行优化")
    print("=" * 60)
    
    demo_seed_cache()
    demo_multiple_seeds_with_cache_and_parallel()
    demo_config_hash()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("\n如何在真实实验中使用：")
    print("1. 使用 run_seed_with_cache 包装单种子运行")
    print("2. 使用 run_multiple_seeds_with_cache 运行多种子并行")
    print("3. 配置用 config_to_hash_prefix 生成前缀，不同配置不会冲突")
    print("=" * 60)


if __name__ == "__main__":
    main()
