"""
core/experiment_utils.py — 实验工具模块

提供种子级缓存、多进程并行等通用实验工具。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")


def _seed_wrapper_dispatch(args):
    """Dispatch wrapper for multiprocessing pickling compatibility.
    
    This needs to be top-level, so we dispatch by argument type.
    """
    import sys
    import inspect
    
    if len(args) == 5 and callable(args[0]):
        seed_fn, seed, experiment_name, cache_prefix, force_rerun = args
        result, cached = run_seed_with_cache(seed_fn, seed, experiment_name, cache_prefix, force_rerun)
        return seed, result, cached
    
    elif len(args) == 6 and isinstance(args[5], dict):
        # Case: no_boundary_cached experiment
        import importlib
        from pathlib import Path
        config, window_size, ewc_lambda, seed, experiment_name, cache_prefix = args
        
        # Import the module dynamically
        module_name = "experiments.continual.no_boundary_cached"
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                module_name,
                Path(__file__).parents[1] / "experiments" / "continual" / "no_boundary_cached.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            module = sys.modules[module_name]
        
        run_seed_fn = module.run_seed
        result, cached = run_seed_with_cache(
            lambda s: run_seed_fn(s, config, window_size, ewc_lambda),
            seed,
            experiment_name,
            cache_prefix
        )
        return seed, result, cached
    
    else:
        raise ValueError(f"Unknown arguments to _seed_wrapper_dispatch: {args}")


def config_to_hash_prefix(config: Any, exclude_fields: Optional[List[str]] = None) -> str:
    """
    将配置对象转换为短哈希前缀，用于缓存区分不同配置。
    
    参数：
        config: 配置对象，可以是dataclass或者字典
        exclude_fields: 不参与哈希的字段列表
    
    返回：
        8字符的十六进制哈希字符串
    """
    if exclude_fields is None:
        exclude_fields = []
    
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "__dict__"):
        config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else config.__dict__.copy()
    else:
        config_dict = dict(config)
    
    for field in exclude_fields:
        if field in config_dict:
            del config_dict[field]
    
    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_json.encode()).hexdigest()[:8]


def run_seed_with_cache(
    seed_fn: Callable[[int], T],
    seed: int,
    experiment_name: str,
    cache_prefix: Optional[str] = None,
    force_rerun: bool = False,
) -> Tuple[T, bool]:
    """
    带缓存的单种子运行。
    
    参数：
        seed_fn: 单种子运行函数，参数是seed，返回结果
        seed: 种子编号
        experiment_name: 实验名称
        cache_prefix: 缓存前缀，用于区分不同配置
        force_rerun: 强制重新运行，忽略缓存
    
    返回：
        (result, from_cache) — 结果和是否来自缓存
    """
    from core.runtime import load_seed_cache, save_seed_cache
    
    if not force_rerun:
        cached = load_seed_cache(experiment_name, seed, cache_prefix)
        if cached is not None:
            return cached, True
    
    result = seed_fn(seed)
    save_seed_cache(result, experiment_name, seed, cache_prefix)
    return result, False


def run_multiple_seeds_with_cache(
    seed_fn: Callable[[int], T],
    seeds: List[int],
    experiment_name: str,
    cache_prefix: Optional[str] = None,
    force_rerun: bool = False,
    num_workers: Optional[int] = None,
    parallel: bool = False,  # Windows下默认不开启并行，避免pickle问题
) -> Tuple[List[T], int]:
    """
    带缓存的多种子运行（支持并行）。
    
    参数：
        seed_fn: 单种子运行函数
        seeds: 种子编号列表
        experiment_name: 实验名称
        cache_prefix: 缓存前缀
        force_rerun: 强制重新运行
        num_workers: 并行进程数，默认是CPU核心数-1
        parallel: 是否开启多进程并行（Windows下建议关闭，避免pickle问题）
    
    返回：
        (results, num_cached) — 结果列表和来自缓存的数量
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    num_cached = 0
    results: List[T] = []
    
    if not parallel or num_workers == 1 or len(seeds) == 1:
        for seed in seeds:
            result, cached = run_seed_with_cache(seed_fn, seed, experiment_name, cache_prefix, force_rerun)
            results.append(result)
            if cached:
                num_cached += 1
    else:
        # 并行模式，Windows下建议谨慎使用
        task_args = [(seed_fn, seed, experiment_name, cache_prefix, force_rerun) for seed in seeds]
        with Pool(processes=num_workers) as pool:
            outputs = pool.map(_seed_wrapper_dispatch, task_args)
        
        outputs_sorted = sorted(outputs, key=lambda x: seeds.index(x[0]))
        for _, result, cached in outputs_sorted:
            results.append(result)
            if cached:
                num_cached += 1
    
    return results, num_cached
