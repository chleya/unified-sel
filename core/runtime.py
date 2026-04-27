"""
core/runtime.py — 路径和结果管理

统一管理结果文件路径、JSON 保存、历史记录格式化。
来源：SEL-Lab core/runtime.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def get_results_path(experiment_name: str) -> Path:
    """返回实验结果目录，自动创建。"""
    p = RESULTS_DIR / experiment_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Path) -> None:
    """保存 JSON 文件，自动处理 numpy 类型。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    """加载 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    """处理 numpy 类型的 JSON 序列化。"""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def timestamp() -> str:
    """返回当前时间戳字符串，用于文件命名。"""
    return time.strftime("%Y%m%d_%H%M%S")


def summarize_runs(runs: List[Dict], key: str = "accuracy") -> Dict:
    """
    汇总多次运行的结果。

    参数：
        runs: 每次运行的结果字典列表
        key: 要汇总的指标名

    返回：
        {mean, std, min, max, n}
    """
    import numpy as np
    values = [r[key] for r in runs if key in r]
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def get_seed_cache_path(experiment_name: str, seed: int, cache_prefix: str | None = None) -> Path:
    """
    返回种子缓存文件的路径。
    
    参数：
        experiment_name: 实验名称
        seed: 种子编号
        cache_prefix: 缓存前缀，用于区分不同的实验配置，
                     不传则只按种子缓存，传入则可以区分不同配置的同种子缓存
    
    返回：
        缓存文件的完整路径
    """
    cache_dir = get_results_path(experiment_name) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if cache_prefix:
        return cache_dir / f"{cache_prefix}_seed_{seed}.json"
    return cache_dir / f"seed_{seed}.json"


def load_seed_cache(experiment_name: str, seed: int, cache_prefix: str | None = None) -> Dict | None:
    """
    加载种子缓存结果。
    
    参数：
        experiment_name: 实验名称
        seed: 种子编号
        cache_prefix: 缓存前缀
    
    返回：
        缓存结果字典，或 None（缓存不存在）
    """
    cache_path = get_seed_cache_path(experiment_name, seed, cache_prefix)
    if cache_path.exists():
        try:
            return load_json(cache_path)
        except Exception:
            return None
    return None


def save_seed_cache(data: Dict, experiment_name: str, seed: int, cache_prefix: str | None = None) -> None:
    """
    保存种子缓存结果。
    
    参数：
        data: 要保存的结果字典
        experiment_name: 实验名称
        seed: 种子编号
        cache_prefix: 缓存前缀
    """
    cache_path = get_seed_cache_path(experiment_name, seed, cache_prefix)
    save_json(data, cache_path)


def clear_expired_seed_cache(experiment_name: str, max_age_days: int = 7) -> int:
    """
    清理过期的种子缓存。
    
    参数：
        experiment_name: 实验名称
        max_age_days: 最大缓存天数
    
    返回：
        清理的缓存文件数量
    """
    import time
    cache_dir = get_results_path(experiment_name) / "cache"
    if not cache_dir.exists():
        return 0
    
    cleaned_count = 0
    current_time = time.time()
    max_age_seconds = max_age_days * 86400
    
    for cache_file in cache_dir.glob("*.json"):
        if current_time - cache_file.stat().st_mtime > max_age_seconds:
            cache_file.unlink()
            cleaned_count += 1
    
    return cleaned_count
