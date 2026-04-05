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
