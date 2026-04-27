"""
weight_graph/utils.py — 工具函数
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(data: Dict[str, Any], path: Path) -> None:
    """保存分析结果为 JSON。处理 numpy 类型。"""
    import json
    from dataclasses import asdict

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return _sanitize(asdict(obj))
        return obj

    text = json.dumps(_sanitize(data), ensure_ascii=False, indent=2)
    Path(path).write_text(text, encoding="utf-8")


def normalize_weights(W: np.ndarray, method: str = "abs") -> np.ndarray:
    """
    归一化权重矩阵。

    方法：
    - "abs": 取绝对值（默认）
    - "minmax": 归一化到 [0, 1]
    - "zscore": 标准化到 mean=0, std=1 然后取绝对值
    """
    if method == "abs":
        return np.abs(W)
    elif method == "minmax":
        w = np.abs(W)
        wmin, wmax = w.min(), w.max()
        if wmax - wmin < 1e-10:
            return np.ones_like(w)
        return (w - wmin) / (wmax - wmin)
    elif method == "zscore":
        w = np.abs(W)
        mean, std = w.mean(), w.std()
        if std < 1e-10:
            return np.ones_like(w)
        return np.abs((w - mean) / std)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def sparsify_percentile(W: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """保留 top percentile% 的权重（按绝对值），其余置零。"""
    threshold = np.percentile(np.abs(W), percentile)
    mask = np.abs(W) >= threshold
    return W * mask


def sparsify_topk(W: np.ndarray, k: int = 32) -> np.ndarray:
    """每行保留 top-k 个绝对值最大的权重。"""
    result = np.zeros_like(W)
    for i in range(W.shape[0]):
        row = np.abs(W[i, :])
        if k >= len(row):
            result[i, :] = W[i, :]
        else:
            idx = np.argsort(row)[-k:]
            result[i, idx] = W[i, idx]
    return result


def sparsify_sigma(W: np.ndarray, n_sigma: float = 2.0) -> np.ndarray:
    """保留 mean + n_sigma * std 以上的权重。"""
    w_abs = np.abs(W)
    threshold = w_abs.mean() + n_sigma * w_abs.std()
    mask = w_abs >= threshold
    return W * mask
