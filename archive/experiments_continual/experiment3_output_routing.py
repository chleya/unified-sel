"""
实验 3：输出层权重与激活路由分析

核心假设：遗忘不仅来自结构权重修改，还来自输出层 (W_out) 干扰和路由混乱。
种子 7 成功是因为学会了任务特定的路由，而 8/9 失败了。

方法：
1. 训练 Task 0 专用模型，得到 W_out_task0
2. 对比持续学习后的 W_out 与 W_out_task0 的余弦相似度
3. 分析 Task 0 和 Task 1 测试时各结构的激活频率分布
4. 计算任务特异性指数 TSI = |freq_0 - freq_1| / (freq_0 + freq_1)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_config import NoBoundaryConfig
from core.learner import UnifiedSELClassifier
from experiments.continual.no_boundary import make_eval_task, stream_sample


@dataclass
class RoutingResult:
    """单个种子的路由分析结果。"""
    seed: int
    forgetting: float
    
    # W_out 分析
    w_out_final_norm: float
    w_out_task0_norm: float
    w_out_cosine_similarity: float  # 与 Task 0 专用 W_out 的相似度
    
    # 路由分析
    task0_activation_counts: Dict[int, int] = field(default_factory=dict)
    task1_activation_counts: Dict[int, int] = field(default_factory=dict)
    task_specificity_index: float = 0.0  # 平均 TSI
    
    # 解释
    interpretation: str = ""
    
    def compute_tsi(self):
        if not self.task0_activation_counts and not self.task1_activation_counts:
            self.task_specificity_index = 0.0
            return
            
        all_ids = set(self.task0_activation_counts.keys()) | set(self.task1_activation_counts.keys())
        tsi_values = []
        for sid in all_ids:
            f0 = self.task0_activation_counts.get(sid, 0)
            f1 = self.task1_activation_counts.get(sid, 0)
            total = f0 + f1
            if total > 0:
                tsi_values.append(abs(f0 - f1) / total)
            else:
                tsi_values.append(0.0)
        self.task_specificity_index = float(np.mean(tsi_values)) if tsi_values else 0.0


def train_task0_only(seed: int, config: NoBoundaryConfig) -> Tuple[np.ndarray, Dict[int, int]]:
    """训练 Task 0 专用模型，返回 W_out 和结构激活计数。"""
    rng = np.random.default_rng(seed)
    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        seed=seed,
        pool_config=config.pool.to_pool_kwargs(),
    )
    
    # 追踪结构激活
    structure_counts: Dict[int, int] = {}
    orig_observe = clf.pool.observe
    def tracked_observe(x: np.ndarray) -> dict:
        res = orig_observe(x)
        sid = res["active_structure"].id
        structure_counts[sid] = structure_counts.get(sid, 0) + 1
        return res
    clf.pool.observe = tracked_observe
    
    # 只在 Task 0 数据上训练
    X, y = make_task(0, config.steps, seed + 1000, config.in_size)
    for i in range(len(X)):
        clf.fit_one(X[i], y[i])
        
    return clf.W_out, structure_counts


def run_routing_analysis(seed: int, config: NoBoundaryConfig, w_out_task0: np.ndarray) -> RoutingResult:
    """运行单个种子的路由分析。"""
    # 运行持续学习
    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        seed=seed,
        pool_config=config.pool.to_pool_kwargs(),
    )
    
    # 追踪 Task 0 和 Task 1 的激活
    task0_counts: Dict[int, int] = {}
    task1_counts: Dict[int, int] = {}
    
    orig_observe = clf.pool.observe
    def tracked_observe(x: np.ndarray) -> dict:
        res = orig_observe(x)
        sid = res["active_structure"].id
        # 这里我们不确知当前是 Task 0 还是 Task 1，
        # 但我们可以在 evaluate 时单独追踪
        return res
    
    clf.pool.observe = tracked_observe
    
    rng = np.random.default_rng(seed)
    X_task_0, y_task_0 = make_task(0, config.eval_samples_per_task, seed + 1000, config.in_size)
    X_task_1, y_task_1 = make_task(1, config.eval_samples_per_task, seed + 2000, config.in_size)
    
    task_0_acc_at_checkpoint = 0.0
    
    for step in range(config.steps):
        progress = step / max(config.steps - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)
        if step + 1 == config.checkpoint_step:
            task_0_acc_at_checkpoint = clf.accuracy(X_task_0, y_task_0)
            
    task_0_final = clf.accuracy(X_task_0, y_task_0)
    forgetting = task_0_acc_at_checkpoint - task_0_final
    
    # W_out 分析
    w_final = clf.W_out
    norm_final = np.linalg.norm(w_final)
    norm_task0 = np.linalg.norm(w_out_task0)
    
    # 余弦相似度
    cos_sim = float(np.dot(w_final.flatten(), w_out_task0.flatten()) / (norm_final * norm_task0 + 1e-8))
    
    # 路由分析：通过输入 Task 0 和 Task 1 的测试数据来追踪激活
    # 临时重置计数
    clf.pool._test_counts_task0 = {}
    clf.pool._test_counts_task1 = {}
    
    def test_tracked_observe(x: np.ndarray, task_id: int) -> dict:
        res = orig_observe(x)
        sid = res["active_structure"].id
        if task_id == 0:
            clf.pool._test_counts_task0[sid] = clf.pool._test_counts_task0.get(sid, 0) + 1
        else:
            clf.pool._test_counts_task1[sid] = clf.pool._test_counts_task1.get(sid, 0) + 1
        return res

    # 注入测试
    original_observe_impl = clf.pool.observe
    for x in X_task_0:
        clf.pool.observe = lambda x=x: test_tracked_observe(x, 0)
        clf.predict(x)  # 触发一次 observe
    for x in X_task_1:
        clf.pool.observe = lambda x=x: test_tracked_observe(x, 1)
        clf.predict(x)
    clf.pool.observe = original_observe_impl
    
    res = RoutingResult(
        seed=seed,
        forgetting=forgetting,
        w_out_final_norm=norm_final,
        w_out_task0_norm=norm_task0,
        w_out_cosine_similarity=cos_sim,
        task0_activation_counts=clf.pool._test_counts_task0,
        task1_activation_counts=clf.pool._test_counts_task1,
    )
    res.compute_tsi()
    
    # 解释
    if res.w_out_cosine_similarity > 0.8 and res.task_specificity_index > 0.4:
        res.interpretation = "高 W_out 对齐 + 高任务特异性 → 成功机制"
    elif res.w_out_cosine_similarity < 0.5:
        res.interpretation = "W_out 严重漂移 → 输出层干扰"
    elif res.task_specificity_index < 0.3:
        res.interpretation = "任务特异性低 → 路由混乱/冲突"
    else:
        res.interpretation = "混合机制"
        
    return res


def make_task(task_id: int, n_samples: int, seed: int, in_size: int):
    """辅助函数：定义在 no_boundary.py 中但这里需要独立调用。"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, in_size))
    boundary = X[:, 0] + X[:, 1]
    if task_id == 0:
        y = (boundary > 0.0).astype(int)
    else:
        y = (boundary < 0.0).astype(int)
    return X, y


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()
    
    config = NoBoundaryConfig()
    
    print("=" * 85)
    print("实验 3：输出层权重与激活路由分析")
    print("=" * 85)
    
    # 1. 先训练 Task 0 专用基准
    print("\n训练 Task 0 专用基准模型...")
    w_out_t0, _ = train_task0_only(seed=args.seeds[0], config=config)
    print("  完成。")
    
    results = []
    for seed in args.seeds:
        print(f"\n运行种子 {seed}...")
        t0 = time.time()
        # 注意：这里的 Task 0 基准用的是同一个种子的 Task 0 模型更公平
        w_t0, _ = train_task0_only(seed=seed, config=config)
        result = run_routing_analysis(seed=seed, config=config, w_out_task0=w_t0)
        results.append(result)
        print(f"  遗忘率: {result.forgetting:.4f}")
        print(f"  W_out 余弦相似度: {result.w_out_cosine_similarity:.4f}")
        print(f"  任务特异性指数 (TSI): {result.task_specificity_index:.4f}")
        print(f"  解释: {result.interpretation}")
        print(f"  耗时: {time.time() - t0:.1f}s")
        
    # 报告
    print("\n" + "-" * 85)
    print(f"{'种子':<8} {'遗忘率':<10} {'W_out 相似度':<12} {'TSI':<10} {'结论'}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: x.forgetting):
        status = "✅" if r.forgetting < 0.1 else "❌"
        print(f"{status} {r.seed:<6} {r.forgetting:<10.4f} {r.w_out_cosine_similarity:<12.4f} {r.task_specificity_index:<10.4f} {r.interpretation}")
        
    # 保存
    output = {
        "experiment": "routing_analysis",
        "per_seed": [
            {
                "seed": r.seed,
                "forgetting": r.forgetting,
                "w_out_cosine_similarity": r.w_out_cosine_similarity,
                "tsi": r.task_specificity_index,
                "interpretation": r.interpretation,
            }
            for r in results
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    from core.runtime import get_results_path, save_json, timestamp
    results_dir = get_results_path("routing_analysis")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
