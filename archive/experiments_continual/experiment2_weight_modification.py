"""
实验 2：结构修改质量分析

核心假设：种子 7 的成功源于"保护性学习"（复用结构时权重变化小），
而种子 8/9 的失败源于"破坏性学习"（复用结构时权重大幅覆盖）。

方法：
1. 在 checkpoint (step 200) 记录所有 Phase 0 结构的权重快照
2. 在实验结束 (step 600) 记录最终权重
3. 计算每个复用结构的相对权重变化：||W_final - W_ckpt|| / ||W_ckpt||
4. 对比成功 vs 失败种子的权重修改模式
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_config import NoBoundaryConfig
from core.learner import UnifiedSELClassifier
from experiments.continual.no_boundary import make_eval_task, stream_sample


@dataclass
class WeightModificationTrace:
    """单个结构的权重修改追踪。"""
    id: int
    created_at_step: int
    is_phase_0: bool
    
    # 权重快照
    w_checkpoint: Optional[np.ndarray] = field(default=None, repr=False)
    w_final: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Phase 1 行为
    phase_1_activations: int = 0
    phase_1_learn_steps: int = 0
    total_phase_1_loss: float = 0.0
    
    # 计算指标
    weight_change_norm: float = 0.0
    relative_weight_change: float = 0.0
    
    def compute_metrics(self):
        if self.w_checkpoint is None or self.w_final is None:
            return
        
        self.weight_change_norm = float(np.linalg.norm(self.w_final - self.w_checkpoint))
        norm_ckpt = np.linalg.norm(self.w_checkpoint) + 1e-8
        self.relative_weight_change = self.weight_change_norm / norm_ckpt


@dataclass
class WeightAnalysisResult:
    """单个种子的权重修改分析结果。"""
    seed: int
    forgetting: float
    task_0_acc: float
    task_1_acc: float
    
    # 结构级追踪
    structure_traces: Dict[int, WeightModificationTrace] = field(default_factory=dict)
    
    @property
    def phase_0_reused_structures(self) -> List[WeightModificationTrace]:
        """在 Phase 0 创建且在 Phase 1 被激活的结构。"""
        return [
            s for s in self.structure_traces.values()
            if s.is_phase_0 and s.phase_1_activations > 0 and s.w_checkpoint is not None
        ]
    
    @property
    def mean_relative_change(self) -> float:
        reused = self.phase_0_reused_structures
        if not reused:
            return 0.0
        return float(np.mean([s.relative_weight_change for s in reused]))
    
    @property
    def max_relative_change(self) -> float:
        reused = self.phase_0_reused_structures
        if not reused:
            return 0.0
        return float(max(s.relative_weight_change for s in reused))
    
    @property
    def low_change_ratio(self) -> float:
        """相对变化 < 0.5 的结构比例（保护性学习）。"""
        reused = self.phase_0_reused_structures
        if not reused:
            return 0.0
        return sum(1 for s in reused if s.relative_weight_change < 0.5) / len(reused)
    
    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "forgetting": self.forgetting,
            "task_0_acc": self.task_0_acc,
            "task_1_acc": self.task_1_acc,
            "n_phase_0_structures": sum(1 for s in self.structure_traces.values() if s.is_phase_0),
            "n_phase_0_reused": len(self.phase_0_reused_structures),
            "mean_relative_change": self.mean_relative_change,
            "max_relative_change": self.max_relative_change,
            "low_change_ratio": self.low_change_ratio,
            "structure_details": {
                str(s.id): {
                    "created_at": s.created_at_step,
                    "phase_1_activations": s.phase_1_activations,
                    "relative_weight_change": s.relative_weight_change,
                }
                for s in self.phase_0_reused_structures
            }
        }


def run_weight_analysis(seed: int, config: NoBoundaryConfig) -> WeightAnalysisResult:
    """运行单个种子的权重修改分析。"""
    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        seed=seed,
        pool_config=config.pool.to_pool_kwargs(),
    )
    
    structure_traces: Dict[int, WeightModificationTrace] = {}
    
    # 钩子 1: 追踪新结构创建
    orig_new = clf.pool._new_structure
    def tracked_new(label: str = "") -> object:
        structure = orig_new(label)
        sid = structure.id
        step = clf.pool.step_count - 1
        structure_traces[sid] = WeightModificationTrace(
            id=sid,
            created_at_step=step,
            is_phase_0=step < config.checkpoint_step,
        )
        return structure
    clf.pool._new_structure = tracked_new
    
    # 初始化已存在的结构
    for s in clf.pool.structures:
        structure_traces[s.id] = WeightModificationTrace(
            id=s.id,
            created_at_step=0,
            is_phase_0=True,
        )
    
    # 钩子 2: 追踪 Phase 1 激活和学习
    orig_observe = clf.pool.observe
    orig_learn = clf.pool.learn_active
    
    def tracked_observe(x: np.ndarray) -> dict:
        result = orig_observe(x)
        step = clf.pool.step_count - 1
        if step >= config.checkpoint_step:
            sid = result["active_structure"].id
            if sid in structure_traces:
                structure_traces[sid].phase_1_activations += 1
        return result
    
    def tracked_learn(active_structure, x, output_error, lr=0.05):
        step = clf.pool.step_count - 1
        if step >= config.checkpoint_step:
            sid = active_structure.id
            if sid in structure_traces:
                structure_traces[sid].phase_1_learn_steps += 1
                structure_traces[sid].total_phase_1_loss += float(np.mean(output_error**2))
        return orig_learn(active_structure, x, output_error, lr)
    
    clf.pool.observe = tracked_observe
    clf.pool.learn_active = tracked_learn
    
    # 运行实验
    rng = np.random.default_rng(seed)
    X_task_0, y_task_0 = make_eval_task(0, config.eval_samples_per_task, seed + 1000, config.in_size)
    X_task_1, y_task_1 = make_eval_task(1, config.eval_samples_per_task, seed + 2000, config.in_size)
    
    task_0_acc_at_checkpoint = 0.0
    
    for step in range(config.steps):
        progress = step / max(config.steps - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)
        
        # Checkpoint 快照
        if step + 1 == config.checkpoint_step:
            task_0_acc_at_checkpoint = clf.accuracy(X_task_0, y_task_0)
            for s in clf.pool.structures:
                if s.id in structure_traces:
                    structure_traces[s.id].w_checkpoint = s.weights.copy()
    
    # 最终快照
    for s in clf.pool.structures:
        if s.id in structure_traces:
            structure_traces[s.id].w_final = s.weights.copy()
    
    # 计算指标
    for trace in structure_traces.values():
        trace.compute_metrics()
    
    # 最终评估
    task_0_final = clf.accuracy(X_task_0, y_task_0)
    task_1_final = clf.accuracy(X_task_1, y_task_1)
    forgetting = task_0_acc_at_checkpoint - task_0_final
    
    return WeightAnalysisResult(
        seed=seed,
        forgetting=forgetting,
        task_0_acc=task_0_final,
        task_1_acc=task_1_final,
        structure_traces=structure_traces,
    )


def print_weight_comparison(results: List[WeightAnalysisResult]):
    """打印权重修改对比报告。"""
    print("\n" + "=" * 85)
    print("实验 2：结构修改质量分析")
    print("=" * 85)
    
    print(f"\n{'种子':<8} {'遗忘率':<10} {'P0复用数':<9} {'平均相对变化':<13} {'最大相对变化':<13} {'保护性学习比例':<14}")
    print("-" * 85)
    
    for r in sorted(results, key=lambda x: x.forgetting):
        status = "✅" if r.forgetting < 0.1 else "❌"
        print(f"{status} {r.seed:<6} {r.forgetting:<10.4f} {len(r.phase_0_reused_structures):<9} "
              f"{r.mean_relative_change:<13.4f} {r.max_relative_change:<13.4f} {r.low_change_ratio:<14.3f}")
    
    print("-" * 85)
    
    success = [r for r in results if r.forgetting < 0.1]
    failure = [r for r in results if r.forgetting >= 0.1]
    
    if success and failure:
        s_mean = np.mean([r.mean_relative_change for r in success])
        f_mean = np.mean([r.mean_relative_change for r in failure])
        s_low = np.mean([r.low_change_ratio for r in success])
        f_low = np.mean([r.low_change_ratio for r in failure])
        
        print(f"\n🔑 关键发现:")
        print(f"  成功种子平均相对变化: {s_mean:.4f}")
        print(f"  失败种子平均相对变化: {f_mean:.4f}")
        
        if s_mean < f_mean:
            print(f"  ✅ 成功种子修改幅度更小 (差异: {f_mean - s_mean:.4f}) → 支持'保护性学习'假设")
        else:
            print(f"  ⚠️  成功种子修改幅度更大 → 证伪'保护性学习'假设")
            
        print(f"\n  成功种子保护性学习比例: {s_low:.3f}")
        print(f"  失败种子保护性学习比例: {f_low:.3f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()
    
    config = NoBoundaryConfig()
    config.seeds = args.seeds
    
    print("=" * 85)
    print("Unified-SEL 实验 2：结构修改质量分析")
    print("=" * 85)
    print(f"种子: {args.seeds}")
    print(f"检查点步数: {config.checkpoint_step}")
    print()
    
    results = []
    for seed in args.seeds:
        print(f"运行种子 {seed}...")
        t0 = time.time()
        result = run_weight_analysis(seed, config)
        elapsed = time.time() - t0
        results.append(result)
        print(f"  遗忘率: {result.forgetting:.4f}")
        print(f"  平均相对变化: {result.mean_relative_change:.4f}")
        print(f"  耗时: {elapsed:.1f}s")
        print()
    
    print_weight_comparison(results)
    
    # 保存结果
    output = {
        "experiment": "weight_modification_analysis",
        "config": config.to_dict(),
        "per_seed": [r.to_dict() for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    from core.runtime import get_results_path, save_json, timestamp
    results_dir = get_results_path("weight_analysis")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
