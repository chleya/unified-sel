"""
实验 1：结构复用模式分析

核心问题：种子 7 是否复用了更多 Phase 0 的旧结构来处理 Task 1？

方法：
1. 追踪每个结构的创建时间和最后激活时间
2. 计算 Phase 0 结构（<200 步创建）在 Phase 1（≥200 步）的激活比例
3. 对比成功种子 vs 失败种子的复用模式
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_config import NoBoundaryConfig
from core.learner import UnifiedSELClassifier
from experiments.continual.no_boundary import make_eval_task, stream_sample


@dataclass
class StructureReuseTrace:
    """单个结构的完整生命周期追踪。"""
    id: int
    created_at_step: int
    is_phase_0: bool  # True if created before checkpoint_step
    
    # 激活记录
    activation_steps: List[int] = field(default_factory=list)
    activations_in_phase_0: int = 0
    activations_in_phase_1: int = 0
    
    # 学习记录
    total_loss: float = 0.0
    n_learn_steps: int = 0
    
    # 状态变化
    utility_history: List[float] = field(default_factory=list)
    tension_history: List[float] = field(default_factory=list)
    
    # 最终状态
    died_at_step: Optional[int] = None
    final_utility: float = 0.0
    
    @property
    def reuse_ratio(self) -> float:
        """Phase 1 激活占总激活的比例。"""
        total = self.activations_in_phase_0 + self.activations_in_phase_1
        return self.activations_in_phase_1 / total if total > 0 else 0.0


@dataclass
class ReuseAnalysisResult:
    """单个种子的复用分析结果。"""
    seed: int
    forgetting: float
    task_0_acc: float
    task_1_acc: float
    checkpoint_step: int
    total_steps: int
    
    # 结构级追踪
    structure_traces: Dict[int, StructureReuseTrace] = field(default_factory=dict)
    
    # 汇总统计
    @property
    def phase_0_structures(self) -> List[StructureReuseTrace]:
        return [s for s in self.structure_traces.values() if s.is_phase_0]
    
    @property
    def phase_1_structures(self) -> List[StructureReuseTrace]:
        return [s for s in self.structure_traces.values() if not s.is_phase_0]
    
    @property
    def phase_0_reuse_ratio(self) -> float:
        """Phase 0 结构在 Phase 1 的激活比例。"""
        phase_0 = self.phase_0_structures
        if not phase_0:
            return 0.0
        
        total_phase_1_activations = sum(
            s.activations_in_phase_1 for s in self.structure_traces.values()
        )
        phase_0_in_phase_1 = sum(
            s.activations_in_phase_1 for s in phase_0
        )
        
        return phase_0_in_phase_1 / total_phase_1_activations if total_phase_1_activations > 0 else 0.0
    
    @property
    def phase_0_survival_rate(self) -> float:
        """Phase 0 结构的存活率。"""
        phase_0 = self.phase_0_structures
        if not phase_0:
            return 0.0
        survived = sum(1 for s in phase_0 if s.died_at_step is None)
        return survived / len(phase_0)
    
    @property
    def phase_0_avg_reuse(self) -> float:
        """Phase 0 结构的平均复用比例。"""
        ratios = [s.reuse_ratio for s in self.phase_0_structures]
        return float(np.mean(ratios)) if ratios else 0.0
    
    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "forgetting": self.forgetting,
            "task_0_acc": self.task_0_acc,
            "task_1_acc": self.task_1_acc,
            "n_structures": len(self.structure_traces),
            "n_phase_0_structures": len(self.phase_0_structures),
            "n_phase_1_structures": len(self.phase_1_structures),
            "phase_0_reuse_ratio": self.phase_0_reuse_ratio,
            "phase_0_survival_rate": self.phase_0_survival_rate,
            "phase_0_avg_reuse": self.phase_0_avg_reuse,
            "structure_details": {
                str(sid): {
                    "created_at": trace.created_at_step,
                    "is_phase_0": trace.is_phase_0,
                    "activations_phase_0": trace.activations_in_phase_0,
                    "activations_phase_1": trace.activations_in_phase_1,
                    "reuse_ratio": trace.reuse_ratio,
                    "died_at": trace.died_at_step,
                    "final_utility": trace.final_utility,
                }
                for sid, trace in self.structure_traces.items()
            }
        }


def run_reuse_analysis(seed: int, config: NoBoundaryConfig) -> ReuseAnalysisResult:
    """运行单个种子的结构复用分析。"""
    # 创建自定义分类器，追踪结构激活
    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        seed=seed,
        pool_config=config.pool.to_pool_kwargs(),
    )
    
    # 追踪所有结构
    structure_traces: Dict[int, StructureReuseTrace] = {}
    
    # 包装 observe 方法来追踪激活
    original_observe = clf.pool.observe
    
    def tracked_observe(x: np.ndarray) -> Dict:
        result = original_observe(x)
        active = result["active_structure"]
        sid = active.id
        
        if sid not in structure_traces:
            structure_traces[sid] = StructureReuseTrace(
                id=sid,
                created_at_step=clf.pool.step_count - 1,
                is_phase_0=(clf.pool.step_count - 1) < config.checkpoint_step,
            )
        
        trace = structure_traces[sid]
        step = clf.pool.step_count - 1
        trace.activation_steps.append(step)
        
        if step < config.checkpoint_step:
            trace.activations_in_phase_0 += 1
        else:
            trace.activations_in_phase_1 += 1
        
        trace.utility_history.append(active.utility)
        trace.tension_history.append(active.tension)
        trace.final_utility = active.utility
        
        return result
    
    clf.pool.observe = tracked_observe
    
    # 包装 learn_active 来追踪学习
    original_learn_active = clf.pool.learn_active
    
    def tracked_learn_active(active_structure, x, output_error, lr=0.05):
        loss = original_learn_active(active_structure, x, output_error, lr)
        sid = active_structure.id
        if sid in structure_traces:
            structure_traces[sid].total_loss += loss
            structure_traces[sid].n_learn_steps += 1
        return loss
    
    clf.pool.learn_active = tracked_learn_active
    
    # 包装 prune 来追踪死亡
    original_prune = clf.pool.prune
    
    def tracked_prune() -> int:
        before = {s.id for s in clf.pool.structures}
        result = original_prune()
        after = {s.id for s in clf.pool.structures}
        
        pruned = before - after
        for sid in pruned:
            if sid in structure_traces:
                structure_traces[sid].died_at_step = clf.pool.step_count - 1
        
        return result
    
    clf.pool.prune = tracked_prune
    
    # 运行实验
    rng = np.random.default_rng(seed)
    X_task_0, y_task_0 = make_eval_task(0, config.eval_samples_per_task, seed + 1000, config.in_size)
    X_task_1, y_task_1 = make_eval_task(1, config.eval_samples_per_task, seed + 2000, config.in_size)
    
    task_0_acc_at_checkpoint = 0.0
    
    for step in range(config.steps):
        progress = step / max(config.steps - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)
        
        if step + 1 == config.checkpoint_step:
            task_0_acc_at_checkpoint = clf.accuracy(X_task_0, y_task_0)
    
    # 最终评估
    task_0_final = clf.accuracy(X_task_0, y_task_0)
    task_1_final = clf.accuracy(X_task_1, y_task_1)
    forgetting = task_0_acc_at_checkpoint - task_0_final
    
    return ReuseAnalysisResult(
        seed=seed,
        forgetting=forgetting,
        task_0_acc=task_0_final,
        task_1_acc=task_1_final,
        checkpoint_step=config.checkpoint_step,
        total_steps=config.steps,
        structure_traces=structure_traces,
    )


def print_reuse_comparison(results: List[ReuseAnalysisResult]):
    """打印复用对比报告。"""
    print("\n" + "=" * 80)
    print("实验 1：结构复用模式分析")
    print("=" * 80)
    
    print(f"\n{'种子':<8} {'遗忘率':<10} {'P0结构':<8} {'P1结构':<8} {'P0复用率':<10} {'P0存活率':<10} {'P0平均复用':<10}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x.forgetting):
        status = "✅" if r.forgetting < 0.1 else "❌"
        print(f"{status} {r.seed:<6} {r.forgetting:<10.4f} {len(r.phase_0_structures):<8} "
              f"{len(r.phase_1_structures):<8} {r.phase_0_reuse_ratio:<10.3f} "
              f"{r.phase_0_survival_rate:<10.3f} {r.phase_0_avg_reuse:<10.3f}")
    
    print("\n" + "-" * 80)
    
    # 找出关键差异
    success = [r for r in results if r.forgetting < 0.1]
    failure = [r for r in results if r.forgetting >= 0.1]
    
    if success and failure:
        success_reuse = np.mean([r.phase_0_reuse_ratio for r in success])
        failure_reuse = np.mean([r.phase_0_reuse_ratio for r in failure])
        
        print(f"\n🔑 关键发现:")
        print(f"  成功种子 P0 复用率: {success_reuse:.3f}")
        print(f"  失败种子 P0 复用率: {failure_reuse:.3f}")
        
        if success_reuse > failure_reuse:
            print(f"  ✅ 成功种子复用了更多旧结构 (差异: {success_reuse - failure_reuse:.3f})")
        else:
            print(f"  ⚠️  失败种子复用了更多旧结构 (差异: {failure_reuse - success_reuse:.3f})")
        
        success_survival = np.mean([r.phase_0_survival_rate for r in success])
        failure_survival = np.mean([r.phase_0_survival_rate for r in failure])
        
        print(f"\n  成功种子 P0 存活率: {success_survival:.3f}")
        print(f"  失败种子 P0 存活率: {failure_survival:.3f}")
        
        if success_survival > failure_survival:
            print(f"  ✅ 成功种子保留了更多旧结构")
        else:
            print(f"  ⚠️  失败种子保留了更多旧结构")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()
    
    config = NoBoundaryConfig()
    config.seeds = args.seeds
    
    print("=" * 80)
    print("Unified-SEL 实验 1：结构复用模式分析")
    print("=" * 80)
    print(f"种子: {args.seeds}")
    print(f"检查点步数: {config.checkpoint_step}")
    print(f"总步数: {config.steps}")
    print()
    
    results = []
    for seed in args.seeds:
        print(f"运行种子 {seed}...")
        t0 = time.time()
        result = run_reuse_analysis(seed, config)
        elapsed = time.time() - t0
        results.append(result)
        print(f"  遗忘率: {result.forgetting:.4f}")
        print(f"  P0 结构: {len(result.phase_0_structures)}")
        print(f"  P0 复用率: {result.phase_0_reuse_ratio:.3f}")
        print(f"  P0 存活率: {result.phase_0_survival_rate:.3f}")
        print(f"  耗时: {elapsed:.1f}s")
        print()
    
    # 对比报告
    print_reuse_comparison(results)
    
    # 保存结果
    output = {
        "experiment": "structure_reuse_analysis",
        "config": config.to_dict(),
        "per_seed": [r.to_dict() for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    from core.runtime import get_results_path, save_json, timestamp
    results_dir = get_results_path("reuse_analysis")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
