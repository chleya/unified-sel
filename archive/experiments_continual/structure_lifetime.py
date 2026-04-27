"""
Unified-SEL 结构寿命分析

目标：
1. 追踪每个结构的创建/死亡时间
2. 绘制结构寿命分布
3. 分析结构寿命与遗忘率的相关性

核心假设：
- 如果结构周转太快 → 旧结构在被充分复用前就死了 → 遗忘
- 如果结构寿命长 → 被充分复用 → 保留更好
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # F:\unified-sel
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_config import NoBoundaryConfig, PoolConfig
from core.learner import UnifiedSELClassifier
from core.runtime import get_results_path, save_json, timestamp


@dataclass
class StructureLifeRecord:
    """单个结构的生命周期记录。"""
    id: int
    created_at_step: int
    died_at_step: Optional[int] = None  # None = 还活着
    age_at_death: Optional[int] = None
    
    @property
    def current_age(self, current_step: int) -> int:
        if self.died_at_step is None:
            return current_step - self.created_at_step
        return self.age_at_death or 0
    
    @property
    def is_alive(self) -> bool:
        return self.died_at_step is None


@dataclass
class LifetimeAnalysisResult:
    """寿命分析结果。"""
    seed: int
    structures: List[StructureLifeRecord]
    total_steps: int
    forgetting: float
    task_0_accuracy_final: float
    task_1_accuracy_final: float
    
    @property
    def alive_structures(self) -> List[StructureLifeRecord]:
        return [s for s in self.structures if s.is_alive]
    
    @property
    def dead_structures(self) -> List[StructureLifeRecord]:
        return [s for s in self.structures if not s.is_alive]
    
    @property
    def mean_lifetime(self) -> float:
        lifetimes = [s.age_at_death for s in self.dead_structures if s.age_at_death is not None]
        return float(np.mean(lifetimes)) if lifetimes else 0.0
    
    @property
    def median_lifetime(self) -> float:
        lifetimes = [s.age_at_death for s in self.dead_structures if s.age_at_death is not None]
        return float(np.median(lifetimes)) if lifetimes else 0.0
    
    @property
    def max_lifetime(self) -> float:
        lifetimes = [s.age_at_death for s in self.dead_structures if s.age_at_death is not None]
        return float(max(lifetimes)) if lifetimes else 0.0
    
    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "total_steps": self.total_steps,
            "n_structures_created": len(self.structures),
            "n_structures_alive": len(self.alive_structures),
            "n_structures_dead": len(self.dead_structures),
            "mean_lifetime": self.mean_lifetime,
            "median_lifetime": self.median_lifetime,
            "max_lifetime": self.max_lifetime,
            "forgetting": self.forgetting,
            "task_0_accuracy_final": self.task_0_accuracy_final,
            "task_1_accuracy_final": self.task_1_accuracy_final,
        }


def run_seed_with_lifetime_tracking(
    seed: int,
    config: NoBoundaryConfig,
) -> LifetimeAnalysisResult:
    """
    运行单个种子，同时追踪结构生命周期。
    """
    from core.pool import StructurePool, Structure
    
    rng = np.random.default_rng(seed)
    
    # 创建自定义的 StructurePool，添加生命周期追踪
    class TrackedStructurePool(StructurePool):
        def __init__(self, *args, **kwargs):
            # 先初始化追踪属性（在 super().__init__ 之前，因为 _new_structure 会用到）
            self.structure_lives: Dict[int, StructureLifeRecord] = {}
            self.step_count = 0
            # 初始化父类（会调用 _new_structure）
            super().__init__(*args, **kwargs)
        
        def _new_structure(self, label: str = "") -> Structure:
            structure = super()._new_structure(label)
            sid = structure.id
            self.structure_lives[sid] = StructureLifeRecord(
                id=sid,
                created_at_step=self.step_count,
            )
            return structure
        
        def prune(self) -> int:
            # 在 prune 之前记录哪些结构会死
            structures_before = {s.id for s in self.structures}
            result = super().prune()
            structures_after = {s.id for s in self.structures}
            
            # 记录死亡
            pruned_ids = structures_before - structures_after
            for sid in pruned_ids:
                if sid in self.structure_lives:
                    record = self.structure_lives[sid]
                    if record.is_alive:
                        record.died_at_step = self.step_count
                        record.age_at_death = self.step_count - record.created_at_step
            
            return result
    
    # 创建分类器
    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        seed=seed,
    )
    
    # 替换为追踪池
    clf.pool = TrackedStructurePool(
        in_size=config.in_size,
        out_size=config.out_size,
        max_structures=config.pool.max_structures,
        initial_structures=1,
        seed=seed,
        surprise_threshold=config.pool.surprise_threshold,
        tension_threshold=config.pool.tension_threshold,
        utility_decay=config.pool.utility_decay,
        utility_prune=config.pool.utility_prune,
        reinforce_amount=config.pool.reinforce_amount,
        clone_perturbation=config.pool.clone_perturbation,
        mature_age=config.pool.mature_age,
        mature_decay_scale=config.pool.mature_decay_scale,
    )
    clf.pool.next_id = 1
    
    # 评估数据
    from experiments.continual.no_boundary import make_eval_task, stream_sample
    
    X_task_0, y_task_0 = make_eval_task(
        task_id=0,
        n_samples=config.eval_samples_per_task,
        seed=seed + 1000,
        in_size=config.in_size,
    )
    X_task_1, y_task_1 = make_eval_task(
        task_id=1,
        n_samples=config.eval_samples_per_task,
        seed=seed + 2000,
        in_size=config.in_size,
    )
    
    # 检查点准确率
    task_0_accuracy_at_checkpoint = 0.0
    
    # 运行流
    for step in range(config.steps):
        progress = step / max(config.steps - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)
        
        # 检查点记录
        if step + 1 == config.checkpoint_step:
            task_0_accuracy_at_checkpoint = clf.accuracy(X_task_0, y_task_0)
    
    # 最终评估
    final_task_0_accuracy = clf.accuracy(X_task_0, y_task_0)
    final_task_1_accuracy = clf.accuracy(X_task_1, y_task_1)
    forgetting = task_0_accuracy_at_checkpoint - final_task_0_accuracy
    
    # 构建结果
    result = LifetimeAnalysisResult(
        seed=seed,
        structures=list(clf.pool.structure_lives.values()),
        total_steps=config.steps,
        forgetting=forgetting,
        task_0_accuracy_final=final_task_0_accuracy,
        task_1_accuracy_final=final_task_1_accuracy,
    )
    
    return result


def analyze_lifetime_forgetting_correlation(
    results: List[LifetimeAnalysisResult],
) -> dict:
    """
    分析结构寿命与遗忘率的相关性。
    """
    if len(results) < 2:
        return {"error": "Need at least 2 seeds for correlation"}
    
    lifetimes = [r.mean_lifetime for r in results]
    forgettings = [r.forgetting for r in results]
    
    # Pearson 相关
    from scipy import stats
    corr, p_value = stats.pearsonr(lifetimes, forgettings)
    
    return {
        "correlation": float(corr),
        "p_value": float(p_value),
        "n_seeds": len(results),
        "interpretation": (
            "负相关：寿命越长，遗忘越少" if corr < -0.5 else
            "正相关：寿命越长，遗忘越多" if corr > 0.5 else
            "弱相关：寿命与遗忘关系不明显"
        ),
    }


def run_experiment(
    num_seeds: int = 5,
    seeds: Optional[List[int]] = None,
) -> dict:
    """
    运行完整寿命分析实验。
    """
    if seeds is None:
        seeds = [7 + i for i in range(num_seeds)]
    
    config = NoBoundaryConfig()
    config.seeds = seeds
    
    print("=" * 60)
    print("Unified-SEL 结构寿命分析实验")
    print("=" * 60)
    print(f"种子: {seeds}")
    print(f"步数: {config.steps}")
    print()
    
    results = []
    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] 运行种子 {seed}...")
        t0 = time.time()
        result = run_seed_with_lifetime_tracking(seed, config)
        elapsed = time.time() - t0
        
        print(f"  结构: {len(result.structures)} 创建, "
              f"{len(result.alive_structures)} 存活, "
              f"{len(result.dead_structures)} 死亡")
        print(f"  平均寿命: {result.mean_lifetime:.1f} 步")
        print(f"  遗忘率: {result.forgetting:.4f}")
        print(f"  耗时: {elapsed:.1f}s")
        print()
        
        results.append(result)
    
    # 相关性分析
    print("-" * 60)
    print("相关性分析:")
    corr_result = analyze_lifetime_forgetting_correlation(results)
    print(f"  寿命-遗忘相关: {corr_result.get('correlation', 'N/A'):.3f}")
    print(f"  p-value: {corr_result.get('p_value', 'N/A'):.4f}")
    print(f"  解释: {corr_result.get('interpretation', 'N/A')}")
    print()
    
    # 构建输出
    output = {
        "experiment": "structure_lifetime_analysis",
        "config": config.to_dict(),
        "per_seed": [r.to_dict() for r in results],
        "correlation": corr_result,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # 保存
    results_dir = get_results_path("structure_lifetime")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    output["saved_to"] = str(output_path)
    
    print(f"结果已保存到: {output_path}")
    return output


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=7)
    args = parser.parse_args()
    
    seeds = [args.start_seed + i for i in range(args.seeds)]
    run_experiment(seeds=seeds)


if __name__ == "__main__":
    main()
