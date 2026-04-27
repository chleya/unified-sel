"""
深度分析：为什么种子 7 成功（负遗忘），而种子 8/9 失败（高遗忘）？

分析维度：
1. 结构激活模式差异
2. W_out 权重变化
3. 张力/惊喜动态
4. 修剪事件分析
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
from core.pool import Structure, StructurePool


@dataclass
class ActivationTrace:
    """单个结构的激活追踪。"""
    structure_id: int
    activation_steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    tensions: List[float] = field(default_factory=list)
    utility_at_step: List[float] = field(default_factory=list)


@dataclass
class SeedDeepAnalysis:
    """单个种子的深度分析结果。"""
    seed: int
    forgetting: float
    task_0_acc: float
    task_1_acc: float
    
    # 结构级追踪
    structure_traces: Dict[int, ActivationTrace] = field(default_factory=dict)
    
    # W_out 权重变化
    w_out_at_checkpoint: Optional[np.ndarray] = None
    w_out_final: Optional[np.ndarray] = None
    w_out_change_magnitude: float = 0.0
    
    # 事件统计
    total_reinforce: int = 0
    total_branch: int = 0
    total_create: int = 0
    total_prune_events: int = 0
    
    # 阶段分析
    phase_0_structures: int = 0  # 200 步前创建的结构数
    phase_1_structures: int = 0  # 200 步后创建的结构数
    phase_0_active_ratio: float = 0.0  # 前 200 步中活跃结构的比例
    phase_1_active_ratio: float = 0.0  # 后 400 步中活跃结构的比例


def run_deep_analysis(seed: int, config: NoBoundaryConfig) -> SeedDeepAnalysis:
    """运行深度分析，追踪所有内部状态。"""
    from experiments.continual.no_boundary import make_eval_task, stream_sample
    
    # 创建带追踪的分类器
    clf = UnifiedSELClassifier(
        in_size=config.in_size,
        out_size=config.out_size,
        lr=config.lr,
        max_structures=config.pool.max_structures,
        evolve_every=config.evolve_every,
        seed=seed,
        pool_config=config.pool.to_pool_kwargs(),
    )
    
    # 添加追踪到 pool
    class TrackedPool(StructurePool):
        def __init__(self, *args, **kwargs):
            # 复制原 pool 的属性
            self.__dict__.update(clf.pool.__dict__)
            self.structure_traces: Dict[int, ActivationTrace] = {}
            self._event_counts = {"reinforce": 0, "branch": 0, "create": 0}
            self._prune_events = 0
            self._w_out_snapshots = []
            
        def observe(self, x: np.ndarray, y: int = None, out_size: int = 2) -> Dict:
            result = super().observe(x, y, out_size)
            active = result["active_structure"]
            sid = active.id
            
            # 追踪激活
            if sid not in self.structure_traces:
                self.structure_traces[sid] = ActivationTrace(structure_id=sid)
            
            trace = self.structure_traces[sid]
            trace.activation_steps.append(self.step_count)
            trace.utility_at_step.append(active.utility)
            
            # 记录事件
            event = result.get("event", "")
            if event in self._event_counts:
                self._event_counts[event] += 1
            
            return result
        
        def learn_active(self, active_structure: Structure, x: np.ndarray, 
                        output_error: np.ndarray, lr: float = 0.05) -> float:
            loss = super().learn_active(active_structure, x, output_error, lr)
            sid = active_structure.id
            if sid in self.structure_traces:
                self.structure_traces[sid].losses.append(loss)
                self.structure_traces[sid].tensions.append(active_structure.tension)
            return loss
        
        def prune(self) -> int:
            before = len(self.structures)
            result = super().prune()
            if before > len(self.structures):
                self._prune_events += 1
            return result
    
    # 替换 pool
    original_pool = clf.pool
    tracked_pool = TrackedPool(
        in_size=original_pool.in_size,
        out_size=original_pool.out_size,
        max_structures=original_pool.max_structures,
        initial_structures=original_pool.structures[0] if original_pool.structures else 1,
        seed=seed,
        surprise_threshold=original_pool.surprise_threshold,
        tension_threshold=original_pool.tension_threshold,
        utility_decay=original_pool.utility_decay,
        utility_prune=original_pool.utility_prune,
        reinforce_amount=original_pool.reinforce_amount,
        clone_perturbation=original_pool.clone_perturbation,
        mature_age=original_pool.mature_age,
        mature_decay_scale=original_pool.mature_decay_scale,
    )
    
    # 复制已有结构的追踪
    for s in original_pool.structures:
        tracked_pool.structure_traces[s.id] = ActivationTrace(structure_id=s.id)
    
    clf.pool = tracked_pool
    
    # 记录检查点 W_out
    rng = np.random.default_rng(seed)
    X_task_0, y_task_0 = make_eval_task(0, config.eval_samples_per_task, seed + 1000, config.in_size)
    X_task_1, y_task_1 = make_eval_task(1, config.eval_samples_per_task, seed + 2000, config.in_size)
    
    task_0_acc_at_checkpoint = 0.0
    
    for step in range(config.steps):
        progress = step / max(config.steps - 1, 1)
        x, y = stream_sample(progress, rng, in_size=config.in_size)
        clf.fit_one(x, y)
        
        # 记录 W_out 快照
        if step == config.checkpoint_step - 1:
            tracked_pool._w_out_snapshots.append(clf.W_out.copy())
            task_0_acc_at_checkpoint = clf.accuracy(X_task_0, y_task_0)
        if step == config.steps - 1:
            tracked_pool._w_out_snapshots.append(clf.W_out.copy())
    
    # 最终评估
    task_0_final = clf.accuracy(X_task_0, y_task_0)
    task_1_final = clf.accuracy(X_task_1, y_task_1)
    forgetting = task_0_acc_at_checkpoint - task_0_final
    
    # 构建分析结果
    result = SeedDeepAnalysis(
        seed=seed,
        forgetting=forgetting,
        task_0_acc=task_0_final,
        task_1_acc=task_1_final,
        structure_traces=tracked_pool.structure_traces,
        total_reinforce=tracked_pool._event_counts["reinforce"],
        total_branch=tracked_pool._event_counts["branch"],
        total_create=tracked_pool._event_counts["create"],
        total_prune_events=tracked_pool._prune_events,
    )
    
    # W_out 变化分析
    if len(tracked_pool._w_out_snapshots) >= 2:
        result.w_out_at_checkpoint = tracked_pool._w_out_snapshots[0]
        result.w_out_final = tracked_pool._w_out_snapshots[1]
        result.w_out_change_magnitude = float(
            np.linalg.norm(result.w_out_final - result.w_out_at_checkpoint)
        )
    
    # 阶段分析
    for sid, trace in result.structure_traces.items():
        if trace.activation_steps:
            first_step = trace.activation_steps[0]
            if first_step < 200:
                result.phase_0_structures += 1
            else:
                result.phase_1_structures += 1
    
    return result


def compare_success_vs_failure(
    success_result: SeedDeepAnalysis,
    failure_results: List[SeedDeepAnalysis],
) -> dict:
    """对比成功与失败种子的差异。"""
    
    comparison = {
        "success_seed": success_result.seed,
        "failure_seeds": [r.seed for r in failure_results],
        "metrics": {},
    }
    
    # 1. W_out 变化对比
    comparison["metrics"]["w_out_change"] = {
        "success": success_result.w_out_change_magnitude,
        "failures_mean": float(np.mean([r.w_out_change_magnitude for r in failure_results])),
        "interpretation": (
            "成功的种子 W_out 变化更小" 
            if success_result.w_out_change_magnitude < np.mean([r.w_out_change_magnitude for r in failure_results])
            else "失败的种子 W_out 变化更小"
        ),
    }
    
    # 2. 事件分布对比
    success_events = np.array([
        success_result.total_reinforce,
        success_result.total_branch,
        success_result.total_create,
    ])
    failure_events_mean = np.mean([
        [r.total_reinforce, r.total_branch, r.total_create]
        for r in failure_results
    ], axis=0)
    
    comparison["metrics"]["event_distribution"] = {
        "success": {
            "reinforce": int(success_result.total_reinforce),
            "branch": int(success_result.total_branch),
            "create": int(success_result.total_create),
        },
        "failures_mean": {
            "reinforce": float(failure_events_mean[0]),
            "branch": float(failure_events_mean[1]),
            "create": float(failure_events_mean[2]),
        },
    }
    
    # 3. 结构数量对比
    comparison["metrics"]["structures"] = {
        "success": len(success_result.structure_traces),
        "failures_mean": float(np.mean([len(r.structure_traces) for r in failure_results])),
    }
    
    # 4. 修剪事件对比
    comparison["metrics"]["prune_events"] = {
        "success": success_result.total_prune_events,
        "failures_mean": float(np.mean([r.total_prune_events for r in failure_results])),
    }
    
    # 5. 阶段分析对比
    comparison["metrics"]["phase_analysis"] = {
        "success": {
            "phase_0_structures": success_result.phase_0_structures,
            "phase_1_structures": success_result.phase_1_structures,
        },
        "failures_mean": {
            "phase_0_structures": float(np.mean([r.phase_0_structures for r in failure_results])),
            "phase_1_structures": float(np.mean([r.phase_1_structures for r in failure_results])),
        },
    }
    
    return comparison


def print_comparison_report(comparison: dict):
    """打印对比报告。"""
    print("\n" + "=" * 70)
    print("种子 7 (成功) vs 种子 8/9 (失败) 深度对比分析")
    print("=" * 70)
    
    print(f"\n✅ 成功种子: {comparison['success_seed']}")
    print(f"❌ 失败种子: {comparison['failure_seeds']}")
    
    print("\n--- 1. W_out 权重变化 ---")
    m = comparison["metrics"]["w_out_change"]
    print(f"  成功种子: {m['success']:.4f}")
    print(f"  失败平均: {m['failures_mean']:.4f}")
    print(f"  解读: {m['interpretation']}")
    
    print("\n--- 2. 事件分布 ---")
    m = comparison["metrics"]["event_distribution"]
    print(f"  {'事件':<12} {'成功':<8} {'失败平均':<8}")
    print(f"  {'reinforce':<12} {m['success']['reinforce']:<8} {m['failures_mean']['reinforce']:<8.1f}")
    print(f"  {'branch':<12} {m['success']['branch']:<8} {m['failures_mean']['branch']:<8.1f}")
    print(f"  {'create':<12} {m['success']['create']:<8} {m['failures_mean']['create']:<8.1f}")
    
    print("\n--- 3. 结构数量 ---")
    m = comparison["metrics"]["structures"]
    print(f"  成功种子: {m['success']}")
    print(f"  失败平均: {m['failures_mean']:.1f}")
    
    print("\n--- 4. 修剪事件 ---")
    m = comparison["metrics"]["prune_events"]
    print(f"  成功种子: {m['success']}")
    print(f"  失败平均: {m['failures_mean']:.1f}")
    
    print("\n--- 5. 阶段分析 ---")
    m = comparison["metrics"]["phase_analysis"]
    print(f"  {'阶段':<20} {'成功':<8} {'失败平均':<8}")
    print(f"  {'Phase 0 (<200步)':<20} {m['success']['phase_0_structures']:<8} {m['failures_mean']['phase_0_structures']:<8.1f}")
    print(f"  {'Phase 1 (>=200步)':<20} {m['success']['phase_1_structures']:<8} {m['failures_mean']['phase_1_structures']:<8.1f}")
    
    print("\n" + "=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-seed", type=int, default=7)
    parser.add_argument("--failure-seeds", type=int, nargs="+", default=[8, 9])
    args = parser.parse_args()
    
    config = NoBoundaryConfig()
    config.seeds = [args.success_seed] + args.failure_seeds
    
    print("=" * 70)
    print("Unified-SEL 深度分析：成功 vs 失败种子对比")
    print("=" * 70)
    print(f"成功种子: {args.success_seed}")
    print(f"失败种子: {args.failure_seeds}")
    print()
    
    results = {}
    for seed in config.seeds:
        print(f"运行种子 {seed}...")
        t0 = time.time()
        result = run_deep_analysis(seed, config)
        elapsed = time.time() - t0
        results[seed] = result
        print(f"  遗忘率: {result.forgetting:.4f}")
        print(f"  W_out 变化: {result.w_out_change_magnitude:.4f}")
        print(f"  耗时: {elapsed:.1f}s")
        print()
    
    # 对比分析
    success_result = results[args.success_seed]
    failure_results = [results[s] for s in args.failure_seeds]
    
    comparison = compare_success_vs_failure(success_result, failure_results)
    
    # 打印报告
    print_comparison_report(comparison)
    
    # 保存结果
    output = {
        "experiment": "success_vs_failure_deep_analysis",
        "comparison": comparison,
        "per_seed": {
            seed: {
                "forgetting": r.forgetting,
                "task_0_acc": r.task_0_acc,
                "task_1_acc": r.task_1_acc,
                "w_out_change": r.w_out_change_magnitude,
                "n_structures": len(r.structure_traces),
                "events": {
                    "reinforce": r.total_reinforce,
                    "branch": r.total_branch,
                    "create": r.total_create,
                    "prune": r.total_prune_events,
                },
                "phase_analysis": {
                    "phase_0_structures": r.phase_0_structures,
                    "phase_1_structures": r.phase_1_structures,
                },
            }
            for seed, r in results.items()
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    from core.runtime import get_results_path, save_json, timestamp
    results_dir = get_results_path("deep_analysis")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(output, output_path)
    
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
