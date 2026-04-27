from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from experiments.meta_controller.baselines import build_controllers


VALID_PROFILES = ("v0", "v01", "v02", "v03", "v03b")
VALID_MASKS = ("surprise", "memory", "cost", "conflict", "drift")


@dataclass(frozen=True)
class AcceptanceRule:
    """Small rule for evaluating one controller against an optional reference."""

    controller: str
    metric: str
    op: str
    value: Optional[float] = None
    reference_controller: Optional[str] = None
    tolerance: float = 0.0

    def evaluate(self, results: Mapping[str, Mapping[str, float]]) -> bool:
        if self.controller not in results:
            raise KeyError(f"Missing controller result: {self.controller}")
        if self.metric not in results[self.controller]:
            raise KeyError(f"Missing metric {self.metric!r} for {self.controller}")
        lhs = float(results[self.controller][self.metric])
        rhs = self._rhs(results)
        if self.op == ">=":
            return lhs + self.tolerance >= rhs
        if self.op == ">":
            return lhs > rhs + self.tolerance
        if self.op == "<=":
            return lhs <= rhs + self.tolerance
        if self.op == "<":
            return lhs + self.tolerance < rhs
        if self.op == "==":
            return abs(lhs - rhs) <= self.tolerance
        raise ValueError(f"Unsupported acceptance operator: {self.op}")

    def _rhs(self, results: Mapping[str, Mapping[str, float]]) -> float:
        if self.reference_controller is not None:
            if self.reference_controller not in results:
                raise KeyError(f"Missing reference result: {self.reference_controller}")
            if self.metric not in results[self.reference_controller]:
                raise KeyError(f"Missing metric {self.metric!r} for {self.reference_controller}")
            return float(results[self.reference_controller][self.metric])
        if self.value is None:
            raise ValueError("AcceptanceRule requires value or reference_controller")
        return float(self.value)


@dataclass(frozen=True)
class RunSpec:
    suite: str
    profile: str
    seed: int
    train_episodes: int
    eval_episodes: int
    controllers: Tuple[str, ...]
    read_cost: Optional[float] = None
    masks: Tuple[str, ...] = ()

    def cli_args(self) -> Tuple[str, ...]:
        args = (
            "--profile",
            self.profile,
            "--mode",
            "train-eval",
            "--train-episodes",
            str(self.train_episodes),
            "--eval-episodes",
            str(self.eval_episodes),
            "--seed",
            str(self.seed),
            "--table",
        )
        if self.read_cost is None:
            return args
        return args + ("--read-cost", f"{self.read_cost:.2f}")


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    profile: str
    seeds: Tuple[int, ...]
    train_episodes: int
    eval_episodes: int
    controllers: Tuple[str, ...]
    masks: Tuple[str, ...] = ()
    read_cost: Optional[float] = None
    acceptance_rules: Tuple[AcceptanceRule, ...] = field(default_factory=tuple)

    def expand_runs(self) -> Tuple[RunSpec, ...]:
        return tuple(
            RunSpec(
                suite=self.name,
                profile=self.profile,
                seed=seed,
                train_episodes=self.train_episodes,
                eval_episodes=self.eval_episodes,
                controllers=self.controllers,
                read_cost=self.read_cost,
                masks=self.masks,
            )
            for seed in self.seeds
        )


def default_suites() -> Tuple[BenchmarkSuite, ...]:
    return (
        BenchmarkSuite(
            name="mainline_acceptance",
            profile="v01",
            seeds=(0, 1, 2),
            train_episodes=240,
            eval_episodes=60,
            controllers=(
                "fixed_rule_controller",
                "planner_necessity_loose_controller",
                "habit_safe_set_h2_controller",
                "random_controller",
            ),
            acceptance_rules=(
                AcceptanceRule("planner_necessity_loose_controller", "task_success", ">=", value=0.99),
                AcceptanceRule(
                    "planner_necessity_loose_controller",
                    "drift_under_horizon",
                    "<=",
                    reference_controller="fixed_rule_controller",
                    tolerance=0.005,
                ),
                AcceptanceRule(
                    "planner_necessity_loose_controller",
                    "planner_calls",
                    "<",
                    reference_controller="fixed_rule_controller",
                ),
                AcceptanceRule(
                    "planner_necessity_loose_controller",
                    "memory_reads",
                    "<=",
                    reference_controller="fixed_rule_controller",
                    tolerance=1.0,
                ),
            ),
        ),
        BenchmarkSuite(
            name="transfer_v02",
            profile="v02",
            seeds=(0, 1, 2),
            train_episodes=240,
            eval_episodes=60,
            controllers=(
                "fixed_rule_controller",
                "planner_necessity_loose_controller",
                "habit_safe_set_h2_controller",
            ),
            acceptance_rules=(
                AcceptanceRule(
                    "planner_necessity_loose_controller",
                    "task_success",
                    ">=",
                    reference_controller="fixed_rule_controller",
                ),
                AcceptanceRule(
                    "planner_necessity_loose_controller",
                    "drift_under_horizon",
                    "<=",
                    reference_controller="fixed_rule_controller",
                ),
                AcceptanceRule(
                    "planner_necessity_loose_controller",
                    "planner_calls",
                    "<",
                    reference_controller="fixed_rule_controller",
                ),
            ),
        ),
        BenchmarkSuite(
            name="read_cost_stress",
            profile="v01",
            seeds=(0, 1, 2),
            train_episodes=240,
            eval_episodes=60,
            read_cost=0.20,
            controllers=(
                "fixed_rule_controller",
                "planner_necessity_loose_controller",
            ),
            acceptance_rules=(
                AcceptanceRule(
                    "planner_necessity_loose_controller",
                    "total_reward",
                    ">",
                    reference_controller="fixed_rule_controller",
                ),
                AcceptanceRule("planner_necessity_loose_controller", "task_success", ">=", value=0.99),
            ),
        ),
        BenchmarkSuite(
            name="b3_signal_masking",
            profile="v01",
            seeds=(0,),
            train_episodes=240,
            eval_episodes=60,
            controllers=("planner_necessity_loose_controller",),
            masks=("surprise", "memory", "conflict", "drift", "core_signals"),
            acceptance_rules=(
                AcceptanceRule("b3_mask_conflict", "task_success", "<", reference_controller="planner_necessity_loose_controller"),
                AcceptanceRule("b3_mask_memory", "task_success", "<", reference_controller="planner_necessity_loose_controller"),
                AcceptanceRule("b3_mask_surprise", "task_success", "<", reference_controller="planner_necessity_loose_controller"),
                AcceptanceRule("b3_mask_core_signals", "task_success", "<", reference_controller="planner_necessity_loose_controller"),
            ),
        ),
        BenchmarkSuite(
            name="long_horizon_drift_v03",
            profile="v03",
            seeds=(0, 1, 2),
            train_episodes=240,
            eval_episodes=60,
            controllers=(
                "fixed_rule_controller",
                "planner_necessity_loose_controller",
                "drift_aware_planner_necessity_controller",
                "drift_aware_planner_necessity_loose_controller",
                "habit_safe_set_controller",
                "shielded_dominance_controller",
            ),
            masks=("drift", "conflict", "core_signals"),
            acceptance_rules=(
                AcceptanceRule(
                    "b3_mask_drift",
                    "drift_under_horizon",
                    ">",
                    reference_controller="planner_necessity_loose_controller",
                ),
                AcceptanceRule(
                    "b3_mask_conflict",
                    "drift_under_horizon",
                    ">",
                    reference_controller="planner_necessity_loose_controller",
                ),
                AcceptanceRule(
                    "b3_mask_core_signals",
                    "drift_under_horizon",
                    ">",
                    reference_controller="planner_necessity_loose_controller",
                ),
                AcceptanceRule(
                    "drift_aware_planner_necessity_controller",
                    "drift_under_horizon",
                    "<=",
                    reference_controller="fixed_rule_controller",
                ),
                AcceptanceRule(
                    "drift_aware_planner_necessity_controller",
                    "task_success",
                    ">=",
                    reference_controller="fixed_rule_controller",
                ),
            ),
        ),
        BenchmarkSuite(
            name="b7_cross_profile_v01",
            profile="v01",
            seeds=(0, 1, 2),
            train_episodes=240,
            eval_episodes=60,
            controllers=(
                "fixed_rule_controller",
                "planner_necessity_loose_controller",
                "drift_aware_planner_necessity_controller",
                "drift_aware_planner_necessity_loose_controller",
            ),
            acceptance_rules=(
                AcceptanceRule(
                    "drift_aware_planner_necessity_controller",
                    "task_success",
                    ">=",
                    reference_controller="fixed_rule_controller",
                ),
                AcceptanceRule(
                    "drift_aware_planner_necessity_controller",
                    "drift_under_horizon",
                    "<=",
                    reference_controller="fixed_rule_controller",
                    tolerance=0.005,
                ),
            ),
        ),
        BenchmarkSuite(
            name="v03b_drift_variant_transfer",
            profile="v03b",
            seeds=(0, 1, 2),
            train_episodes=240,
            eval_episodes=60,
            controllers=(
                "fixed_rule_controller",
                "planner_necessity_loose_controller",
                "drift_aware_planner_necessity_controller",
                "drift_aware_planner_necessity_loose_controller",
            ),
            acceptance_rules=(
                AcceptanceRule(
                    "drift_aware_planner_necessity_controller",
                    "task_success",
                    ">=",
                    reference_controller="fixed_rule_controller",
                ),
                AcceptanceRule(
                    "drift_aware_planner_necessity_controller",
                    "drift_under_horizon",
                    "<=",
                    reference_controller="fixed_rule_controller",
                    tolerance=0.005,
                ),
                AcceptanceRule(
                    "drift_aware_planner_necessity_controller",
                    "planner_calls",
                    "<=",
                    reference_controller="fixed_rule_controller",
                    tolerance=1.0,
                ),
            ),
        ),
        BenchmarkSuite(
            name="b7_cross_profile_v02",
            profile="v02",
            seeds=(0, 1, 2),
            train_episodes=240,
            eval_episodes=60,
            controllers=(
                "fixed_rule_controller",
                "planner_necessity_loose_controller",
                "drift_aware_planner_necessity_controller",
                "drift_aware_planner_necessity_loose_controller",
            ),
            acceptance_rules=(
                AcceptanceRule(
                    "drift_aware_planner_necessity_controller",
                    "task_success",
                    ">=",
                    reference_controller="fixed_rule_controller",
                ),
                AcceptanceRule(
                    "drift_aware_planner_necessity_controller",
                    "drift_under_horizon",
                    "<=",
                    reference_controller="fixed_rule_controller",
                    tolerance=0.005,
                ),
            ),
        ),
    )


def suite_by_name(name: str) -> BenchmarkSuite:
    suites = {suite.name: suite for suite in default_suites()}
    if name not in suites:
        raise KeyError(f"Unknown benchmark suite: {name}")
    return suites[name]


def expand_suites(suites: Iterable[BenchmarkSuite] | None = None) -> Tuple[RunSpec, ...]:
    selected = tuple(default_suites() if suites is None else suites)
    return tuple(run for suite in selected for run in suite.expand_runs())


def validate_suite(suite: BenchmarkSuite, controller_names: Sequence[str] | None = None) -> None:
    available = set(controller_names or build_controllers(seed=0).keys())
    if suite.profile not in VALID_PROFILES:
        raise ValueError(f"Unknown profile {suite.profile!r}")
    missing = sorted(set(suite.controllers) - available)
    if missing:
        raise ValueError(f"Unknown controllers in {suite.name}: {', '.join(missing)}")
    invalid_masks = sorted(mask for mask in suite.masks if mask not in VALID_MASKS and mask != "core_signals")
    if invalid_masks:
        raise ValueError(f"Unknown masks in {suite.name}: {', '.join(invalid_masks)}")


def validate_suites(suites: Iterable[BenchmarkSuite] | None = None) -> None:
    for suite in tuple(default_suites() if suites is None else suites):
        validate_suite(suite)


def evaluate_acceptance(
    suite: BenchmarkSuite,
    results: Mapping[str, Mapping[str, float]],
) -> Dict[str, bool]:
    return {
        f"{rule.controller}.{rule.metric}.{rule.op}": rule.evaluate(results)
        for rule in suite.acceptance_rules
    }


def controller_families() -> Dict[str, Tuple[str, ...]]:
    return {
        "anchors": (
            "fixed_rule_controller",
            "oracle_macro_controller",
            "random_controller",
        ),
        "single_mode": (
            "habit_only",
            "planner_always",
            "memory_always",
        ),
        "learned_flat": (
            "learned_contextual_bandit",
            "imitation_controller",
        ),
        "dominance": (
            "habit_safe_set_h2_controller",
            "planner_necessity_loose_controller",
            "drift_aware_planner_necessity_controller",
            "drift_aware_planner_necessity_loose_controller",
        ),
    }
