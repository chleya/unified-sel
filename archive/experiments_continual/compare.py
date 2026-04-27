"""
Compare Unified-SEL, FixedNetwork, and EWC results.

Default behavior stays compatible with the previous script:
- if no explicit paths are provided, the latest JSON result from each group is used

New behavior:
- explicit files or directories can be passed for each group
- directories are expanded to all JSON files within them
- raw seed-level samples are preserved when available
- per-run summary statistics and simple statistical tests are reported when enough samples exist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment_config import CompareConfig
from core.runtime import RESULTS_DIR, get_results_path, load_json, save_json, timestamp


DEFAULT_COMPARE_CONFIG = CompareConfig()


def latest_json(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON result files found in {directory}")
    return candidates[-1]


def flatten_path_args(groups: Optional[Sequence[Sequence[str]]]) -> List[Path]:
    if not groups:
        return []
    paths: List[Path] = []
    for group in groups:
        for item in group:
            paths.append(Path(item))
    return paths


def read_numeric(mapping: Dict, *keys: str) -> Optional[float]:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, dict):
            if "mean" in value:
                return float(value["mean"])
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    return None


def resolve_inputs(
    provided: List[Path],
    fallback_dir: Path,
) -> List[Path]:
    if not provided:
        return [latest_json(fallback_dir)]

    resolved: List[Path] = []
    for item in provided:
        if item.is_dir():
            resolved.extend(sorted(p for p in item.glob("*.json") if p.is_file()))
        elif item.is_file():
            resolved.append(item)
        else:
            raise FileNotFoundError(f"Result path not found: {item}")

    if not resolved:
        raise FileNotFoundError(f"No JSON result files found from inputs: {provided}")
    return resolved


def metric_from_summary(summary_value) -> float:
    if isinstance(summary_value, dict):
        return float(summary_value.get("mean", 0.0))
    return float(summary_value)


def extract_samples(payload: Dict) -> Dict[str, List[float]]:
    """
    Extract comparable numeric samples from a result payload.

    Baseline files currently contain a single run at the top level.
    Unified no-boundary files contain a top-level summary plus a per-seed `runs` list.
    """
    samples: Dict[str, List[float]] = {
        "avg_accuracy": [],
        "task_0_final_accuracy": [],
        "task_1_final_accuracy": [],
        "forgetting_task_0": [],
        "max_structures_seen": [],
        "recent_mean_loss": [],
        "recent_mean_surprise": [],
    }

    if "runs" in payload and isinstance(payload["runs"], list) and payload["runs"]:
        for run in payload["runs"]:
            if not isinstance(run, dict):
                continue
            task_0 = read_numeric(run, "task_0_accuracy_final", "task_0_accuracy_after_task_1", "task_0_accuracy_after_task_0")
            task_1 = read_numeric(run, "task_1_accuracy_final", "task_1_accuracy_after_task_1")
            forgetting = read_numeric(run, "forgetting_task_0")
            max_structures_seen = read_numeric(run, "max_structures_seen")
            recent_mean_loss = read_numeric(run, "recent_mean_loss")

            if task_0 is not None:
                samples["task_0_final_accuracy"].append(task_0)
            if task_1 is not None:
                samples["task_1_final_accuracy"].append(task_1)
            if task_0 is not None and task_1 is not None:
                samples["avg_accuracy"].append((task_0 + task_1) / 2.0)
            if forgetting is not None:
                samples["forgetting_task_0"].append(forgetting)
            if max_structures_seen is not None:
                samples["max_structures_seen"].append(max_structures_seen)
            if recent_mean_loss is not None:
                samples["recent_mean_loss"].append(recent_mean_loss)
            if "final_stats" in run and isinstance(run["final_stats"], dict):
                final_stats = run["final_stats"]
                recent_surprise = read_numeric(final_stats, "recent_avg_surprise", "avg_tension")
                if recent_surprise is not None:
                    samples["recent_mean_surprise"].append(recent_surprise)
        return {k: v for k, v in samples.items() if v}

    # Single-result fallback: baseline files and older result shapes
    summary = payload["summary"] if "summary" in payload and isinstance(payload["summary"], dict) else {}

    task_0 = read_numeric(
        payload,
        "task_0_accuracy_after_task_1",
        "task_0_accuracy_after_task_0",
        "task_0_accuracy_final",
    )
    if task_0 is None:
        task_0 = read_numeric(summary, "task_0_accuracy_final")

    task_1 = read_numeric(
        payload,
        "task_1_accuracy_after_task_1",
        "task_1_accuracy_final",
    )
    if task_1 is None:
        task_1 = read_numeric(summary, "task_1_accuracy_final")

    if task_0 is not None:
        samples["task_0_final_accuracy"].append(task_0)
    if task_1 is not None:
        samples["task_1_final_accuracy"].append(task_1)
    if task_0 is not None and task_1 is not None:
        samples["avg_accuracy"].append((task_0 + task_1) / 2.0)

    forgetting = read_numeric(payload, "forgetting_task_0")
    if forgetting is None:
        forgetting = read_numeric(summary, "forgetting_task_0")
    if forgetting is not None:
        samples["forgetting_task_0"].append(forgetting)

    max_structures_seen = read_numeric(payload, "max_structures_seen")
    if max_structures_seen is None:
        max_structures_seen = read_numeric(summary, "max_structures_seen")
    if max_structures_seen is not None:
        samples["max_structures_seen"].append(max_structures_seen)

    recent_mean_loss = read_numeric(payload, "recent_mean_loss")
    if recent_mean_loss is not None:
        samples["recent_mean_loss"].append(recent_mean_loss)

    if "final_stats" in payload and isinstance(payload["final_stats"], dict):
        recent_surprise = read_numeric(payload["final_stats"], "recent_avg_surprise", "avg_tension")
        if recent_surprise is not None:
            samples["recent_mean_surprise"].append(recent_surprise)

    if "aggregate_event_counts" in summary:
        event_counts = summary["aggregate_event_counts"]
        if isinstance(event_counts, dict):
            total = float(sum(float(v) for v in event_counts.values()))
            samples["recent_mean_loss"].append(total)

    return {k: v for k, v in samples.items() if v}


def summarize_samples(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    summary = {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "n": int(arr.size),
    }
    if arr.size >= 2:
        summary["ci95_low"], summary["ci95_high"] = bootstrap_mean_ci(arr)
    return summary


def bootstrap_mean_ci(values: np.ndarray, n_resamples: int = 2000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    resampled_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(values, size=values.size, replace=True)
        resampled_means[i] = float(sample.mean())
    low = float(np.quantile(resampled_means, 0.025))
    high = float(np.quantile(resampled_means, 0.975))
    return low, high


def bootstrap_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 0,
) -> Optional[Tuple[float, float]]:
    if len(a) == 0 or len(b) == 0:
        return None
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return None

    if a_arr.size == 1 and b_arr.size == 1:
        diff = float(a_arr.mean() - b_arr.mean())
        return diff, diff

    rng = np.random.default_rng(seed)
    resampled_diffs = np.empty(n_resamples, dtype=float)
    if a_arr.size >= 2 and b_arr.size >= 2:
        for i in range(n_resamples):
            sample_a = rng.choice(a_arr, size=a_arr.size, replace=True)
            sample_b = rng.choice(b_arr, size=b_arr.size, replace=True)
            resampled_diffs[i] = float(sample_a.mean() - sample_b.mean())
    elif a_arr.size >= 2:
        baseline = float(b_arr.mean())
        for i in range(n_resamples):
            sample_a = rng.choice(a_arr, size=a_arr.size, replace=True)
            resampled_diffs[i] = float(sample_a.mean() - baseline)
    else:
        baseline = float(a_arr.mean())
        for i in range(n_resamples):
            sample_b = rng.choice(b_arr, size=b_arr.size, replace=True)
            resampled_diffs[i] = float(baseline - sample_b.mean())

    low = float(np.quantile(resampled_diffs, 0.025))
    high = float(np.quantile(resampled_diffs, 0.975))
    return low, high


def permutation_p_value(a: Sequence[float], b: Sequence[float], n_resamples: int = 5000, seed: int = 0) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    observed = float(a_arr.mean() - b_arr.mean())
    pooled = np.concatenate([a_arr, b_arr])
    na = a_arr.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_resamples):
        permuted = rng.permutation(pooled)
        diff = float(permuted[:na].mean() - permuted[na:].mean())
        if abs(diff) >= abs(observed):
            count += 1
    return float((count + 1) / (n_resamples + 1))


def cohen_d(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    pooled_var = ((a_arr.size - 1) * a_arr.var(ddof=1) + (b_arr.size - 1) * b_arr.var(ddof=1))
    pooled_var /= max(a_arr.size + b_arr.size - 2, 1)
    if pooled_var <= 0:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / np.sqrt(pooled_var))


def load_inputs(paths: List[Path]) -> List[Tuple[Path, Dict]]:
    return [(path, load_json(path)) for path in paths]


def build_group_report(label: str, path_data: List[Tuple[Path, Dict]]) -> Dict:
    paths = [str(path) for path, _ in path_data]
    samples_by_metric: Dict[str, List[float]] = {}
    for _, payload in path_data:
        extracted = extract_samples(payload)
        for metric, values in extracted.items():
            samples_by_metric.setdefault(metric, []).extend(values)

    metrics = {metric: summarize_samples(values) for metric, values in samples_by_metric.items()}
    return {
        "label": label,
        "paths": paths,
        "samples": samples_by_metric,
        "metrics": metrics,
    }


def compare_samples(
    unified_samples: Sequence[float],
    reference_samples: Sequence[float],
    higher_is_better: bool,
) -> Dict[str, Optional[float]]:
    unified_arr = np.asarray(list(unified_samples), dtype=float)
    reference_arr = np.asarray(list(reference_samples), dtype=float)

    unified_mean = float(unified_arr.mean()) if unified_arr.size else 0.0
    reference_mean = float(reference_arr.mean()) if reference_arr.size else 0.0
    delta = unified_mean - reference_mean

    ci = bootstrap_delta_ci(unified_arr, reference_arr)
    p_value = permutation_p_value(unified_arr, reference_arr)
    effect_size = cohen_d(unified_arr, reference_arr)

    if higher_is_better:
        direction_ok = delta > 0
    else:
        direction_ok = delta < 0

    ci_supports_direction = None
    if ci is not None:
        low, high = ci
        if higher_is_better:
            ci_supports_direction = low > 0
        else:
            ci_supports_direction = high < 0

    statistically_supported = None
    if p_value is not None:
        statistically_supported = p_value < 0.05 and direction_ok
    elif ci_supports_direction is not None:
        statistically_supported = ci_supports_direction

    return {
        "unified_mean": unified_mean,
        "reference_mean": reference_mean,
        "delta_mean": delta,
        "delta_ci95_low": ci[0] if ci is not None else None,
        "delta_ci95_high": ci[1] if ci is not None else None,
        "p_value": p_value,
        "effect_size_cohen_d": effect_size,
        "direction_ok": direction_ok,
        "statistically_supported": statistically_supported,
        "unified_n": int(unified_arr.size),
        "reference_n": int(reference_arr.size),
        "method": (
            "permutation_test"
            if p_value is not None
            else ("bootstrap_delta_ci" if ci is not None else "insufficient_samples")
        ),
    }


def build_comparison(fixed_report: Dict, ewc_report: Dict, unified_report: Dict) -> Dict:
    fixed_metrics = fixed_report["metrics"]
    ewc_metrics = ewc_report["metrics"]
    unified_metrics = unified_report["metrics"]

    fixed_samples = fixed_report["samples"]
    ewc_samples = ewc_report["samples"]
    unified_samples = unified_report["samples"]

    fixed_avg_accuracy = fixed_metrics["avg_accuracy"]["mean"]
    ewc_avg_accuracy = ewc_metrics["avg_accuracy"]["mean"]
    unified_avg_accuracy = unified_metrics["avg_accuracy"]["mean"]

    fixed_forgetting = fixed_metrics["forgetting_task_0"]["mean"]
    ewc_forgetting = ewc_metrics["forgetting_task_0"]["mean"]
    unified_forgetting = unified_metrics["forgetting_task_0"]["mean"]
    unified_max_structures = unified_metrics.get("max_structures_seen", {"mean": 0.0})

    accuracy_vs_ewc = compare_samples(
        unified_samples.get("avg_accuracy", []),
        ewc_samples.get("avg_accuracy", []),
        higher_is_better=True,
    )
    forgetting_vs_ewc = compare_samples(
        unified_samples.get("forgetting_task_0", []),
        ewc_samples.get("forgetting_task_0", []),
        higher_is_better=False,
    )

    accuracy_pass = accuracy_vs_ewc["direction_ok"]
    forgetting_pass = forgetting_vs_ewc["direction_ok"]

    if accuracy_pass and forgetting_pass:
        overall = "pass"
    elif accuracy_pass or forgetting_pass:
        overall = "partial_pass"
    else:
        overall = "fail"

    significance_support = [accuracy_vs_ewc["statistically_supported"], forgetting_vs_ewc["statistically_supported"]]
    if all(item is True for item in significance_support):
        significance_status = "pass"
    elif any(item is False for item in significance_support):
        significance_status = "fail"
    else:
        significance_status = "insufficient"

    comparison = {
        "config": DEFAULT_COMPARE_CONFIG.to_dict(),
        "sources": {
            "fixed": fixed_report["paths"],
            "ewc": ewc_report["paths"],
            "unified_sel": unified_report["paths"],
        },
        "metrics": {
            "fixed": fixed_metrics,
            "ewc": ewc_metrics,
            "unified_sel": unified_metrics,
        },
        "headline": {
            "avg_accuracy": {
                "fixed": fixed_avg_accuracy,
                "ewc": ewc_avg_accuracy,
                "unified_sel": unified_avg_accuracy,
            },
            "forgetting_task_0": {
                "fixed": fixed_forgetting,
                "ewc": ewc_forgetting,
                "unified_sel": unified_forgetting,
            },
            "preliminary_decision": {
                "unified_beats_ewc_on_avg_accuracy": accuracy_pass,
                "unified_beats_ewc_on_forgetting": forgetting_pass,
            },
        },
        "goal_report": {
            "avg_accuracy_vs_ewc": {
                "status": "pass" if accuracy_pass else "fail",
                "higher_is_better": True,
                **accuracy_vs_ewc,
            },
            "forgetting_vs_ewc": {
                "status": "pass" if forgetting_pass else "fail",
                "higher_is_better": False,
                **forgetting_vs_ewc,
            },
            "statistical_significance": {
                "avg_accuracy": accuracy_vs_ewc,
                "forgetting": forgetting_vs_ewc,
                "status": significance_status,
            },
            "overall": overall,
        },
        "analysis": {
            "unified_avg_accuracy_minus_ewc": accuracy_vs_ewc["delta_mean"],
            "unified_forgetting_minus_ewc": forgetting_vs_ewc["delta_mean"],
            "unified_max_structures_seen_mean": unified_max_structures["mean"],
            "accuracy_sample_sizes": {
                "unified_sel": accuracy_vs_ewc["unified_n"],
                "ewc": accuracy_vs_ewc["reference_n"],
            },
            "forgetting_sample_sizes": {
                "unified_sel": forgetting_vs_ewc["unified_n"],
                "ewc": forgetting_vs_ewc["reference_n"],
            },
        },
    }
    return comparison


def default_or_explicit(paths: List[Path], fallback_dir: Path) -> List[Path]:
    return resolve_inputs(paths, fallback_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed",
        action="append",
        nargs="+",
        help="Explicit fixed baseline result files or directories. Can be repeated.",
    )
    parser.add_argument(
        "--ewc",
        action="append",
        nargs="+",
        help="Explicit EWC result files or directories. Can be repeated.",
    )
    parser.add_argument(
        "--unified",
        action="append",
        nargs="+",
        help="Explicit Unified-SEL result files or directories. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path for the comparison JSON.",
    )
    args = parser.parse_args()

    fixed_inputs = flatten_path_args(args.fixed)
    ewc_inputs = flatten_path_args(args.ewc)
    unified_inputs = flatten_path_args(args.unified)

    fixed_paths = default_or_explicit(fixed_inputs, RESULTS_DIR / "baseline_fixed")
    ewc_paths = default_or_explicit(ewc_inputs, RESULTS_DIR / "baseline_ewc")
    unified_paths = default_or_explicit(unified_inputs, RESULTS_DIR / "continual_no_boundary")

    fixed_report = build_group_report("fixed", load_inputs(fixed_paths))
    ewc_report = build_group_report("ewc", load_inputs(ewc_paths))
    unified_report = build_group_report("unified_sel", load_inputs(unified_paths))

    comparison = build_comparison(fixed_report, ewc_report, unified_report)

    output_dir = get_results_path("analysis_compare")
    output_path = Path(args.output) if args.output else output_dir / f"{timestamp()}.json"
    save_json(comparison, output_path)

    print(json.dumps(comparison, indent=2))
    print(f"\nSaved comparison: {output_path}")


if __name__ == "__main__":
    main()
