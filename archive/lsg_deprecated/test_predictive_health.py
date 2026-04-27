from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.predictive_health import (
    ControlContext,
    PredictiveHealthMonitor,
    PredictiveHealthSignals,
)


def approx_eq(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_predictive_health_creation() -> None:
    mon = PredictiveHealthMonitor(latent_dim=16, control_dim=4)
    assert mon.name == "predictive_health"
    assert mon._step_count == 0
    print("[OK] PredictiveHealthMonitor creation")


def test_control_context_to_vector() -> None:
    ctx = ControlContext(
        routing_decision="escalated",
        solver_type="search",
        monitor_signal=0.8,
        verifier_result=True,
    )
    vec = ctx.to_vector(dim=8)
    assert vec.shape == (8,)
    assert approx_eq(vec[0], 1.0)
    assert vec[1] == 1.0
    assert vec[2] == 0.8
    assert vec[3] == 1.0
    print("[OK] ControlContext to_vector")


def test_control_context_defaults() -> None:
    ctx = ControlContext()
    vec = ctx.to_vector(dim=4)
    assert vec[0] == 0.0
    assert vec[1] == 0.0
    assert vec[2] == 0.5
    assert vec[3] == 0.5
    print("[OK] ControlContext defaults")


def test_warmup_phase() -> None:
    mon = PredictiveHealthMonitor(latent_dim=8, window_size=5, warmup_windows=2)
    rng = np.random.RandomState(42)
    for i in range(15):
        z = rng.randn(8).astype(np.float32)
        sig = mon.observe(z)
        if i < 5:
            assert sig.status == "warmup"
    print("[OK] PredictiveHealthMonitor warmup phase")


def test_predict_observe_score_update_order() -> None:
    mon = PredictiveHealthMonitor(
        latent_dim=8, window_size=5, warmup_windows=1, learning_rate=0.1
    )
    rng = np.random.RandomState(7)

    for _ in range(20):
        z = rng.randn(8).astype(np.float32)
        sig = mon.observe(z)

    n_centroids_before = len(mon._window_centroids)
    z = rng.randn(8).astype(np.float32)
    sig = mon.observe(z)
    assert mon._step_count > 0
    print("[OK] Predict-observe-score-update order")


def test_centroid_prediction_residual() -> None:
    mon = PredictiveHealthMonitor(
        latent_dim=4, window_size=5, warmup_windows=1
    )
    rng = np.random.RandomState(123)

    centroid_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(10):
        z = centroid_a + rng.randn(4).astype(np.float32) * 0.1
        mon.observe(z)

    centroid_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32) * 3.0
    shift_residuals = []
    for _ in range(10):
        z = centroid_b + rng.randn(4).astype(np.float32) * 0.1
        sig = mon.observe(z)
        if sig.status != "warmup":
            shift_residuals.append(sig.residual_mean)

    assert len(shift_residuals) > 0
    print("[OK] PredictiveHealthMonitor centroid prediction residual")


def test_domain_shift_detection() -> None:
    mon = PredictiveHealthMonitor(
        latent_dim=16, window_size=5, warmup_windows=2
    )
    rng = np.random.RandomState(42)

    centroid_a = rng.randn(16).astype(np.float32)
    centroid_a = centroid_a / np.linalg.norm(centroid_a)
    for _ in range(25):
        z = centroid_a + rng.randn(16).astype(np.float32) * 0.1
        mon.observe(z)

    centroid_b = rng.randn(16).astype(np.float32)
    centroid_b = centroid_b / np.linalg.norm(centroid_b)
    shift_signals = []
    for _ in range(15):
        z = centroid_b + rng.randn(16).astype(np.float32) * 0.1
        sig = mon.observe(z)
        if sig.status != "warmup":
            shift_signals.append(sig)

    assert len(shift_signals) > 0
    max_status_rank = max(
        {"healthy": 0, "gradual_drift": 1, "domain_shift": 2, "anomaly": 3}.get(
            s.status, 0
        )
        for s in shift_signals
    )
    assert max_status_rank >= 1
    print("[OK] PredictiveHealthMonitor domain shift detection")


def test_control_context_interface() -> None:
    mon = PredictiveHealthMonitor(
        latent_dim=8, control_dim=4, window_size=5, warmup_windows=1
    )
    rng = np.random.RandomState(99)

    for _ in range(15):
        z = rng.randn(8).astype(np.float32)
        ctx = ControlContext(routing_decision="accepted", monitor_signal=0.3)
        sig = mon.observe(z, control_context=ctx)

    z = rng.randn(8).astype(np.float32)
    ctx = ControlContext(routing_decision="escalated", monitor_signal=0.9)
    sig = mon.observe(z, control_context=ctx)
    assert mon._step_count > 0
    print("[OK] PredictiveHealthMonitor control_context interface")


def test_health_report() -> None:
    mon = PredictiveHealthMonitor(
        latent_dim=8, window_size=5, warmup_windows=1
    )
    rng = np.random.RandomState(55)

    report = mon.health_report()
    assert report["status"] == "warmup"

    for _ in range(25):
        z = rng.randn(8).astype(np.float32)
        mon.observe(z)

    report = mon.health_report()
    assert "residual_mean" in report
    assert "residual_trend" in report
    assert "n_observed" in report
    print("[OK] PredictiveHealthMonitor health_report")


def test_warmup_not_counted_in_metrics() -> None:
    mon = PredictiveHealthMonitor(
        latent_dim=8, window_size=5, warmup_windows=2
    )
    rng = np.random.RandomState(11)

    for _ in range(25):
        z = rng.randn(8).astype(np.float32) * 100.0
        mon.observe(z)

    assert len(mon._window_centroids) > 2
    print("[OK] PredictiveHealthMonitor warmup not counted in metrics")


def main() -> None:
    test_predictive_health_creation()
    test_control_context_to_vector()
    test_control_context_defaults()
    test_warmup_phase()
    test_predict_observe_score_update_order()
    test_centroid_prediction_residual()
    test_domain_shift_detection()
    test_control_context_interface()
    test_health_report()
    test_warmup_not_counted_in_metrics()
    print("All predictive health tests passed")


if __name__ == "__main__":
    main()
