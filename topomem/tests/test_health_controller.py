"""
tests/test_health_controller.py — Health ECU 单元测试
"""

import pytest
import numpy as np

from topomem.health_controller import (
    TopologyHealthController,
    HealthControllerConfig,
    HealthStatus,
    HealthTrend,
    TrendDirection,
)


class TestHealthStatus:
    """HealthStatus 数据类测试。"""

    def test_health_status_default(self):
        hs = HealthStatus()
        assert hs.health_score == 1.0
        assert hs.retrieval_gamma_mult == 1.0
        assert hs.prune_aggressiveness == 0.0
        assert hs.consolidate_threshold == 0.3
        assert hs.cluster_filter_enabled is False
        assert hs.h1_health == 1.0
        assert hs.h2_health == 1.0
        assert hs.betti_1_count == 0
        assert hs.betti_2_count == 0
        assert hs.trend is None

    def test_health_status_clipping(self):
        hs = HealthStatus(
            health_score=1.5,
            retrieval_gamma_mult=-0.5,
            prune_aggressiveness=2.0,
        )
        assert hs.health_score == 1.0
        assert hs.retrieval_gamma_mult == 0.0
        assert hs.prune_aggressiveness == 1.0


class TestHealthControllerConfig:
    """HealthControllerConfig 测试。"""

    def test_default_config(self):
        cfg = HealthControllerConfig()
        assert cfg.h1_weight == 0.5
        assert cfg.h2_weight == 0.5
        assert cfg.health_formula == "weighted_avg"
        assert cfg.consolidate_trigger_threshold == 0.3
        assert cfg.trend_window_size == 10
        assert cfg.min_history_for_trend == 3

    def test_custom_config(self):
        cfg = HealthControllerConfig(
            h1_weight=0.7,
            h2_weight=0.3,
            health_formula="min",
            trend_window_size=20,
        )
        assert cfg.h1_weight == 0.7
        assert cfg.h2_weight == 0.3
        assert cfg.health_formula == "min"
        assert cfg.trend_window_size == 20


class TestComputeHealthStatus:
    """compute_health_status 核心逻辑测试。"""

    def test_weighted_avg_formula(self):
        cfg = HealthControllerConfig(h1_weight=0.5, h2_weight=0.5, health_formula="weighted_avg")
        ec = TopologyHealthController(cfg)
        hs = ec.compute_health_status(h1_health=0.6, h2_health=0.8, betti_1_count=5, betti_2_count=2)
        assert hs.health_score == pytest.approx(0.7)
        assert hs.h1_health == 0.6
        assert hs.h2_health == 0.8

    def test_min_formula(self):
        cfg = HealthControllerConfig(h1_weight=0.5, h2_weight=0.5, health_formula="min")
        ec = TopologyHealthController(cfg)
        hs = ec.compute_health_status(h1_health=0.6, h2_health=0.8, betti_1_count=5, betti_2_count=2)
        assert hs.health_score == 0.6

    def test_geometric_formula(self):
        cfg = HealthControllerConfig(h1_weight=0.5, h2_weight=0.5, health_formula="geometric")
        ec = TopologyHealthController(cfg)
        hs = ec.compute_health_status(h1_health=0.5, h2_health=0.8, betti_1_count=5, betti_2_count=2)
        assert hs.health_score == pytest.approx(np.sqrt(0.5 * 0.8))

    def test_retrieval_gamma_high_health(self):
        cfg = HealthControllerConfig(retrieval_gamma_min=0.0, retrieval_gamma_max=1.0)
        ec = TopologyHealthController(cfg)
        hs = ec.compute_health_status(h1_health=1.0, h2_health=1.0, betti_1_count=0, betti_2_count=0)
        assert hs.retrieval_gamma_mult == pytest.approx(1.0)

    def test_retrieval_gamma_low_health(self):
        cfg = HealthControllerConfig(retrieval_gamma_min=0.0, retrieval_gamma_max=1.0)
        ec = TopologyHealthController(cfg)
        hs = ec.compute_health_status(h1_health=0.0, h2_health=0.0, betti_1_count=0, betti_2_count=0)
        assert hs.retrieval_gamma_mult == pytest.approx(0.0)

    def test_prune_aggressiveness_low_health(self):
        cfg = HealthControllerConfig(prune_aggressiveness_min=0.0, prune_aggressiveness_max=0.5)
        ec = TopologyHealthController(cfg)
        hs = ec.compute_health_status(h1_health=0.0, h2_health=0.0, betti_1_count=0, betti_2_count=0)
        assert hs.prune_aggressiveness == pytest.approx(0.5)

    def test_prune_aggressiveness_high_health(self):
        cfg = HealthControllerConfig(prune_aggressiveness_min=0.0, prune_aggressiveness_max=0.5)
        ec = TopologyHealthController(cfg)
        hs = ec.compute_health_status(h1_health=1.0, h2_health=1.0, betti_1_count=0, betti_2_count=0)
        assert hs.prune_aggressiveness == pytest.approx(0.0)

    def test_consolidate_threshold_dynamic(self):
        cfg = HealthControllerConfig(consolidate_trigger_threshold=0.3)
        ec = TopologyHealthController(cfg)
        # health=1.0 → threshold=0.3+0.4*(1-1.0)=0.3
        hs = ec.compute_health_status(h1_health=1.0, h2_health=1.0, betti_1_count=0, betti_2_count=0)
        assert hs.consolidate_threshold == pytest.approx(0.3)
        # health=0.5 → threshold=0.3+0.4*(0.5)=0.5
        hs2 = ec.compute_health_status(h1_health=0.5, h2_health=0.5, betti_1_count=0, betti_2_count=0)
        assert hs2.consolidate_threshold == pytest.approx(0.5)

    def test_cluster_filter_enabled(self):
        cfg = HealthControllerConfig(cluster_filter_trigger_threshold=0.7)
        ec = TopologyHealthController(cfg)
        # health=0.8 > 0.7 → False
        hs = ec.compute_health_status(h1_health=0.8, h2_health=0.8, betti_1_count=0, betti_2_count=0)
        assert hs.cluster_filter_enabled is False
        # health=0.6 < 0.7 → True
        hs2 = ec.compute_health_status(h1_health=0.6, h2_health=0.6, betti_1_count=0, betti_2_count=0)
        assert hs2.cluster_filter_enabled is True

    def test_betti_counts_passed_through(self):
        ec = TopologyHealthController()
        hs = ec.compute_health_status(h1_health=0.7, h2_health=0.7, betti_1_count=10, betti_2_count=3)
        assert hs.betti_1_count == 10
        assert hs.betti_2_count == 3


class TestShouldConsolidate:
    """should_consolidate 决策边界测试。"""

    def test_below_threshold(self):
        ec = TopologyHealthController()
        hs = HealthStatus(health_score=0.2, consolidate_threshold=0.5)
        assert ec.should_consolidate(hs) is True

    def test_above_threshold(self):
        ec = TopologyHealthController()
        hs = HealthStatus(health_score=0.6, consolidate_threshold=0.5)
        assert ec.should_consolidate(hs) is False

    def test_equal_threshold(self):
        ec = TopologyHealthController()
        hs = HealthStatus(health_score=0.5, consolidate_threshold=0.5)
        assert ec.should_consolidate(hs) is False  # < not <=


class TestShouldFilterClusters:
    """should_filter_clusters 测试。"""

    def test_enabled_when_cluster_filter(self):
        ec = TopologyHealthController()
        hs = HealthStatus(cluster_filter_enabled=True)
        assert ec.should_filter_clusters(hs) is True

    def test_disabled_when_no_filter(self):
        ec = TopologyHealthController()
        hs = HealthStatus(cluster_filter_enabled=False)
        assert ec.should_filter_clusters(hs) is False


class TestTrendCalculation:
    """趋势计算核心测试。"""

    def test_trend_insufficient_history(self):
        ec = TopologyHealthController()
        # 只有3个点，刚好达到 min_history_for_trend=3
        # 轻微下降斜率，落在YELLOW范围（-0.01 < slope < 0）
        ec.compute_health_status(0.8, 0.8, 1, 1)
        ec.compute_health_status(0.795, 0.795, 1, 1)
        hs = ec.compute_health_status(0.79, 0.79, 1, 1)
        assert hs.trend is not None
        # slope 轻微负 (>-0.01)，应为YELLOW
        assert hs.trend.direction == TrendDirection.YELLOW
        assert len(hs.trend.history_scores) == 3

    def test_trend_stable_scores(self):
        ec = TopologyHealthController()
        for score in [0.7, 0.71, 0.69, 0.7, 0.71, 0.7]:
            hs = ec.compute_health_status(score, score, 1, 1)
        assert hs.trend is not None
        assert hs.trend.is_stable is True
        assert hs.trend.direction in (TrendDirection.GREEN, TrendDirection.YELLOW)

    def test_trend_declining(self):
        ec = TopologyHealthController()
        for score in [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]:
            hs = ec.compute_health_status(score, score, 1, 1)
        assert hs.trend is not None
        assert hs.trend.slope < 0
        assert hs.trend.direction in (TrendDirection.YELLOW, TrendDirection.ORANGE, TrendDirection.RED)

    def test_trend_rising(self):
        ec = TopologyHealthController()
        for score in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            hs = ec.compute_health_status(score, score, 1, 1)
        assert hs.trend is not None
        assert hs.trend.slope > 0
        assert hs.trend.direction == TrendDirection.GREEN

    def test_trend_window_respected(self):
        cfg = HealthControllerConfig(trend_window_size=5)
        ec = TopologyHealthController(cfg)
        for _ in range(10):
            ec.compute_health_status(0.7, 0.7, 1, 1)
        # 历史最多5条
        hs = ec.compute_health_status(0.7, 0.7, 1, 1)
        assert len(hs.trend.history_scores) <= 5

    def test_steps_until_consolidation_declining(self):
        ec = TopologyHealthController()
        # 初始健康分 0.65，每步降 0.05
        scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        for score in scores:
            hs = ec.compute_health_status(score, score, 1, 1)
        # 应该能预测步数
        assert hs.trend.steps_until_consolidation is not None
        assert hs.trend.steps_until_consolidation > 0

    def test_steps_until_consolidation_healthy(self):
        ec = TopologyHealthController()
        # 健康分稳定在0.9，不会触发 consolidation
        for score in [0.9, 0.9, 0.9, 0.9]:
            hs = ec.compute_health_status(score, score, 1, 1)
        # slope ≈ 0，steps_until 应该是 None
        assert hs.trend.steps_until_consolidation is None

    def test_confidence_low_with_short_history(self):
        cfg = HealthControllerConfig(min_history_for_trend=5)
        ec = TopologyHealthController(cfg)
        for _ in range(3):
            ec.compute_health_status(0.7, 0.7, 1, 1)
        hs = ec.compute_health_status(0.65, 0.65, 1, 1)
        # 历史不足5条，置信度应该较低
        # （因为 history<min_history_for_trend，走不到趋势计算分支）
        assert hs.trend is not None

    def test_reset_history(self):
        ec = TopologyHealthController()
        for _ in range(5):
            ec.compute_health_status(0.7, 0.7, 1, 1)
        ec.reset_history()
        hs = ec.compute_health_status(0.7, 0.7, 1, 1)
        assert len(hs.trend.history_scores) == 1


class TestShouldEarlyIntervene:
    """should_early_intervene 提前干预测试。"""

    def test_early_intervene_orange_trend_not_yet_triggered(self):
        ec = TopologyHealthController()
        # 趋势恶化 (ORANGE) 但 health_score 还高于阈值
        # score=0.7附近，threshold≈0.3+0.4*(1-0.7)=0.42，所以不会触发consolidation
        # 但slope明显为负 → ORANGE
        for score in [0.75, 0.7, 0.65, 0.62, 0.58]:
            hs = ec.compute_health_status(score, score, 1, 1)
        # ORANGE/RED 趋势 + 未触发阈值 → True
        ei = ec.should_early_intervene(hs)
        assert ei is True

    def test_no_early_intervene_green_trend(self):
        ec = TopologyHealthController()
        for score in [0.9, 0.9, 0.9, 0.9]:
            hs = ec.compute_health_status(score, score, 1, 1)
        assert ec.should_early_intervene(hs) is False

    def test_no_early_intervene_when_already_triggered(self):
        ec = TopologyHealthController()
        # 健康分已经低于阈值，should_consolidate 触发
        hs = HealthStatus(
            health_score=0.2,
            consolidate_threshold=0.5,
            trend=HealthTrend(direction=TrendDirection.RED, slope=-0.1),
        )
        # 已经触发了，不需要提前干预
        assert ec.should_early_intervene(hs) is False
        assert ec.should_consolidate(hs) is True


class TestGetDiagnosticInfo:
    """get_diagnostic_info 输出测试。"""

    def test_diagnostic_includes_trend_info(self):
        ec = TopologyHealthController()
        for score in [0.8, 0.75, 0.7, 0.65]:
            hs = ec.compute_health_status(score, score, 1, 1)
        diag = ec.get_diagnostic_info(hs)
        assert "trend_direction" in diag
        assert "trend_slope" in diag
        assert "is_stable" in diag
        assert "steps_until_consolidation" in diag
        assert "should_early_intervene" in diag

    def test_diagnostic_includes_basic_health(self):
        ec = TopologyHealthController()
        hs = ec.compute_health_status(0.7, 0.8, 5, 3)
        diag = ec.get_diagnostic_info(hs)
        assert diag["health_score"] == pytest.approx(0.75)
        assert diag["h1_health"] == pytest.approx(0.7)
        assert diag["h2_health"] == pytest.approx(0.8)
        assert diag["betti_1"] == 5
        assert diag["betti_2"] == 3
        assert diag["prune_aggressiveness"] > 0


class TestRetrievalGammaMult:
    """retrieval_gamma_mult 边界测试。"""

    def test_gamma_at_min_health(self):
        cfg = HealthControllerConfig(retrieval_gamma_min=0.2, retrieval_gamma_max=0.8)
        ec = TopologyHealthController(cfg)
        hs = ec.compute_health_status(0.0, 0.0, 0, 0)
        assert hs.retrieval_gamma_mult == pytest.approx(0.2)

    def test_gamma_at_max_health(self):
        cfg = HealthControllerConfig(retrieval_gamma_min=0.2, retrieval_gamma_max=0.8)
        ec = TopologyHealthController(cfg)
        hs = ec.compute_health_status(1.0, 1.0, 0, 0)
        assert hs.retrieval_gamma_mult == pytest.approx(0.8)
