"""Tests for Router-LSG bridge integration."""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    CandidateState,
    RewritePhase,
    RewriteSystemState,
)
from core.router_lsg_bridge import (
    DriftSignal,
    RouterDecision,
    RouterLSGBridge,
    RoutingPolicyState,
)


def _make_bridge() -> RouterLSGBridge:
    lsg_state = RewriteSystemState(bandwidth_limit=3)
    policy = RoutingPolicyState()
    return RouterLSGBridge(
        lsg_state=lsg_state,
        policy=policy,
        escalation_rate_threshold=0.5,
        window_size=5,
    )


class TestRouterDecisions:
    def test_low_escalation_rate_no_proposal(self):
        bridge = _make_bridge()
        for i in range(5):
            decision = RouterDecision(
                task_id=f"t{i}",
                decision_path="accept_after_verify",
                monitor_signal=0.6,
                confidence=0.7,
                verified=True,
                escalated=False,
                revised=False,
            )
            result = bridge.record_decision(decision)
            assert result is None

    def test_high_escalation_rate_creates_proposal(self):
        bridge = _make_bridge()
        proposal_ids = []
        for i in range(6):
            decision = RouterDecision(
                task_id=f"t{i}",
                decision_path="escalate_low_confidence",
                monitor_signal=0.2,
                confidence=0.3,
                verified=False,
                escalated=True,
                revised=False,
            )
            pid = bridge.record_decision(decision)
            if pid:
                proposal_ids.append(pid)

        assert len(proposal_ids) >= 1
        assert len(bridge.lsg_state.revision_log) >= 1

    def test_conservative_mode_prevents_duplicate_proposals(self):
        bridge = _make_bridge()
        for i in range(6):
            decision = RouterDecision(
                task_id=f"t{i}",
                decision_path="escalate_low_confidence",
                monitor_signal=0.2,
                confidence=0.3,
                verified=False,
                escalated=True,
                revised=False,
            )
            bridge.record_decision(decision)

        first_count = len(bridge.lsg_state.revision_log)

        for i in range(6):
            decision = RouterDecision(
                task_id=f"t{i+10}",
                decision_path="escalate_low_confidence",
                monitor_signal=0.2,
                confidence=0.3,
                verified=False,
                escalated=True,
                revised=False,
            )
            bridge.record_decision(decision)

        assert len(bridge.lsg_state.revision_log) == first_count


class TestDriftSignals:
    def test_domain_shift_creates_proposal(self):
        bridge = _make_bridge()
        signal = DriftSignal(
            status="domain_shift_detected",
            centroid_drift=0.713,
            window_size=5,
            task_count_in_window=5,
        )
        pid = bridge.record_drift_signal(signal)
        assert pid is not None
        assert len(bridge.lsg_state.revision_log) == 1

    def test_no_drift_no_proposal(self):
        bridge = _make_bridge()
        signal = DriftSignal(
            status="healthy",
            centroid_drift=0.043,
            window_size=5,
            task_count_in_window=5,
        )
        pid = bridge.record_drift_signal(signal)
        assert pid is None

    def test_drift_latch_prevents_duplicate(self):
        bridge = _make_bridge()
        signal1 = DriftSignal(
            status="domain_shift_detected",
            centroid_drift=0.713,
            window_size=5,
            task_count_in_window=5,
        )
        pid1 = bridge.record_drift_signal(signal1)
        assert pid1 is not None

        signal2 = DriftSignal(
            status="gradual_drift",
            centroid_drift=0.091,
            window_size=5,
            task_count_in_window=5,
        )
        pid2 = bridge.record_drift_signal(signal2)
        assert pid2 is None


class TestApprovalAndExecution:
    def test_approve_and_execute_escalation_proposal(self):
        bridge = _make_bridge()
        assert bridge.policy.conservative_mode is False
        assert bridge.policy.version == 1

        for i in range(6):
            decision = RouterDecision(
                task_id=f"t{i}",
                decision_path="escalate_low_confidence",
                monitor_signal=0.2,
                confidence=0.3,
                verified=False,
                escalated=True,
                revised=False,
            )
            pid = bridge.record_decision(decision)
            if pid:
                executed = bridge.approve_and_execute_proposal(pid)
                assert executed is not None
                assert executed.execution_executed is True

        assert bridge.policy.conservative_mode is True
        assert bridge.policy.version == 2
        assert bridge.policy.escalation_threshold == 0.25

    def test_approve_and_execute_drift_proposal(self):
        bridge = _make_bridge()
        assert bridge.policy.drift_latch is False

        signal = DriftSignal(
            status="domain_shift_detected",
            centroid_drift=0.713,
            window_size=5,
            task_count_in_window=5,
        )
        pid = bridge.record_drift_signal(signal)
        assert pid is not None

        executed = bridge.approve_and_execute_proposal(pid)
        assert executed is not None
        assert executed.execution_executed is True

        assert bridge.policy.drift_latch is True
        assert bridge.policy.version == 2

    def test_unapproved_proposal_does_not_change_policy(self):
        bridge = _make_bridge()

        for i in range(6):
            decision = RouterDecision(
                task_id=f"t{i}",
                decision_path="escalate_low_confidence",
                monitor_signal=0.2,
                confidence=0.3,
                verified=False,
                escalated=True,
                revised=False,
            )
            bridge.record_decision(decision)

        assert bridge.policy.conservative_mode is False
        assert bridge.policy.version == 1


class TestRollbackViaBridge:
    def test_approve_and_rollback_escalation_proposal(self):
        bridge = _make_bridge()
        assert bridge.policy.conservative_mode is False
        assert bridge.policy.version == 1

        for i in range(6):
            decision = RouterDecision(
                task_id=f"t{i}",
                decision_path="escalate_low_confidence",
                monitor_signal=0.2,
                confidence=0.3,
                verified=False,
                escalated=True,
                revised=False,
            )
            pid = bridge.record_decision(decision)
            if pid:
                executed = bridge.approve_and_execute_proposal(pid)
                assert executed is not None

        assert bridge.policy.conservative_mode is True
        assert bridge.policy.version == 2

        proposal_id = bridge.lsg_state.revision_log[0].event_id
        rollback = bridge.approve_and_rollback_proposal(proposal_id)
        assert rollback is not None
        assert rollback.rollback_executed is True

        assert bridge.policy.conservative_mode is False
        assert bridge.policy.version == 1
        assert bridge.policy.escalation_threshold == 0.35

    def test_approve_and_rollback_drift_proposal(self):
        bridge = _make_bridge()
        signal = DriftSignal(
            status="domain_shift_detected",
            centroid_drift=0.713,
            window_size=5,
            task_count_in_window=5,
        )
        pid = bridge.record_drift_signal(signal)
        executed = bridge.approve_and_execute_proposal(pid)
        assert executed is not None

        assert bridge.policy.drift_latch is True
        assert bridge.policy.version == 2

        rollback = bridge.approve_and_rollback_proposal(pid)
        assert rollback is not None
        assert rollback.rollback_executed is True

        assert bridge.policy.drift_latch is False
        assert bridge.policy.version == 1
