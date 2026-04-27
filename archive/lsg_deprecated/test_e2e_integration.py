"""End-to-end integration test: Router -> Bridge -> LSG -> Policy Change -> Rollback

This test exercises the complete three-system flow:
1. Capability Router makes per-task decisions
2. Bridge aggregates decisions into policy-relevant signals
3. LSG governs whether proposals are approved and executed
4. Policy changes only happen through LSG-controlled version increment
5. Rollback reverses policy changes through LSG-controlled version decrement
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    RewriteSystemState,
    check_revision_execution_log_invariants,
    check_rollback_log_invariants,
)
from core.router_lsg_bridge import (
    DriftSignal,
    RouterDecision,
    RouterLSGBridge,
    RoutingPolicyState,
)


class TestEndToEndEscalationFlow:
    """Simulate: tasks arrive, escalation rate rises, policy changes, then recovers."""

    def test_full_escalation_lifecycle(self):
        state = RewriteSystemState(bandwidth_limit=3)
        policy = RoutingPolicyState()
        bridge = RouterLSGBridge(
            lsg_state=state,
            policy=policy,
            escalation_rate_threshold=0.5,
            window_size=5,
        )

        assert policy.version == 1
        assert policy.conservative_mode is False
        assert policy.escalation_threshold == 0.35

        for i in range(5):
            decision = RouterDecision(
                task_id=f"easy_{i}",
                decision_path="accept",
                monitor_signal=0.8,
                confidence=0.9,
                verified=True,
                escalated=False,
                revised=False,
            )
            bridge.record_decision(decision)

        assert len(state.revision_log) == 0
        assert policy.conservative_mode is False

        for i in range(6):
            decision = RouterDecision(
                task_id=f"hard_{i}",
                decision_path="escalate_low_confidence",
                monitor_signal=0.2,
                confidence=0.3,
                verified=False,
                escalated=True,
                revised=False,
            )
            pid = bridge.record_decision(decision)

        assert len(state.revision_log) >= 1
        assert policy.conservative_mode is False

        proposal_id = state.revision_log[0].event_id
        executed = bridge.approve_and_execute_proposal(proposal_id)

        assert executed is not None
        assert executed.execution_executed is True
        assert policy.conservative_mode is True
        assert policy.version == 2
        assert policy.escalation_threshold == 0.25

        rev_inv = check_revision_execution_log_invariants(state)
        assert rev_inv["passed"] is True

        for i in range(5):
            decision = RouterDecision(
                task_id=f"recovery_{i}",
                decision_path="accept_after_verify",
                monitor_signal=0.7,
                confidence=0.8,
                verified=True,
                escalated=False,
                revised=False,
            )
            bridge.record_decision(decision)

        rollback = bridge.approve_and_rollback_proposal(proposal_id)
        assert rollback is not None
        assert rollback.rollback_executed is True
        assert policy.conservative_mode is False
        assert policy.version == 1
        assert policy.escalation_threshold == 0.35

        rb_inv = check_rollback_log_invariants(state)
        assert rb_inv["passed"] is True


class TestEndToEndDriftFlow:
    """Simulate: domain shift detected, drift latch enabled, then cleared."""

    def test_full_drift_lifecycle(self):
        state = RewriteSystemState(bandwidth_limit=3)
        policy = RoutingPolicyState()
        bridge = RouterLSGBridge(
            lsg_state=state,
            policy=policy,
            escalation_rate_threshold=0.5,
            window_size=5,
        )

        assert policy.drift_latch is False

        drift_signal = DriftSignal(
            status="domain_shift_detected",
            centroid_drift=0.713,
            window_size=5,
            task_count_in_window=5,
        )
        pid = bridge.record_drift_signal(drift_signal)

        assert pid is not None
        assert len(state.revision_log) == 1
        assert policy.drift_latch is False

        executed = bridge.approve_and_execute_proposal(pid)
        assert executed is not None
        assert policy.drift_latch is True
        assert policy.version == 2

        healthy_signal = DriftSignal(
            status="healthy",
            centroid_drift=0.03,
            window_size=5,
            task_count_in_window=5,
        )
        bridge.record_drift_signal(healthy_signal)

        rollback = bridge.approve_and_rollback_proposal(pid)
        assert rollback is not None
        assert policy.drift_latch is False
        assert policy.version == 1


class TestEndToEndCombinedFlow:
    """Simulate: both escalation and drift signals, with independent rollback."""

    def test_combined_escalation_and_drift(self):
        state = RewriteSystemState(bandwidth_limit=3)
        policy = RoutingPolicyState()
        bridge = RouterLSGBridge(
            lsg_state=state,
            policy=policy,
            escalation_rate_threshold=0.5,
            window_size=5,
        )

        for i in range(6):
            decision = RouterDecision(
                task_id=f"hard_{i}",
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

        assert policy.conservative_mode is True
        assert policy.version == 2

        drift_signal = DriftSignal(
            status="domain_shift_detected",
            centroid_drift=0.713,
            window_size=5,
            task_count_in_window=5,
        )
        pid_drift = bridge.record_drift_signal(drift_signal)
        assert pid_drift is not None

        executed_drift = bridge.approve_and_execute_proposal(pid_drift)
        assert executed_drift is not None
        assert policy.drift_latch is True
        assert policy.version == 3

        rollback_drift = bridge.approve_and_rollback_proposal(pid_drift)
        assert rollback_drift is not None
        assert policy.drift_latch is False
        assert policy.version == 2
        assert policy.conservative_mode is True

        escalation_proposal_id = state.revision_log[0].event_id
        rollback_esc = bridge.approve_and_rollback_proposal(escalation_proposal_id)
        assert rollback_esc is not None
        assert policy.conservative_mode is False
        assert policy.version == 1

        rev_inv = check_revision_execution_log_invariants(state)
        rb_inv = check_rollback_log_invariants(state)
        assert rev_inv["passed"] is True
        assert rb_inv["passed"] is True
