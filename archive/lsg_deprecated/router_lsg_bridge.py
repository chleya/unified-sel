"""
Bridge: Capability Router decisions → LSG revision proposals.

Core idea:
  - Router decisions (accept/verify/escalate) are ephemeral per-task events.
  - Only certain decisions should propose changes to long-term system state.
  - LSG governs whether those proposals are approved and executed.

What triggers a revision proposal:
  1. Escalation rate exceeds threshold → propose policy adjustment
  2. Batch drift detected → propose conservative mode
  3. Monitor signal distribution shift → propose threshold recalibration

What does NOT trigger a revision proposal:
  - Single task accept/verify/escalate (per-task, no state change)
  - Confidence score on individual output (ephemeral)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.rewrite_dynamics import (
    CandidateState,
    CommitEvent,
    RewritePhase,
    RewriteSystemState,
    RevisionProposalEvent,
    RevisionExecutionEvent,
    RollbackEvent,
    draft_revision_execution_event,
    draft_rollback_event,
    execute_revision_event,
    execute_rollback_event,
    record_revision_execution_draft,
)


@dataclass(frozen=True)
class RouterDecision:
    """A single routing decision from Capability Router."""

    task_id: str
    decision_path: str
    monitor_signal: float
    confidence: float
    verified: bool
    escalated: bool
    revised: bool
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class DriftSignal:
    """A batch-level drift signal from TopoMem OBD."""

    status: str
    centroid_drift: float
    window_size: int
    task_count_in_window: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class RoutingPolicyState:
    """Current routing policy parameters (LSG-governed)."""

    escalation_threshold: float = 0.35
    verify_threshold: float = 0.50
    conservative_mode: bool = False
    drift_latch: bool = False
    version: int = 1


class RouterLSGBridge:
    """Bridge between Capability Router decisions and LSG governance.

    Flow:
      1. Router makes per-task decisions (accept/verify/escalate)
      2. Bridge aggregates decisions into policy-relevant signals
      3. When a signal crosses a threshold, bridge creates a RevisionProposalEvent
      4. LSG gates approve/reject the proposal
      5. If approved, bridge drafts and executes a RevisionExecutionEvent
      6. RoutingPolicyState is updated via LSG-controlled version increment
    """

    def __init__(
        self,
        lsg_state: RewriteSystemState,
        policy: RoutingPolicyState,
        escalation_rate_threshold: float = 0.5,
        drift_proposal_threshold: float = 0.3,
        window_size: int = 10,
    ):
        self.lsg_state = lsg_state
        self.policy = policy
        self.escalation_rate_threshold = escalation_rate_threshold
        self.drift_proposal_threshold = drift_proposal_threshold
        self.window_size = window_size

        self._recent_decisions: List[RouterDecision] = []
        self._recent_drift_signals: List[DriftSignal] = []
        self._pending_proposals: List[str] = []

    def record_decision(self, decision: RouterDecision) -> Optional[str]:
        """Record a routing decision. Returns proposal_event_id if a revision proposal was created."""

        self._recent_decisions.append(decision)

        if len(self._recent_decisions) > self.window_size * 2:
            self._recent_decisions = self._recent_decisions[-self.window_size:]

        escalation_rate = self._compute_escalation_rate()

        if escalation_rate > self.escalation_rate_threshold:
            has_pending = any(
                "escalation_rate" in p.reason
                for p in self.lsg_state.revision_log
                if not p.revision_executed
            )
            if not has_pending and not self.policy.conservative_mode:
                proposal_id = self._propose_conservative_mode(
                    reason=f"escalation_rate={escalation_rate:.2f}>{self.escalation_rate_threshold}"
                )
                if proposal_id:
                    self._pending_proposals.append(proposal_id)
                return proposal_id

        return None

    def record_drift_signal(self, signal: DriftSignal) -> Optional[str]:
        """Record a drift signal. Returns proposal_event_id if a revision proposal was created."""

        self._recent_drift_signals.append(signal)

        if signal.status in ("domain_shift_detected", "gradual_drift"):
            has_pending = any(
                "drift_status" in p.reason
                for p in self.lsg_state.revision_log
                if not p.revision_executed
            )
            if not has_pending and not self.policy.drift_latch:
                proposal_id = self._propose_drift_latch(
                    reason=f"drift_status={signal.status}, centroid_drift={signal.centroid_drift:.3f}"
                )
                if proposal_id:
                    self._pending_proposals.append(proposal_id)
                return proposal_id

        return None

    def approve_and_execute_proposal(self, proposal_event_id: str) -> Optional[RevisionExecutionEvent]:
        """Approve and execute a pending proposal through LSG."""

        matches = [p for p in self.lsg_state.revision_log if p.event_id == proposal_event_id]
        if not matches:
            return None

        proposal = matches[0]

        object.__setattr__(proposal, "approval_open", True)

        event = record_revision_execution_draft(
            self.lsg_state,
            proposal_event_id=proposal.event_id,
        )

        executed = execute_revision_event(
            self.lsg_state,
            execution_event_id=event.event_id,
        )

        self._apply_policy_change(proposal.reason)

        if proposal_event_id in self._pending_proposals:
            self._pending_proposals.remove(proposal_event_id)

        return executed

    def approve_and_rollback_proposal(self, proposal_event_id: str) -> Optional[RollbackEvent]:
        """Approve and rollback a previously executed proposal through LSG.

        This reverses the effect of a revision execution by decrementing
        the candidate version and restoring previous disturbance/stability.
        """
        matches = [p for p in self.lsg_state.revision_log if p.event_id == proposal_event_id]
        if not matches:
            return None

        proposal = matches[0]

        object.__setattr__(proposal, "approval_open", True)

        rollback = draft_rollback_event(
            self.lsg_state,
            proposal_event_id=proposal.event_id,
        )

        executed = execute_rollback_event(
            self.lsg_state,
            rollback_event_id=rollback.event_id,
        )

        self._revert_policy_change(proposal.reason)

        if proposal_event_id in self._pending_proposals:
            self._pending_proposals.remove(proposal_event_id)

        return executed

    def _compute_escalation_rate(self) -> float:
        """Compute escalation rate over recent window."""
        if not self._recent_decisions:
            return 0.0

        recent = self._recent_decisions[-self.window_size:]
        escalated = sum(1 for d in recent if d.escalated)
        return escalated / len(recent)

    def _ensure_routing_policy_candidate(self):
        """Create routing_policy candidate with commit event if not present."""
        candidate_id = "routing_policy"
        if candidate_id not in self.lsg_state.candidates:
            candidate = CandidateState(
                candidate_id=candidate_id,
                disturbance=0.3,
                stability=0.7,
                phase=RewritePhase.ACKNOWLEDGED,
                committed=True,
                version=self.policy.version,
            )
            self.lsg_state.candidates[candidate_id] = candidate
            self.lsg_state.active_candidate_ids.append(candidate_id)

            commit = CommitEvent(
                event_id=f"commit_{candidate_id}",
                step_index=0,
                candidate_id=candidate_id,
                from_phase="commit_review",
                to_phase="acknowledged",
                disturbance=0.3,
                stability=0.7,
                ratio=0.429,
                evidence_open=True,
                constitution_open=True,
                log_ready=True,
                commit_executed=True,
                candidate_version=self.policy.version,
            )
            self.lsg_state.commit_log.append(commit)

    def _propose_conservative_mode(self, reason: str) -> Optional[str]:
        """Create a revision proposal to enter conservative mode."""

        self._ensure_routing_policy_candidate()
        candidate_id = "routing_policy"

        candidate = self.lsg_state.candidates[candidate_id]
        proposal = RevisionProposalEvent(
            event_id=f"rp_conservative_{int(time.time()*1000)}",
            step_index=len(self.lsg_state.commit_log),
            candidate_id=candidate_id,
            reason=reason,
            disturbance=0.6,
            stability=0.4,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            approval_open=False,
            revision_executed=False,
            target_version=candidate.version,
        )
        self.lsg_state.revision_log.append(proposal)
        return proposal.event_id

    def _propose_drift_latch(self, reason: str) -> Optional[str]:
        """Create a revision proposal to enable drift latch."""

        self._ensure_routing_policy_candidate()
        candidate_id = "routing_policy"
        if candidate_id not in self.lsg_state.candidates:
            candidate = CandidateState(
                candidate_id=candidate_id,
                disturbance=0.3,
                stability=0.7,
                phase=RewritePhase.ACKNOWLEDGED,
                committed=True,
                version=self.policy.version,
            )
            self.lsg_state.candidates[candidate_id] = candidate
            self.lsg_state.active_candidate_ids.append(candidate_id)

        candidate = self.lsg_state.candidates[candidate_id]
        proposal = RevisionProposalEvent(
            event_id=f"rp_drift_latch_{int(time.time()*1000)}",
            step_index=len(self.lsg_state.commit_log),
            candidate_id=candidate_id,
            reason=reason,
            disturbance=0.5,
            stability=0.5,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            approval_open=False,
            revision_executed=False,
            target_version=candidate.version,
        )
        self.lsg_state.revision_log.append(proposal)
        return proposal.event_id

    def _apply_policy_change(self, reason: str):
        """Apply policy change after LSG execution approval."""

        self.policy.version += 1

        if "escalation_rate" in reason:
            self.policy.conservative_mode = True
            self.policy.escalation_threshold = 0.25
            self.policy.verify_threshold = 0.40

        if "drift_status" in reason:
            self.policy.drift_latch = True

    def _revert_policy_change(self, reason: str):
        """Revert policy change after LSG rollback approval."""

        self.policy.version = max(1, self.policy.version - 1)

        if "escalation_rate" in reason:
            self.policy.conservative_mode = False
            self.policy.escalation_threshold = 0.35
            self.policy.verify_threshold = 0.50

        if "drift_status" in reason:
            self.policy.drift_latch = False
