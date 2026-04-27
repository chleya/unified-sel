"""Tests for LSG Phase 25: RevisionExecutionEvent state mutation."""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    CandidateState,
    CandidateObservation,
    CommitEvent,
    RewriteDynamicsConfig,
    RewritePhase,
    RewriteSystemState,
    RevisionProposalEvent,
    RevisionExecutionEvent,
    propose_revision_for_acknowledged_candidate,
    draft_revision_execution_event,
    record_revision_execution_draft,
    execute_revision_event,
    check_revision_execution_event_against_state,
    check_revision_execution_log_invariants,
)


def _make_acknowledged_state() -> RewriteSystemState:
    """Create a state with one acknowledged candidate at version 1."""
    state = RewriteSystemState(bandwidth_limit=3)
    candidate = CandidateState(
        candidate_id="c1",
        disturbance=0.5,
        stability=0.5,
        phase=RewritePhase.ACKNOWLEDGED,
        committed=True,
        version=1,
    )
    state.candidates["c1"] = candidate
    state.active_candidate_ids.append("c1")
    return state


def _make_commit_event(candidate_id: str = "c1") -> CommitEvent:
    return CommitEvent(
        event_id="commit_1",
        step_index=0,
        candidate_id=candidate_id,
        from_phase=RewritePhase.COMMIT_REVIEW.value,
        to_phase=RewritePhase.ACKNOWLEDGED.value,
        disturbance=0.5,
        stability=0.5,
        ratio=1.0,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
        commit_executed=True,
        candidate_version=1,
    )


def _make_revision_proposal(
    candidate_id: str = "c1",
    approved: bool = True,
    executed: bool = False,
) -> RevisionProposalEvent:
    return RevisionProposalEvent(
        event_id="rev_prop_1",
        step_index=1,
        candidate_id=candidate_id,
        reason="test revision",
        disturbance=0.7,
        stability=0.6,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
        approval_open=approved,
        revision_executed=executed,
        target_version=1,
    )


class TestRevisionExecution:
    """Phase 25: approved execution transitions candidate to version 2."""

    def test_execute_approved_revision_increments_version(self):
        state = _make_acknowledged_state()
        state.commit_log.append(_make_commit_event())

        proposal = _make_revision_proposal(approved=True, executed=False)
        state.revision_log.append(proposal)

        # Draft execution event
        event = record_revision_execution_draft(
            state,
            proposal_event_id=proposal.event_id,
        )
        assert event.execution_executed is False
        assert event.from_version == 1
        assert event.to_version == 2

        # Execute
        executed_event = execute_revision_event(state, execution_event_id=event.event_id)

        # Verify state mutation
        candidate = state.candidates["c1"]
        assert candidate.version == 2
        assert candidate.disturbance == 0.7
        assert candidate.stability == 0.6
        assert executed_event.execution_executed is True

        # Verify proposal marked executed
        assert proposal.revision_executed is True

    def test_execute_without_approval_fails(self):
        state = _make_acknowledged_state()
        state.commit_log.append(_make_commit_event())

        proposal = _make_revision_proposal(approved=False, executed=False)
        state.revision_log.append(proposal)

        with pytest.raises(ValueError, match="requires an approved revision proposal"):
            draft_revision_execution_event(
                state,
                proposal_event_id=proposal.event_id,
            )

    def test_execute_already_executed_fails(self):
        state = _make_acknowledged_state()
        state.commit_log.append(_make_commit_event())

        proposal = _make_revision_proposal(approved=True, executed=False)
        state.revision_log.append(proposal)

        event = record_revision_execution_draft(
            state,
            proposal_event_id=proposal.event_id,
        )

        # First execution succeeds
        execute_revision_event(state, execution_event_id=event.event_id)

        # Second execution fails
        with pytest.raises(ValueError, match="already executed"):
            execute_revision_event(state, execution_event_id=event.event_id)

    def test_execute_version_mismatch_fails(self):
        state = _make_acknowledged_state()
        state.commit_log.append(_make_commit_event())

        # Candidate is version 1, but proposal targets version 2
        proposal = RevisionProposalEvent(
            event_id="rev_prop_2",
            step_index=1,
            candidate_id="c1",
            reason="wrong version",
            disturbance=0.7,
            stability=0.6,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            approval_open=True,
            revision_executed=False,
            target_version=2,  # Wrong version
        )
        state.revision_log.append(proposal)

        with pytest.raises(ValueError, match="target_version must match current candidate version"):
            draft_revision_execution_event(
                state,
                proposal_event_id=proposal.event_id,
            )

    def test_invariants_pass_after_execution(self):
        state = _make_acknowledged_state()
        state.commit_log.append(_make_commit_event())

        proposal = _make_revision_proposal(approved=True, executed=False)
        state.revision_log.append(proposal)

        event = record_revision_execution_draft(
            state,
            proposal_event_id=proposal.event_id,
        )
        execute_revision_event(state, execution_event_id=event.event_id)

        invariants = check_revision_execution_log_invariants(state)
        assert invariants["passed"] is True
        assert invariants["executed_revision_execution_count"] == 1

    def test_execution_log_tracks_executed_events(self):
        state = _make_acknowledged_state()
        state.commit_log.append(_make_commit_event())

        proposal = _make_revision_proposal(approved=True, executed=False)
        state.revision_log.append(proposal)

        event = record_revision_execution_draft(
            state,
            proposal_event_id=proposal.event_id,
        )
        execute_revision_event(state, execution_event_id=event.event_id)

        assert len(state.revision_execution_log) == 1
        assert state.revision_execution_log[0].execution_executed is True
