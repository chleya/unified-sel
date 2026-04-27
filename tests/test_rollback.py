"""Tests for LSG rollback protocol (Phase 26)."""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    check_rollback_log_invariants,
    check_revision_execution_log_invariants,
)


def _make_state_with_versioned_candidate(version=2):
    state = RewriteSystemState()
    candidate = CandidateState(
        candidate_id="c1",
        disturbance=0.6,
        stability=0.4,
        phase=RewritePhase.ACKNOWLEDGED,
        committed=True,
        version=version,
    )
    state.candidates["c1"] = candidate
    state.active_candidate_ids.append("c1")

    commit = CommitEvent(
        event_id="commit_c1",
        step_index=0,
        candidate_id="c1",
        from_phase="commit_review",
        to_phase="acknowledged",
        disturbance=0.3,
        stability=0.7,
        ratio=0.429,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
        commit_executed=True,
        candidate_version=1,
    )
    state.commit_log.append(commit)

    if version >= 2:
        proposal = RevisionProposalEvent(
            event_id="rp_c1_v2",
            step_index=1,
            candidate_id="c1",
            reason="escalation_rate=0.60>0.50",
            disturbance=0.6,
            stability=0.4,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            approval_open=True,
            revision_executed=False,
            target_version=1,
        )
        state.revision_log.append(proposal)

        exec_event = RevisionExecutionEvent(
            event_id="re_c1_v2",
            step_index=1,
            proposal_event_id="rp_c1_v2",
            candidate_id="c1",
            from_version=1,
            to_version=2,
            disturbance=0.6,
            stability=0.4,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            approval_open=True,
            execution_executed=True,
        )
        state.revision_execution_log.append(exec_event)

    return state


class TestRollbackEvent:
    def test_rollback_event_requires_version_decrement(self):
        with pytest.raises(ValueError, match="from_version - 1"):
            RollbackEvent(
                event_id="rb_1",
                step_index=0,
                proposal_event_id="rp_1",
                candidate_id="c1",
                from_version=2,
                to_version=3,
                disturbance=0.3,
                stability=0.7,
                evidence_open=True,
                constitution_open=True,
                log_ready=True,
                approval_open=True,
                rollback_executed=False,
            )

    def test_rollback_from_version_1_impossible(self):
        with pytest.raises(ValueError):
            RollbackEvent(
                event_id="rb_1",
                step_index=0,
                proposal_event_id="rp_1",
                candidate_id="c1",
                from_version=1,
                to_version=0,
                disturbance=0.3,
                stability=0.7,
                evidence_open=True,
                constitution_open=True,
                log_ready=True,
                approval_open=True,
                rollback_executed=False,
            )

    def test_rollback_requires_approval(self):
        with pytest.raises(ValueError, match="approval_open"):
            RollbackEvent(
                event_id="rb_1",
                step_index=0,
                proposal_event_id="rp_1",
                candidate_id="c1",
                from_version=2,
                to_version=1,
                disturbance=0.3,
                stability=0.7,
                evidence_open=True,
                constitution_open=True,
                log_ready=True,
                approval_open=False,
                rollback_executed=False,
            )


class TestDraftRollback:
    def test_draft_rollback_from_approved_proposal(self):
        state = _make_state_with_versioned_candidate(version=2)
        rollback = draft_rollback_event(state, proposal_event_id="rp_c1_v2")
        assert rollback.from_version == 2
        assert rollback.to_version == 1
        assert rollback.rollback_executed is False
        assert len(state.rollback_log) == 1

    def test_draft_rollback_restores_previous_state(self):
        state = _make_state_with_versioned_candidate(version=2)
        rollback = draft_rollback_event(state, proposal_event_id="rp_c1_v2")
        assert rollback.disturbance == 0.3
        assert rollback.stability == 0.7

    def test_draft_rollback_requires_approved_proposal(self):
        state = _make_state_with_versioned_candidate(version=2)
        state.revision_log[0] = RevisionProposalEvent(
            event_id="rp_c1_v2",
            step_index=1,
            candidate_id="c1",
            reason="test",
            disturbance=0.6,
            stability=0.4,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            approval_open=False,
            revision_executed=False,
            target_version=1,
        )
        with pytest.raises(ValueError, match="approved"):
            draft_rollback_event(state, proposal_event_id="rp_c1_v2")

    def test_draft_rollback_version_1_impossible(self):
        state = _make_state_with_versioned_candidate(version=1)
        proposal = RevisionProposalEvent(
            event_id="rp_c1_v1",
            step_index=0,
            candidate_id="c1",
            reason="test",
            disturbance=0.3,
            stability=0.7,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            approval_open=True,
            revision_executed=False,
            target_version=1,
        )
        state.revision_log.append(proposal)
        with pytest.raises(ValueError, match="version 1"):
            draft_rollback_event(state, proposal_event_id="rp_c1_v1")


class TestExecuteRollback:
    def test_execute_rollback_decrements_version(self):
        state = _make_state_with_versioned_candidate(version=2)
        rollback = draft_rollback_event(state, proposal_event_id="rp_c1_v2")
        executed = execute_rollback_event(state, rollback_event_id=rollback.event_id)
        assert executed.rollback_executed is True
        assert state.candidates["c1"].version == 1
        assert state.candidates["c1"].disturbance == 0.3
        assert state.candidates["c1"].stability == 0.7

    def test_execute_rollback_double_fails(self):
        state = _make_state_with_versioned_candidate(version=2)
        rollback = draft_rollback_event(state, proposal_event_id="rp_c1_v2")
        execute_rollback_event(state, rollback_event_id=rollback.event_id)
        with pytest.raises(ValueError, match="already executed"):
            execute_rollback_event(state, rollback_event_id=rollback.event_id)

    def test_execute_rollback_version_mismatch_fails(self):
        state = _make_state_with_versioned_candidate(version=2)
        rollback = draft_rollback_event(state, proposal_event_id="rp_c1_v2")
        state.candidates["c1"].version = 5
        with pytest.raises(ValueError, match="validation failed"):
            execute_rollback_event(state, rollback_event_id=rollback.event_id)

    def test_rollback_log_invariants_pass_after_execution(self):
        state = _make_state_with_versioned_candidate(version=2)
        rollback = draft_rollback_event(state, proposal_event_id="rp_c1_v2")
        execute_rollback_event(state, rollback_event_id=rollback.event_id)
        result = check_rollback_log_invariants(state)
        assert result["passed"] is True
        assert result["executed_rollback_count"] == 1


class TestRollbackAndRevisionCoexistence:
    def test_revision_then_rollback_roundtrip(self):
        state = _make_state_with_versioned_candidate(version=1)
        state.candidates["c1"].disturbance = 0.3
        state.candidates["c1"].stability = 0.7

        proposal = RevisionProposalEvent(
            event_id="rp_c1_rev",
            step_index=1,
            candidate_id="c1",
            reason="policy update",
            disturbance=0.6,
            stability=0.4,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            approval_open=True,
            revision_executed=False,
            target_version=1,
        )
        state.revision_log.append(proposal)

        exec_event = record_revision_execution_draft(
            state, proposal_event_id="rp_c1_rev"
        )
        execute_revision_event(state, execution_event_id=exec_event.event_id)

        assert state.candidates["c1"].version == 2
        assert state.candidates["c1"].disturbance == 0.6
        assert state.candidates["c1"].stability == 0.4

        rollback_proposal = RevisionProposalEvent(
            event_id="rp_c1_rollback",
            step_index=2,
            candidate_id="c1",
            reason="rollback needed",
            disturbance=0.3,
            stability=0.7,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            approval_open=True,
            revision_executed=False,
            target_version=2,
        )
        state.revision_log.append(rollback_proposal)

        rollback = draft_rollback_event(state, proposal_event_id="rp_c1_rollback")
        execute_rollback_event(state, rollback_event_id=rollback.event_id)

        assert state.candidates["c1"].version == 1
        assert state.candidates["c1"].disturbance == 0.3
        assert state.candidates["c1"].stability == 0.7

        rev_result = check_revision_execution_log_invariants(state)
        rb_result = check_rollback_log_invariants(state)
        assert rev_result["passed"] is True
        assert rb_result["passed"] is True
