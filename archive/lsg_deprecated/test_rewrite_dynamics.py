from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rewrite_dynamics import (
    CandidateObservation,
    CommitEvent,
    ProposalEnvelope,
    ProxyVector,
    RevisionExecutionEvent,
    RevisionProposalEvent,
    RewriteDynamicsConfig,
    RewritePhase,
    check_revision_execution_log_invariants,
    check_revision_execution_event_against_state,
    check_revision_log_invariants,
    compute_governance_scalars,
    count_phase_flips,
    draft_revision_execution_event,
    record_revision_execution_draft,
    observation_from_proxy,
    observation_from_proposal,
    project_commit_event_to_cee,
    propose_revision_for_acknowledged_candidate,
    proposal_audit_record,
    simulate_case,
)


def obs(candidate_id, d, s, e=False, k=True, log=True):
    return CandidateObservation(
        candidate_id=candidate_id,
        disturbance_observed=d,
        stability_observed=s,
        evidence_open=e,
        constitution_open=k,
        log_ready=log,
    )


def test_core_dataclasses_reject_invalid_values():
    invalid_cases = [
        (
            lambda: CandidateObservation("c", 1.2, 0.1, False, True, True),
            "disturbance_observed",
        ),
        (
            lambda: CandidateObservation("c", 0.2, 0.1, "false", True, True),
            "evidence_open",
        ),
        (
            lambda: ProxyVector(0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1),
            "a1_institutional_level",
        ),
        (
            lambda: ProxyVector(0.1, 0.1, True, 0.1, 0.1, 0.1, 0.1, 0.1),
            "n1_goal_loss_if_ignored",
        ),
        (
            lambda: ProposalEnvelope(
                "",
                "c",
                "summary",
                ProxyVector(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            ),
            "proposal_id",
        ),
        (
            lambda: ProposalEnvelope(
                "p",
                "c",
                "summary",
                ProxyVector(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                requested_evidence_open="yes",
            ),
            "requested_evidence_open",
        ),
        (
            lambda: RewriteDynamicsConfig(alpha=1.2),
            "alpha",
        ),
        (
            lambda: RewriteDynamicsConfig(theta_verify_ratio=3.0, theta_commit_ratio=2.0),
            "theta_verify_ratio",
        ),
        (
            lambda: RewriteDynamicsConfig(bandwidth_limit=0),
            "bandwidth_limit",
        ),
    ]
    for factory, expected in invalid_cases:
        try:
            factory()
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{expected} should fail")
    print("[OK] core dataclasses reject invalid values")


def test_commit_event_rejects_invalid_values():
    valid_kwargs = {
        "event_id": "ce_test",
        "step_index": 1,
        "candidate_id": "candidate",
        "from_phase": RewritePhase.COMMIT_REVIEW.value,
        "to_phase": RewritePhase.ACKNOWLEDGED.value,
        "disturbance": 0.9,
        "stability": 0.1,
        "ratio": 9.0,
        "evidence_open": True,
        "constitution_open": True,
        "log_ready": True,
        "commit_executed": True,
    }
    assert CommitEvent(**valid_kwargs).commit_executed is True
    invalid_cases = [
        ({**valid_kwargs, "step_index": -1}, "step_index"),
        ({**valid_kwargs, "ratio": -0.1}, "ratio"),
        ({**valid_kwargs, "evidence_open": "true"}, "evidence_open"),
        ({**valid_kwargs, "evidence_open": False}, "all gates open"),
        ({**valid_kwargs, "event_id": ""}, "event_id"),
        ({**valid_kwargs, "candidate_version": 0}, "candidate_version"),
    ]
    for kwargs, expected in invalid_cases:
        try:
            CommitEvent(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{expected} should fail")
    print("[OK] commit event rejects invalid values")


def test_temporary_spike_never_acknowledges():
    case = [
        [obs("temp", 0.1, 0.8, e=False)],
        [obs("temp", 0.9, 0.8, e=False)],
        [obs("temp", 0.1, 0.8, e=False)],
        [obs("temp", 0.1, 0.8, e=False)],
    ]
    state, _ = simulate_case(case)
    assert not state.candidates["temp"].committed
    assert len(state.commit_log) == 0
    print("[OK] temporary spike never acknowledges")


def test_sustained_drift_acknowledges():
    case = [
        [obs("drift", 0.2, 0.8, e=False)],
        [obs("drift", 0.7, 0.7, e=False)],
        [obs("drift", 0.8, 0.5, e=True)],
        [obs("drift", 0.85, 0.35, e=True)],
        [obs("drift", 0.9, 0.25, e=True)],
    ]
    state, timeline = simulate_case(case)
    phases = [
        row["candidates"]["drift"]["phase"]
        for row in timeline
    ]
    assert RewritePhase.COMMIT_REVIEW.value in phases or state.candidates["drift"].committed
    assert state.candidates["drift"].committed
    assert len(state.commit_log) == 1
    assert state.commit_log[0].evidence_open
    assert state.commit_log[0].constitution_open
    assert state.commit_log[0].log_ready
    print("[OK] sustained drift acknowledges")


def test_candidate_commits_once_after_gates_open_late():
    case = [
        [obs("late_gate", 0.2, 0.8, e=False, k=True, log=True)],
        [obs("late_gate", 0.8, 0.4, e=False, k=True, log=True)],
        [obs("late_gate", 0.9, 0.2, e=False, k=True, log=True)],
        [obs("late_gate", 0.9, 0.2, e=True, k=True, log=True)],
        [obs("late_gate", 0.9, 0.2, e=True, k=True, log=True)],
    ]
    state, timeline = simulate_case(case, RewriteDynamicsConfig(alpha=1.0))
    assert state.candidates["late_gate"].committed
    assert state.candidates["late_gate"].phase == RewritePhase.ACKNOWLEDGED
    assert len(state.commit_log) == 1
    assert state.commit_log[0].step_index == 4
    assert timeline[-1]["commit_events"] == 1
    print("[OK] candidate commits once after gates open late")


def test_acknowledged_candidate_is_absorbing_under_later_observations():
    case = [
        [obs("absorbing", 0.9, 0.2, e=True, k=True, log=True)],
        [obs("absorbing", 0.0, 1.0, e=False, k=False, log=False)],
        [obs("absorbing", 0.1, 0.9, e=False, k=False, log=False)],
    ]
    state, timeline = simulate_case(case, RewriteDynamicsConfig(alpha=1.0))
    candidate = state.candidates["absorbing"]
    assert candidate.committed
    assert candidate.phase == RewritePhase.ACKNOWLEDGED
    assert len(state.commit_log) == 1
    first = timeline[0]["candidates"]["absorbing"]
    last = timeline[-1]["candidates"]["absorbing"]
    assert last["disturbance"] == first["disturbance"]
    assert last["stability"] == first["stability"]
    assert all(
        row["candidates"]["absorbing"]["phase"] == RewritePhase.ACKNOWLEDGED.value
        for row in timeline
    )
    print("[OK] acknowledged candidate is absorbing under later observations")


def test_revision_event_rejects_invalid_execution():
    valid_kwargs = {
        "event_id": "rpe_test",
        "step_index": 1,
        "candidate_id": "candidate",
        "reason": "correct stale acknowledgement",
        "disturbance": 0.2,
        "stability": 0.9,
        "evidence_open": True,
        "constitution_open": True,
        "log_ready": True,
        "approval_open": False,
        "revision_executed": False,
    }
    assert RevisionProposalEvent(**valid_kwargs).revision_executed is False
    invalid_cases = [
        ({**valid_kwargs, "event_id": ""}, "event_id"),
        ({**valid_kwargs, "step_index": -1}, "step_index"),
        ({**valid_kwargs, "reason": ""}, "reason"),
        ({**valid_kwargs, "disturbance": 1.2}, "disturbance"),
        ({**valid_kwargs, "approval_open": "true"}, "approval_open"),
        ({**valid_kwargs, "target_version": 0}, "target_version"),
        ({**valid_kwargs, "revision_executed": True}, "Executed revision requires"),
    ]
    for kwargs, expected in invalid_cases:
        try:
            RevisionProposalEvent(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{expected} should fail")
    print("[OK] revision proposal event rejects invalid execution")


def test_revision_execution_event_is_schema_only():
    valid_kwargs = {
        "event_id": "ree_test",
        "step_index": 1,
        "proposal_event_id": "rpe_test",
        "candidate_id": "candidate",
        "from_version": 1,
        "to_version": 2,
        "disturbance": 0.2,
        "stability": 0.9,
        "evidence_open": True,
        "constitution_open": True,
        "log_ready": True,
        "approval_open": True,
        "execution_executed": False,
    }
    assert RevisionExecutionEvent(**valid_kwargs).to_version == 2
    invalid_cases = [
        ({**valid_kwargs, "event_id": ""}, "event_id"),
        ({**valid_kwargs, "proposal_event_id": ""}, "proposal_event_id"),
        ({**valid_kwargs, "from_version": 0}, "from_version"),
        ({**valid_kwargs, "to_version": 3}, "to_version"),
        ({**valid_kwargs, "approval_open": False}, "approval_open"),
    ]
    for kwargs, expected in invalid_cases:
        try:
            RevisionExecutionEvent(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{expected} should fail")
    print("[OK] revision execution event is schema-only")


def test_revision_proposal_for_acknowledged_candidate_is_audit_only():
    state, _ = simulate_case([
        [obs("revision_target", 0.9, 0.2, e=True, k=True, log=True)],
    ], RewriteDynamicsConfig(alpha=1.0))
    candidate = state.candidates["revision_target"]
    before = (
        candidate.disturbance,
        candidate.stability,
        candidate.phase,
        candidate.committed,
        candidate.version,
        len(state.commit_log),
    )

    event = propose_revision_for_acknowledged_candidate(
        state,
        candidate_id="revision_target",
        reason="later evidence says this anchor should be reconsidered",
        disturbance_observed=0.05,
        stability_observed=0.95,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
    )
    after_candidate = state.candidates["revision_target"]
    after = (
        after_candidate.disturbance,
        after_candidate.stability,
        after_candidate.phase,
        after_candidate.committed,
        after_candidate.version,
        len(state.commit_log),
    )

    assert before == after
    assert event.candidate_id == "revision_target"
    assert event.target_version == after_candidate.version
    assert state.commit_log[0].candidate_version == after_candidate.version
    assert event.revision_executed is False
    assert len(state.revision_log) == 1
    revision_invariants = check_revision_log_invariants(state)
    assert revision_invariants["passed"] is True
    assert revision_invariants["approved_revision_count"] == 0
    assert revision_invariants["executed_revision_count"] == 0
    print("[OK] acknowledged revision proposal is audit-only")


def test_draft_revision_execution_event_requires_approved_proposal_and_does_not_mutate():
    state, _ = simulate_case([
        [obs("execution_target", 0.9, 0.2, e=True, k=True, log=True)],
    ], RewriteDynamicsConfig(alpha=1.0))
    proposal = propose_revision_for_acknowledged_candidate(
        state,
        candidate_id="execution_target",
        reason="approved revision can be drafted but not executed",
        disturbance_observed=0.05,
        stability_observed=0.95,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
        approval_open=True,
    )
    candidate = state.candidates["execution_target"]
    before = (
        candidate.disturbance,
        candidate.stability,
        candidate.version,
        len(state.commit_log),
        len(state.revision_log),
    )
    execution = draft_revision_execution_event(
        state,
        proposal_event_id=proposal.event_id,
    )
    after_candidate = state.candidates["execution_target"]
    after = (
        after_candidate.disturbance,
        after_candidate.stability,
        after_candidate.version,
        len(state.commit_log),
        len(state.revision_log),
    )
    assert before == after
    assert execution.proposal_event_id == proposal.event_id
    assert execution.from_version == 1
    assert execution.to_version == 2
    assert execution.execution_executed is False
    assert proposal.revision_executed is False
    execution_invariants = check_revision_execution_event_against_state(state, execution)
    assert execution_invariants["passed"] is True
    print("[OK] draft revision execution event requires approved proposal and does not mutate")


def test_record_revision_execution_draft_adds_log_without_mutating():
    state, _ = simulate_case([
        [obs("execution_log_target", 0.9, 0.2, e=True, k=True, log=True)],
    ], RewriteDynamicsConfig(alpha=1.0))
    proposal = propose_revision_for_acknowledged_candidate(
        state,
        candidate_id="execution_log_target",
        reason="approved revision can be logged as an execution draft",
        disturbance_observed=0.05,
        stability_observed=0.95,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
        approval_open=True,
    )
    candidate = state.candidates["execution_log_target"]
    before = (candidate.disturbance, candidate.stability, candidate.version)
    event = record_revision_execution_draft(
        state,
        proposal_event_id=proposal.event_id,
    )
    after_candidate = state.candidates["execution_log_target"]
    after = (after_candidate.disturbance, after_candidate.stability, after_candidate.version)
    assert before == after
    assert event.proposal_event_id == proposal.event_id
    assert len(state.revision_execution_log) == 1
    execution_log_invariants = check_revision_execution_log_invariants(state)
    assert execution_log_invariants["passed"] is True
    assert execution_log_invariants["num_revision_execution_events"] == 1
    assert execution_log_invariants["executed_revision_execution_count"] == 0
    print("[OK] record revision execution draft adds log without mutating")


def test_draft_revision_execution_event_rejects_unapproved_proposal():
    state, _ = simulate_case([
        [obs("unapproved_execution_target", 0.9, 0.2, e=True, k=True, log=True)],
    ], RewriteDynamicsConfig(alpha=1.0))
    proposal = propose_revision_for_acknowledged_candidate(
        state,
        candidate_id="unapproved_execution_target",
        reason="unapproved revision cannot be drafted for execution",
        disturbance_observed=0.05,
        stability_observed=0.95,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
        approval_open=False,
    )
    try:
        draft_revision_execution_event(state, proposal_event_id=proposal.event_id)
    except ValueError as exc:
        assert "approved" in str(exc)
    else:
        raise AssertionError("execution draft should reject unapproved proposal")
    print("[OK] draft revision execution event rejects unapproved proposal")


def test_revision_proposal_rejects_version_mismatch():
    state, _ = simulate_case([
        [obs("versioned_target", 0.9, 0.2, e=True, k=True, log=True)],
    ], RewriteDynamicsConfig(alpha=1.0))
    try:
        propose_revision_for_acknowledged_candidate(
            state,
            candidate_id="versioned_target",
            reason="stale version should not revise current candidate",
            disturbance_observed=0.1,
            stability_observed=0.9,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
            target_version=2,
        )
    except ValueError as exc:
        assert "target_version" in str(exc)
    else:
        raise AssertionError("revision proposal should reject stale target version")
    print("[OK] revision proposal rejects version mismatch")


def test_revision_proposal_rejects_unacknowledged_candidate():
    state, _ = simulate_case([
        [obs("pending_target", 0.4, 0.8, e=False, k=True, log=True)],
    ], RewriteDynamicsConfig(alpha=1.0))
    try:
        propose_revision_for_acknowledged_candidate(
            state,
            candidate_id="pending_target",
            reason="not acknowledged yet",
            disturbance_observed=0.9,
            stability_observed=0.1,
            evidence_open=True,
            constitution_open=True,
            log_ready=True,
        )
    except ValueError as exc:
        assert "already be acknowledged" in str(exc)
    else:
        raise AssertionError("revision proposal should reject unacknowledged target")
    print("[OK] revision proposal rejects unacknowledged candidate")


def test_protected_boundary_never_acknowledges():
    case = [
        [obs("protected", 0.9, 0.9, e=True, k=False)],
        [obs("protected", 0.95, 0.9, e=True, k=False)],
        [obs("protected", 0.9, 0.85, e=True, k=False)],
    ]
    state, timeline = simulate_case(case)
    assert not state.candidates["protected"].committed
    assert len(state.commit_log) == 0
    phases = {row["candidates"]["protected"]["phase"] for row in timeline}
    assert RewritePhase.ACKNOWLEDGED.value not in phases
    print("[OK] protected boundary never acknowledges")


def test_hysteresis_reduces_phase_flips():
    hysteretic_cfg = RewriteDynamicsConfig(alpha=1.0)
    single_threshold_cfg = RewriteDynamicsConfig(
        alpha=1.0,
        theta_fg_enter=0.05,
        theta_fg_exit=0.05,
    )
    values = [
        (0.54, 0.47),
        (0.56, 0.46),
        (0.53, 0.47),
        (0.57, 0.46),
        (0.52, 0.47),
        (0.58, 0.46),
    ]
    case = [[obs("chatter", d, s, e=False)] for d, s in values]
    _, hysteretic_timeline = simulate_case(case, hysteretic_cfg)
    _, single_timeline = simulate_case(case, single_threshold_cfg)
    assert count_phase_flips(hysteretic_timeline, "chatter") < count_phase_flips(
        single_timeline, "chatter"
    )
    print("[OK] hysteresis reduces phase flips")


def test_bandwidth_limit_blocks_inactive_acknowledgement():
    cfg = RewriteDynamicsConfig(alpha=1.0, bandwidth_limit=3)
    observations = []
    for i in range(10):
        observations.append(obs(f"c{i}", 0.9 - i * 0.03, 0.25, e=False, k=True, log=True))

    state, _ = simulate_case([observations], cfg)
    assert len(state.active_candidate_ids) == 3
    assert state.active_candidate_ids == ["c0", "c1", "c2"]
    assert len(state.commit_log) == 0
    assert sum(1 for c in state.candidates.values() if c.committed) == 0
    print("[OK] bandwidth limit blocks inactive acknowledgement")


def test_bandwidth_limit_commits_only_top_active_candidates():
    cfg = RewriteDynamicsConfig(alpha=1.0, bandwidth_limit=3)
    observations = []
    for i in range(5):
        observations.append(obs(f"bw{i}", 0.95 - i * 0.05, 0.2, e=True, k=True, log=True))
    state, timeline = simulate_case([observations], cfg)
    assert len(state.commit_log) == 3
    assert [event.candidate_id for event in state.commit_log] == ["bw0", "bw1", "bw2"]
    assert state.candidates["bw0"].committed
    assert state.candidates["bw1"].committed
    assert state.candidates["bw2"].committed
    assert not state.candidates["bw3"].committed
    assert not state.candidates["bw4"].committed
    assert not set(timeline[-1]["active_candidate_ids"]) & {"bw0", "bw1", "bw2"}
    print("[OK] bandwidth limit commits only top active candidates")


def test_no_commit_without_each_gate():
    case = [
        [obs("no_evidence", 0.9, 0.25, e=False, k=True, log=True)],
        [obs("no_constitution", 0.9, 0.25, e=True, k=False, log=True)],
        [obs("no_log", 0.9, 0.25, e=True, k=True, log=False)],
    ]
    state, _ = simulate_case(case, RewriteDynamicsConfig(alpha=1.0))
    assert not state.candidates["no_evidence"].committed
    assert not state.candidates["no_constitution"].committed
    assert not state.candidates["no_log"].committed
    assert len(state.commit_log) == 0
    print("[OK] no commit without each gate")


def test_proxy_vector_maps_to_governance_scalars():
    proxy = ProxyVector(
        u1_conflict=0.5,
        u2_mismatch=0.5,
        n1_goal_loss_if_ignored=0.4,
        n2_commitment_carry_cost=0.5,
        a1_institutional_level=0.8,
        a2_current_anchor_strength=0.5,
        p1_dependency_fanout=0.2,
        p2_rollback_cost=0.3,
    )
    scalars = compute_governance_scalars(proxy)
    assert abs(scalars.U - 0.75) < 1e-9
    assert abs(scalars.N - 0.70) < 1e-9
    assert abs(scalars.A - 0.40) < 1e-9
    assert abs(scalars.P - 0.44) < 1e-9
    assert 0.0 <= scalars.disturbance <= 1.0
    assert 0.0 <= scalars.stability <= 1.0
    assert scalars.R_raw > 0.0
    print("[OK] proxy vector maps to governance scalars")


def test_candidate_evidence_gate_is_separate_from_current_anchor():
    proxy = ProxyVector(
        u1_conflict=0.9,
        u2_mismatch=0.9,
        n1_goal_loss_if_ignored=0.9,
        n2_commitment_carry_cost=0.9,
        a1_institutional_level=0.1,
        a2_current_anchor_strength=0.1,
        p1_dependency_fanout=0.1,
        p2_rollback_cost=0.1,
    )
    observation = observation_from_proxy(
        "candidate_evidence_closed",
        proxy,
        evidence_open=False,
        constitution_open=True,
        log_ready=True,
    )
    state, _ = simulate_case([[observation]], RewriteDynamicsConfig(alpha=1.0))
    assert state.candidates["candidate_evidence_closed"].phase == RewritePhase.COMMIT_REVIEW
    assert not state.candidates["candidate_evidence_closed"].committed
    assert len(state.commit_log) == 0
    print("[OK] candidate evidence gate is separate from current anchor")


def test_proxy_driven_sustained_drift_acknowledges():
    proxies = [
        ProxyVector(0.1, 0.1, 0.2, 0.1, 0.9, 0.9, 0.7, 0.7),
        ProxyVector(0.5, 0.4, 0.6, 0.4, 0.8, 0.7, 0.5, 0.4),
        ProxyVector(0.8, 0.7, 0.8, 0.7, 0.4, 0.4, 0.3, 0.2),
        ProxyVector(0.9, 0.8, 0.9, 0.8, 0.2, 0.2, 0.2, 0.1),
    ]
    case = [
        [observation_from_proxy(
            "proxy_drift",
            proxy,
            evidence_open=i >= 2,
            constitution_open=True,
            log_ready=True,
        )]
        for i, proxy in enumerate(proxies)
    ]
    state, _ = simulate_case(case, RewriteDynamicsConfig(alpha=1.0))
    assert state.candidates["proxy_drift"].committed
    assert len(state.commit_log) == 1
    print("[OK] proxy-driven sustained drift acknowledges")


def test_scripted_sweep_separates_families():
    from experiments.capability.rewrite_dynamics_sweep import (
        protected_boundary,
        run_family,
        small_disturbance_stream,
        sustained_drift,
        temporary_spike,
    )

    seeds = [7, 42, 123]
    cfg = RewriteDynamicsConfig()
    temporary = run_family("temporary_spike", temporary_spike, seeds, cfg)
    small = run_family("small_disturbance", small_disturbance_stream, seeds, cfg)
    drift = run_family("sustained_drift", sustained_drift, seeds, cfg)
    protected = run_family("protected_boundary", protected_boundary, seeds, cfg)

    assert temporary["commit_rate"] == 0.0
    assert small["commit_rate"] == 0.0
    assert protected["commit_rate"] == 0.0
    assert drift["commit_rate"] >= 0.8
    print("[OK] scripted sweep separates families")


def test_cee_projection_preserves_commitment_revision_boundary():
    case = [
        [obs("cee_projection", 0.2, 0.8, e=False)],
        [obs("cee_projection", 0.8, 0.5, e=True)],
        [obs("cee_projection", 0.9, 0.25, e=True)],
    ]
    state, _ = simulate_case(case, RewriteDynamicsConfig(alpha=1.0))
    assert len(state.commit_log) == 1

    projection = project_commit_event_to_cee(
        state.commit_log[0],
        source_state_id="ws_0",
        resulting_state_id="ws_1",
    )
    commitment = projection.commitment_event
    revision = projection.revision_event

    assert commitment["schema_version"] == "cee.commitment.v1"
    assert commitment["event_type"] == "commitment"
    assert commitment["commitment_kind"] == "internal_commit"
    assert commitment["success"] is True
    assert commitment["requires_approval"] is False

    assert revision["schema_version"] == "cee.revision.v1"
    assert revision["event_type"] == "revision"
    assert revision["prior_state_id"] == "ws_0"
    assert revision["resulting_state_id"] == "ws_1"
    assert revision["caused_by_event_id"] == commitment["event_id"]
    assert revision["deltas"][0]["target_kind"] == "anchor_add"
    assert revision["new_anchor_fact_summaries"]
    print("[OK] CEE projection preserves commitment/revision boundary")


def test_cee_projection_rejects_unexecuted_or_closed_gate_event():
    from core.rewrite_dynamics import CommitEvent

    try:
        CommitEvent(
            event_id="bad",
            step_index=1,
            candidate_id="bad_candidate",
            from_phase="commit_review",
            to_phase="acknowledged",
            disturbance=0.9,
            stability=0.2,
            ratio=4.5,
            evidence_open=False,
            constitution_open=True,
            log_ready=True,
            commit_executed=True,
        )
        raise AssertionError("closed-gate executed acknowledgement should fail construction")
    except ValueError:
        pass

    unexecuted = CommitEvent(
        event_id="bad_unexecuted",
        step_index=1,
        candidate_id="bad_candidate",
        from_phase="commit_review",
        to_phase="commit_review",
        disturbance=0.9,
        stability=0.2,
        ratio=4.5,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
        commit_executed=False,
    )
    try:
        project_commit_event_to_cee(
            unexecuted,
            source_state_id="ws_0",
            resulting_state_id="ws_1",
        )
        raise AssertionError("projection should reject unexecuted commit event")
    except ValueError:
        pass
    print("[OK] CEE projection rejects invalid commit events")


def test_cee_roundtrip_if_available():
    cee_src = PROJECT_ROOT.parent / "cognitive-execution-engine" / "src"
    if not cee_src.exists():
        print("[SKIP] CEE roundtrip unavailable")
        return

    if str(cee_src) not in sys.path:
        sys.path.insert(0, str(cee_src))

    from cee_core.commitment import CommitmentEvent
    from cee_core.event_log import EventLog
    from cee_core.revision import ModelRevisionEvent

    case = [
        [obs("cee_roundtrip_test", 0.2, 0.8, e=False)],
        [obs("cee_roundtrip_test", 0.8, 0.45, e=True)],
        [obs("cee_roundtrip_test", 0.9, 0.25, e=True)],
    ]
    state, _ = simulate_case(case, RewriteDynamicsConfig(alpha=1.0))
    projection = project_commit_event_to_cee(
        state.commit_log[0],
        source_state_id="ws_0",
        resulting_state_id="ws_1",
    )

    commitment = CommitmentEvent.from_dict(projection.commitment_event)
    revision = ModelRevisionEvent.from_dict(projection.revision_event)
    log = EventLog()
    log.append(commitment)
    log.append(revision)
    replayed = log.replay_world_state()
    expected_anchor = projection.revision_event["new_anchor_fact_summaries"][0]

    assert revision.caused_by_event_id == commitment.event_id
    assert replayed.state_id == "ws_1"
    assert expected_anchor in replayed.anchored_fact_summaries
    print("[OK] CEE roundtrip if available")


def test_model_proposal_cannot_open_gates_or_thresholds():
    proposal = ProposalEnvelope(
        proposal_id="proposal_1",
        candidate_id="proposal_candidate",
        candidate_summary="Please commit this immediately.",
        proxy=ProxyVector(
            u1_conflict=0.95,
            u2_mismatch=0.95,
            n1_goal_loss_if_ignored=0.95,
            n2_commitment_carry_cost=0.95,
            a1_institutional_level=0.1,
            a2_current_anchor_strength=0.1,
            p1_dependency_fanout=0.1,
            p2_rollback_cost=0.1,
        ),
        requested_evidence_open=True,
        requested_constitution_open=True,
        requested_log_ready=True,
        requested_threshold_update={"theta_commit_ratio": 0.01},
    )
    audit = proposal_audit_record(proposal)
    assert audit["authority_requests_ignored"] is True
    assert audit["requested_threshold_update"] == {"theta_commit_ratio": 0.01}

    observation = observation_from_proposal(
        proposal,
        evidence_open=False,
        constitution_open=False,
        log_ready=False,
    )
    state, _ = simulate_case([[observation]], RewriteDynamicsConfig(alpha=1.0))
    candidate = state.candidates["proposal_candidate"]
    assert candidate.phase == RewritePhase.COMMIT_REVIEW
    assert not candidate.committed
    assert len(state.commit_log) == 0
    print("[OK] model proposal cannot open gates or thresholds")


def test_model_proposal_can_commit_only_when_external_gates_open():
    proposal = ProposalEnvelope(
        proposal_id="proposal_2",
        candidate_id="proposal_allowed",
        candidate_summary="Candidate with external evidence.",
        proxy=ProxyVector(0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1),
    )
    observation = observation_from_proposal(
        proposal,
        evidence_open=True,
        constitution_open=True,
        log_ready=True,
    )
    state, _ = simulate_case([[observation]], RewriteDynamicsConfig(alpha=1.0))
    assert state.candidates["proposal_allowed"].committed
    assert len(state.commit_log) == 1
    print("[OK] model proposal can commit only when external gates open")


def run_all() -> None:
    test_core_dataclasses_reject_invalid_values()
    test_commit_event_rejects_invalid_values()
    test_temporary_spike_never_acknowledges()
    test_sustained_drift_acknowledges()
    test_candidate_commits_once_after_gates_open_late()
    test_acknowledged_candidate_is_absorbing_under_later_observations()
    test_revision_event_rejects_invalid_execution()
    test_revision_execution_event_is_schema_only()
    test_revision_proposal_for_acknowledged_candidate_is_audit_only()
    test_draft_revision_execution_event_requires_approved_proposal_and_does_not_mutate()
    test_record_revision_execution_draft_adds_log_without_mutating()
    test_draft_revision_execution_event_rejects_unapproved_proposal()
    test_revision_proposal_rejects_version_mismatch()
    test_revision_proposal_rejects_unacknowledged_candidate()
    test_protected_boundary_never_acknowledges()
    test_hysteresis_reduces_phase_flips()
    test_bandwidth_limit_blocks_inactive_acknowledgement()
    test_bandwidth_limit_commits_only_top_active_candidates()
    test_no_commit_without_each_gate()
    test_proxy_vector_maps_to_governance_scalars()
    test_candidate_evidence_gate_is_separate_from_current_anchor()
    test_proxy_driven_sustained_drift_acknowledges()
    test_scripted_sweep_separates_families()
    test_cee_projection_preserves_commitment_revision_boundary()
    test_cee_projection_rejects_unexecuted_or_closed_gate_event()
    test_cee_roundtrip_if_available()
    test_model_proposal_cannot_open_gates_or_thresholds()
    test_model_proposal_can_commit_only_when_external_gates_open()
    print("All rewrite dynamics tests passed")


if __name__ == "__main__":
    run_all()
