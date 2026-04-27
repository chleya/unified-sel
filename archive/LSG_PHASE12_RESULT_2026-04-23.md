# LSG Phase 12 Result - Core Dataclass Boundary Hardening

Date: 2026-04-23

## Decision

Phase 12 continues the boundary audit by moving validation into the core LSG dataclasses.

Phase 11 hardened replay/capture explicit-state parsing.  Phase 12 hardens direct Python construction paths.

## Why This Was Worth Doing

Before this phase, strict checks existed in JSON parsing and mediation, but core dataclasses still accepted invalid direct values such as:

```text
CandidateObservation(evidence_open="false")
ProxyVector(n1_goal_loss_if_ignored=True)
RewriteDynamicsConfig(alpha=1.2)
ProposalEnvelope(proposal_id="")
```

These are not normal user-facing paths, but they matter because tests, experiments, provider adapters, and future code can construct these objects directly.

## Fix

Updated:

- `core/rewrite_dynamics.py`

Added `__post_init__` validation to:

- `RewriteDynamicsConfig`
- `CandidateObservation`
- `ProxyVector`
- `ProposalEnvelope`

### RewriteDynamicsConfig

Now checks:

- `alpha` in `[0, 1]`
- `epsilon > 0`
- `delta_max > 0` when present
- `bandwidth_limit` is integer and `>= 1`
- thresholds are nonnegative
- `theta_verify_ratio <= theta_commit_ratio`
- `theta_protected_stability` in `[0, 1]`

### CandidateObservation

Now checks:

- `candidate_id` is non-empty
- `disturbance_observed` in `[0, 1]`
- `stability_observed` in `[0, 1]`
- `evidence_open`, `constitution_open`, `log_ready` are real booleans

### ProxyVector

Now checks all 8 proxy fields are numeric scores in `[0, 1]`:

- `u1_conflict`
- `u2_mismatch`
- `n1_goal_loss_if_ignored`
- `n2_commitment_carry_cost`
- `a1_institutional_level`
- `a2_current_anchor_strength`
- `p1_dependency_fanout`
- `p2_rollback_cost`

Boolean values are rejected as numeric scores.

### ProposalEnvelope

Now checks:

- `proposal_id`, `candidate_id`, `candidate_summary`, `proposal_origin` are non-empty
- `proxy` is a `ProxyVector`
- requested gate fields are booleans when present
- requested threshold updates are a dict with non-empty string keys and nonnegative numeric values

## Added Regression Tests

Updated:

- `tests/test_rewrite_dynamics.py`

New check:

```text
core dataclasses reject invalid values
```

It covers invalid observations, invalid proxy scores, invalid proposal envelopes, and invalid dynamics config.

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_dynamics.py F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\tests\test_rewrite_provider_capture.py
python F:\unified-sel\tests\test_rewrite_live_provider_capture.py
python F:\unified-sel\tests\test_rewrite_source_distribution.py
python F:\unified-sel\experiments\capability\rewrite_proposal_replay_fixture.py --label phase12_dataclass_hardening
python F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py --label phase12_dataclass_hardening
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
All rewrite dynamics tests passed
All rewrite proposal provider tests passed
All rewrite proxy mediator tests passed
All rewrite proposal replay tests passed
All rewrite proxy calibration tests passed
All rewrite provider capture tests passed
All rewrite live provider capture tests passed
All rewrite source distribution tests passed
All smoke tests passed
```

The first parallel `py_compile` attempt hit a Windows `__pycache__` rename conflict while tests were writing pyc files concurrently.  Re-running `py_compile` serially passed.  This was not a code failure.

The smoke run emitted an unrelated Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

This phase strengthens the invariant:

> every trusted LSG boundary object rejects malformed authority, proxy, and threshold values at construction time.

This reduces the chance that future experiments bypass validation by constructing internal objects directly.

## What Is Still Worth Verifying

Remaining useful hardening targets:

- validate `CommitEvent` construction directly, not only projection
- distinguish expected schema interceptions from unexpected schema failures in source distribution summaries
- add config sweep sanity tests for invalid threshold combinations beyond `theta_verify_ratio <= theta_commit_ratio`
- add adversarial replay cases with duplicate IDs and conflicting repeated candidates
- add multi-step replay capture where gates open over time rather than in a single row

