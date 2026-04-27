# LSG Milestone Report

Date: 2026-04-23

## 1. Project Claim

LSG is a minimal learned state governance kernel.

It does not try to build a stronger agent.  It separates formal state rewrite from ordinary inference and makes rewrite authority pass through explicit gates, logs, mediation, calibration, and replay.

The current architecture is:

```text
candidate disturbance
-> proxy vector U/N/A/P
-> D/S rewrite dynamics
-> hysteresis and bandwidth control
-> evidence/constitution/log gates
-> commit event
-> CEE-style projection
```

For model/provider integration:

```text
provider proposal
-> schema validation
-> mediation
-> calibration
-> replay
-> source distribution comparison
```

The invariant is:

> model output is proposal evidence, not state rewrite authority.

## 2. Concept Lock

The project intentionally moved away from surface terms like attention, salience, router, or action selection.

The current conceptual core is:

```text
disturbance pressure D
vs
order self-stability S
```

The engineering layer exposes this through:

```text
U = incompatibility pressure
N = cost of ignoring the candidate
A = acknowledgement depth of current formal state
P = propagation burden of current formal state
R = U*N / (A*P + epsilon)
```

In implementation, `D` and `S` are tracked directly:

```text
D = U*N
S = A*P
```

`R` remains useful as a diagnostic rewrite qualification ratio, but commit authority does not come from `R` alone.

## 3. Implemented Phases

### Phase 0-3: Rewrite Dynamics Core

Implemented:

- D/S rewrite dynamics
- candidate phases
- hysteresis
- bandwidth limits
- explicit gates
- commit log
- CEE projection and roundtrip
- model proposal boundary

Key result:

```text
temporary_false_acknowledgement_rate: 0
sustained_drift_acknowledgement_rate: 1
protected_boundary_false_acknowledgements: 0
proxy_drift_acknowledgement_rate: 1
phase_flip_rate: 0
commits_without_log: 0
```

### Phase 4: Proxy Mediation Boundary

Implemented:

- model-suggested proxy vs system-owned explicit proxy separation
- system override of `a1` and `p1`
- gate and threshold requests are audit-only

Key result:

```text
closed_gates_do_not_commit: true
model_authority_requests_ignored: true
open_system_gates_can_commit: true
```

### Phase 5: Proxy Calibration Fixture

Implemented:

- proxy delta metrics
- aligned, overclaim, and underclaim fixtures
- commit outcome evaluation after mediation

Key result:

```text
num_cases: 3
passed: true
false_commit_count: 0
missed_commit_count: 0
override_rate: 0.6666666666666666
authority_request_rate: 0.3333333333333333
mean_abs_delta: 0.5666666666666667
max_abs_delta: 0.85
```

### Phase 6: Local Proposal Replay Dataset

Implemented:

- local proposal replay fixture dataset
- schema validation to replay
- failure classification

Failure classes:

- `schema_error`
- `authority_request`
- `proxy_disagreement`
- `false_commit`
- `missed_commit`

Key result:

```text
num_cases: 4
passed: true
failed_count: 0
false_commit_count: 0
missed_commit_count: 0
schema_error_count: 1
authority_request_count: 1
proxy_disagreement_count: 1
```

### Phase 7: Provider Output Capture Protocol

Implemented:

- provider prompt
- captured provider JSONL
- capture-to-replay conversion
- offline replay of provider outputs

Key result:

```text
num_cases: 2
passed: true
false_commit_count: 0
missed_commit_count: 0
schema_error_count: 1
```

### Phase 8: Optional Live Provider Capture Adapter

Implemented:

- optional live provider adapter
- environment-only key/url/model
- dry-run default
- fake transport tests
- live output must be captured and replayed

Required environment for MiniMax:

```text
MINIMAX_API_KEY
MINIMAX_API_URL
MINIMAX_MODEL
```

No smoke test makes live provider calls.

### Phase 9: Source Distribution Comparison

Implemented:

- proposal source summaries
- cross-source comparison
- default comparison between hand-authored replay and provider capture fixture

Key result:

```text
hand_authored_replay:
  num_cases: 4
  schema_error_rate: 0.25
  authority_request_rate: 0.25
  proxy_disagreement_rate: 0.25
  false_commit_count: 0
  missed_commit_count: 0

provider_capture_fixture:
  num_cases: 2
  schema_error_rate: 0.5
  authority_request_rate: 0.0
  proxy_disagreement_rate: 0.0
  false_commit_count: 0
  missed_commit_count: 0
```

## 4. Code Map

Core modules:

```text
core/rewrite_dynamics.py
core/rewrite_proposal_provider.py
core/rewrite_proxy_mediator.py
core/rewrite_proxy_calibration.py
core/rewrite_proposal_replay.py
core/rewrite_provider_capture.py
core/rewrite_live_provider_capture.py
core/rewrite_source_distribution.py
```

Tests:

```text
tests/test_rewrite_dynamics.py
tests/test_rewrite_proposal_provider.py
tests/test_rewrite_proxy_mediator.py
tests/test_rewrite_proxy_calibration.py
tests/test_rewrite_proposal_replay.py
tests/test_rewrite_provider_capture.py
tests/test_rewrite_live_provider_capture.py
tests/test_rewrite_source_distribution.py
tests/smoke_test.py
```

Experiments:

```text
experiments/capability/rewrite_dynamics_sanity.py
experiments/capability/rewrite_dynamics_sweep.py
experiments/capability/rewrite_dynamics_cee_projection.py
experiments/capability/rewrite_dynamics_cee_roundtrip.py
experiments/capability/rewrite_dynamics_proposal_boundary.py
experiments/capability/rewrite_proposal_schema_validation.py
experiments/capability/rewrite_proxy_mediation_sanity.py
experiments/capability/rewrite_proxy_calibration_fixture.py
experiments/capability/rewrite_proposal_replay_fixture.py
experiments/capability/rewrite_provider_capture_replay.py
experiments/capability/rewrite_live_provider_capture.py
experiments/capability/rewrite_source_distribution_compare.py
```

Data fixtures:

```text
data/lsg/proposal_replay_v0.json
data/lsg/provider_capture_v0.jsonl
```

Phase notes:

```text
LSG_PHASE0_3_RESULT_2026-04-23.md
LSG_PHASE4_RESULT_2026-04-23.md
LSG_PHASE5_RESULT_2026-04-23.md
LSG_PHASE6_RESULT_2026-04-23.md
LSG_PHASE7_RESULT_2026-04-23.md
LSG_PHASE8_RESULT_2026-04-23.md
LSG_PHASE9_RESULT_2026-04-23.md
```

## 5. Current Guarantees

These are engineering guarantees from the current tests and fixtures, not mathematical guarantees over all possible systems.

### G1. No formal commit without gates

Formal commit requires:

```text
evidence_open == true
constitution_open == true
log_ready == true
```

### G2. Model proposals cannot open gates

Provider JSON may request:

```text
requested_evidence_open
requested_constitution_open
requested_log_ready
requested_threshold_update
```

These are audit-only and ignored by the effective observation path.

### G3. System-owned proxy fields override model values

The mediation layer keeps these system-owned:

```text
a1_institutional_level
p1_dependency_fanout
evidence_open
constitution_open
log_ready
```

Optional explicit values may also override:

```text
u1_conflict
u2_mismatch
n1_goal_loss_if_ignored
n2_commitment_carry_cost
a2_current_anchor_strength
p2_rollback_cost
```

### G4. Hysteresis reduces one-step phase flipping

The rewrite dynamics use smoothing, slope limits, and hysteresis bands to prevent direct single-threshold jitter.

### G5. Replay catches schema and authority problems before commit

Replay classifies:

```text
schema_error
authority_request
proxy_disagreement
false_commit
missed_commit
```

### G6. Live provider integration is outside smoke and trusted path

Live provider calls are optional, explicit, environment-backed, and captured before replay.

## 6. Non-Guarantees

The current project does not yet guarantee:

- real-world correctness of proxy values
- learned proxy-head calibration
- robust adversarial prompt resistance for live providers
- theorem-level proof over arbitrary dynamics
- production-safe MiniMax integration
- automatic memory design search
- end-to-end task performance improvement
- human-level attention modeling
- prevention of semantic drift outside the defined commit path

These are future work, not current claims.

## 7. Command Checklist

### Core unit tests

```text
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\tests\test_rewrite_provider_capture.py
python F:\unified-sel\tests\test_rewrite_live_provider_capture.py
python F:\unified-sel\tests\test_rewrite_source_distribution.py
```

### Core experiments

```text
python F:\unified-sel\experiments\capability\rewrite_dynamics_sanity.py
python F:\unified-sel\experiments\capability\rewrite_dynamics_sweep.py --label manual
python F:\unified-sel\experiments\capability\rewrite_dynamics_proposal_boundary.py
python F:\unified-sel\experiments\capability\rewrite_proposal_schema_validation.py
python F:\unified-sel\experiments\capability\rewrite_proxy_mediation_sanity.py --label manual
python F:\unified-sel\experiments\capability\rewrite_proxy_calibration_fixture.py --label manual
python F:\unified-sel\experiments\capability\rewrite_proposal_replay_fixture.py --label manual
python F:\unified-sel\experiments\capability\rewrite_provider_capture_replay.py --label manual
python F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py --label manual
```

### CEE compatibility

```text
python F:\unified-sel\experiments\capability\rewrite_dynamics_cee_projection.py
python F:\unified-sel\experiments\capability\rewrite_dynamics_cee_roundtrip.py
```

### Live provider dry run

```text
python F:\unified-sel\experiments\capability\rewrite_live_provider_capture.py --label dry_run
```

### Full smoke

```text
python F:\unified-sel\tests\smoke_test.py
```

## 8. Safe MiniMax Usage Recipe

Live MiniMax should only be used through:

```text
experiments/capability/rewrite_live_provider_capture.py
```

Required environment:

```text
MINIMAX_API_KEY
MINIMAX_API_URL
MINIMAX_MODEL
```

Dry run first:

```text
python F:\unified-sel\experiments\capability\rewrite_live_provider_capture.py --label minimax_dry_run
```

Live call shape:

```text
python F:\unified-sel\experiments\capability\rewrite_live_provider_capture.py --live --provider-name minimax --capture-id minimax_case_001 --label minimax_case_001 --observation-summary "..." --current-order-summary "..." --goal-summary "..." --evidence-open --constitution-open --log-ready --expected-committed
```

The command writes:

```text
data/lsg/provider_captures/minimax_minimax_case_001.jsonl
results/capability_generalization/rewrite_live_provider_capture_minimax_minimax_case_001.json
```

Rule:

> Do not consume live provider output directly.  Only consume replay summaries and captured JSONL.

## 9. How To Extend

### Add a new replay case

Edit:

```text
data/lsg/proposal_replay_v0.json
```

Then run:

```text
python F:\unified-sel\experiments\capability\rewrite_proposal_replay_fixture.py --label new_case
python F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py --label new_case
```

### Add a new capture file

Write JSONL with:

```text
capture_id
provider_name
request
prompt
raw_model_json
explicit
expected_committed or expected_error
```

Then run:

```text
python F:\unified-sel\experiments\capability\rewrite_provider_capture_replay.py --capture-file <path> --label capture_eval
python F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py --source new_capture:capture_jsonl:<path> --label capture_compare
```

### Add a new provider

Do not add it to the commit path.

Add it behind capture:

```text
provider -> raw_model_json -> ProviderCaptureRecord -> JSONL -> replay
```

## 10. Current Status

LSG is now a working v0 kernel with:

- rewrite dynamics
- explicit commit gates
- model proposal boundary
- proxy mediation
- calibration fixtures
- replay dataset
- provider capture
- optional live adapter
- source distribution comparison

It is not a paper-ready theorem system yet.

It is a runnable engineering scaffold for studying:

> when a candidate content has enough rewrite qualification to alter formal state, and how to prevent model proposals from becoming rewrite authority.

