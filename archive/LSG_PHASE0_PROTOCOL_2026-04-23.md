# LSG Phase 0 Protocol

Date: 2026-04-23
Status: deterministic simulator protocol

This protocol defines the first executable experiment for the LSG project.

Phase 0 tests only:

- two-variable dynamics
- phase transitions
- formal acknowledgement boundary
- bandwidth limit
- no bypass commit

It does not test LLM performance or learned proxy quality.

---

## 1. Files

Create:

```text
core/rewrite_dynamics.py
tests/test_rewrite_dynamics.py
experiments/capability/rewrite_dynamics_sanity.py
```

Output:

```text
results/rewrite_dynamics_sanity/<timestamp>.json
```

---

## 2. Input Case Schema

Each scripted case is a sequence of observations:

```json
{
  "case_id": "temporary_spike",
  "candidate_id": "c1",
  "steps": [
    {
      "disturbance_observed": 0.1,
      "stability_observed": 0.8,
      "evidence_open": false,
      "constitution_open": true,
      "log_ready": true
    }
  ]
}
```

Required fields per step:

- `disturbance_observed`
- `stability_observed`
- `evidence_open`
- `constitution_open`
- `log_ready`

Optional fields:

- `expected_phase`
- `note`
- `candidate_id`

---

## 3. Candidate State

Minimum:

```python
@dataclass
class CandidateState:
    candidate_id: str
    disturbance: float
    stability: float
    disturbance_velocity: float
    stability_velocity: float
    phase: str
    evidence_open: bool
    constitution_open: bool
    log_ready: bool
    committed: bool = False
```

---

## 4. System State

Minimum:

```python
@dataclass
class RewriteSystemState:
    candidates: dict[str, CandidateState]
    active_candidate_ids: list[str]
    bandwidth_limit: int
    commit_log: list[CommitEvent]
    step_index: int
```

Default:

```text
bandwidth_limit = 3
```

---

## 5. Commit Event

Minimum:

```python
@dataclass
class CommitEvent:
    event_id: str
    step_index: int
    candidate_id: str
    from_phase: str
    to_phase: str
    disturbance: float
    stability: float
    ratio: float
    evidence_open: bool
    constitution_open: bool
    log_ready: bool
    commit_executed: bool
```

---

## 6. Update Rule

Smooth observed values:

```text
D_next = (1 - alpha) * D_prev + alpha * D_observed
S_next = (1 - alpha) * S_prev + alpha * S_observed
```

Then:

```text
dD = D_next - D_prev
dS = S_next - S_prev
ratio = D_next / (S_next + epsilon)
margin = D_next - S_next
```

Defaults:

```text
alpha = 0.3
epsilon = 1e-6
```

Optional after baseline:

```text
Delta_max = 0.25
```

---

## 7. Phase Rule

Initial thresholds:

```text
theta_bg_enter = 0.20
theta_fg_enter = 0.15
theta_fg_exit = 0.05
theta_verify_ratio = 1.25
theta_commit_ratio = 2.00
theta_protected_stability = 0.70
```

Base phase selection:

```text
if D < theta_bg_enter and S >= D:
    phase = suppressed
elif margin < theta_fg_enter:
    phase = background
elif ratio >= theta_commit_ratio and S < theta_protected_stability:
    phase = commit_review
elif ratio >= theta_verify_ratio:
    phase = verify
else:
    phase = foreground
```

Acknowledgement rule:

```text
if phase == commit_review
and evidence_open
and constitution_open
and log_ready:
    phase = acknowledged
    append CommitEvent
```

Protected-boundary override:

```text
if D high and S >= theta_protected_stability:
    phase cannot be acknowledged
```

---

## 8. Required Cases

### Case A: Temporary Spike

Stream:

```text
D_obs: 0.1, 0.9, 0.1, 0.1
S_obs: 0.8, 0.8, 0.8, 0.8
E:     0,   0,   0,   0
K:     1,   1,   1,   1
Log:   1,   1,   1,   1
```

Expected:

```text
acknowledged never appears
```

### Case B: Sustained Drift

Stream:

```text
D_obs: 0.2, 0.7, 0.8, 0.85, 0.9
S_obs: 0.8, 0.7, 0.5, 0.35, 0.25
E:     0,   0,   1,   1,    1
K:     1,   1,   1,   1,    1
Log:   1,   1,   1,   1,    1
```

Expected:

```text
commit_review appears
acknowledged appears
```

### Case C: Protected Boundary

Stream:

```text
D_obs: 0.9, 0.95, 0.9
S_obs: 0.9, 0.9,  0.85
E:     1,   1,    1
K:     0,   0,    0
Log:   1,   1,    1
```

Expected:

```text
acknowledged never appears
```

### Case D: Threshold Chattering

Stream:

```text
ratio and margin oscillate around foreground/verify boundary
```

Expected:

```text
hysteretic rule has fewer phase flips than single-threshold baseline
acknowledged never appears
```

### Case E: Bandwidth Competition

Setup:

```text
10 candidates
bandwidth_limit = 3
```

Expected:

```text
len(active_candidate_ids) <= 3 at every step
no inactive candidate can be acknowledged
```

---

## 9. Required Metrics

Experiment JSON must include:

```json
{
  "num_cases": 0,
  "temporary_false_acknowledgement_rate": 0.0,
  "sustained_drift_acknowledgement_rate": 0.0,
  "protected_boundary_false_acknowledgements": 0,
  "phase_flip_rate": 0.0,
  "single_threshold_phase_flip_rate": 0.0,
  "max_active_candidates": 0,
  "bandwidth_limit": 3,
  "commit_events": 0,
  "commits_without_log": 0
}
```

---

## 10. Pass Criteria

Phase 0 passes if:

```text
temporary_false_acknowledgement_rate == 0
protected_boundary_false_acknowledgements == 0
commits_without_log == 0
max_active_candidates <= bandwidth_limit
sustained_drift_acknowledgement_rate > 0
phase_flip_rate < single_threshold_phase_flip_rate
```

---

## 11. Fail Criteria

Phase 0 fails if:

```text
temporary spike acknowledges
protected boundary acknowledges
candidate acknowledges without evidence, constitution, or log
active candidates exceed bandwidth limit
sustained drift can never acknowledge
hysteresis does not reduce phase flips
```

---

## 12. Validation Commands

Expected commands after implementation:

```powershell
cd F:\unified-sel
python -m py_compile core\rewrite_dynamics.py tests\test_rewrite_dynamics.py experiments\capability\rewrite_dynamics_sanity.py
python tests\test_rewrite_dynamics.py
python experiments\capability\rewrite_dynamics_sanity.py
```

