# LSG Action Guide

Date: 2026-04-23
Status: execution guide for next project start

This guide turns the current LSG / attentional governance discussion into an execution sequence.
It should be read together with:

- `LSG_ATTENTION_INSPIRATION_2026-04-23.md`
- `REWRITE_QUALIFICATION_DYNAMICS_v0.md`
- `LSG_ENGINEERING_SPEC_v0.md`
- `LSG_THEOREM_NOTE_v0.md`

---

## 1. Project Start Position

Start from the narrowest viable version:

```text
Two-variable rewrite qualification simulator.
```

Do not start with:

- LLM calls
- learned heads
- CEE integration
- human attention claims
- large benchmark claims
- memory design search

The first implementation should be deterministic.

---

## 2. Core Variables

Every candidate content receives:

```text
D_t = disturbance rewrite pressure
S_t = order self-stability
```

Derived quantities:

```text
margin_t = D_t - S_t
ratio_t = D_t / (S_t + epsilon)
```

Only these quantities should drive phase movement in Phase 0.

The earlier `R_t` can be treated as:

```text
R_t = ratio_t
```

or as a later compatibility wrapper:

```text
R_t = (U_t * N_t) / (A_t * P_t + epsilon)
```

Do not introduce more primary metrics until the two-variable version fails.

---

## 3. Minimal Candidate State

Implement this first:

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

Allowed phases:

```text
suppressed
background
foreground
verify
commit_review
acknowledged
rejected
```

Do not implement a full attention state object in Phase 0.
Foreground, verify, and commit-review are phases, not separate agent modules.

---

## 4. Minimal System State

Implement:

```python
@dataclass
class RewriteSystemState:
    candidates: dict[str, CandidateState]
    active_candidate_ids: list[str]
    bandwidth_limit: int
    commit_log: list[CommitEvent]
    step_index: int
```

Hard invariants:

```text
len(active_candidate_ids) <= bandwidth_limit
no acknowledged candidate without CommitEvent
no committed=True unless phase == acknowledged
```

Default:

```text
bandwidth_limit = 3
```

---

## 5. Phase Rules

Use simple hysteretic phase rules.

Suggested initial thresholds:

```text
theta_bg_enter = 0.20
theta_fg_enter = 0.15       # margin threshold
theta_fg_exit  = 0.05
theta_verify_ratio = 1.25
theta_commit_ratio = 2.00
theta_protected_stability = 0.70
```

Rules:

```text
if D < theta_bg_enter and S >= D:
    suppressed

elif margin < theta_fg_enter:
    background

elif ratio >= theta_commit_ratio and S < theta_protected_stability:
    commit_review

elif ratio >= theta_verify_ratio:
    verify

else:
    foreground
```

Acknowledgement rule:

```text
phase == commit_review
and evidence_open == true
and constitution_open == true
and log_ready == true
=> acknowledged
```

Protected boundary rule:

```text
D high and S high
=> verify or defer, never direct acknowledged
```

In Phase 0, `defer` can be represented as `verify` or `background`.

---

## 6. Time Dynamics

Use exponential smoothing:

```python
D_next = (1 - alpha) * D_prev + alpha * D_observed
S_next = (1 - alpha) * S_prev + alpha * S_observed
```

Then compute velocities:

```python
dD = D_next - D_prev
dS = S_next - S_prev
```

Suggested initial:

```text
alpha = 0.3
```

Add slope clipping only after baseline behavior is visible:

```text
Delta_max = 0.25
```

---

## 7. Minimal Commit Event

Implement only:

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

Do not import CEE yet.
Mirror its discipline, not its whole codebase.

CEE can be integrated later once the simulator proves the two-variable behavior.

---

## 8. First Files To Create

Create:

```text
core/rewrite_dynamics.py
tests/test_rewrite_dynamics.py
experiments/capability/rewrite_dynamics_sanity.py
```

Do not touch:

- CEE
- TopoMem Chroma data
- existing Capability Router code
- learned governance specs beyond adding references

---

## 9. Phase 0 Test Cases

### Test 1: Temporary Spike

Input stream:

```text
D: 0.1, 0.9, 0.1, 0.1
S: 0.8, 0.8, 0.8, 0.8
evidence_open: false
constitution_open: true
log_ready: true
```

Expected:

```text
never acknowledged
phase may enter verify but returns/demotes
```

### Test 2: Sustained Drift

Input stream:

```text
D: 0.2, 0.7, 0.8, 0.85, 0.9
S: 0.8, 0.7, 0.5, 0.35, 0.25
evidence_open: false, false, true, true, true
constitution_open: true
log_ready: true
```

Expected:

```text
reaches commit_review
then acknowledged
```

### Test 3: Protected Boundary

Input stream:

```text
D: 0.9, 0.95, 0.9
S: 0.9, 0.9, 0.85
evidence_open: true
constitution_open: false
log_ready: true
```

Expected:

```text
verify or background
never acknowledged
```

### Test 4: Threshold Chattering

Input stream:

```text
D/S oscillates around foreground threshold
```

Expected:

```text
with hysteresis: fewer phase flips than single-threshold baseline
no acknowledged phase
```

### Test 5: Bandwidth Competition

Input:

```text
10 candidates, bandwidth_limit = 3
```

Expected:

```text
active_candidate_ids never exceeds 3
top candidates by ratio/margin retained
protected high-S candidates route to verify, not acknowledgement
```

---

## 10. Sanity Experiment Output

`experiments/capability/rewrite_dynamics_sanity.py` should write:

```text
results/rewrite_dynamics_sanity/<timestamp>.json
```

Minimum JSON:

```json
{
  "num_cases": 0,
  "temporary_false_acknowledgement_rate": 0.0,
  "sustained_drift_acknowledgement_rate": 0.0,
  "protected_boundary_false_acknowledgements": 0,
  "phase_flip_rate": 0.0,
  "max_active_candidates": 0,
  "commit_events": 0,
  "commits_without_log": 0
}
```

---

## 11. Pass / Fail Criteria

Phase 0 passes only if:

```text
temporary_false_acknowledgement_rate == 0
protected_boundary_false_acknowledgements == 0
commits_without_log == 0
max_active_candidates <= bandwidth_limit
sustained_drift_acknowledgement_rate > 0
```

Phase 0 fails if:

```text
any candidate reaches acknowledged without evidence_open, constitution_open, and log_ready
temporary spike commits
protected boundary commits
active candidates exceed bandwidth limit
sustained drift can never reach acknowledgement
```

---

## 12. What To Borrow From CEE Later

After Phase 0 passes, inspect and possibly reuse:

```text
F:\cognitive-execution-engine\src\cee_core\commitment.py
F:\cognitive-execution-engine\src\cee_core\revision.py
F:\cognitive-execution-engine\src\cee_core\event_log.py
F:\cognitive-execution-engine\tests\test_red_line_invariants.py
```

Reusable ideas:

- append-only event log
- commitment event before formal revision
- model revision event as actual state mutation record
- replayed state excludes denied/pending changes
- provenance on every committed fact
- requires_approval blocks revision

Do not copy CEE wholesale.
Use it as the commit-boundary reference.

---

## 13. What To Borrow From Hypergraph Bistability Later

After Phase 0/1, inspect:

```text
F:\hypergraph_bistability\experiments\research\multi_stability_core.py
F:\hypergraph_bistability\experiments\research\control_inhibition.py
F:\hypergraph_bistability\experiments\research\control_*.py
```

Reusable ideas:

- stable basins
- order parameters
- inhibition/control experiments
- transition under perturbation

Use this for interpretation and later visualization, not for the first simulator.

---

## 14. Do Not Do Yet

Do not:

- train proxy heads
- add LLM calls
- claim biological realism
- claim human consciousness modeling
- integrate CEE directly
- publish broad paper framing
- optimize downstream benchmark performance
- use TopoMem as a per-task router
- let candidate proposals change thresholds

---

## 15. Next Concrete Command Sequence

Once implementation starts:

```powershell
cd F:\unified-sel
python -m py_compile core/rewrite_dynamics.py tests/test_rewrite_dynamics.py experiments/capability/rewrite_dynamics_sanity.py
python tests/test_rewrite_dynamics.py
python experiments/capability/rewrite_dynamics_sanity.py
```

Before implementation starts, keep the project in document-only mode.

---

## 16. One-Sentence Direction

Build the smallest deterministic simulator showing that formal state acknowledgement can be controlled by two continuous variables:

```text
disturbance rewrite pressure D_t
vs
order self-stability S_t
```

under evidence, constitutional, logging, and bandwidth constraints.

