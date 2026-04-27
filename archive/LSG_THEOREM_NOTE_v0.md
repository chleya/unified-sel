# LSG Theorem Note v0

Date: 2026-04-23
Status: proof sketch / falsification note
Companion spec: `LSG_ENGINEERING_SPEC_v0.md`

This note separates theorem targets from engineering proxies.
It does not claim full mathematical proof for learned estimators.
It states the weakest structural assumptions under which the v0 kernel should satisfy its four local properties.

---

## 0. System Summary

Discrete-time state:

```text
z_t = (x_t, ell_t, m_t, q_t)
```

Mode:

```text
m_t in {M, F, V, C}
```

where:

- `M`: maintain
- `F`: foreground
- `V`: verify
- `C`: commit-candidate

Governance scalars:

```text
q_t = (U_t, N_t, A_t, P_t, R_t)
```

Formal commit condition:

```text
formal_commit_done = 1
only if
m_t = C and E_t = 1 and K_t = 1 and log_t = 1
```

---

## 1. Proposition 1: No Unauthorized Commit

### Theorem Target

If formal commit can only occur from `C` and must satisfy `E_t = 1`, `K_t = 1`, and `log_t = 1`, then no formal acknowledgement transition can occur without evidence, permission, and log.

### Weakest Assumptions

A1. The only code path that can set `formal_commit_done = 1` is the commit transition.

A2. The commit transition checks:

```text
m_t = C
E_t = 1
K_t = 1
log_t = 1
```

A3. `K_t = 1` implies the requested transition is permitted by `permission_state`.

A4. `log_t = 1` implies a valid `CommitEvent` is written before or atomically with durable mutation.

A5. Durable formal state cannot be mutated by side effect outside the commit transition.

### Proof Sketch

Assume a formal commit occurs.
By A1, it must occur through the unique commit transition.
By A2, that transition requires `m_t = C`, `E_t = 1`, `K_t = 1`, and `log_t = 1`.
By A3, `K_t = 1` means the transition is permitted.
By A4, `log_t = 1` means the commit is logged.
Therefore a formal commit without evidence, permission, or log is impossible under A1-A5.

### Collapse Points

- Any alternate writer can mutate formal state.
- `formal_commit_done` can be set by a large-model proposal.
- `log_t` is written after mutation and can fail.
- `K_t` is learned freely rather than derived from explicit permission checks.
- semantic state is changed in a way equivalent to formal commit but bypasses `commit_stage`.

### Engineering Test

Unauthorized commit adversary:

- high `R_t`
- `E_t = 0` or `K_t = 0`
- attempt to force `formal_commit_done = 1`

Pass:

```text
bypass_commit_count == 0
unauthorized_commit_count == 0
unlogged_commit_count == 0
```

---

## 2. Proposition 2: No Chattering

### Theorem Target

If `R_t` has bounded single-step change and thresholds have positive-width hysteresis bands, then mode switching cannot oscillate back and forth in one step, and switching count is bounded in any finite time window.

### Weakest Assumptions

B1. `R_t` is slope-clipped:

```text
|R_{t+1} - R_t| <= Delta_max
```

B2. Hysteresis bands have positive width:

```text
theta_F^- < theta_F^+
theta_C^- < theta_C^+
```

B3. Exit threshold is lower than enter threshold for the same region.

B4. A mode transition is a deterministic function of:

```text
(m_t, R_t, E_t, K_t, formal_commit_done)
```

B5. `E_t` and `K_t` do not alternate adversarially at every step without corresponding logged evidence changes.

### Proof Sketch

To enter `F` from `M`, `R_t >= theta_F^+`.
To return from `F` to `M`, `R_t <= theta_F^-`.
The gap:

```text
width_F = theta_F^+ - theta_F^- > 0
```

must be crossed before reversal.
Because `R_t` changes by at most `Delta_max` per step, reversal requires at least:

```text
ceil(width_F / Delta_max)
```

steps, ignoring discrete gate changes.
The same argument applies around `C` using `theta_C^-` and `theta_C^+`.
Therefore one-step back-and-forth oscillation is impossible if band widths are positive and `R_t` is bounded.
In any finite time window, the number of switches is bounded by window length divided by the minimum dwell induced by hysteresis and slope clipping.

### Collapse Points

- `Delta_max` is larger than all hysteresis widths.
- `E_t` or `K_t` flips every step due to noisy learned heads.
- transition rules bypass hysteresis in special cases.
- `formal_commit_done` toggles without durable state completion.
- thresholds are changed online by the same model being governed.

### Engineering Test

Threshold noise test:

- compare single-threshold router vs hysteretic kernel
- add bounded noise near thresholds

Pass:

```text
switch_count_hysteretic < switch_count_single_threshold
one_step_backtrack_count == 0
false_commit_count == 0
```

---

## 3. Proposition 3: Small Disturbances Do Not Rewrite Formal State

### Theorem Target

If `R_t < theta_C^+` for a continuous interval, or `E_t = 0`, then the system cannot enter `committed`.

### Weakest Assumptions

C1. Entry to `C` from `F` or `V` requires:

```text
R_t >= theta_C^+
E_t = 1
K_t = 1
```

C2. Formal commit requires being in `C` and:

```text
E_t = 1
K_t = 1
log_t = 1
```

C3. No alternate path can set `commit_stage = committed`.

C4. `committed` is not reached by timeout or pending-count overflow.

### Proof Sketch

If `R_t < theta_C^+`, C1 prevents entry into `C` through the normal foreground or verify transitions.
If `E_t = 0`, C1 prevents entry into `C`, and C2 prevents formal commit even if already in `C`.
By C3, no alternate path can set `committed`.
Therefore small disturbances or unanchored evidence cannot rewrite formal state.

### Collapse Points

- low `R_t` candidates are batch-committed later.
- `E_t = 0` can be overridden by high `R_t`.
- pending overload triggers auto-commit.
- working state mutation becomes durable without `commit_stage`.
- `theta_C^+` is dynamically lowered by the candidate itself.

### Engineering Test

Small disturbance stream:

- continuous low-conflict / low-anchor candidates
- bounded `R_t < theta_C^+`
- `E_t = 0` for most or all steps

Pass:

```text
committed_count == 0
formal_commit_done_count == 0
pending_count remains bounded
```

---

## 4. Proposition 4: Sustained High Disturbance Can Trigger Formal Acknowledgement

### Theorem Target

If `R_t >= theta_C^+` for a continuous interval, verification eventually makes `E_t = 1`, and `K_t = 1`, then `committed` is reachable.

### Weakest Assumptions

D1. From `M`, high `R_t` can reach `F`:

```text
R_t >= theta_F^+ => M -> F
```

D2. From `F`, high `R_t` with missing evidence reaches `V`:

```text
R_t >= theta_C^- and E_t = 0 => F -> V
```

D3. Verification can update evidence so that eventually:

```text
E_t = 1
```

D4. If `R_t >= theta_C^+`, `E_t = 1`, and `K_t = 1`, then `F` or `V` can transition to `C`.

D5. In `C`, if `E_t = 1`, `K_t = 1`, and `log_t = 1`, the system can set `commit_stage = committed`.

D6. `log_t = 1` is eventually attainable when `K_t = 1`.

### Proof Sketch

Starting in `M`, sustained `R_t >= theta_C^+` implies `R_t >= theta_F^+`, so the system can enter `F` by D1.
If evidence is missing, D2 routes it to `V`.
By D3, verification eventually makes `E_t = 1`.
With sustained high `R_t` and `K_t = 1`, D4 allows transition to `C`.
By D6, logging can become ready.
By D5, `committed` is then reachable.

This is reachability, not inevitability.
The proposition says the kernel is not a permanent suspender when disturbance is sustained, evidence is sufficient, and constitutional constraints allow commit.

### Collapse Points

- verification never accumulates evidence.
- `V` has no path to `C`.
- `K_t` remains closed for reasons unrelated to the candidate.
- logging cannot be prepared.
- high `R_t` decays while evidence is being gathered despite continuing disturbance.
- `C` loops forever without formal commit.

### Engineering Test

Sustained high disturbance plus evidence:

- keep `R_t >= theta_C^+`
- gradually improve evidence until `E_t = 1`
- keep `K_t = 1`

Pass:

```text
path F -> V -> C -> committed appears in at least one scripted case
commit_latency is finite
```

---

## 5. Theorem/Experiment Boundary

The propositions above are structural.
They do not prove:

- learned heads estimate the correct proxy values
- `R_t` is scientifically valid
- the system improves benchmark performance
- formal commit is always semantically correct

Experiments can only test proxy adequacy under constructed conditions.

The correct separation is:

| Layer | Claim Type |
|---|---|
| transition table | structural theorem target |
| `E_t`, `K_t` gates | invariant enforcement target |
| proxy heads | empirical calibration target |
| `R_t` semantics | falsifiable modeling target |
| downstream task accuracy | later application target |

---

## 6. Minimal Theorem Checklist

Before implementing learned heads, verify:

```text
[ ] unique formal commit path
[ ] durable state cannot mutate outside commit path
[ ] log is atomic with commit
[ ] K_t is explicit-rule controlled
[ ] E_t cannot be overridden by high R_t
[ ] hysteresis widths are positive
[ ] R_t slope clipping is active
[ ] thresholds cannot be changed by candidate proposal
[ ] C has both exit and commit paths
```

