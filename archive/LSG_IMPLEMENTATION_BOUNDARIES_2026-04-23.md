# LSG Implementation Boundaries

Date: 2026-04-23
Status: hard boundaries for Phase 0

This file defines what the first implementation may and may not do.

---

## 1. Phase 0 Allowed Work

Allowed files to create:

```text
core/rewrite_dynamics.py
tests/test_rewrite_dynamics.py
experiments/capability/rewrite_dynamics_sanity.py
results/rewrite_dynamics_sanity/*.json
```

Allowed behavior:

- deterministic simulator
- scripted test streams
- two-variable `D/S` dynamics
- hysteretic phase transitions
- bandwidth limit
- commit event logging
- JSON result artifact

---

## 2. Phase 0 Forbidden Work

Do not:

- call LLM APIs
- train proxy heads
- use network resources
- integrate CEE runtime
- modify CEE files
- modify existing Capability Router behavior
- modify TopoMem data or Chroma files
- modify CEE approval policies
- add autonomous self-modification
- add human-attention biological claims to code
- let candidate proposals change thresholds
- let any model decide `constitution_open`
- let any model decide `log_ready`
- let foreground/verify mutate durable state

---

## 3. Formal State Boundary

Only `acknowledged` may represent formal state promotion.

Forbidden:

```text
foreground -> durable state
verify -> durable state
background -> durable state
suppressed -> durable state
candidate proposal -> threshold change
candidate proposal -> gate override
```

Required:

```text
acknowledged requires CommitEvent
acknowledged requires evidence_open
acknowledged requires constitution_open
acknowledged requires log_ready
```

---

## 4. Gate Boundary

In Phase 0:

```text
evidence_open
constitution_open
log_ready
```

are scripted booleans.

They are not learned.
They are not inferred by a large model.
They are not changed by candidate text.

Later phases may compute them from explicit statistics, but they remain gate conditions.

---

## 5. Threshold Boundary

Thresholds are config values.

Forbidden:

- candidate modifies thresholds
- LLM proposes threshold and it is applied automatically
- test case changes threshold mid-run without explicit experiment config

Allowed later:

- offline threshold sweep
- hand-reviewed threshold update
- scripted ablation comparing thresholds

---

## 6. Bandwidth Boundary

Active candidates must obey:

```text
len(active_candidate_ids) <= bandwidth_limit
```

No exception in Phase 0.

If more candidates qualify, the simulator must select top candidates by a deterministic priority rule.

Suggested priority:

```text
ratio first
margin second
disturbance third
candidate_id tie-break
```

---

## 7. Result Boundary

Result artifact must be machine-readable JSON.

Do not manually edit result JSON.

Required location:

```text
results/rewrite_dynamics_sanity/
```

Required fields are listed in `LSG_PHASE0_PROTOCOL_2026-04-23.md`.

---

## 8. Git Boundary

Do not stage the whole dirty worktree.

When committing Phase 0, stage only:

```text
core/rewrite_dynamics.py
tests/test_rewrite_dynamics.py
experiments/capability/rewrite_dynamics_sanity.py
results/rewrite_dynamics_sanity/<intentional-small-result>.json
LSG_*.md
REWRITE_QUALIFICATION_DYNAMICS_v0.md
```

Do not stage:

```text
topomem/data/chromadb/*
CEP_CC_*.md
META_CONTROLLER_*.md
large generated caches
```

---

## 9. Done Definition

Phase 0 is done when:

```text
python -m py_compile ... passes
python tests/test_rewrite_dynamics.py passes
python experiments/capability/rewrite_dynamics_sanity.py writes JSON
Phase 0 pass criteria are satisfied
```

No benchmark performance claim is required.

No LLM result is required.

