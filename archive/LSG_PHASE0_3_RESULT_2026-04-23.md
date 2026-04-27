# LSG Phase 0-3 Result

Date: 2026-04-23
Status: implementation result summary

This report summarizes the current executable LSG prototype.

---

## 1. Implemented Files

Core:

```text
core/rewrite_dynamics.py
core/rewrite_proposal_provider.py
```

Tests:

```text
tests/test_rewrite_dynamics.py
tests/test_rewrite_proposal_provider.py
tests/smoke_test.py
```

Experiments:

```text
experiments/capability/rewrite_dynamics_sanity.py
experiments/capability/rewrite_dynamics_sweep.py
experiments/capability/rewrite_dynamics_cee_projection.py
experiments/capability/rewrite_dynamics_cee_roundtrip.py
experiments/capability/rewrite_dynamics_proposal_boundary.py
```

---

## 2. What Is Proven So Far

### Phase 0: Two-Variable Dynamics

Implemented:

- `D_t`: disturbance rewrite pressure
- `S_t`: order self-stability
- phase selection
- hysteresis
- bandwidth limit
- acknowledgement gate

Verified:

- temporary spike does not acknowledge
- sustained drift can acknowledge
- protected boundary does not acknowledge
- hysteresis reduces phase flips
- active candidates obey bandwidth limit
- no commit without evidence, constitution, and log gates

### Phase 1: `U/N/A/P/R` Interface

Implemented:

- `ProxyVector`
- `GovernanceScalars`
- `compute_governance_scalars`
- `observation_from_proxy`

Verified:

- proxy values map deterministically to `U/N/A/P/R`
- `R_t` is compatible with `D/S`
- candidate evidence gate is separate from current-state anchoring
- proxy-driven sustained drift can acknowledge

### Phase 2: Multi-Seed Scripted Sweep

Implemented:

- `rewrite_dynamics_sweep.py`

Verified across 10 seeds:

```text
temporary_spike commit_rate = 0.0
small_disturbance commit_rate = 0.0
sustained_drift commit_rate = 1.0
protected_boundary commit_rate = 0.0
```

### Phase 3: CEE Boundary Alignment

Implemented:

- CEE-style projection from LSG `CommitEvent`
- round-trip through CEE `CommitmentEvent.from_dict`
- round-trip through CEE `ModelRevisionEvent.from_dict`
- replay through CEE `EventLog.replay_world_state`

Verified:

- causal link is preserved
- projected revision replays into CEE `WorldState`
- anchor appears in replayed `anchored_fact_summaries`

### Proposal-Only Boundary

Implemented:

- `ProposalEnvelope`
- `ProposalRequest`
- `ProposalProvider`
- `MockProposalProvider`
- `MiniMaxProposalProvider` skeleton

Verified:

- model proposal cannot open `E/K/log` gates
- model proposal cannot change thresholds
- model proposal can only commit when external gates are open
- MiniMax provider is currently skeleton-only and makes no network call

---

## 3. Latest Result Artifacts

Sanity:

```text
results/rewrite_dynamics_sanity/rewrite_dynamics_sanity_20260423_072805.json
```

Sweep:

```text
results/rewrite_dynamics_sweep/rewrite_dynamics_sweep_phase2_smoke_20260423_083719.json
```

CEE round-trip:

```text
results/rewrite_dynamics_cee_roundtrip/rewrite_dynamics_cee_roundtrip_20260423_084817.json
```

Proposal boundary:

```text
results/rewrite_dynamics_proposal_boundary/rewrite_dynamics_proposal_boundary_20260423_085428.json
```

---

## 4. Validation Commands

Commands run successfully:

```powershell
python -m py_compile core\rewrite_dynamics.py core\rewrite_proposal_provider.py tests\test_rewrite_dynamics.py tests\test_rewrite_proposal_provider.py
python tests\test_rewrite_dynamics.py
python tests\test_rewrite_proposal_provider.py
python experiments\capability\rewrite_dynamics_sanity.py
python experiments\capability\rewrite_dynamics_sweep.py --label phase2_smoke
python experiments\capability\rewrite_dynamics_cee_projection.py
python experiments\capability\rewrite_dynamics_cee_roundtrip.py
python experiments\capability\rewrite_dynamics_proposal_boundary.py
python tests\smoke_test.py
```

---

## 5. MiniMax Integration Conditions

MiniMax may be added only as a proposal provider.

Allowed:

- generate `candidate_summary`
- suggest proxy values
- explain proposed evidence needs
- propose local candidate text

Forbidden:

- open `evidence_open`
- open `constitution_open`
- set `log_ready`
- change thresholds at runtime
- set `committed`
- bypass CEE-style commitment boundary

Before enabling network-backed MiniMax:

1. define response JSON schema
2. validate response into `ProposalEnvelope`
3. reject malformed responses
4. ignore all authority requests
5. log proposal audit record
6. keep deterministic mock provider tests

---

## 6. Current Claim

The current result supports this narrow claim:

```text
Formal state acknowledgement can be controlled by a two-variable
disturbance/stability model with explicit evidence, constitutional,
logging, bandwidth, and proposal-only boundaries.
```

It does not yet prove:

- learned proxy heads are valid
- MiniMax proposals improve proxy quality
- human-like attention is biologically realistic
- downstream task accuracy improves

---

## 7. Next Recommended Step

Next step:

```text
Add network-backed MiniMaxProposalProvider behind the existing ProposalProvider
contract, but keep all gates external and deterministic.
```

If network access or key handling is not ready, continue with:

```text
JSON-schema validation and malformed proposal rejection tests.
```

