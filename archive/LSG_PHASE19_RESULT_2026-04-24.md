# LSG Phase 19 Result - Audit-Only Revision Proposals

Date: 2026-04-24

## Decision

Phase 19 makes acknowledged-state revision explicit:

```text
ordinary observation after acknowledgement: ignored
revision request after acknowledgement: recorded as audit-only RevisionProposalEvent
state mutation after acknowledgement: not supported in Phase 19
```

This preserves the existing absorbing acknowledgement invariant while allowing later corrections to be represented without silent state rewrite.

## Implemented Artifacts

Updated:

- `core/rewrite_dynamics.py`
  - added `RevisionProposalEvent`
  - added `RewriteSystemState.revision_log`
  - added `propose_revision_for_acknowledged_candidate`
  - added `check_revision_log_invariants`
  - added `revision_events` to `simulate_case` timeline rows

- `tests/test_rewrite_dynamics.py`
  - added validation for invalid executed revision events
  - added audit-only revision proposal test for acknowledged candidates
  - added rejection test for unacknowledged revision targets

- `tests/smoke_test.py`
  - added smoke coverage proving a revision proposal does not mutate acknowledged candidate state or append acknowledgement commits

## Result

Core invariant:

```text
acknowledged candidate + later revision proposal
=> revision_log += 1
=> commit_log unchanged
=> candidate disturbance/stability/phase unchanged
=> revision_executed == false
```

Unsupported behavior remains blocked:

```text
revision proposal for unacknowledged candidate: rejected
executed revision without all gates and approval: rejected
executed revision state transition: not implemented in Phase 19
```

## Validation

Commands run:

```text
python tests\test_rewrite_dynamics.py
python tests\smoke_test.py
```

Observed status:

```text
All rewrite dynamics tests passed
All smoke tests passed
```

Attempted command:

```text
python -m py_compile core\rewrite_dynamics.py tests\test_rewrite_dynamics.py tests\smoke_test.py
```

Observed issue:

```text
[WinError 5] Access denied while replacing core\__pycache__\rewrite_dynamics.cpython-314.pyc
```

This appears to be a Windows `__pycache__` write/lock issue. Import-based unit and smoke validation passed.

## Interpretation

Phase 19 closes the biggest remaining semantic gap in acknowledged-state handling:

> LSG can now distinguish "a new observation tried to overwrite an acknowledged state" from "a formal revision request was made against an acknowledged state."

The former is ignored by the absorbing-state rule. The latter is durable audit data, but not an executed state change.

## Remaining Useful Work

- define the separate approval/execution protocol for actual rollback or revision
- add revision proposal cases to sequence replay fixtures
- add mixed fuzz coverage combining refill, identity collisions, and revision proposals
