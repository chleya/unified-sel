# Capability Benchmark Report

**Suite**: code | **Protocol**: monitor_repair_triage | **Monitor**: semantic
**Seed**: 7 | **Tasks**: 3 | **Solver**: search
**Schema**: 1.0 | **Data source**: real_verification | **Cost model**: assumed (hardcoded units)
**WARNING: Oracle assumption**: Escalation path uses OracleSolver (100% success by assumption)

## Summary

| Metric | Value |
|--------|-------|
| Success Rate | 1.0000 |
| Mean Cost (assumed units) | 1.5000 |
| Escalation Rate | 0.0000 |
| Revision Rate | 1.0000 |
| Verifier Rate | 1.0000 |
| Direct Escalation Rate | 0.0000 |
| Accepted w/o Verifier | 0.0000 |
| Mean Routing Signal | 0.5367 |

## Per-Family Breakdown

| Family | Count | Success | Cost | Escalation | Revision |
|--------|-------|---------|------|------------|----------|
| code | 3 | 1.000 | 1.500 | 0.000 | 0.000 |

## Route Trace

| Decision Path | Count |
|---------------|-------|
| accept_after_guarded_low_repairable_signal_revision | 2 |
| accept_after_repairable_signal_revision | 1 |

## Failures

No failures — all tasks resolved successfully.
