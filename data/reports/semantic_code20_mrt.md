# Capability Benchmark Report

**Suite**: code | **Protocol**: monitor_repair_triage | **Monitor**: semantic
**Seed**: 7 | **Tasks**: 20 | **Solver**: search
**Schema**: N/A | **Data source**: N/A | **Cost model**: N/A

## Summary

| Metric | Value |
|--------|-------|
| Success Rate | 1.0000 |
| Mean Cost (assumed units) | 1.3750 |
| Escalation Rate | 0.0000 |
| Revision Rate | 0.6500 |
| Verifier Rate | 1.0000 |
| Direct Escalation Rate | 0.0000 |
| Accepted w/o Verifier | 0.1000 |
| Mean Routing Signal | 0.5735 |

## Per-Family Breakdown

| Family | Count | Success | Cost | Escalation | Revision |
|--------|-------|---------|------|------------|----------|
| code | 20 | 1.000 | 1.375 | 0.000 | 0.000 |

## Route Trace

| Decision Path | Count |
|---------------|-------|
| accept_after_repairable_signal_revision | 8 |
| accept_after_guarded_low_repairable_signal_revision | 5 |
| accept_after_repairable_signal_verify | 5 |
| accept_low_monitor_signal | 2 |

## Failures

No failures — all tasks resolved successfully.
