# CEP-CC V0.1e Two-Stage Compression Fine-Tuning - 2026-04-21

## Purpose

V0.1e tests whether compression works better after a high-bandwidth continuous protocol is already learned.

Question:

> If a learned high-bandwidth protocol is established first, can compression fine-tuning simplify communication geometry without destroying task performance?

## Implementation

Files changed:

- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/metrics.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `train_existing`
- `run_two_stage_compression`
- CLI flag: `--two-stage-compression`
- CLI flags:
  - `--stage1-episodes`
  - `--stage2-episodes`
- table delta columns:
  - `delta_task_accuracy`
  - `delta_comm_energy`
  - `delta_comm_l1`
  - `delta_comm_effective_dim`
  - `delta_target_cluster_alignment`

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `11 passed`
- Pytest cache warning only.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --two-stage-compression --seeds 0,1,2 --stage1-episodes 300 --stage2-episodes 120 --batch-size 128 --lambda-sweep 0.001,0.003,0.006,0.01 --lr 0.005 --table
```

## Results

| run | eval accuracy | train accuracy | comm energy | comm l1 | comm effective dim | prototype reuse | compactness | target alignment | delta acc | delta energy | delta l1 | delta dim | delta align |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 0 stage1 high-bandwidth | 0.668 | 0.680 | 0.774 | 0.843 | 4.336 | 0.186 | 2.714 | 0.552 | n/a | n/a | n/a | n/a | n/a |
| seed 0 stage2 lambda 0.001 | 0.713 | 0.680 | 0.775 | 0.845 | 4.461 | 0.178 | 2.589 | 0.570 | 0.045 | 0.000 | 0.001 | 0.124 | 0.018 |
| seed 0 stage2 lambda 0.003 | 0.707 | 0.680 | 0.769 | 0.841 | 4.435 | 0.211 | 2.589 | 0.581 | 0.039 | -0.005 | -0.003 | 0.099 | 0.029 |
| seed 0 stage2 lambda 0.006 | 0.703 | 0.680 | 0.760 | 0.833 | 4.412 | 0.191 | 2.582 | 0.569 | 0.035 | -0.015 | -0.010 | 0.076 | 0.018 |
| seed 0 stage2 lambda 0.01 | 0.703 | 0.688 | 0.746 | 0.823 | 4.401 | 0.240 | 2.316 | 0.516 | 0.035 | -0.028 | -0.020 | 0.065 | -0.036 |
| seed 1 stage1 high-bandwidth | 0.859 | 0.836 | 0.727 | 0.810 | 4.225 | 0.184 | 2.076 | 0.614 | n/a | n/a | n/a | n/a | n/a |
| seed 1 stage2 lambda 0.001 | 0.879 | 0.891 | 0.739 | 0.818 | 4.539 | 0.201 | 1.974 | 0.614 | 0.020 | 0.012 | 0.008 | 0.314 | 0.000 |
| seed 1 stage2 lambda 0.003 | 0.877 | 0.891 | 0.737 | 0.817 | 4.534 | 0.201 | 1.974 | 0.612 | 0.018 | 0.010 | 0.007 | 0.308 | -0.002 |
| seed 1 stage2 lambda 0.006 | 0.881 | 0.891 | 0.735 | 0.816 | 4.526 | 0.203 | 1.976 | 0.614 | 0.021 | 0.008 | 0.006 | 0.301 | -0.000 |
| seed 1 stage2 lambda 0.01 | 0.881 | 0.891 | 0.733 | 0.814 | 4.516 | 0.209 | 1.977 | 0.616 | 0.021 | 0.006 | 0.004 | 0.291 | 0.002 |
| seed 2 stage1 high-bandwidth | 0.828 | 0.875 | 0.702 | 0.793 | 4.623 | 0.150 | 2.259 | 0.647 | n/a | n/a | n/a | n/a | n/a |
| seed 2 stage2 lambda 0.001 | 0.863 | 0.852 | 0.724 | 0.808 | 4.851 | 0.184 | 2.055 | 0.593 | 0.035 | 0.022 | 0.015 | 0.229 | -0.054 |
| seed 2 stage2 lambda 0.003 | 0.863 | 0.852 | 0.722 | 0.806 | 4.846 | 0.184 | 2.030 | 0.601 | 0.035 | 0.019 | 0.013 | 0.224 | -0.046 |
| seed 2 stage2 lambda 0.006 | 0.865 | 0.852 | 0.717 | 0.803 | 4.838 | 0.184 | 2.029 | 0.594 | 0.037 | 0.015 | 0.010 | 0.215 | -0.053 |
| seed 2 stage2 lambda 0.01 | 0.863 | 0.852 | 0.712 | 0.799 | 4.826 | 0.184 | 2.028 | 0.601 | 0.035 | 0.009 | 0.006 | 0.204 | -0.047 |

## Interpretation

Two-stage fine-tuning solves the task-collapse problem:

- all seeds improve task accuracy after stage-2 fine-tuning.
- seed 0 improves from `0.668` to `0.703-0.713`.
- seed 1 improves from `0.859` to `0.877-0.881`.
- seed 2 improves from `0.828` to `0.863-0.865`.

However, the run does not establish compression-induced simplification:

- communication energy and L1 mostly increase or barely decrease.
- effective dimension increases on every stage-2 run.
- target alignment is mixed:
  - seed 0 mostly improves.
  - seed 1 is flat.
  - seed 2 drops.

The stage-2 objective is currently acting more like continued task fine-tuning with a weak compression penalty than like real protocol compression.

## Decision

C1e partially succeeds:

- success: staged training avoids compressed-from-scratch collapse.
- failure: staged training does not simplify communication geometry.

Do not move to C2 semantic stability.

Next work should remain in C1:

- increase or schedule compression in stage 2.
- consider lower stage-2 learning rate.
- consider freezing Listener and compressing Speaker only.
- consider explicit target for communication geometry, such as stronger effective-dimension penalty or covariance shrinkage.

Recommended next branch:

- C1f: Stronger/Frozen Two-Stage Compression Ablation

