# CEP-CC V0.1 Scaffold - 2026-04-21

## Purpose

V0.1 starts the CEP-CC project as executable code.

The goal of this version is scaffold completion, not a positive proto-symbol claim yet.

Question for the full V0.1 experiment:

> Under communication compression, do continuous communication trajectories form stable reusable proto-symbol partitions while preserving task performance?

## Implemented

Files added:

- `experiments/cep_cc/env.py`
- `experiments/cep_cc/models.py`
- `experiments/cep_cc/losses.py`
- `experiments/cep_cc/metrics.py`
- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Project docs already exist:

- `experiments/cep_cc/README.md`
- `experiments/cep_cc/PROJECT_SPEC.md`
- `experiments/cep_cc/ROADMAP.md`
- `experiments/cep_cc/IMPLEMENTATION_PLAN.md`

## Environment

Implemented `ContinuousObjectSelectionEnv`.

Batch fields:

- `speaker_obs`: full continuous object state plus goal.
- `listener_obs`: position-only object projection plus partial goal.
- `target_index`: final object choice target.
- `latent_factors`: task-relevant continuous factors for analysis.
- `object_states`: full object states.
- `goal`: continuous goal vector.

No symbolic communication labels are provided to the agents.

## Models

Implemented:

- `Speaker`
- `Listener`
- `RandomProjectionSpeaker`
- `SpeakerListenerSystem`

Communication is continuous:

```text
comm shape = batch x 8 x 6
```

No token vocabulary, vector quantizer, categorical communication action, or next-token objective is used.

## Losses

Implemented:

- task cross-entropy over final object choice.
- communication energy.
- communication L1 sparsity.
- communication smoothness.
- communication effective dimension.
- state L1.

Effective dimension uses a stable participation-ratio proxy:

```text
DimEff = (sum eigenvalues)^2 / sum(eigenvalues^2)
```

Zero-variance communication returns `0.0`.

## Metrics

Implemented:

- task accuracy.
- communication energy.
- communication L1.
- communication effective dimension.
- prototype reuse rate.
- cluster compactness.
- target/cluster alignment.

The clustering metric is an internal k-means fallback and does not require scikit-learn.

## CLI

Example:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 80 --batch-size 64 --lambda-sweep 0.0,0.001,0.003 --seed 0 --table
```

Baselines:

```powershell
python -m experiments.cep_cc.run_experiment --baseline no-communication --seed 0 --table
python -m experiments.cep_cc.run_experiment --baseline high-bandwidth --seed 0 --table
python -m experiments.cep_cc.run_experiment --baseline random-projection --seed 0 --table
```

## Sanity Runs

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `6 passed`
- Pytest cache warning only.

Small CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 3 --batch-size 12 --lambda-sweep 0.0,0.001 --seed 0 --table
```

Result:

| run | task accuracy | comm energy | comm l1 | comm effective dim | prototype reuse | compactness | target alignment |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_communication | 0.229 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| lambda_0 | 0.229 | 0.023 | 0.128 | 3.006 | 0.208 | 2.232 | 0.532 |
| lambda_0.001 | 0.229 | 0.022 | 0.127 | 2.765 | 0.208 | 2.414 | 0.572 |

Longer high-bandwidth sanity:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 250 --batch-size 64 --baseline high-bandwidth --seed 1 --table
```

Result:

| run | task accuracy | comm energy | comm l1 | comm effective dim | prototype reuse | compactness | target alignment |
|---|---:|---:|---:|---:|---:|---:|---:|
| high-bandwidth | 0.297 | 0.086 | 0.238 | 1.103 | 0.145 | 5.634 | 0.412 |

## Interpretation

The code scaffold is working.

There is not yet a positive proto-symbol emergence result.

Current sanity runs show:

- communication is continuous and measurable.
- compression lowers communication magnitude and effective dimension.
- the task signal is not yet strong enough to claim communication necessity.
- high-bandwidth training only slightly exceeds random/no-communication in the quick run.

## Decision

Do not move to semantic stability or compositionality yet.

Next work should be C1b:

- calibrate the V0.1 environment and training loop until high-bandwidth learned communication clearly beats no-communication.
- only then run the compression sweep as a real proto-symbol emergence test.

Candidate C1b changes:

- simplify the first target rule so hidden continuous factors are learnable faster.
- add object-wise listener heads that combine partial object observations with decoded communication state.
- add a longer but bounded train/eval profile.
- add a positive-control test where Speaker sends a continuous teacher signal derived from hidden factors without discretization.

