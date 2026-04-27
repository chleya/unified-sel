# CEP-CC Implementation Plan

## V0.1 File Plan

Recommended files:

```text
experiments/cep_cc/
  __init__.py
  env.py
  models.py
  losses.py
  metrics.py
  run_experiment.py
  README.md
  PROJECT_SPEC.md
  ROADMAP.md
  IMPLEMENTATION_PLAN.md
tests/
  test_cep_cc_protocol.py
```

## Step 1: Environment

Implement `ContinuousObjectSelectionEnv`.

API:

```python
batch = env.sample_batch(batch_size, seed=None)
```

Batch fields:

- `speaker_obs`: full continuous object state.
- `listener_obs`: partial object state.
- `target_index`: final object choice target.
- `latent_factors`: continuous task-relevant invariants for analysis only.

No symbolic labels should be given to the agents.

## Step 2: Models

Implement:

- `Speaker`
- `Listener`
- `SpeakerListenerSystem`

Speaker returns:

```python
comm: FloatTensor[batch, T_c, d_c]
speaker_state: FloatTensor[batch, d_z]
```

Listener returns:

```python
object_logits: FloatTensor[batch, n_objects]
listener_state: FloatTensor[batch, d_z]
```

## Step 3: Losses

Implement:

- `task_loss`
- `communication_energy`
- `communication_sparsity`
- `communication_smoothness`
- `effective_dimension`
- `state_l1`

V0.1 default:

```text
L = L_task
  + lambda_energy * L_energy
  + lambda_sparse * L_sparse
  + lambda_dimeff * L_dimeff
```

## Step 4: Metrics

Implement:

- task accuracy
- communication energy
- communication L1
- communication effective dimension
- prototype reuse rate
- silhouette score if scikit-learn is available
- fallback clustering proxy if not available

Do not make scikit-learn mandatory for the first smoke test.

## Step 5: CLI

Command shape:

```powershell
python -m experiments.cep_cc.run_experiment --mode train-eval --episodes 2000 --batch-size 128 --lambda-sweep 0.0,0.001,0.003,0.01,0.03 --seed 0 --table
```

Baselines:

```powershell
python -m experiments.cep_cc.run_experiment --baseline no-communication --seed 0 --table
python -m experiments.cep_cc.run_experiment --baseline high-bandwidth --seed 0 --table
python -m experiments.cep_cc.run_experiment --baseline random-projection --seed 0 --table
python -m experiments.cep_cc.run_experiment --baseline teacher-signal --seed 0 --table
```

## Step 6: Tests

Minimum tests:

- environment emits correct tensor shapes.
- no communication tensor is discrete or integer-coded.
- forward pass returns object logits and continuous communication.
- one tiny train loop reduces task loss on a fixed seed.
- metrics summarize communication complexity.

## V0.1 Stop Condition

Stop after the first lambda sweep.

Write:

- success vs lambda.
- effective dimension vs lambda.
- prototype reuse vs lambda.
- cluster/invariant alignment if available.

Do not proceed to compositionality until proto-symbol emergence is visible or clearly falsified.
