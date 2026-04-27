# CEP-CC

Compressed Emergent Protocol under Continuous Communication.

## Research Question

Can a continuous multi-agent dynamical system develop language-like structure without tokens, text data, conditional probability generation, or next-token prediction?

The first claim to test:

> Interaction pressure plus compression pressure plus memory pressure can induce stable, reusable, symbol-like partitions in continuous communication trajectories.

## Non-Goals

This project does not try to build an LLM.

It does not train on language data.

It does not use a token vocabulary or a next-token prediction objective.

It does not claim that early trajectory clusters are language. The minimum language-like standard is:

- prototype reuse
- semantic stability across reparameterized environments
- local compositionality under segment intervention

## Minimal System

Two agents:

- Speaker A observes the full continuous world state.
- Listener B observes only a partial projection and must act.

World state:

```text
w_t = objects + goal + context
```

Object state:

```text
o_k = (x, y, vx, vy, q)
```

Communication:

```text
y in R^(T_c x d_c)
T_c = 8
d_c = 6
```

The communication signal is a continuous trajectory, not a token sequence.

## Training Objective

```text
L = L_task + L_comm + L_state + L_temp
```

Where:

- `L_task`: listener selects the target object.
- `L_comm`: communication energy, sparsity, and low-dimensionality pressure.
- `L_state`: internal state compression pressure.
- `L_temp`: consistency pressure for semantically similar situations.

## Core Metrics

Task:

- task success
- generalization success
- noisy-channel success

Compression:

- communication energy
- communication effective dimension
- trajectory complexity

Proto-symbol emergence:

- trajectory clustering silhouette
- Davies-Bouldin index
- prototype reuse rate
- within/between trajectory distance ratio

Semantic stability:

- cluster alignment with task-relevant latent invariants
- probe accuracy from communication to invariant
- invariance under nuisance transformations

Compositionality:

- segment swap targeted effect
- segment swap off-target effect
- `CompScore = targeted_effect / off_target_effect`

Phase transition:

- sweep compression coefficient lambda
- track success, effective dimension, silhouette, reuse rate, semantic stability, and compositionality

## First Milestone

V0.1:

- implement continuous object-selection environment.
- implement Speaker/Listener with continuous communication.
- run baseline no-compression and compressed communication.
- run continuous teacher-signal positive control.
- report whether compression induces stable reusable trajectory clusters.
