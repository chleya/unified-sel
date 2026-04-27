# CEP-CC Project Spec

## Hypothesis

Language-like structure can emerge as a compressed coordination protocol in continuous dynamical systems.

The target causal chain is:

```text
interaction pressure
+ compression pressure
+ memory or temporal consistency pressure
=> symbol-like trajectory partitions
=> compositional protocol
=> language-like dynamics
```

## Mechanism Replacement

Do not model:

```text
x_(t+1) ~ P(x_(t+1) | x_<=t)
```

Model continuous state evolution:

```text
z_(t+1) = F(z_t, u_t, m_t)
y_t = G(z_t)
```

Listener receives continuous communication:

```text
z_B_(t+1) = F_B(z_B_t, u_B_t, y_t, m_B_t)
```

No dictionary, no token vocabulary, no categorical communication action.

## Environment V0.1

Scene:

- `n_objects = 4`
- 2D positions in `[-1, 1]`
- velocities in `[-0.25, 0.25]`
- continuous object attribute `q in [-1, 1]`
- continuous global goal vector `g`

Speaker observation:

- all object positions
- all velocities
- all `q`
- goal vector

Listener observation:

- object positions only
- optionally partial goal projection

Target rule:

```text
target = argmax_k score(o_k, objects, g)
```

Initial rule family should be continuous and relation-sensitive:

```text
score_k =
  a1 * dot(position_k, g_position)
+ a2 * q_k * g_q
+ a3 * nearest_neighbor_velocity_alignment_k
+ a4 * relation_to_context_anchor_k
```

The system may know only task reward/target during training; it must not receive symbolic rule names.

## Communication

Speaker emits:

```text
Y = [y_1, ..., y_Tc]
Y in R^(8 x 6)
```

Constraints:

- amplitude clipping or tanh output
- energy penalty
- L1 sparsity penalty
- effective-dimension penalty on batch communication covariance
- optional temporal smoothness penalty

## Models

Start simple.

Speaker:

- MLP encoder over object set
- permutation-invariant pooling
- GRU or small recurrent cell to emit `T_c` continuous vectors

Listener:

- MLP encoder over partial object set
- GRU over communication trajectory
- object-wise scoring head

Allowed:

- MLP, RNN, GRU, ODE-RNN later

Avoid in V0.1:

- Transformers
- attention-heavy architectures
- discrete bottlenecks
- vector quantization

Reason: V0.1 tests whether continuous compression alone induces partitions.

## Loss

Task:

```text
L_task = cross_entropy(object_scores, target_index)
```

This uses a discrete task choice at the final action, not discrete communication.

Communication:

```text
L_energy = mean(Y^2)
L_sparse = mean(abs(Y))
L_smooth = mean((Y_t - Y_(t-1))^2)
L_dimeff = effective_dimension(cov(flatten(Y)))
```

State:

```text
L_state = mean(abs(z_A)) + mean(abs(z_B))
```

Consistency:

```text
L_temp = mean(||phi(Y_i) - phi(Y_j)||^2)
```

Only add `L_temp` after the base environment runs. V0.1 can begin with `L_task + L_comm`.

## Baselines

A. No communication.

- Listener acts from partial observation only.
- Establishes communication necessity.

B. High-bandwidth communication.

- Weak or zero compression.
- Tests whether structure is compression-induced.

C. Random projection communication.

- Speaker output replaced by fixed random projection of speaker observation.
- Tests whether learned protocol matters.

D. Continuous teacher-signal positive control.

- Speaker output replaced by continuous task-score trajectory derived from hidden factors.
- Tests whether the Listener and task can use continuous communication at all.
- This is not evidence of emergent protocol.

E. Discrete token communication.

- Small codebook baseline.
- Used only as comparison, not as the main mechanism.

## Acceptance For V0.1

Minimum pass:

- compressed communication beats no-communication on task success.
- compressed communication uses lower effective dimension than high-bandwidth communication.
- compressed communication shows higher prototype reuse or clustering quality than high-bandwidth communication.

Minimum reject:

- no task gain over no-communication.
- no reusable trajectory structure under compression.
- trajectory clusters do not align with any task-relevant invariant.
