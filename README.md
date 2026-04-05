# Unified-SEL

Unified-SEL is a research codebase for continual learning without explicit task-boundary signals.

Its core hypothesis is:

> A learner should be able to detect environmental change from its own internal signals,
> create or split structure when needed, and preserve useful old structure without being told
> that "the task has changed".

The project is therefore aimed first at **endogenous task-boundary formation**, not at being merely a stronger generic continual-learning baseline.

## Real Relationship Between the Three Source Projects

This repository is **not** a direct merge of three existing projects.
It is a new research-oriented integration that draws different strengths from each source:

- `F:\sel-lab`
  - Provides the main continual-learning line
  - Contributes DFA-style local learning
  - Contributes tension-driven structural evolution
  - Contributes the strongest existing experimental framing for sequential learning and forgetting

- `F:\SDAS`
  - Provides the surprise-driven structure-pool idea
  - Contributes the `observe -> reinforce / branch / create` pattern
  - Contributes the idea that unfamiliar inputs should trigger new structure without external task labels

- `F:\fcrs_mis`
  - Provides engineering reference rather than the same learning algorithm
  - Contributes stricter type organization, pool abstraction, validation, capacity management, and vectorized implementation style

## Design Position

The intended design is:

- use `SEL-Lab` as the learning backbone
- use `SDAS` as the source of surprise-driven structure creation
- use `FCRS` as the engineering and abstraction reference

So the project should be understood as:

`SEL-Lab learning core` + `SDAS surprise mechanism` + `FCRS engineering discipline`

not as a literal code merge.

## Current State

Current `unified-sel` is still an early skeleton implementation.

What exists now:

- `core/structure.py`
- `core/pool.py`
- `core/learner.py`
- `core/runtime.py`
- minimal baseline placeholders in `experiments/baselines/`
- `tests/smoke_test.py`

What is true right now:

- the repository already reflects the intended integration direction
- the smoke test passes
- the code is not yet feature-aligned with the fuller mechanisms in the three source projects
- some planned experiment and analysis entrypoints described in `AGENTS.md` still need to be implemented

## Target Mechanism

Unified-SEL is built around two complementary internal signals:

- `tension`
  - asks whether an existing structure is saturated or struggling
  - high tension suggests cloning or splitting work across structures

- `surprise`
  - asks whether the current input fits any known structure
  - high surprise suggests creating a new structure

These two signals answer different questions:

- `tension`: "Is the current structure enough?"
- `surprise`: "Have I seen this kind of situation before?"

The research claim of Unified-SEL depends on combining both.

## Research Goal

The main research target is to show that, in a no-boundary continual-learning setting:

- Unified-SEL can form task-boundary structure from internal signals alone
- these endogenous boundaries remain useful under drift and capacity pressure
- the resulting boundary dynamics help explain both retention and transfer

Performance against baselines still matters, but as validation rather than identity:

- Unified-SEL should remain competitive with EWC on average accuracy
- Unified-SEL should improve forgetting relative to its current best-known result
- comparisons should eventually hold across multiple random seeds with stronger statistical support

So the primary claim is:

`endogenous boundary formation as the mechanism`

not:

`a slightly better continual-learning baseline`

## Workflow

See `AGENTS.md` for the project operating rules and `STATUS.md` for the current task.
