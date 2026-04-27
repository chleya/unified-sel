# F Drive Integration Risk Assessment - 2026-04-20

## Verdict

Directly wiring the existing F-drive projects into the learned meta-controller experiment is not recommended.

The projects are useful, but many are research prototypes with dirty worktrees, partial implementations, encoding issues, cached outputs, and unclear package boundaries. Treat them as reference material and module donors, not as stable dependencies.

The first experiment should remain inside `F:\unified-sel` and copy or reimplement only the smallest needed ideas after review.

## Why Direct Integration Is Risky

### 1. Dirty Worktrees

Several candidate projects have large uncommitted or untracked changes.

Observed:

- `F:\cognitive-execution-engine`: many modified source files, tests, new modules, and inaccessible cache/artifact directories.
- `F:\SDAS`: modified source files, deleted files, generated caches, new experiment outputs, new modules.
- `F:\sel-lab`: smaller but still has modified files and cache permission warnings.
- `F:\diff_world`: not a git repository, so there is no clean version boundary.

Risk:

- Importing or vendoring directly may capture an unstable intermediate state.
- Test failures could come from unrelated local edits.
- Later updates could silently change experiment behavior.

Decision:

- No `pip install -e` from these projects into the main meta-controller experiment.
- No cross-repo imports in V0.
- Copy only reviewed, minimal code snippets or reimplement from scratch.

### 2. Package Boundary Is Uneven

Observed:

- `cognitive-execution-engine` has a proper `pyproject.toml` and `src` layout, but its current tree is heavily modified.
- `SDAS` has `setup.py` and light dependencies, but source and tests are in active flux.
- `sel-lab` has a large internal research structure but not a clean external API for this experiment.
- `diff_world` is minimal and readable but lacks package metadata and has mojibake in docs/comments.

Risk:

- Direct imports would couple V0 to research-internal paths and assumptions.
- Refactoring or cleaning one donor project could break the integration experiment.

Decision:

- Define fresh interfaces in `unified-sel/experiments/meta_controller/`.
- Use donor projects only to inform implementation.

### 3. Encoding / Documentation Quality Issues

Observed:

- Some Chinese docs and comments render as mojibake in `diff_world` and `SDAS` files.

Risk:

- Documentation cannot be trusted without opening source and verifying behavior.
- Copying comments or docs directly would pollute new project files.

Decision:

- New meta-controller files should be ASCII unless there is a clear reason otherwise.
- Use clean English docstrings and test names in the new experiment.

### 4. Experimental Claims Are Not Transferable

Observed:

- Existing projects report useful experiments, but they target different claims:
  - SEL selector/router benchmarks
  - SDAS structure reuse/adaptation
  - CEE policy-guarded runtime and state transitions
  - DiffWorld delta prediction

Risk:

- Reusing their results as evidence for the learned unified mechanism would be overclaiming.
- Their success does not imply the meta-controller has learned causal arbitration.

Decision:

- Existing results can motivate design.
- The learned meta-controller needs its own baselines, metrics, ablations, and transfer tests.

## Project-by-Project Integration Grade

| Project | Direct Import? | Copy Minimal Code? | Use As Design Reference? | Risk |
|---|---:|---:|---:|---|
| `F:\cognitive-execution-engine` | No for V0 | Maybe later | Yes | High |
| `F:\sel-lab` | No for V0 | Yes, selected benchmark patterns | Yes | Medium |
| `F:\SDAS` | No for V0 | Maybe selected structure-pool ideas | Yes | High |
| `F:\diff_world` | No | Yes, delta predictor idea | Yes | Medium-low |
| `F:\FCRS-World` / `F:\fcrs-v5` | No for V0 | Maybe later | Yes | Medium |
| `F:\SDAS-LLM\sdas-structure-memory` | No for V0 | Yes, if small JSON memory is needed | Yes | Medium |
| `F:\system_stability` | No | Rarely | Yes, metrics only | Medium |
| `F:\highway-agent-kernel` | No | Rarely | Yes, invariant/policy style | Medium |

## Recommended Rule

Use a three-tier integration rule.

### Tier 0: Read-Only Reference

Default for all F-drive projects.

Allowed:

- read source
- read tests
- copy design patterns into notes
- cite concepts in local docs

Forbidden:

- direct imports from sibling project paths
- installing sibling repos as editable packages
- depending on their current test status

### Tier 1: Minimal Vendoring

Allowed only when one function/class is small, dependency-light, and behavior can be covered by local tests.

Requirements:

- copied into `unified-sel/experiments/meta_controller/` or a clearly scoped local module
- rewritten or cleaned where appropriate
- local tests added
- source project noted in comments or docs

Examples:

- delta-prediction idea from `diff_world`
- benchmark matrix style from `sel-lab`
- JSON memory shape from `SDAS-LLM`

### Tier 2: Real Integration

Allowed only after V0 proves the experimental shape.

Requirements:

- donor project has a clean git state or a pinned commit/branch
- public API is identified
- smoke tests pass in isolation
- integration tests pass in `unified-sel`
- behavior is covered by ablation so the donor module does not hide the meta-controller effect

Examples:

- CEE-style event log and replay
- SDAS structure pool
- FCRS predictive compression

## What To Build First

Build V0 without cross-project imports.

Directory:

```text
F:\unified-sel\experiments\meta_controller\
```

Initial files:

```text
README.md
env.py
modules.py
meta_controller.py
baselines.py
metrics.py
run_experiment.py
report.py
```

Use local, simple implementations:

- symbolic regime-shift environment
- table or rule-based habit policy
- short-horizon planner
- small list/KV episodic memory
- simple predictor for surprise
- contextual-bandit or tabular-Q meta-controller

Then compare:

- `habit_only`
- `planner_always`
- `memory_always`
- `fixed_rule_controller`
- `random_controller`
- `learned_contextual_bandit`

## When To Reuse Each Project

### Use `diff_world` Early, But By Reimplementation

Best for:

- prediction error
- delta prediction
- surprise signal

How:

- Reimplement a small delta predictor locally.
- Do not import from `F:\diff_world`.

### Use `sel-lab` For Benchmark Discipline

Best for:

- benchmark family organization
- ablation matrices
- selector/router comparison style

How:

- Copy the reporting pattern, not the whole framework.
- Keep V0 experiment independent.

### Use `cognitive-execution-engine` After V0

Best for:

- event log
- replayable state transitions
- policy-mediated memory/state commitment
- audit trail

How:

- After V0, extract a minimal event schema.
- Do not wire in the full runtime until the meta-controller effect is already measurable.

### Use `SDAS` After V0.1

Best for:

- structure pool
- action-driven update
- regime adaptation ideas

How:

- Use as candidate replacement for habit/memory modules after the V0 baseline is stable.
- Do not make SDAS the first environment or controller.

### Use `system_stability` Only For Metrics

Best for:

- drift
- reset
- stability / recovery vocabulary

How:

- Translate concepts into local metrics.
- Do not import its runtime.

## Practical Acceptance Gate Before Any Donor Code Enters V0

A donor module can enter only if all are true:

1. It has fewer than two external dependencies, or dependencies already exist in `unified-sel`.
2. The needed behavior can be explained in one paragraph.
3. The copied/reimplemented version is under about 150 lines.
4. A local test covers its behavior.
5. Removing it in an ablation still leaves the experiment runnable.
6. It does not make the learned meta-controller's causal role harder to isolate.

## Final Recommendation

Do not "接入" the old projects first.

First build the experiment spine in `unified-sel`, with deliberately small local modules. Then promote donor ideas one at a time only when they improve a specific measured weakness:

- poor surprise signal -> borrow from `diff_world`
- weak benchmark/reporting -> borrow from `sel-lab`
- poor audit/replay -> borrow from `cognitive-execution-engine`
- weak habit/adaptation -> borrow from `SDAS`
- weak drift metrics -> borrow from `system_stability`

This keeps the causal question clean: if the experiment works, the result belongs to the learned meta-controller, not to an accidental mixture of unfinished systems.
