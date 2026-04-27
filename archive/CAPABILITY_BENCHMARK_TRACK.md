# CAPABILITY_BENCHMARK_TRACK

## Purpose

This file defines the first executable benchmark track for the project's long-range capability line.

The goal is not to prove that the current system already has strong reasoning or coding ability.
The goal is to create a clean evaluation scaffold for the next stage:

- local-only solving
- local + verification
- local + escalation

## Current Scope

The first scaffold includes two task families:

1. `reasoning`
   - short-horizon arithmetic reasoning
   - exact-match evaluation

2. `code`
   - micro code-repair tasks
   - hidden-test evaluation

3. `mixed`
   - combines both families

## Local Capability Module

The benchmark now supports two local-solver modes:

1. `heuristic`
   - simple baseline
   - useful as a low-control comparison

2. `search`
   - current default
   - exact symbolic solving for reasoning tasks
   - verifier-guided patch search for code-repair tasks

This `search` solver is the first real local capability module in the repository.

## Protocols

### `local_only`

- one local attempt
- no retry
- no escalation

### `local_verify`

- local attempt
- explicit verifier check
- one local revision pass on verifier failure

### `local_escalate`

- local attempt
- escalate when confidence is low or verification fails
- oracle fallback stands in for a future stronger remote model

## Why This Matters

This benchmark moves the project from:

- anti-forgetting only

to:

- capability under control policy
- verification value
- escalation cost vs success tradeoff

## Current Implementation

- core scaffold:
  - [capability_benchmark.py](F:/unified-sel/core/capability_benchmark.py)
- runnable entrypoint:
  - [benchmark.py](F:/unified-sel/experiments/capability/benchmark.py)

Hardened reference runs:

- search `local_only`:
  - [20260409_155001.json](F:/unified-sel/results/capability_benchmark/20260409_155001.json)
- search `local_verify`:
  - [20260409_155030.json](F:/unified-sel/results/capability_benchmark/20260409_155030.json)
- search `local_escalate`:
  - [20260409_155624.json](F:/unified-sel/results/capability_benchmark/20260409_155624.json)

In the current hardened 8-task mixed sample:

- `local_only`
  - success rate `0.75`
  - mean cost `1.0`
- `local_verify`
  - success rate `0.875`
  - mean cost `1.2875`
  - revision rate `0.25`
- `local_escalate`
  - success rate `1.0`
  - mean cost `2.225`
  - escalation rate `0.25`

Interpretation:

- the benchmark no longer saturates under the search-based local solver
- `reverse_words` is now a verifier-helpful task:
  - `local_only` fails it
  - `local_verify` repairs it locally
- `dedupe_sorted` is now the clean escalation-pressure task:
  - current local search does not repair it
  - `local_escalate` resolves it through oracle fallback
- under the current protocol, `local_escalate` also escalates `reverse_words` because it does not attempt a local revision before fallback

This is the first clean three-level separation in the capability line:

- local solving has visible limits
- verification has measurable standalone value
- escalation has measurable residual value after local failure

First routing-policy comparison references:

- `confidence_threshold` at `0.90`:
  - [20260409_162606.json](F:/unified-sel/results/capability_benchmark/20260409_162606.json)
- `verifier_first`:
  - [20260409_162635.json](F:/unified-sel/results/capability_benchmark/20260409_162635.json)
- `escalation_first`:
  - [20260409_162655.json](F:/unified-sel/results/capability_benchmark/20260409_162655.json)
- routing comparison note:
  - [CAPABILITY_ROUTING_COMPARISON_2026-04-09.md](F:/unified-sel/results/CAPABILITY_ROUTING_COMPARISON_2026-04-09.md)

Initial routing result on the same hardened 8-task mixed sample:

- `confidence_threshold` (`0.90`)
  - success rate `0.75`
  - mean cost `1.0`
  - verifier rate `0.0`
- `verifier_first`
  - success rate `1.0`
  - mean cost `1.75`
  - escalation rate `0.125`
  - revision rate `0.25`
- `escalation_first`
  - success rate `1.0`
  - mean cost `2.225`
  - escalation rate `0.25`

Initial routing interpretation:

- naive confidence-only control fails on this benchmark because the local module is overconfident on wrong code answers
- `verifier_first` dominates `escalation_first` on this sample:
  - same final success
  - lower escalation rate
  - lower mean cost
- the current benchmark is now strong enough to compare control logic, not only solver strength

Confidence-threshold sweep result:

- threshold `0.90` / `0.94`
  - success `0.75`
  - mean cost `1.0`
- threshold `0.95` / `0.99`
  - success `1.0`
  - mean cost `2.9`
  - escalation rate `0.5`
- threshold `1.00`
  - success `1.0`
  - mean cost `4.8`
  - escalation rate `1.0`

Interpretation:

- confidence routing is not a smooth Pareto on this scaffold
- the local confidence values are too clustered to separate:
  - easy local code wins
  - suspicious visible-test-only answers

First surprise-like routing reference:

- `surprise_gate` at `0.50`:
  - [20260409_165805.json](F:/unified-sel/results/capability_benchmark/20260409_165805.json)

First surprise-like routing result:

- success rate `1.0`
- mean cost `1.6`
- escalation rate `0.125`
- verifier rate `0.25`

Interpretation:

- a model-external routing signal already beats confidence-only routing on this benchmark
- on the current sample it also beats `verifier_first` on mean cost while preserving success
- the current version is now based on structured search diagnostics rather than note-string matching

Monitor split references:

- `monitor_gate confidence`:
  - [20260409_171444.json](F:/unified-sel/results/capability_benchmark/20260409_171444.json)
- `monitor_gate diagnostic`:
  - [20260409_171501.json](F:/unified-sel/results/capability_benchmark/20260409_171501.json)
- `monitor_gate hybrid`:
  - [20260409_171516.json](F:/unified-sel/results/capability_benchmark/20260409_171516.json)
- `monitor_gate external`:
  - [20260409_173450.json](F:/unified-sel/results/capability_benchmark/20260409_173450.json)

Monitor comparison interpretation:

- under a fixed `monitor_gate` policy:
  - `confidence` reaches success `1.0`, mean cost `1.65`
  - `diagnostic` reaches success `1.0`, mean cost `1.6`
  - `hybrid` reaches success `1.0`, mean cost `1.6`
  - `external` reaches success `1.0`, mean cost `1.6`
- current reading:
  - the gain is not mostly from more complicated gate logic
  - the gain is from better risk structure in the diagnostic-family signals
  - hybrid does not currently beat diagnostic, so the diagnostic signal is already the main useful component
  - the external monitor matching diagnostic is an important step:
    - high-quality routing no longer depends on solver-internal fields alone

Harder separation probe:

- new tasks:
  - `normalize_spaces`
  - `normalize_commas`
  - `normalize_pipes`
- purpose:
  - create ambiguous visible-pass cases where a changed but wrong patch can look safe to an external monitor
- references:
  - `diagnostic` harder probe:
    - [20260409_183559.json](F:/unified-sel/results/capability_benchmark/20260409_183559.json)
  - `external` harder probe:
    - [20260409_183616.json](F:/unified-sel/results/capability_benchmark/20260409_183616.json)
  - reinforced `diagnostic` harder probe on `code-8`:
    - [20260409_185136.json](F:/unified-sel/results/capability_benchmark/20260409_185136.json)
  - reinforced `external` harder probe on `code-8`:
    - [20260409_185147.json](F:/unified-sel/results/capability_benchmark/20260409_185147.json)
  - reinforced `counterfactual` harder probe on `code-8`:
    - [20260409_185159.json](F:/unified-sel/results/capability_benchmark/20260409_185159.json)
  - `behavioral` harder probe on `code-8` with non-mirrored challenge tests:
    - [20260410_011709.json](F:/unified-sel/results/capability_benchmark/20260410_011709.json)
  - stress `external` probe on `code-9`:
    - [20260410_062640.json](F:/unified-sel/results/capability_benchmark/20260410_062640.json)
  - stress `counterfactual` probe on `code-9`:
    - [20260410_062651.json](F:/unified-sel/results/capability_benchmark/20260410_062651.json)
  - stress `behavioral` probe on `code-9`:
    - [20260410_062712.json](F:/unified-sel/results/capability_benchmark/20260410_062712.json)
  - stress `surface` probe on `code-9`:
    - [20260410_073827.json](F:/unified-sel/results/capability_benchmark/20260410_073827.json)
  - stress `external` probe on `mixed-18`:
    - [20260410_062757.json](F:/unified-sel/results/capability_benchmark/20260410_062757.json)
  - serial `behavioral` confirmation on `mixed-18`:
    - [20260410_063059.json](F:/unified-sel/results/capability_benchmark/20260410_063059.json)
  - serial `counterfactual` confirmation on `mixed-18`:
    - [20260410_063108.json](F:/unified-sel/results/capability_benchmark/20260410_063108.json)
  - `surface` confirmation on `mixed-18`:
    - [20260410_073837.json](F:/unified-sel/results/capability_benchmark/20260410_073837.json)
  - expanded stress `external` probe on `code-10`:
    - [20260410_075609.json](F:/unified-sel/results/capability_benchmark/20260410_075609.json)
  - expanded stress `behavioral` probe on `code-10`:
    - [20260410_075624.json](F:/unified-sel/results/capability_benchmark/20260410_075624.json)
  - expanded stress `surface` probe on `code-10`:
    - [20260410_075638.json](F:/unified-sel/results/capability_benchmark/20260410_075638.json)
  - expanded stress `counterfactual` probe on `code-10`:
    - [20260410_075650.json](F:/unified-sel/results/capability_benchmark/20260410_075650.json)
  - expanded stress `external` probe on `mixed-20`:
    - [20260410_075709.json](F:/unified-sel/results/capability_benchmark/20260410_075709.json)
  - expanded stress `behavioral` probe on `mixed-20`:
    - [20260410_075720.json](F:/unified-sel/results/capability_benchmark/20260410_075720.json)
  - expanded stress `surface` probe on `mixed-20`:
    - [20260410_075946.json](F:/unified-sel/results/capability_benchmark/20260410_075946.json)
  - expanded stress `counterfactual` probe on `mixed-20`:
    - [20260410_080000.json](F:/unified-sel/results/capability_benchmark/20260410_080000.json)
  - expanded stress `semantic` probe on `code-10`:
    - [20260410_082442.json](F:/unified-sel/results/capability_benchmark/20260410_082442.json)
  - expanded stress `semantic` probe on `mixed-20`:
    - [20260410_082457.json](F:/unified-sel/results/capability_benchmark/20260410_082457.json)
  - generalized stress `external` probe on `code-11`:
    - [20260410_084244.json](F:/unified-sel/results/capability_benchmark/20260410_084244.json)
  - generalized stress `behavioral` probe on `code-11`:
    - [20260410_084255.json](F:/unified-sel/results/capability_benchmark/20260410_084255.json)
  - generalized stress `surface` probe on `code-11`:
    - [20260410_084305.json](F:/unified-sel/results/capability_benchmark/20260410_084305.json)
  - generalized stress `semantic` probe on `code-11`:
    - [20260410_084315.json](F:/unified-sel/results/capability_benchmark/20260410_084315.json)
  - generalized stress `counterfactual` probe on `code-11`:
    - [20260410_084326.json](F:/unified-sel/results/capability_benchmark/20260410_084326.json)
  - generalized stress `external` probe on `mixed-22`:
    - [20260410_084349.json](F:/unified-sel/results/capability_benchmark/20260410_084349.json)
  - generalized stress `behavioral` probe on `mixed-22`:
    - [20260410_084401.json](F:/unified-sel/results/capability_benchmark/20260410_084401.json)
  - generalized stress `surface` probe on `mixed-22`:
    - [20260410_084413.json](F:/unified-sel/results/capability_benchmark/20260410_084413.json)
  - generalized stress `semantic` probe on `mixed-22`:
    - [20260410_084432.json](F:/unified-sel/results/capability_benchmark/20260410_084432.json)
  - generalized stress `counterfactual` probe on `mixed-22`:
    - [20260410_084442.json](F:/unified-sel/results/capability_benchmark/20260410_084442.json)

Harder-probe interpretation:

- on the original mixed sample, `external` matched `diagnostic`
- on the harder `code-7` sample, they separate:
  - `diagnostic`: success `1.0`
  - `external`: success `0.8571`
- on the reinforced `code-8` sample, the separation becomes cleaner:
  - `diagnostic`: success `1.0`, mean cost `1.7875`
  - `counterfactual`: success `1.0`, mean cost `1.7875`
  - `behavioral`: success `1.0`, mean cost `1.7875`
  - `external`: success `0.75`, mean cost `1.6625`
- on the `mixed-16` run that includes the same harder code block:
  - `diagnostic`: success `1.0`, mean cost `1.39375`
  - `counterfactual`: success `1.0`, mean cost `1.39375`
  - `behavioral`: success `1.0`, mean cost `1.39375`
  - `external`: success `0.875`, mean cost `1.33125`
- on the new stress `code-9` run with `normalize_pipes` added:
  - `counterfactual`: success `1.0`, mean cost `1.7555555555555555`
  - `behavioral`: success `1.0`, mean cost `1.7555555555555555`
  - `surface`: success `1.0`, mean cost `1.7555555555555555`
  - `external`: success `0.6666666666666666`, mean cost `1.588888888888889`
- on the new stress `mixed-18` run with the same code block:
  - `counterfactual`: success `1.0`, mean cost `1.3777777777777778`
  - `behavioral`: success `1.0`, mean cost `1.3777777777777778`
  - `surface`: success `1.0`, mean cost `1.3777777777777778`
  - `external`: success `0.8333333333333334`, mean cost `1.3222222222222224`
- on the expanded stress `code-10` run with `count_positive` added:
  - `counterfactual`: success `1.0`, mean cost `1.73`
  - `behavioral`: success `0.9`, mean cost `1.6800000000000002`
  - `surface`: success `0.9`, mean cost `1.6800000000000002`
  - `external`: success `0.6`, mean cost `1.53`
- on the expanded stress `mixed-20` run with the same ambiguity task embedded:
  - `counterfactual`: success `1.0`, mean cost `1.365`
  - `behavioral`: success `0.95`, mean cost `1.34`
  - `surface`: success `0.95`, mean cost `1.34`
  - `external`: success `0.8`, mean cost `1.2650000000000001`
- on the expanded stress `semantic` runs:
  - `code-10`: success `1.0`, mean cost `1.73`
  - `mixed-20`: success `1.0`, mean cost `1.365`
- on the generalized stress with `count_negative` added:
  - `code-11`
    - `external`: success `0.5454545454545454`, mean cost `1.481818181818182`
    - `behavioral`: success `0.8181818181818182`, mean cost `1.6181818181818182`
    - `surface`: success `0.8181818181818182`, mean cost `1.6181818181818182`
    - `semantic`: success `1.0`, mean cost `1.7090909090909092`
    - `counterfactual`: success `1.0`, mean cost `1.7090909090909092`
  - `mixed-22`
    - `external`: success `0.7727272727272727`, mean cost `1.240909090909091`
    - `behavioral`: success `0.9090909090909091`, mean cost `1.309090909090909`
    - `surface`: success `0.9090909090909091`, mean cost `1.309090909090909`
    - `semantic`: success `1.0`, mean cost `1.3545454545454545`
    - `counterfactual`: success `1.0`, mean cost `1.3545454545454545`
- this is the desired benchmark behavior:
  - the scaffold can now expose the residual value of solver-process diagnostics
  - `behavioral` and `surface` are now the cleanest answer-only baselines
  - their repeated-separator results no longer rely on probe cases copied from hidden tests
  - `counterfactual` remains the strongest ambiguity-enumeration baseline
  - `behavioral` now also generalizes to a third repeated-separator normalization family without needing hidden-test mirroring
  - the repeated-separator behavioral probes are now inferred from visible input/output shape rather than from `buggy_code` string matching
  - the new `surface` monitor matches `behavioral` on the separator-heavy stress while removing dependence on `bug_type` labels
  - `external` now fails all three separator-normalization ambiguity tasks:
    - `normalize_spaces`
    - `normalize_commas`
    - `normalize_pipes`
  - the new `count_positive` task introduces a different ambiguity family:
    - multiple changed patches pass the visible test
    - answer-only monitors still accept the wrong visible-pass repair
    - `counterfactual` detects the ambiguity and routes to verification plus revise
  - `behavioral` and `surface` therefore share the same blind spot on the expanded stress
  - `semantic` closes that blind spot using only surface-level zero-boundary probes inferred from visible I/O
  - `semantic` now matches `counterfactual` on the expanded stress set without enumerating repair candidates
  - adding `count_negative` shows this is not a one-task patch:
    - the same zero-boundary probe idea transfers from positive-count to negative-count ambiguity
    - `semantic` still matches `counterfactual` on both code-only and mixed runs
  - `counterfactual` remains the strongest ambiguity-enumeration baseline

Operational note:

- result filenames currently use second-level timestamps
- do not run multiple `experiments/capability/benchmark.py` commands in parallel when you need reliable saved files
- run benchmark commands serially

## Intended Next Use

1. keep this benchmark fixed for one cycle so routing comparisons are meaningful
2. keep the current confidence sweep and surprise-gate result as the reference routing baseline
3. keep replacing heuristic signal pieces with more principled external diagnostics
4. compare signal quality directly under a fixed gate:
   - raw confidence
   - surprise-like external signal
   - hybrid monitor
5. use `monitor_gate behavioral` / `monitor_gate surface` as the current separator-focused answer-only baselines, `monitor_gate semantic` as the current strongest surface-level baseline, and `monitor_gate counterfactual` as the ambiguity-enumeration baseline
6. use `code-14` and `mixed-28` as the current top-tier routing probes, keep `code-13` / `mixed-26` as the surface-semantic saturation predecessor checkpoint, and keep `code-10` / `mixed-20` as the first ambiguity-separation references
7. replace the oracle fallback later with a real stronger model or toolchain
8. only add more hard tasks if escalation remains too trivial after routing-policy evaluation

## Threshold-Comparator Extension

`count_gt_two` was added as a third comparator-boundary ambiguity family after the earlier zero-boundary probes:

- visible example:
  - `[3, 4, 1] -> 2`
- wrong visible-pass repair:
  - `count_nonstrict_gt_two_fix`
  - `return sum(1 for x in nums if x >= threshold)`
- correct repair:
  - `count_strict_gt_two_fix`
  - `return sum(1 for x in nums if x > threshold)`
- hidden checks:
  - `[2, 3, 1] -> 1`
  - `[2, 2, 1] -> 0`

Valid post-redesign references:

- `code-12`
  - `external`: [20260410_090406.json](F:/unified-sel/results/capability_benchmark/20260410_090406.json)
    - success `0.5`
    - mean cost `1.4416666666666667`
  - `surface`: [20260410_090419.json](F:/unified-sel/results/capability_benchmark/20260410_090419.json)
    - success `0.75`
    - mean cost `1.5666666666666667`
  - `behavioral`: [20260410_090830.json](F:/unified-sel/results/capability_benchmark/20260410_090830.json)
    - success `0.75`
    - mean cost `1.5666666666666667`
  - `semantic`: [20260410_090842.json](F:/unified-sel/results/capability_benchmark/20260410_090842.json)
    - success `1.0`
    - mean cost `1.6916666666666667`
  - `counterfactual`: [20260410_090854.json](F:/unified-sel/results/capability_benchmark/20260410_090854.json)
    - success `1.0`
    - mean cost `1.6916666666666667`
- `mixed-24`
  - `external`: [20260410_090914.json](F:/unified-sel/results/capability_benchmark/20260410_090914.json)
    - success `0.75`
    - code-family success `0.5`
    - mean cost `1.2208333333333334`
  - `behavioral`: [20260410_090929.json](F:/unified-sel/results/capability_benchmark/20260410_090929.json)
    - success `0.875`
    - code-family success `0.75`
    - mean cost `1.2833333333333334`
  - `surface`: [20260410_090943.json](F:/unified-sel/results/capability_benchmark/20260410_090943.json)
    - success `0.875`
    - code-family success `0.75`
    - mean cost `1.2833333333333334`
  - `semantic`: [20260410_090956.json](F:/unified-sel/results/capability_benchmark/20260410_090956.json)
    - success `1.0`
    - code-family success `1.0`
    - mean cost `1.3458333333333332`
  - `counterfactual`: [20260410_091018.json](F:/unified-sel/results/capability_benchmark/20260410_091018.json)
    - success `1.0`
    - code-family success `1.0`
    - mean cost `1.3458333333333332`

Interpretation:

- `count_gt_two` is the same ambiguity shape as `count_positive` and `count_negative`, but it removes the zero-boundary shortcut and forces the monitor to reason about a non-zero comparator threshold
- `behavioral` and `surface` fail all three comparator-boundary families:
  - `count_positive`
  - `count_negative`
  - `count_gt_two`
- `semantic` still matches `counterfactual` without enumerating repair candidates
- the new family increases separation relative to `code-11` / `mixed-22`:
  - `external` drops from `0.5455` to `0.5` on the code-only run
  - `behavioral` and `surface` drop from `0.8182` to `0.75` on the code-only run
  - `behavioral` and `surface` drop from `0.9091` to `0.875` on the mixed run
  - `semantic` and `counterfactual` remain at `1.0`
- this is the cleanest current stress set for judging whether a routing signal can distinguish answer-only visible-pass ambiguity from genuine low-risk local success

Current benchmark recommendation:

- use `monitor_gate behavioral` and `monitor_gate surface` as the answer-only baselines
- use `monitor_gate semantic` as the strongest current surface-level baseline
- use `monitor_gate counterfactual` as the ambiguity-enumeration reference
- use `code-13` and `mixed-26` as the current canonical comparator-and-parity probes
- keep `code-12` / `mixed-24` as the comparator-boundary predecessor checkpoint
- keep `code-11` / `mixed-22` as the earlier comparator-generalization checkpoint
- keep `code-10` / `mixed-20` as the first ambiguity-separation checkpoint

## Parity Extension

`count_even` was added as a fourth visible-pass ambiguity family after the comparator-boundary set:

- visible example:
  - `[2, 3, 4, 5] -> 2`
- wrong visible-pass repair:
  - `count_odd_fix`
  - `return sum(1 for x in nums if x % 2 != 0)`
- correct repair:
  - `count_even_fix`
  - `return sum(1 for x in nums if x % 2 == 0)`
- hidden checks:
  - `[0, 1, -1] -> 1`
  - `[2, 4, 6] -> 3`

Valid parity-extension references:

- `code-13`
  - `external`: [20260410_092833.json](F:/unified-sel/results/capability_benchmark/20260410_092833.json)
    - success `0.46153846153846156`
    - mean cost `1.4384615384615385`
  - `behavioral`: [20260410_092851.json](F:/unified-sel/results/capability_benchmark/20260410_092851.json)
    - success `0.6923076923076923`
    - mean cost `1.5615384615384615`
  - `surface`: [20260410_092911.json](F:/unified-sel/results/capability_benchmark/20260410_092911.json)
    - success `0.6923076923076923`
    - mean cost `1.5615384615384615`
  - `semantic`: [20260410_092941.json](F:/unified-sel/results/capability_benchmark/20260410_092941.json)
    - success `1.0`
    - mean cost `1.6846153846153846`
  - `counterfactual`: [20260410_093024.json](F:/unified-sel/results/capability_benchmark/20260410_093024.json)
    - success `1.0`
    - mean cost `1.6846153846153846`
- `mixed-26`
  - `external`: [20260410_093039.json](F:/unified-sel/results/capability_benchmark/20260410_093039.json)
    - success `0.7307692307692307`
    - code-family success `0.46153846153846156`
    - mean cost `1.2153846153846153`
  - `behavioral`: [20260410_093054.json](F:/unified-sel/results/capability_benchmark/20260410_093054.json)
    - success `0.8461538461538461`
    - code-family success `0.6923076923076923`
    - mean cost `1.273076923076923`
  - `surface`: [20260410_093149.json](F:/unified-sel/results/capability_benchmark/20260410_093149.json)
    - success `0.8461538461538461`
    - code-family success `0.6923076923076923`
    - mean cost `1.273076923076923`
  - `semantic`: [20260410_093208.json](F:/unified-sel/results/capability_benchmark/20260410_093208.json)
    - success `1.0`
    - code-family success `1.0`
    - mean cost `1.3346153846153845`
  - `counterfactual`: [20260410_093229.json](F:/unified-sel/results/capability_benchmark/20260410_093229.json)
    - success `1.0`
    - code-family success `1.0`
    - mean cost `1.3346153846153845`

Parity-extension interpretation:

- `count_even` is not a threshold-boundary task:
  - the ambiguity is parity semantics, not comparator strictness
  - visible I/O still leaves two plausible changed repairs
- `behavioral` and `surface` fail again:
  - they accept the visible-pass `count_odd_fix`
  - they still do not represent parity ambiguity at the surface level
- `semantic` now closes both families:
  - comparator ambiguity
  - parity ambiguity
- `counterfactual` remains tied with `semantic`, so ambiguity enumeration is still the strongest general non-privileged baseline
- relative to `code-12` / `mixed-24`, the benchmark becomes strictly sharper:
  - `external` drops from `0.5` to `0.4615` on code-only
  - `behavioral` and `surface` drop from `0.75` to `0.6923` on code-only
  - `behavioral` and `surface` drop from `0.875` to `0.8462` on mixed
  - `semantic` and `counterfactual` remain at `1.0`

Updated benchmark recommendation:

- use `monitor_gate behavioral` and `monitor_gate surface` as the answer-only baselines
- use `monitor_gate semantic` as the strongest current surface-level baseline
- use `monitor_gate counterfactual` as the ambiguity-enumeration reference
- use `code-13` and `mixed-26` as the current canonical monitor-comparison probes
- keep `code-12` / `mixed-24` as the comparator-boundary predecessor checkpoint
- keep `code-10` / `mixed-20` as the first ambiguity-separation checkpoint

## Zero-Role Extension

`count_nonzero` was added as a fifth visible-pass ambiguity family to probe the current limit of `semantic`:

- visible example:
  - `[1, 0, -2] -> 2`
- wrong visible-pass repair:
  - `count_nonnegative_zero_fix`
  - `return sum(1 for x in nums if x >= 0)`
- correct repair:
  - `count_nonzero_fix`
  - `return sum(1 for x in nums if x != 0)`
- hidden checks:
  - `[0, -1, 0] -> 1`
  - `[-2, -3, 0] -> 2`

Valid zero-role references:

- `code-14`
  - `external`: [20260410_095359.json](F:/unified-sel/results/capability_benchmark/20260410_095359.json)
    - success `0.42857142857142855`
    - mean cost `1.3785714285714286`
  - `behavioral`: [20260410_095400.json](F:/unified-sel/results/capability_benchmark/20260410_095400.json)
    - success `0.6428571428571429`
    - mean cost `1.4857142857142858`
  - `surface`: [20260410_095402.json](F:/unified-sel/results/capability_benchmark/20260410_095402.json)
    - success `0.6428571428571429`
    - mean cost `1.4857142857142858`
  - `semantic`: [20260410_095403.json](F:/unified-sel/results/capability_benchmark/20260410_095403.json)
    - success `0.9285714285714286`
    - mean cost `1.6285714285714286`
  - `counterfactual`: [20260410_095404.json](F:/unified-sel/results/capability_benchmark/20260410_095404.json)
    - success `1.0`
    - mean cost `1.6642857142857144`
- `mixed-28`
  - `external`: [20260410_095424.json](F:/unified-sel/results/capability_benchmark/20260410_095424.json)
    - success `0.7142857142857143`
    - code-family success `0.42857142857142855`
    - mean cost `1.1892857142857143`
  - `behavioral`: [20260410_095426.json](F:/unified-sel/results/capability_benchmark/20260410_095426.json)
    - success `0.8214285714285714`
    - code-family success `0.6428571428571429`
    - mean cost `1.2428571428571427`
  - `surface`: [20260410_095427.json](F:/unified-sel/results/capability_benchmark/20260410_095427.json)
    - success `0.8214285714285714`
    - code-family success `0.6428571428571429`
    - mean cost `1.2428571428571427`
  - `semantic`: [20260410_095429.json](F:/unified-sel/results/capability_benchmark/20260410_095429.json)
    - success `0.9642857142857143`
    - code-family success `0.9285714285714286`
    - mean cost `1.3142857142857143`
  - `counterfactual`: [20260410_095430.json](F:/unified-sel/results/capability_benchmark/20260410_095430.json)
    - success `1.0`
    - code-family success `1.0`
    - mean cost `1.332142857142857`

Zero-role interpretation:

- `count_nonzero` is the first family that cleanly separates `semantic` from `counterfactual`
- the failure mechanism is specific:
  - `semantic` has comparator and parity ambiguity handling
  - it does not yet represent the semantic role of zero itself
  - it therefore accepts `count_nonnegative_zero_fix` as a low-risk visible-pass repair
- this creates a sharper hierarchy:
  - `behavioral` and `surface` remain answer-only baselines
  - `semantic` is now the strongest current surface-level monitor, but no longer saturated
  - `counterfactual` remains the strongest ambiguity-enumeration baseline
- relative to `code-13` / `mixed-26`, the new probe is strictly more informative:
  - `semantic` drops from `1.0` to `0.9286` on code-only
  - `semantic` drops from `1.0` to `0.9643` on mixed
  - `counterfactual` stays at `1.0`
  - weaker monitors all degrade further by one additional failure

Zero-role closure:

- targeted zero-role ambiguity support was then added to `semantic`
- updated semantic references:
  - `code-14`: [20260410_100749.json](F:/unified-sel/results/capability_benchmark/20260410_100749.json)
    - success `1.0`
    - mean cost `1.6642857142857144`
  - `mixed-28`: [20260410_100818.json](F:/unified-sel/results/capability_benchmark/20260410_100818.json)
    - success `1.0`
    - code-family success `1.0`
    - mean cost `1.332142857142857`
- updated interpretation:
  - the zero-role gap was real, but it was not fundamental
  - a targeted surface-semantic extension closes it without using repair candidate enumeration
  - `semantic` is therefore again tied with `counterfactual` on the strongest current probe

Updated benchmark recommendation:

- use `monitor_gate behavioral` and `monitor_gate surface` as the answer-only baselines
- use `monitor_gate semantic` as the strongest current surface-level baseline
- use `monitor_gate counterfactual` as the ambiguity-enumeration reference
- use `code-14` and `mixed-28` as the current top-tier monitor-comparison probes
- keep `code-13` / `mixed-26` as the predecessor saturation checkpoint for the current `semantic` monitor
- keep `code-12` / `mixed-24` as the comparator-boundary predecessor checkpoint
