# Capability Mainline Conclusion and Next Validation Plan

**Date**: 2026-04-11
**Status**: Active mainline note
**Scope**: capability routing benchmark

---

## 1. Mainline conclusion

As of the 2026-04-10 vowel closure round, the capability-routing mainline has reached the following result:

> `semantic` now matches `counterfactual` on the current benchmark without repair candidate enumeration, using only surface-level ambiguity handling plus visible-pass lexical/structural probes.

This is no longer just evidence that `semantic` is better than `surface` or `behavioral`.
It is evidence for a stronger claim:

> on the current benchmark, a non-enumerative surface-semantic routing monitor can saturate the same top-tier probe that previously required ambiguity enumeration.

---

## 2. Current fixed benchmark point

The current fixed top-tier probe is:

- `code-20`
- `mixed-40`

Current main policy baseline:

- `monitor_repair_triage semantic`

Current saturation references:

- `monitor_repair_triage counterfactual`
- `monitor_repair_triage diagnostic`

Latest closure round:

- family closed: lexical vowel ambiguity
- visible-pass wrong repair: `count_words_starting_with_vowel_fix`
- intended repair: `count_words_with_vowel_fix`

Key semantic references:

- `code-20`
  - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_171643.json`
  - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_171603.json`
- `mixed-40`
  - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_171708.json`
  - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_171559.json`

Observed semantic summary:

- `code-20`
  - `monitor_gate semantic`: success `1.0`, mean cost `1.615`
  - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.59`
- `mixed-40`
  - `monitor_gate semantic`: success `1.0`, mean cost `1.3075`
  - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.295`

Practical interpretation:

- `semantic` is now the strongest current surface-level routing monitor.
- `monitor_repair_triage semantic` remains the simplest current mainline policy baseline.
- `counterfactual` remains the ambiguity-enumeration reference, not the only top-tier-successful monitor.

---

## 3. Boundary of the claim

This result does **not** mean routing is solved in general.

It means something narrower and more defensible:

- on the current synthetic-but-hardened benchmark,
- across the currently exposed eight ambiguity families,
- `semantic` can now absorb the relevant ambiguity signal without candidate enumeration.

Therefore the correct boundary statement is:

> the current benchmark no longer demonstrates a necessary gap between surface-semantic routing and ambiguity-enumeration routing.

What is still unproven:

- whether `semantic` generalizes to genuinely new ambiguity regimes
- whether it has learned a reusable routing principle rather than an accumulated family inventory
- whether the same result survives lexical restatement and probe paraphrase

---

## 4. Mainline decision

The benchmark should now be frozen for one cycle.

Immediate policy:

- do **not** add a ninth ambiguity family by default
- do **not** keep chasing closure unless a new family creates a genuinely new semantic regime
- treat `code-20` / `mixed-40` as the fixed mainline probe while running validation experiments

Reason:

Once `semantic` matches `counterfactual` at this point, the highest-value next step is no longer another closure. The highest-value next step is to test whether the result is robust and non-fragile.

---

## 5. Next validation plan

### P0. Held-out family validation

Goal:

- test whether `semantic` generalizes beyond the exact currently closed family inventory

Design:

- build leave-one-family-out evaluation slices
- for each slice, remove one ambiguity family from the semantic design target
- measure whether the monitor still routes correctly on neighboring but non-identical tasks

Decision value:

- distinguishes reusable routing structure from benchmark-specific closure accumulation

### P0. Lexical / probe paraphrase validation

Goal:

- test whether `semantic` tracks ambiguity structure rather than literal probe phrasing

Design:

- rewrite visible tests and task wording while preserving the same intended ambiguity
- vary names, token order, lexical surface forms, and example choices
- keep hidden target semantics unchanged

Decision value:

- checks whether current success is structural or probe-template dependent

### P1. Policy invariance check

Goal:

- verify that the mainline conclusion is not an artifact of one policy only

Design:

- keep the benchmark fixed
- compare `semantic`, `counterfactual`, and `diagnostic` under the same policy families already used in the benchmark
- confirm whether equality remains at both success and cost level

Decision value:

- separates signal-family claims from policy-layer claims

### P1. Ninth family gate

A new ambiguity family should be designed only if all of the following hold:

- it creates a real failure for frozen `semantic`
- it does not collapse into any current closed family
- it remains a plausible ambiguity family under visible-pass evidence
- it is not merely a lexical restatement of an already closed regime

---

## 6. Current paper-grade wording

Recommended wording for the main result:

> On the hardened capability-routing benchmark, the `semantic` monitor matches the `counterfactual` reference up through `code-20` / `mixed-40` while avoiding repair candidate enumeration. The result shows that, on the current probe set, surface-semantic ambiguity handling is sufficient to recover the same routing decisions previously achieved by ambiguity enumeration.

Recommended boundary wording:

> This is not evidence that routing is solved in general. It is evidence that the current benchmark no longer forces a gap between non-enumerative surface-semantic routing and enumeration-based routing.

---

## 7. Operational next step

The next mainline execution step is:

1. freeze `code-20` / `mixed-40`
2. write the main result into paper-facing materials
3. design a held-out / paraphrase validation round
4. only after that, decide whether a ninth family is necessary
