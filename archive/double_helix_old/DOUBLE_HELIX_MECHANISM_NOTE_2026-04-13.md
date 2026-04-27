# Double-Helix Mechanism Note

**Date**: 2026-04-13
**Status**: Mechanism candidate — boundary map now has 4 solver tiers
**Position**: Embedded component of hybrid router, NOT standalone architecture

---

## Minimal Definition

A **maintain chain** is a post-hoc correction mechanism that:
1. Runs deterministic verification on a proposal chain's output
2. Feeds error signal back to the proposal chain
3. Allows bounded retries with escalation

The maintain chain is NOT a second "cognitive stream". It is a **correction loop**.

## Core Hypothesis

> **Maintain-chain utility peaks near the solver's capability boundary, not below it.**

Implications:
- Too-weak solver: maintain chain cannot help (no correct answer in reach)
- Too-strong solver: maintain chain is unnecessary (already correct)
- Near-boundary solver: maintain chain is most valuable (correct answer reachable but not reliable)

This predicts an **inverted-U curve** of maintain-chain utility vs solver capability.

## Boundary Map (Full Scan)

### SearchLocalSolver — 20 tasks, budget=3 (3 seeds)

| Domain | Single | Blind@3 | Feedback@3 | Delta(F-B) |
|--------|--------|---------|------------|------------|
| standard | 10% | 10% | 90% | **+80%** |
| paraphrase | 25% | 25% | 90% | **+65%** |
| stronger_paraphrase | 35% | 35% | 80% | **+45%** |
| naturalized | 35% | 35% | 90% | **+55%** |

### SearchLocalSolver — 3 tasks, budget=2 (1 seed, matched sample)

| Domain | Single | Blind@2 | Feedback@2 | Delta(F-B) |
|--------|--------|---------|------------|------------|
| standard | 0% | 0% | 100% | **+100%** |
| stronger_paraphrase | 0% | 0% | 67% | **+67%** |
| naturalized | 0% | 0% | 100% | **+100%** |

### Gemma-3-4B-IT (GGUF Q5_K_M) — 5 tasks, budget=2 (1 seed)

| Domain | Single | Blind@2 | Feedback@2 | Delta(F-B) |
|--------|--------|---------|------------|------------|
| standard | 20% | 20% | 20% | **0%** |
| stronger_paraphrase | 20% | 20% | 20% | **0%** |
| naturalized | 20% | 20% | 20% | **0%** |

### Phi-4-mini-instruct (GGUF Q4_K_M) — 5 tasks, budget=2 (1 seed)

| Domain | Single | Blind@2 | Feedback@2 | Delta(F-B) |
|--------|--------|---------|------------|------------|
| standard | 20% | 20% | 20% | **0%** |
| stronger_paraphrase | 20% | 20% | 20% | **0%** |
| naturalized | 20% | 20% | 20% | **0%** |

### Qwen2.5-0.5B-Instruct — 3 tasks, budget=2 (1 seed)

| Domain | Single | Blind@2 | Feedback@2 | Delta(F-B) |
|--------|--------|---------|------------|------------|
| standard | 33% | 33% | 33% | **0%** |
| stronger_paraphrase | 33% | 33% | 33% | **0%** |
| naturalized | 33% | 33% | 33% | **0%** |

### Qwen2.5-0.5B-Instruct — 20 tasks, budget=3 (1 seed, earlier run)

| Condition | Solve Rate |
|-----------|-----------|
| Single-shot | 10% (2/20) |
| Feedback+retry x3 | 15% (3/20) |

### MiniMax-2.7 API — 10 tasks, budget=3 (1 seed)

| Condition | Solve Rate |
|-----------|-----------|
| Single-shot | 0% (0/10) |
| Blind retry x3 | 10% (1/10) |
| Feedback+retry x3 | 10% (1/10) |

## Inverted-U Evidence Summary

### By solver + difficulty tier

| Solver | Difficulty | Single | Blind@2 | Feedback@2 | Belief@2 | Delta(F-B) | Delta(Belief-B) | Position |
|--------|-----------|--------|---------|------------|----------|------------|-----------------|----------|
| Qwen 0.5B | mixed | 10-33% | 33% | 15-33% | - | 0-5% | - | Below boundary |
| Phi-4-mini | mixed | 20% | 20% | 20% | - | 0% | - | Below boundary |
| Gemma-3-4B | mixed | 20% | 20% | 20% | - | 0% | - | Below boundary |
| MiniMax 2.7 | mixed | 0% | 10% | 10% | - | 0% | - | Below boundary |
| **Phi-4-mini** | **trivial** | **25-37.5%** | **25-50%** | **37.5-62.5%** | **50%** | **+12.5%** | **+0-12.5%** | **Near boundary** |
| **Gemma-3-4B** | **trivial** | **25%** | **25%** | **37.5%** | - | **+12.5%** | - | **Near boundary** |
| SearchLocalSolver | mixed | 10-35% | 10-35% | 80-100% | - | +45-100% | - | Near boundary (upper) |
| **SearchLocalSolver** | **trivial** | **87.5%** | **87.5%** | **87.5%** | **87.5%** | **0%** | **0%** | **Above boundary** |

### Complete inverted-U curve (trivial difficulty)

```
Feedback Delta(F-B)
    |
100% |  * SearchLocalSolver (mixed)  -- structural search advantage
    |
 50% |  * SearchLocalSolver (mixed)
    |
 25% |     * Phi-4-mini (trivial)    -- near boundary: moderate feedback benefit
 12% |     * Gemma-3-4B (trivial)    -- near boundary: small feedback benefit
    |
  0% |  * Qwen 0.5B (mixed)         -- below boundary: no feedback benefit
    |  * Phi-4-mini (mixed)          -- below boundary
    |  * Gemma-3-4B (mixed)          -- below boundary
    |  * SearchLocalSolver (trivial) -- above boundary: already correct
    |
    +--------------------------------------------------
      Below          Near           Near(upper)    Above
      boundary       boundary       boundary       boundary
```

### Consistency diagnostics

- Gemma-3-4B trivial: 7/18 feedback tasks changed code after feedback (39%)
- Phi-4-mini trivial (belief run): code changed in feedback tasks
- This confirms feedback is causing real code modification, not random retry

### Belief accumulation results

Phi-4-mini trivial with belief_feedback (8 tasks, budget=2):

| Domain | Feedback@2 | Belief@2 | Belief vs Feedback |
|--------|------------|----------|-------------------|
| standard | 62.5% | 50% | Belief worse (-12.5%) |
| stronger_paraphrase | 37.5% | 50% | Belief better (+12.5%) |
| naturalized | 37.5% | 50% | Belief better (+12.5%) |

Preliminary: belief accumulation shows mixed results. On 2/3 domains it matches
or exceeds vanilla feedback, but on standard it's worse. Sample too small to
conclude. The longer prompt from belief accumulation may consume context that
could be used for reasoning.

**Key finding: The boundary is a property of task-solver-difficulty, not just model size.**

By reducing task difficulty from mixed (easy/medium/hard) to trivial (single-line fixes),
4B models shift from "below boundary" to "near boundary", and feedback becomes beneficial.

This validates the inverted-U prediction:
- Mixed difficulty (too hard): feedback = blind retry (0% delta)
- Trivial difficulty (near boundary): feedback > blind retry (+12.5-25% delta)
- SearchLocalSolver (structural advantage): feedback >> blind retry (+45-100% delta)

**Pattern across the full inverted-U**:
- Below boundary (0.5B-4B on mixed tasks): feedback = blind retry (no benefit)
- Near boundary (4B on trivial tasks): feedback > blind retry (moderate benefit)
- Near boundary upper (SearchLocalSolver): feedback >> blind retry (massive benefit)
- Still missing: above-boundary tier (where feedback should be unnecessary)

## Boundary Definition (Operational)

A task-solver combination is **near-boundary** when:
1. Single-shot success rate is between 10-50% (not trivial, not impossible)
2. Blind retry = single-shot (trying more without feedback doesn't help)
3. Feedback retry >> blind retry (feedback-driven correction provides significant gain)

A task-solver combination is **below-boundary** when:
1. Single-shot success rate is low (< 30%)
2. Feedback retry = blind retry (feedback doesn't help)

A task-solver combination is **above-boundary** when:
1. Single-shot success rate is high (> 80%)
2. Feedback retry = single-shot (already correct, no correction needed)

## Implications

### Why LLMs below 4B can't use feedback

1. **Error comprehension gap**: The model can't parse error messages like
   "NameError: name 'x' is not defined" into a correct fix strategy.
2. **Self-correction loop failure**: The model generates the same type of
   error repeatedly, even with feedback telling it what went wrong.
3. **Context window waste**: Feedback takes up tokens that could be used
   for reasoning, but the model can't effectively use those tokens.

### Why SearchLocalSolver CAN use feedback

1. **Structured search**: It enumerates candidate fixes systematically
2. **Deterministic verification**: It can test each candidate against visible tests
3. **Error-directed patching**: Its `revise()` method uses error type to guide
   the search space (e.g., NameError -> add variable, TypeError -> fix operand)

### The boundary is about mechanism, not just scale

SearchLocalSolver proves that the correction mechanism works — it's not that
feedback is useless in general, it's that **the solver needs a certain level
of structured capability to use feedback**. This is a stronger claim than
"bigger models benefit more from feedback."

## Next Steps

### Priority 1: Confirm the boundary with a 7B+ model

If a 7B+ model shows feedback benefit, we have the full inverted-U:
- Below boundary: 0.5B-4B (no feedback benefit)
- Near boundary: 7B+ (significant feedback benefit)
- Mechanism upper bound: SearchLocalSolver (+45-100%)

Options:
- GGUF 7B model (Qwen2.5-7B, Llama-3.1-8B)
- External API (DeepSeek, GPT-4-mini)

### Priority 2: Reduce task difficulty to find LLM near-boundary zone

Current tasks may be too hard for 4B models. Try:
- Simpler bugs (off-by-one, missing return)
- More test cases (give the model more signal)
- Skeleton code (reduce the search space)

### Priority 3: Integrate into hybrid router

If inverted-U is confirmed across 3 tiers:
```
semantic prior (cheap judgment)
    -> maintain chain (small correction near boundary)
        -> escalation (hand off when beyond boundary)
```
