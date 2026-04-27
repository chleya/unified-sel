# Contributing to Granularity-Aligned Metacognition for LLMs

Thank you for your interest in this research codebase. This document provides guidelines for contributing.

## Project Scope

This is a **research codebase**, not a production service. Contributions should align with the current research direction:

- **Paper line**: Boundary-local amplification (inverted-U feedback benefit pattern)
- **Tool line**: Capability Router benchmark for accept/verify/escalate routing decisions
- **Methodology line**: The Granularity Alignment Principle

## How to Contribute

### Reporting Issues

When reporting bugs or issues, please include:

1. **Environment**: Python version, OS, numpy version
2. **Reproduction steps**: Minimal command or script to reproduce
3. **Expected vs actual behavior**
4. **Smoke test result**: Run `python tests/smoke_test.py` and attach output

### Pull Request Guidelines

1. **One change at a time** — Keep PRs focused and small
2. **Run smoke tests** — All PRs must pass `python tests/smoke_test.py`
3. **Update documentation** — If you change behavior, update README.md or relevant docs
4. **No hidden-test leakage** — Public benchmark files (`.public.jsonl`) must not contain `hidden_tests`, `fixed_code`, or `expected_route`
5. **No oracle overclaim** — If your change affects escalation paths, label oracle assumptions explicitly

### Code Style

- Follow PEP 8
- Use type hints for new functions
- Add docstrings for public APIs
- Keep archived modules importable (see `tests/smoke_test.py` for compatibility checks)

## Red-Line Rules (Non-Negotiable)

1. **No oracle overclaim**: Escalation path success rates assume oracle. Do not claim 100% escalation success as real.
2. **No simulated cost as real cost**: `cost_units` are abstract. Label all cost conclusions "based on assumed cost model".
3. **No hidden-test leakage**: Public benchmark files must exclude hidden tests and answers.
4. **No self-awareness validated claim**: The self-aware LLM direction is a future narrative, not validated.
5. **No new synthetic solver experiments without preflight**: Check EXPERIMENT_LOG.md and STATUS.md before running new synthetic experiments.

## Development Setup

```bash
pip install -r requirements.txt
python tests/smoke_test.py
```

## Questions?

Open an issue or refer to:
- [README.md](README.md) — Project overview
- [STATUS.md](STATUS.md) — Current progress and truth table
- [AGENTS.md](AGENTS.md) — Project direction and red-line rules
