# WebVM Reuse Assessment - 2026-04-21

Source inspected:

- `F:\workspace-ideas\webvm-main`

## Verdict

WebVM has indirect reference value, but it should not enter the current meta-controller experiment as a dependency.

It is not useful for the V0/V1 causal core:

- no learnable meta-controller
- no memory arbitration logic
- no habit/planner dominance policy
- no long-horizon drift benchmark
- no direct Python experiment substrate

It is useful as a future interface/sandbox reference:

- browser-contained Linux-like execution environment
- terminal plus graphical display abstraction
- tool-use loop around shell/screenshot/mouse/keyboard actions
- telemetry surfaces such as CPU, disk activity, latency
- config-driven disk image and command boot path
- service-worker/cross-origin-isolation engineering for heavy browser runtime

## Most Relevant Ideas

### 1. Agent-Observable Runtime Surface

`src/lib/WebVM.svelte` exposes:

- terminal I/O
- graphical display sizing
- CPU activity
- disk activity
- disk latency
- process-count callback

This is relevant if the meta-controller later needs a richer external environment where control actions have visible cost and delayed effects.

### 2. Tool-Use Harness

`src/lib/anthropic.js` implements an AI loop over:

- shell command tool
- screenshot tool
- mouse/keyboard actions
- stateful message history

The exact provider integration is not reusable for this project, but the shape is useful:

> controller chooses action -> environment executes -> observation/tool result returns -> next control action

This resembles the outer loop we may eventually need for interactive agent benchmarks.

### 3. Cost And Activity Telemetry

The CPU/disk activity tracking is a practical reminder that future experiments can expose real control costs:

- process count
- CPU active time
- disk latency
- network availability
- tool latency

These could become `control_cost_estimate` inputs in a browser-agent benchmark.

### 4. Configurable Environment Images

The config files define:

- disk image URL/type
- command and args
- environment variables
- working directory
- display needs

This is a useful pattern for future benchmark environments:

```text
task profile -> boot config -> observation/action surface -> metrics
```

## Risks

- CheerpX/WebVM licensing is not suitable for casual vendoring.
- The project depends on heavy browser/WebAssembly infrastructure.
- It is front-end/runtime oriented, not a controlled research benchmark.
- Browser networking and disk images add reproducibility risk.
- Directly integrating it would distract from the current causal question.
- The included AI integration uses browser-side API keys and provider-specific computer-use tooling; it should not be copied into the current experiment.

## Recommendation

Do not add WebVM to the current F-drive integration shortlist.

Keep it as a future branch:

- `B8 Browser Agent Sandbox`

Possible future use:

1. after the Python synthetic benchmarks stabilize
2. define a small WebVM task suite with shell-level objectives
3. expose observations as terminal/screenshot/tool telemetry
4. let the meta-controller choose among:
   - cheap shell habit
   - planner/tool reasoning
   - memory read/write
   - recovery mode
5. measure cost, recovery, drift, and intervention rates

For now, WebVM should remain reference-only.

