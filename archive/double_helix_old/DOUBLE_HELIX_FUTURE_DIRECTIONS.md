# Double-Helix Future Directions

**Status**: Long-term extensions, NOT current priority
**Position**: Archive for ideas that exceed current evidence level

---

These directions are preserved for future reference but should NOT drive current work.
Current work is defined in DOUBLE_HELIX_MECHANISM_NOTE_2026-04-13.md.

---

## 1. Closed-Loop Embodied Environment

The original deep-research-report proposed Gymnasium-based "mortal agent" environments.
This is interesting but requires:
- Custom environment design (energy decay + task completion)
- Real-time decision loop (not batch inference)
- Embodied state representation

**Why deferred**: Current evidence doesn't justify this investment. The mechanism must first prove itself in the simpler code-repair setting.

## 2. ROS2 / Robotics Integration

Using real robot state as the "environment" for the maintain chain.
The maintain chain would monitor sensor health, battery, and task progress.

**Why deferred**: Hardware dependency, high engineering cost, no current evidence that the mechanism transfers to continuous control.

## 3. Long-Term Self-Sustaining Agent

An agent that maintains its own memory, health, and knowledge over extended time periods.
The maintain chain becomes a "metabolic" process that runs continuously.

**Why deferred**: This is the ultimate vision but requires solving many sub-problems first. The current mechanism note addresses only the first sub-problem: "does feedback correction work near the capability boundary?"

## 4. Dual-Chain Cognitive Architecture

Two parallel processing streams (planning + maintaining) with shared memory and arbitration.
Inspired by dual-process theory in cognitive science.

**Why deferred**: The "dual-chain" framing overclaims current evidence. What we have is a correction loop, not a second cognitive stream. If the mechanism proves valuable, this framing can be revisited.

## 5. TopoMem ECU as Maintain-Chain Signal Source

Using H1/H2 health signals from TopoMem to trigger maintain-chain interventions.
When embedding-space geometry degrades, the maintain chain activates.

**Why deferred**: TopoMem ECU signals are sample-level geometric quantities. The maintain chain operates at the attempt level. The granularity mismatch needs resolution before integration.

---

## Revisit Criteria

Any of these directions should be reactivated only when:
1. The core mechanism (feedback correction near boundary) is validated in the 4-group experiment
2. The mechanism is integrated into the hybrid router on A-track
3. There is a specific, testable hypothesis that requires the extension
