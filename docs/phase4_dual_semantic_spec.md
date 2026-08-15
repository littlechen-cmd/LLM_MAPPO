# Phase 4 Dual-Semantic Contract

Phase 4 uses two independent offline LLM preferences. Both are bounded soft
targets in `[0, 1]`; neither may assign an AGV, prescribe a per-step action,
emit an A* path, or control a charging station.

## Frozen semantics

- `task_commitment`: willingness to continue the original transport task.
  A low value means transport progress should pause or be abandoned for safety,
  maintenance, charging, or yielding. It never means willingness to reach the
  current charging target.
- `local_assertiveness`: willingness to claim immediate passage in the local
  coordination context. A low value means yield or wait; a high value means
  proceed when the rule layer and action mask permit it.

The decision precedence is:

1. hard rule-layer safety;
2. battery safety;
3. loaded-versus-empty local yielding;
4. task priority, where `A > B > C`;
5. ordinary transport progress.

## Reference cases

| Scenario | Task commitment | Local assertiveness |
| --- | ---: | ---: |
| Low battery, diverting to charge, clear route | 0.1 | 0.8 |
| Empty AGV yielding to a loaded AGV in a narrow corridor | 0.6 | 0.2 |
| High-priority AGV at an intersection | 0.9 | 0.85 |
| Low-priority AGV facing a high-priority AGV | 0.5 | 0.2 |
| Charging-station exit congestion | context dependent | 0.2-0.4 |
| Normal unobstructed transport | 0.6-0.8 | 0.7-0.9 |

## Strict JSON response

```json
{
  "task_commitment": 0.1,
  "task_reason": "Battery safety overrides transport progress.",
  "local_assertiveness": 0.8,
  "coordination_reason": "The charging route is unobstructed."
}
```

## Collection gates

1. Collect five labels per scenario type and review all 25 records.
2. Reject any schema violation, forbidden control instruction, reversed
   priority ordering, or contradiction between a score and its reason.
3. Require the controlled direction checks:
   `low_battery.task_commitment <= 0.3`,
   `priority_conflict.local_assertiveness >= 0.7`,
   `narrow_corridor.local_assertiveness <= 0.3`, and
   `station_exit.local_assertiveness <= 0.4`.
4. Only after the pilot passes, collect the frozen 400-record split.
5. Audit at least 10% of the formal dataset before training.

The prior single-engagement dataset is retained as a rejected `v1` artifact and
must not be loaded by the dual-semantic trainer.
