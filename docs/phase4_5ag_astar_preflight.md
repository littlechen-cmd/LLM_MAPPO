# Phase 4 5-AGV A* Preflight

The Phase 4 environment is `medium / 5 AGV` with a two-cell outer highway
ring, eight corner charging stations, dynamic batches every 40 steps, batch
sizes `[4, 8]`, and a completion target of 50 deliveries. It is intentionally a
new scale track rather than a direct comparison with the Phase 3b 3-AGV result.

## Result

Command:

```powershell
& 'D:\Anaconda3\envs\py310\python.exe' eval\evaluate_dynamic_ingress_astar.py `
  --config configs\phase4_llm_distillation.yaml `
  --seeds 0 1 2 `
  --episodes-per-seed 2 `
  --output artifacts\phase4_5ag_astar_gate_smoke_3x2.json
```

The six local episodes achieved a completion rate of `1.0`, zero collisions,
and zero terminating deadlocks. Each episode completed the target before the
1,000-step limit. At reset, the A and B batches each contained at least four
tasks; eight charging stations remained available for five AGVs.

## Diagnostic Risk

The expert recorded 344-362 path-livelock events and 1-7 state-repeat events
per seed. These events were recovered by the current replanning and yielding
logic, so they did not trigger the terminal deadlock metric. They must still be
reported alongside Phase 4 training and visual replay, especially at corridor
exits, picking stations, and charging-station exits.

## Decision

This preflight permits offline DeepSeek label collection. It does not validate
the learned Phase 4 policy and does not remove the requirement for a later
matched 5-AGV rule-engagement plus A* KL baseline.
