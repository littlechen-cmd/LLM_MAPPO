# Phase 2 Medium Three-AGV Expert Baseline

## Status

The medium single-AGV warm-start gate passed, so the project has entered the
three-AGV avoidance and charging-coordination stage. MAPPO training has not
started for this stage because the expert-demonstration gate is not yet met.

## Local Baseline

The test used native Windows, local CPU, `py310`,
`llm-mappo-medium-3ag-v1`, three AGVs, 400 maximum steps, and 100 seeds:

```powershell
python train/train_phase2.py --config configs/phase2_medium_3ag_astar_bc.yaml
```

The initial independent A* actions completed 94.0% of requested tasks but
produced 1.74 true collisions per episode. A conservative one-step resolver
was added for head-on moves, competing target cells, and moving into an AGV
that will not vacate its cell. It eliminated true collisions and blocked
forwards, but completed 97.3% of requested tasks; some task chains still
stall before all three deliveries finish.

## Gate

Do not collect demonstrations or start the 800-episode MAPPO run until the
expert reaches 100% task completion, pickup/delivery equality, zero true
collisions, and zero blocked forwards across the 100-episode local check.
The next implementation task is reservation-aware multi-agent A* with
time-expanded cell reservations and an explicit yielding/replanning policy.
