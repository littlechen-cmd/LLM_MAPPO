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
forwards, but completed 97.3% of requested tasks.

The current controller replaces that one-step repair with rolling-horizon
time-expanded A*. Higher-priority AGVs reserve cells and directed edges for a
16-step window; lower-priority AGVs plan around those reservations, can wait,
and are replanned every environment step. It completed 99.0% of requested
tasks across 100 local episodes with zero true collisions and zero blocked
forwards. Three seeds (50, 85, and 96) still ended at the 400-step limit with
one loaded, assigned AGV unable to finish delivery.

## Gate

Do not collect demonstrations or start the 800-episode MAPPO run until the
expert reaches 100% task completion, pickup/delivery equality, zero true
collisions, and zero blocked forwards across the 100-episode local check.
The next implementation task is a loaded-AGV recovery policy that detects a
repeated reserved-path prefix, temporarily parks lower-priority idle AGVs,
then replans the loaded AGV with released reservations.

## MAPPO Trial Outcome

After the 100-episode expert check exceeded the configured `0.95` curriculum
gate, a local CPU 800-episode MAPPO run was started using
`configs/phase2_medium_3ag_astar_bc.yaml`. Successful expert trajectories were
used for behavior cloning; failed expert episodes were excluded from the BC
dataset. The BC-only validation reached `0.9833` completion with zero
collisions, but the final PPO checkpoint regressed:

| Metric | Result | Gate |
| --- | ---: | ---: |
| Task completion rate | 0.820 | >= 0.95 |
| Mean collisions per episode | 3.940 | <= 2.00 |
| Deadlock rate | 0.385 | <= 0.05 |
| Success-rate standard deviation | 0.078 | <= 0.10 |

The PPO result is therefore **No-Go**. The checkpoint is retained for
diagnostics at `artifacts/phase2_medium_3ag_astar_bc/seed_007/checkpoint_final.pt`;
do not use it as a 3-AGV baseline. The next correction must preserve the
reservation-aware coordination during RL, for example with a decaying
reservation-policy KL term or a protected coordination head, before another
800-episode trial.

## Failed-Seed Visualization

Generate GIFs and per-step traces for the three known failures:

```powershell
python visualize.py --controller expert --seeds 50 85 96 --record-gif --trace
```

Outputs are written to:

```text
artifacts/visualizations/
```

To visualize a selected seed or change the frame rate:

```powershell
python visualize.py --controller expert --seeds 50 --fps 8 --record-gif --trace --output-dir artifacts/failed_seed_050
```
