# Phase 2 Small-Map A* Behavior-Cloning Feasibility Run

## Scope

This is the Phase 2 single-AGV feasibility curriculum, not a multi-AGV result.
It uses `llm-mappo-small-1ag-v1` (10x20), one requested shelf, a 200-step
limit, the collision classification fix, oracle interaction masks, and
execution-time A* waypoint observations.

## Local Environment

- Host: local Windows workstation, native Conda `py310` environment.
- Device: CPU, PyTorch limited to one thread for this small MLP workload.
- No 4080S, Linux, WSL, or CUDA server was used.
- Command:

```powershell
python train/train_phase2.py --config configs/phase2_small_astar_bc.yaml
python eval/evaluate_phase2.py artifacts/phase2_small_astar_bc/seed_007/checkpoint_final.pt --episodes-per-seed 20
```

## Curriculum

Before MAPPO, the deterministic A* controller completed 100 local episodes:
`task_completion_rate=1.0`, matched pickups and deliveries, zero collisions,
and zero blocked forwards. It then supplied 12,000 masked, smooth action
preference labels. The shared actor was behavior-cloned for 30 epochs with
class-balanced samples. Its independent 20-episode warm-start gate was
`task_completion_rate=1.0`; the configured 0.80 minimum therefore passed.

## Result

The CPU run completed 800 MAPPO episodes in about 72 seconds (27,026 steps).
The final episode completed its one task in 23 steps with zero collisions,
zero blocked forwards, and reward `4.95`. The final checkpoint is:

```text
artifacts/phase2_small_astar_bc/seed_007/checkpoint_final.pt
```

The final ten-seed evaluation used 20 episodes per seed (200 episodes total):

| Metric | Result | Gate |
| --- | ---: | ---: |
| Task completion rate | 1.00 | >= 0.95 |
| Mean collisions per episode | 0.00 | <= 2.00 |
| Deadlock rate | 0.00 | <= 0.05 |
| Success-rate standard deviation | 0.00 | <= 0.10 |

All small single-AGV gates passed. The next experiment must add difficulty
incrementally (medium map before multi-AGV coordination); do not move to the
4080S until that local feasibility gate also passes.

## Visualization

Record a deterministic, non-flashing GIF directly from the saved checkpoint:

```powershell
python visualize.py artifacts/phase2_small_astar_bc/seed_007/checkpoint_final.pt --fps 5
```

This opens one stable real-time software-rendered window. Add `--record-gif`
to save a replay, or `--no-live --record-gif` for a headless recording. Use
`--seeds 3 7 11` to replay several seeds sequentially in the same window.
