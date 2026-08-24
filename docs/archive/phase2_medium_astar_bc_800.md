# Phase 2 Medium-Map A* Behavior-Cloning Feasibility Run

## Scope

This local curriculum extends the successful small single-AGV setup to the
medium RWARE map (`llm-mappo-medium-3ag-v1`) while overriding it to one AGV.
It is the required prerequisite for the following three-AGV avoidance and
charging-coordination stage.

## Local Environment

- Host: local Windows workstation, native Conda `py310` environment.
- Device: CPU with one PyTorch thread.
- No 4080S, Linux, WSL, CUDA, or server execution was used.

```powershell
python train/train_phase2.py --config configs/phase2_medium_astar_bc.yaml
python eval/evaluate_phase2.py artifacts/phase2_medium_astar_bc/seed_007/checkpoint_final.pt --episodes-per-seed 20
```

## Curriculum Result

The A* expert completed 100/100 validation episodes with matching pickups and
deliveries, zero collisions, and zero blocked forwards. It then collected
12,000 demonstrations across 364 completed episodes. The 30-epoch,
class-balanced behavior-cloning warm start passed its independent 20-episode
gate with completion `1.00`.

The subsequent CPU MAPPO run completed 800 episodes in about 271 seconds
(31,118 steps). The final episode completed its task in 35 steps with zero
collisions and reward `4.94`. Its checkpoint is:

```text
artifacts/phase2_medium_astar_bc/seed_007/checkpoint_final.pt
```

## Ten-Seed Evaluation

Twenty deterministic episodes were evaluated per seed (200 total):

| Metric | Result | Gate |
| --- | ---: | ---: |
| Task completion rate | 0.995 | >= 0.95 |
| Mean collisions per episode | 0.000 | <= 2.00 |
| Deadlock rate | 0.005 | <= 0.05 |
| Success-rate standard deviation | 0.015 | <= 0.10 |

All medium single-AGV Go/No-Go gates passed. One seed had one deadlocked
episode; the aggregate deadlock rate remains below the configured gate. The
next local experiment is the medium three-AGV A* safety and coordination gate.
