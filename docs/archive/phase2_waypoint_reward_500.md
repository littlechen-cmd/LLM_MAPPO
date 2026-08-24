# Phase 2 Waypoint Reward Feasibility Comparison

## Scope

This is a local CPU feasibility experiment, not a Phase 2 Go/No-Go evaluation.
Both runs used `py310`, seed `7`, the medium dynamic warehouse, one AGV, one
priority-B task, a 400-step episode limit, and identical PPO settings. Only the
reward for a waypoint-distance reduction differed. Each run trained for 500
episodes from a fresh initialization.

| Waypoint reward | Output directory | Wall time | Environment steps |
| --- | --- | ---: | ---: |
| `0.01` | `artifacts/phase2_waypoint_001/seed_007` | 587.8 s | 95,419 |
| `0.05` | `artifacts/phase2_waypoint_005/seed_007` | 856.6 s | 144,500 |

## Results

| Metric | `0.01` | `0.05` |
| --- | ---: | ---: |
| Completed tasks | 0 | 0 |
| Successful episodes | 0 / 500 | 0 / 500 |
| Mean task completion rate | 0.000 | 0.000 |
| Mean episode reward | -1.524 | -19.045 |
| Mean collisions | 0.118 | 10.764 |
| Deadlock rate | 62.8% | 43.0% |
| Final value loss | 0.007 | 33.918 |
| Mean value loss, final 10 updates | 0.015 | 8.378 |

## Decision

Neither setting is feasible: reducing the waypoint reward alone did not produce
a single completed task. The `0.01` setting is less unstable and has far fewer
collisions, but it still fails the basic one-AGV task-completion criterion.
Do not continue this policy to 3 AGVs, 5,000 episodes, ten seeds, or the 4080S.

The next local experiment must address task-level progress and valid
`TOGGLE_LOAD` exploration before changing training scale. The unexpectedly high
one-AGV collision count in the `0.05` run should also be diagnosed before any
further reward comparison.
