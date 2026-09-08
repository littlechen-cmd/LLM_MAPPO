# LLM-MAPPO Dynamic Warehouse

This repository combines the RWARE Gymnasium environment with the staged
LLM-MAPPO research implementation. The current architecture and experiment
contract are documented in `specs/mission.md`, `specs/tech-stack.md`,
`specs/roadmap.md`, and `TASKS.md`.

## Development Setup

Always use the canonical Python 3.10 interpreter recorded in
`specs/tech-stack.md`. From the repository root, install the editable package
and local development/training extras:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pip install -e ".[dev,train]"
& "D:\Anaconda3\envs\py310\python.exe" -m pytest
& "D:\Anaconda3\envs\py310\python.exe" -m flake8 rware llm_mappo eval train scripts figures/core
```

Use the configured `py310` interpreter when `conda` is not available on `PATH`. Training outputs belong in `artifacts/`; keep API credentials in an untracked `.env` file.

## Current Research Core

The current convergence-recovery work centres on a pure MAPPO baseline with a
feature-pruned centralized critic, Reward-v2, and a controlled comparison of
DirectGoal against an execution-time waypoint hint. The owner-run baseline
entry point is:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" scripts/run_mappo_fp_baseline.py --help
```

The frozen R1 baseline configuration is
`configs/optimization/r1_mappo_fp_dg_3ag_lowload.yaml`. Training outputs are
written below `artifacts/`; raw formal three-dimensional semantic labels are
kept at `artifacts/formal/` and must not be edited in place.

