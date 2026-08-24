# LLM-MAPPO Dynamic Warehouse

This repository combines the RWARE Gymnasium environment with the staged
LLM-MAPPO dynamic-warehouse research implementation. The current architecture
and experiment contract are documented in `specs/mission.md`,
`specs/tech-stack.md`, `specs/roadmap.md`, `TASKS.md`, and
`plan/experiment-protocol.md`.

## Development Setup

Use Python 3.10. From the repository root, install the editable package and the local development/training extras:

```powershell
python -m pip install -e ".[dev,train]"
python -m pytest
python -m flake8 rware llm_mappo eval train scripts figures/core
```

Use the configured `py310` interpreter when `conda` is not available on `PATH`. Training outputs belong in `artifacts/`; keep API credentials in an untracked `.env` file.

## Phase 1 Dynamic Warehouse

Import `rware`, then create `llm-mappo-medium-3ag-v1` through Gymnasium for the medium three-AGV baseline. It adds FIFO priority batches, a hard 10% task-assignment battery filter, three charging stations, collision penalties, and three-step picking locks while preserving the original `rware-*` environments.

The Phase 1 reference configuration is `configs/phase1_medium_3ag.yaml`.

## Phase 2 CTDE MAPPO Baseline

Phase 2 adds a custom PyTorch CTDE MAPPO implementation. A shared Actor receives
each AGV's local RWARE observation plus the current oracle waypoint, battery,
nearby AGV, and compact global features. During training, the Critic uses
attention pooling over all AGV observations. The Phase 2 adapter fixes all
initial tasks to priority `B`, disables later task batches, and retains three
charging stations for the three-AGV stage.

```powershell
python train/train_phase2.py --agents 1 --episodes 5000
python train/train_phase2.py --agents 3 --episodes 5000
python eval/evaluate_phase2.py artifacts/phase2/seed_007/checkpoint_final.pt
```

For the local 1-AGV waypoint-reward feasibility comparison, run the two isolated
500-episode configurations:

```powershell
python train/train_phase2.py --config configs/phase2_waypoint_001.yaml
python train/train_phase2.py --config configs/phase2_waypoint_005.yaml
```

Runs write configuration, checkpoints, per-episode CSV, per-update CSV,
TensorBoard events, and `summary.json` beneath `artifacts/phase2/`. The evaluator
uses ten seeds by default and reports the completion, collision, deadlock, and
success-rate variance gates from `configs/phase2_mappo.yaml`.

