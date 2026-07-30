# Repository Guidelines

## Project Structure & Module Organization

The Git repository is the workspace root. `requirement.md` records working constraints, and `大规模动态仓储LLM-MAPPO总体方案.md` is the architecture reference. The `rware/` package implements the Gymnasium environment; `warehouse.py` contains core dynamics, `rendering.py` visualization, and `rware/utils/` spaces and wrappers. Tests live in `tests/`, images in `docs/img/`, and `human_play.py` is the interactive runner. Keep MAPPO, LLM-teacher, A* teacher, training, and evaluation code in separate modules.

## Build, Test, and Development Commands

Run commands from the repository root in the Conda `py310` environment:

```powershell
conda activate py310
python -m pip install -e ".[dev,train]"  # editable package and tooling
python -m pytest                    # full test suite
python -m pytest tests/test_env.py  # focused environment tests
python -m flake8 rware human_play.py
python human_play.py --env rware-tiny-2ag-v2
python -m build                     # package artifacts; install build first
```

Prefer CPU development on the MateBook. Use the 4080S server only when episode count or runtime makes local training impractical.

## Coding Style & Naming Conventions

Use UTF-8, LF endings, a final newline, four spaces, and a maximum line length of 89. Follow PEP 8: `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Flake8 ignores `E226`, `E302`, and `E41`; tests are excluded by its configuration but should remain readable. Avoid mixing refactors with behavioral work.

## Testing Guidelines

Tests use pytest and follow `test_*.py` with `test_*` functions. Add deterministic regression tests for movement, goals, transitions, wrappers, seeding, and safety constraints. For learning components, record seeds, configuration, checkpoint, environment ID, and evaluation metrics. Follow the proposal's Go/No-Go criteria; do not claim improvements from one seed.

## Architecture & Agent Workflow

Treat rule-layer safety checks as hard constraints. LLM output may adjust priority semantics, not assign agents or emit per-step actions. A* supplies training preferences and execution-time waypoint observations; MAPPO remains the action policy. Maintain a root `TASKS.md` checklist as work progresses. Ask before resolving ambiguous requirements, and report permission or network blockers instead of bypassing them.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects such as `update render function for gymnasium compatibility` and `fix minor code issues`. Keep commits focused; create a Git checkpoint after substantial changes. Pull requests should describe behavior, list validation commands and results, link the relevant phase or issue, and include screenshots/GIFs for rendering changes plus metrics/configuration for training changes.
