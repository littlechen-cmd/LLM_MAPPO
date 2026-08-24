# Repository Guidelines

## Project Structure & Module Organization

The Git repository is the workspace root. `specs/mission.md`,
`specs/tech-stack.md`, and `specs/roadmap.md` record the architecture and
experiment contract, while `TASKS.md` tracks phase progress.
The `rware/` package implements the Gymnasium environment; `warehouse.py`
contains core dynamics, `rendering.py` visualization, and `rware/utils/`
spaces and wrappers. Tests live in `tests/`, images in `docs/img/`, and
`visualize.py` is the deterministic replay entry point. Keep MAPPO,
LLM-teacher, A* teacher, training, and evaluation code in separate modules.

## Build, Test, and Development Commands

Run commands from the repository root in the Conda `py310` environment:

```powershell
conda activate py310
python -m pip install -e ".[dev,train]"  # editable package and tooling
python -m pytest                    # full test suite
python -m pytest tests/test_env.py  # focused environment tests
python -m flake8 rware llm_mappo eval train scripts figures/core
python visualize.py --help
python -m build                     # package artifacts; install build first
```

Prefer CPU development on the MateBook. Use the A6000 server when episode count
or runtime makes local training impractical.

## Coding Style & Naming Conventions

Use UTF-8, LF endings, a final newline, four spaces, and a maximum line length of 89. Follow PEP 8: `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Flake8 ignores `E226`, `E302`, and `E41`; tests are excluded by its configuration but should remain readable. Avoid mixing refactors with behavioral work.

## Testing Guidelines

Tests use pytest and follow `test_*.py` with `test_*` functions. Add deterministic regression tests for movement, goals, transitions, wrappers, seeding, and safety constraints. For learning components, record seeds, configuration, checkpoint, environment ID, and evaluation metrics. Follow the proposal's Go/No-Go criteria; do not claim improvements from one seed.

## Architecture & Agent Workflow

Treat rule-layer safety checks as hard constraints. LLM output may adjust priority semantics, not assign agents or emit per-step actions. A* supplies training preferences and execution-time waypoint observations; MAPPO remains the action policy. Maintain a root `TASKS.md` checklist as work progresses. Ask before resolving ambiguous requirements, and report permission or network blockers instead of bypassing them.

### Project Roles And Authority

- The project owner approves major architecture and experiment decisions, runs
  long training, multi-seed evaluation, and long replay jobs, and publishes Git
  changes.
- The core architect maintains project direction, the three constitution files
  under `specs/`, and `TASKS.md`; decomposes work; checks consistency across code, experiments, and
  paper claims; reviews evidence; and provides recommendations. The architect
  may make small documentation, planning, and configuration corrections but
  delegates substantial code implementation to the project engineer.
- The project engineer implements architect-approved task packets, adds tests,
  runs short validations, prepares owner-run commands, and creates focused local
  commits. The engineer must not modify the constitution files under `specs/`, mark `TASKS.md` items
  complete, change frozen experimental decisions, push, or merge without explicit
  owner or architect authorization.
- Completion of an engineering implementation does not by itself pass a Gate.
  The architect updates `TASKS.md` only after reviewing the required evidence and
  the corresponding feature-spec and Roadmap acceptance criteria.
- Every completed `TASKS.md` subtask must update root `CHANGELOG.md` in the same
  commit. Use the standard filename `CHANGELOG.md`, newest date first, with one
  concise entry per meaningful change.
- Engineer handoffs must state the task ID, implementation scope, changed files,
  validation commands and results, unfinished checks, owner-run long commands,
  known risks, prohibited claims, commit ID, and worktree status.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects such as `update render function for gymnasium compatibility` and `fix minor code issues`. Keep commits focused; create a Git checkpoint after substantial changes. Pull requests should describe behavior, list validation commands and results, link the relevant phase or issue, and include screenshots/GIFs for rendering changes plus metrics/configuration for training changes.
