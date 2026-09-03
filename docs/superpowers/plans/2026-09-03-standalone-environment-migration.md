# Standalone Environment Migration Implementation Plan

> **For implementation:** Execute the task groups in order in the local target
> repository. Do not create a worktree and do not use the Linux server.

**Goal:** Extract the usable warehouse simulation into a new, independently
installable local project at `D:\codeProject\LLM_MAPPO`, with a clean public
API and no learning, teacher, calibration, experiment, or historical-phase
code.

**Architecture:** The target project contains a physical warehouse layer and a
thin Gymnasium-compatible environment layer. The latter constructs the fixed
DirectGoal 613D observation, Reward-v2 team reward, legal-action masks, and
episode metrics. Rendering is read-only. Configuration and scripts depend only
on the public environment API. Dependencies flow one way:
`warehouse -> observation/reward -> environment -> rendering/config/scripts`.

**Source design:**
`docs/superpowers/specs/2026-09-03-standalone-environment-design.md`

**Target stack:** Python 3.10; Gymnasium; NumPy; PyYAML; Pillow/imageio for
headless frame and GIF output; pytest. The existing Windows interpreter
`D:\Anaconda3\envs\py310\python.exe` is used directly. The target project must
not depend on PyTorch, TensorBoard, MAPPO, an A* implementation, an LLM SDK, or
the current repository as an installed dependency.

## Non-negotiable boundaries

- Work only in the local target `D:\codeProject\LLM_MAPPO` after confirming it
  is empty. Never overwrite a pre-existing target file without new owner
  approval.
- The current repository remains a read-only source of behaviour. No source
  code, source artifacts, or source Git history is moved or deleted.
- The target uses a new Git repository and a new history.
- Target filenames are functional only: `environment.py`, `observation.py`,
  `reward.py`, `rendering.py`, `config.py`, and `warehouse/`. Do not introduce
  names containing `phase2`, `phase3`, `reward_v2`, `legacy`, `o1`, `o2`, `e1`,
  `e2`, or `r1`.
- Preserve only DirectGoal 613D. Do not copy the waypoint schema, no-goal-hint
  schema, planner instrumentation, planning snapshots, or any execution-time
  A* path/waypoint logic.
- Preserve only Reward-v2. Do not expose a reward-version switch or copy the
  `legacy-v1` formula.
- No target module may import `llm_mappo`, `rware`, or `scripts` from the old
  repository at runtime.
- Do not move experimental artifacts, checkpoints, labels, TensorBoard files,
  statistical reports, paper material, or server runbooks.

## Public target contract

The target package is `llm_mappo_env`. Its only environment-facing public
symbols are:

```python
from llm_mappo_env import EnvironmentConfig, WarehouseEnvironment

environment = WarehouseEnvironment(EnvironmentConfig(...))
observations, info = environment.reset(seed=7)
masks = environment.action_masks()              # bool [n_agents, 5]
observations, team_reward, terminated, truncated, info = environment.step(actions)
metrics = environment.metrics.as_dict()
```

`WarehouseEnvironment` follows the Gymnasium reset/step return convention.
`info["metrics"]` contains the same dictionary as `environment.metrics.as_dict()`.
`actions` is one five-action integer per AGV in stable agent-index order:
`NOOP`, `FORWARD`, `LEFT`, `RIGHT`, `TOGGLE_LOAD`. The physical mask is always
present but remains a recommendation/legality aid; the environment still
performs its own hard physical checks.

The `render()` method produces an RGB array in headless mode. `render_episode.py`
serializes a deterministic episode to GIF without opening a window.

## Task group 1 — Create the independent local package shell

**Files created**

- `D:\codeProject\LLM_MAPPO\pyproject.toml`
- `D:\codeProject\LLM_MAPPO\README.md`
- `D:\codeProject\LLM_MAPPO\src\llm_mappo_env\__init__.py`
- `D:\codeProject\LLM_MAPPO\src\llm_mappo_env\config.py`
- `D:\codeProject\LLM_MAPPO\tests\test_package_contract.py`
- package, scripts, configs, docs, and tests directories listed in the design.

1. Read-only check that `D:\codeProject\LLM_MAPPO` exists and has no children.
   If it has any child, stop before writing it and report the complete listing.
2. Initialise Git inside that empty directory. Create a minimal setuptools
   project with `src` layout, Python `>=3.10`, runtime dependencies
   `gymnasium`, `numpy`, `PyYAML`, and `Pillow`, and an optional `dev` extra
   containing pytest. Do not add ML packages.
3. Write a failing package-contract test first. It imports
   `EnvironmentConfig` and `WarehouseEnvironment`, asserts that target imports
   do not load `torch`, and asserts the action count constant is five.
4. Add an initial `EnvironmentConfig` dataclass and package exports sufficient
   for that test. Configuration validation must reject non-positive agent
   counts, invalid battery thresholds, and an invalid batch range.
5. Run, in the target repository:

   ```powershell
   D:\Anaconda3\envs\py310\python.exe -m pip install -e ".[dev]"
   D:\Anaconda3\envs\py310\python.exe -m pytest tests\test_package_contract.py
   ```

   Commit only this target scaffolding with `Initialize standalone warehouse environment package`.

## Task group 2 — Move the physical warehouse model into `warehouse/`

**Source authority**

- `rware/warehouse.py`
- `rware/utils/`
- the portions of `rware/rendering.py` needed by the future renderer.

**Target files**

- `src/llm_mappo_env/warehouse/__init__.py`
- `src/llm_mappo_env/warehouse/model.py`
- `src/llm_mappo_env/warehouse/tasks.py`
- `src/llm_mappo_env/warehouse/spaces.py`
- any small local utility modules needed to keep these imports self-contained.

1. Before copying behaviour, add failing tests for a seeded reset, stable
   agent IDs/order, five discrete actions, a dynamic batch being created, and
   a task priority weight being preserved. Use a four-AGV deterministic
   configuration so failures are reproducible.
2. Extract the map grid, `Action`, `Direction`, AGV, shelf, task queue,
   dynamic-ingress scheduling, priority labels, collision/block handling,
   battery use, charging, energy death, and recovery mechanics. Split the
   source logically into `model.py` and `tasks.py`; retain low-level spaces in
   `spaces.py` only where needed.
3. Rename imports and types for a self-contained target package. Keep source
   action enum integer values exactly (`NOOP=0`, `FORWARD=1`, `LEFT=2`,
   `RIGHT=3`, `TOGGLE_LOAD=4`) so replay actions remain unambiguous.
4. Retain physical environment events required by reward and metrics:
   task completed, blocked forward, collisions, pickups, charging, battery
   depletion, recovery, and dynamic task creation. Do not retain any expert,
   planner, reservation, yielding, or student-policy fields.
5. Run the targeted test file and commit the target changes with
   `Extract physical warehouse dynamics and task ingress`.

## Task group 3 — Install the single DirectGoal observation and Reward-v2

**Source authority**

- DirectGoal branch of `llm_mappo/optimization_observation.py`
- Reward-v2 branch of `llm_mappo/reward_v2.py`
- goal selection, charging-target, nearby/global feature helpers from
  `llm_mappo/phase2.py`.

**Target files**

- `src/llm_mappo_env/observation.py`
- `src/llm_mappo_env/reward.py`
- `tests/test_observation.py`
- `tests/test_reward.py`

1. Write failing observation tests that assert a five-AGV reset returns
   `float32` `[5, 613]`, the direct goal displacement occupies positions
   `586:588`, and positions `588:595` are zero. These tests establish the
   immutable DirectGoal contract without importing a planner.
2. Implement `build_direct_goal_observation()` as the only exported builder.
   It takes raw warehouse observation and physical features, constructs the
   two normalized target displacements followed by seven fixed zero slots, and
   returns `float32`. Do not define an observation schema enum: there is only
   one schema in the target.
3. Write failing reward tests for: task completion receives the full weighted
   team term; pickup receives its priority-weighted local term; unchanged task
   identity enables signed distance progress; a switched target receives zero
   progress; blocked forward has the Reward-v2 penalty; `NOOP` costs `0.01`.
4. Implement the single reward formula in `reward.py` using the existing
   constants and `RewardGoalSnapshot`. Delete the old reward-version API while
   extracting: callers cannot select or reference a legacy formula.
5. Run both test files and commit with
   `Add DirectGoal observations and team reward`.

## Task group 4 — Build the clean Gymnasium environment adapter

**Source authority**

- DirectGoal and Reward-v2 paths of `llm_mappo/phase2.py`
- applicable checks from `tests/test_phase2.py`,
  `tests/test_optimization_observation.py`, and `tests/test_r1_reward_v2.py`.

**Target files**

- `src/llm_mappo_env/environment.py`
- `src/llm_mappo_env/__init__.py`
- `tests/test_environment.py`
- `tests/test_battery_and_metrics.py`

1. Write failing reset/step tests for the public contract: standard Gymnasium
   returns, observations `[n_agents, 613]`, a scalar finite team reward,
   stable `info["metrics"]`, and no cross-agent action reordering.
2. Implement `EpisodeMetrics` with exactly the required physical fields:
   completed/created/target tasks, completion rate, collisions, blocked
   forwards, deadlock, pickup, battery/charge/recovery counters, minimum
   battery, step count, reward, and success. `as_dict()` is the canonical
   serialization used by scripts and tests.
3. Implement `WarehouseEnvironment(gym.Env)` around the extracted warehouse.
   Its target selection is limited to assigned pickup shelf, delivery goal, or
   charging station based on the retained battery rule. It must calculate the
   pre/post goal snapshots, Reward-v2 team reward, episode termination, and
   metric updates exactly once per joint action.
4. Implement `action_masks()` as shape `[n_agents, 5]`: dead or picking-locked
   AGVs allow only `NOOP`; a physically required pickup allows only
   `TOGGLE_LOAD`; all other actions remain available for the underlying model
   to validate. There is no A*-derived, priority-derived, reservation-derived,
   or coordination-derived masking.
5. Write and run deterministic tests for required pickup masking, blocked
   forward, charging exposure, battery death/recovery, dynamic ingress, and
   reproducible reset. Confirm the implementation contains no imports from
   `planner`, `shadow`, `calibration`, `semantic`, `mappo`, `ppo`, `torch`, or
   `llm_mappo`. Commit with
   `Add standalone Gymnasium warehouse environment`.

## Task group 5 — Add environment-only configuration and headless tools

**Source authority**

- physical portion of `configs/optimization/r1_4agv_lowload.yaml`
- the frozen formal-environment values used by the current E1/E2 runs
  (five AGVs, dynamic ingress, DirectGoal, Reward-v2, battery scale 1.10,
  charge thresholds 0.30/0.80)
- `llm_mappo/visualization.py` and required source rendering helpers.

**Target files**

- `src/llm_mappo_env/config.py`
- `src/llm_mappo_env/rendering.py`
- `configs/4agv_lowload.yaml`
- `configs/5agv_formal.yaml`
- `scripts/run_random_policy.py`
- `scripts/render_episode.py`
- `tests/test_config.py`
- `tests/test_rendering.py`

1. Write failing YAML tests that load both profiles and assert they only carry
   environment fields. Explicitly reject training fields such as `seed`,
   `rollout_length`, `num_env_workers`, `reward_version`, or any teacher/LLM
   setting.
2. Implement `load_environment_config(path)` and
   `create_environment(config)`. The 4-AGV LowLoad profile preserves: interval
   40, batch range `[2, 4]`, queue 4, target 20, max steps 1000, battery scale
   1.10, charge thresholds 0.30/0.80. The 5-AGV Formal profile preserves the
   formal physical environment only, with queue 8 and target 50; it contains
   no training budget, seed list, or experimental group.
3. Implement a headless renderer returning a nonblank `uint8` RGB frame and a
   GIF writer. It may read state but must not mutate it, instantiate a GUI, or
   depend on display variables.
4. Implement `run_random_policy.py`: load one YAML, reset with an optional
   seed, choose only actions whose mask bit is true, print final metric JSON,
   and close the environment. Implement `render_episode.py` similarly with an
   explicit GIF output path, frame stride, optional seed, and no policy/model
   loading.
5. Run both scripts once for each YAML configuration, run the config/rendering
   tests, and commit with
   `Add profiles and headless environment tools`.

## Task group 6 — Final extraction audit, documentation, and handoff

**Target files**

- `docs/migration-map.md`
- `docs/handoff.md`
- `README.md`
- test files required for any gap found in the audit.

1. Add `docs/migration-map.md`, listing every source family, target module,
   retained responsibilities, and deliberately excluded responsibilities.
   It must specifically record the exclusion of training, all teachers,
   calibration, LLM labels, artifacts, old waypoint observations, and legacy
   rewards.
2. Add `docs/handoff.md` for the next project: package installation, profile
   selection, random-run command, GIF command, public API, fixed action order,
   DirectGoal 613D meaning, Reward-v2 scope, and known limitations. It must
   state that the repository does not provide a policy, training method, or
   performance claim.
3. Update the README with the same short installation and two command examples.
   It must not cite O1/O2/O3/E1/E2/R1 results or imply that an algorithm has
   been validated.
4. Run the complete target suite and two acceptance runs:

   ```powershell
   D:\Anaconda3\envs\py310\python.exe -m pytest
   D:\Anaconda3\envs\py310\python.exe scripts\run_random_policy.py --config configs\4agv_lowload.yaml --seed 9107
   D:\Anaconda3\envs\py310\python.exe scripts\render_episode.py --config configs\5agv_formal.yaml --seed 7 --output artifacts\formal_preview.gif
   ```

   Check that both runs finish, metrics are finite, the GIF exists and has at
   least one frame, and the package import leaves `torch` unloaded.
5. Run a content boundary audit with ripgrep for forbidden imports/terms and
   manually classify any legitimate documentation mention. Commit the final
   documentation and tests with `Document standalone environment handoff`.

## Final acceptance checklist

- [ ] `D:\codeProject\LLM_MAPPO` is an independent Git repository and is
      installable without the old repository.
- [ ] Both YAML profiles construct an environment with DirectGoal `[n, 613]`,
      five action masks, dynamic ingress, priority, battery/charging/death/
      recovery, and Reward-v2 only.
- [ ] Headless random replay and GIF creation work for the two profiles.
- [ ] The target test suite passes under the canonical `py310` interpreter.
- [ ] There is no target runtime dependency on Torch, a policy/trainer,
      A*, calibration, LLM/semantic labeling, artifacts, checkpoints, or old
      phase/legacy code.
- [ ] Target `docs/migration-map.md` and `docs/handoff.md` make the new
      project's boundaries and operation clear to its next maintainer.
