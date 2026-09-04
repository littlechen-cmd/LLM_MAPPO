# MAPPO-FP-DG Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the optimization-route baseline with a teacher-free MAPPO implementation whose value function uses a project-specific Feature-Pruned agent-specific global state.

**Architecture:** A shared feed-forward actor consumes only the 613D DirectGoal observation. A shared centralized critic consumes one FP state per robot and produces one value per robot; rollout storage, GAE, PPO and value normalization preserve the environment, time and robot axes. A dedicated baseline runner reuses the spawned CPU environment pattern without importing any A*, LLM, semantic or reward-calibration component.

**Tech Stack:** Python 3.10, PyTorch, NumPy, Gymnasium, pytest, YAML.

**Spec:** `specs/2026-09-02-r1-convergence-recovery/requirements.md`

## Global Constraints

- The only long run is `3 AGV LowLoad + DirectGoal + 500000 cumulative environment steps`.
- The actor, runner and checkpoint must contain no A*, LLM, semantic or reward-calibration module.
- The FP state contains current environment truth only, removes duplicated features, and never contains teacher or future information.
- The canonical Windows interpreter is `D:\Anaconda3\envs\py310\python.exe`.
- Long training and evaluation are owner-run on physical GPU 0.

---

### Task 1: FP state contract

**Files:**
- Create: `llm_mappo/mappo_baseline/fp_state.py`
- Create: `llm_mappo/mappo_baseline/__init__.py`
- Test: `tests/test_mappo_fp_baseline.py`

**Interfaces:**
- Produces: `FPStateSchema`, `build_fp_state(environment) -> np.ndarray`, shape `[agents, fp_dim]`.

- [x] Write tests for fixed shape, agent-specific rows, death masking, deterministic output, and forbidden-field absence.
- [x] Run the focused test and verify it fails because the module is absent.
- [x] Implement the normalized common-state and agent-specific suffix builders.
- [x] Run the focused tests and verify they pass.

### Task 2: Canonical actor, critic and ValueNorm

**Files:**
- Create: `llm_mappo/mappo_baseline/networks.py`
- Create: `llm_mappo/mappo_baseline/value_norm.py`
- Test: `tests/test_mappo_fp_baseline.py`

**Interfaces:**
- Produces: `MAPPOActor`, `MAPPOFPCritic`, `ValueNorm`.

- [x] Write failing tests for actor/critic shapes, orthogonal initialization, independent values and ValueNorm round trips.
- [x] Implement the feed-forward networks and running value normalization.
- [x] Run focused tests and verify they pass.

### Task 3: Per-agent rollout and GAE

**Files:**
- Create: `llm_mappo/mappo_baseline/storage.py`
- Test: `tests/test_mappo_fp_baseline.py`

**Interfaces:**
- Produces: `MAPPOBatch`, `MAPPORolloutStorage.compute_returns(...)`.

- [x] Write failing tests proving GAE does not cross workers or episodes and remains per-agent.
- [x] Implement axis-preserving storage, bootstrap masks, bad-transition masks and active masks.
- [x] Run focused tests and verify they pass.

### Task 4: PPO trainer

**Files:**
- Create: `llm_mappo/mappo_baseline/trainer.py`
- Test: `tests/test_mappo_fp_baseline.py`

**Interfaces:**
- Produces: `MAPPOConfig`, `MAPPOTrainer.act`, `MAPPOTrainer.update`, checkpoint state methods.

- [x] Write failing tests for clipped policy/value losses, one-minibatch five-epoch updates, separate optimizers and active masks.
- [x] Implement the canonical MAPPO update with ValueNorm, Huber loss, value clipping and gradient clipping.
- [x] Run focused tests and verify they pass.

### Task 5: Teacher-free vector runner and owner entry point

**Files:**
- Create: `llm_mappo/mappo_baseline/vector_env.py`
- Create: `llm_mappo/mappo_baseline/runner.py`
- Create: `configs/optimization/r1_mappo_fp_dg_3ag_lowload.yaml`
- Create: `scripts/run_mappo_fp_baseline.py`
- Test: `tests/test_mappo_fp_runner.py`

**Interfaces:**
- Produces: owner CLI, resumable checkpoint, TensorBoard/CSV/JSON evidence, deterministic evaluation hook.

- [x] Write failing contract tests for exact environment, 500k cumulative steps, 16×64 rollout accounting, zero forbidden imports and checkpoint identity.
- [x] Implement persistent spawned CPU workers that return observations, FP states, masks, rewards, terminations and complete episode records.
- [x] Implement the centralized learner, atomic evidence and exact resume boundary.
- [x] Run focused tests and verify they pass.

### Task 6: Integration, governance and handoff

**Files:**
- Modify: `specs/2026-09-02-r1-convergence-recovery/plan.md`
- Modify: `specs/2026-09-02-r1-convergence-recovery/requirements.md`
- Modify: `specs/2026-09-02-r1-convergence-recovery/validation.md`
- Modify: `specs/roadmap.md`
- Modify: `TASKS.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: the owner-run Linux command only after local focused validation succeeds.

- [x] Run focused MAPPO-FP-DG tests with the canonical interpreter.
- [x] Run affected existing baseline/environment tests.
- [x] Run Flake8 on changed Python files.
- [x] Record implementation status without claiming the 500k capability Gate passed.
- [x] Create one focused commit and report the exact bundle synchronization commands.
