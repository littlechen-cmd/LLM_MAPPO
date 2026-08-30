# O2 Calibration Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the owner-run, resumable six-run O2 calibration experiment and its
pre-run parity smoke and Go/No-Go analyzer without changing the frozen MAPPO, Pure
Motion A*, Reward Calibration, environment, reward, or energy contracts.

**Architecture:** O2 is a new experiment layer around the already approved O0/O1
components. It owns matrix expansion, evidence, checkpoint/resume, PPO rollout
orchestration and Gate aggregation; existing core modules remain behaviorally
unchanged. The owner runs all 150000-step jobs on Linux GPU 0, while local work is
limited to deterministic short tests and smoke runs.

**Tech Stack:** Python 3.10, PyTorch 2.10, NumPy, PyYAML, Gymnasium, pytest.

**Spec:** `docs/architecture/o0-reward-calibrated-heterogeneous-distillation.md`
section 13.1-13.2, `specs/roadmap.md` Phase O2, and `TASKS.md` Phase O2.

## Global Constraints

- The only O2 matrix is `MAPPO-DG/RC-AStarKD × 107/117/127 × 150000` real
  environment steps; LLMKD is disabled and Fixed-AStarKD receives no long run.
- The environment remains the frozen 5-AGV canonical environment with dynamic
  ingress 40, batch size 4-8, explicit initial priority label `A`, queue 8, task
  target 50, max steps 1000, deadlock 180, and energy `1.10/0.30/0.80`. The label
  must be explicit because an episode can contain at most 26 batches (two initial
  plus 24 scheduled), requiring the complete `A`--`Z` range.
- O2 must verify a passing `o1-gate-receipt-v1` and matching O1 summary before
  creating run artifacts.
- Long training and evaluation are owner-only; local commands use
  `D:\Anaconda3\envs\py310\python.exe` and short budgets only.
- Every O2 run is one joint 5-AGV environment: `150000` means joint environment
  transitions. Training actions are stochastic; O1's deterministic argmax is only a
  Gate/shadow rule.
- PPO is frozen to rollout `512`, gamma `0.99`, GAE lambda `0.95`, clip `0.20`, value
  coefficient `0.50`, entropy coefficient `0.01`, learning rate `3e-4`, max gradient
  norm `0.50`, `4` epochs and minibatch `64`.
- Both O2 groups use the same 61D semantic interface but receive all-zero semantic
  observations and have semantic loss fixed to zero. MAPPO-DG makes zero Teacher
  queries, shadow calls and EMA updates; only RC-AStarKD invokes calibration.
- Checkpoints occur every `10000` transitions and resume only when code commit,
  configuration and seed match exactly. The six formal runs execute sequentially on
  physical GPU 0 after the P1 preflight/lease; a completed run does not make O2 Go
  until the aggregate analyzer passes.
- O2 artifacts live only below `artifacts/optimization/o2_calibration/`; ordinary
  logs never contain full state or Teacher arrays.
- No source under `llm_mappo/mappo.py`, `llm_mappo/environment.py`,
  `llm_mappo/reward_calibration.py`, or `llm_mappo/pure_motion_teacher.py` may change.

---

### Task 1: O1 evidence closure and O2 frozen matrix

**Files:**
- Create: `configs/optimization/o2_calibration.yaml`
- Create: `llm_mappo/o2_contract.py`
- Test: `tests/test_o2_contract.py`
- Modify: `TASKS.md`, `CHANGELOG.md`, `plan/progress.md`

**Interfaces:**
- Produces: `O2ExperimentConfig.from_yaml(path)`,
  `O2RunSpec(group: str, seed: int, real_env_steps: int)`,
  `expand_o2_matrix(config) -> tuple[O2RunSpec, ...]`, and
  `verify_o1_authorization(run_dir) -> dict`.

- [x] Write tests asserting the exact six-run Cartesian matrix, 150000-step budget,
  disabled LLMKD, no Fixed long run, canonical environment/energy fields, and rejection
  of unknown or changed fields.
- [x] Run `D:\Anaconda3\envs\py310\python.exe -m pytest tests/test_o2_contract.py -q`
  and confirm failure because the O2 contract module/config do not exist.
- [x] Implement immutable config parsing, exact matrix expansion and O1
  summary/receipt/hash verification without creating artifacts on failure.
- [x] Mark the independently evidenced P1/O1 prerequisites complete, record commit
  `7c305ea24cdca34467c2e7e8a5a9d66ba1133d1e` and the Gate metrics, but keep O2 pending.
- [x] Run the focused test and `git diff --check`.

### Task 2: MAPPO-compatible O2 rollout and update adapter

**Files:**
- Create: `llm_mappo/o2_training.py`
- Test: `tests/test_o2_training.py`

**Interfaces:**
- Consumes: `O0StudentActor`, `O0CentralizedCritic`,
  `OptimizationRolloutBuffer`, `LinearEnvStepSchedule`, `RewardCalibrator`, and
  `OptimizationTrainer` helper adapters.
- Produces: `O2Rollout`, `O2PPOUpdater.update(rollout, last_value) -> dict`, and
  `O2Trainer.run(max_steps: int | None = None) -> dict`.

- [x] Write tests proving clipped PPO/GAE uses stored old log-probabilities and values,
  MAPPO-DG has exactly zero A*KD weight and zero shadow calls, RC-AStarKD applies
  `lambda_A*m_valid*m_calib*c_reward`, LLM loss is zero in both groups, terminal/deadlock
  boundaries reset consistently, and all reported losses/gradients are finite.
- [x] Run the focused tests and confirm failure because the O2 training interfaces do
  not exist.
- [x] Implement the experiment-layer rollout/update adapter using the existing frozen
  network and mathematical components; do not edit legacy or O0 core algorithm files.
- [x] Run `D:\Anaconda3\envs\py310\python.exe -m pytest tests/test_o2_training.py
  tests/test_optimization_buffer.py tests/test_reward_calibration.py -q`.

### Task 3: Compact evidence, checkpoints and resume

**Files:**
- Create: `llm_mappo/o2_evidence.py`
- Test: `tests/test_o2_evidence.py`

**Interfaces:**
- Produces: `O2EvidenceWriter`, `save_o2_checkpoint`, `load_o2_checkpoint`, and
  `compute_throughput_grid(episodes, grid) -> list[dict]`.
- Output: `run_manifest.json`, `teacher_step_counts.csv`, `teacher_events.jsonl`,
  `updates.csv`, `episodes.csv`, `throughput_grid.csv`, `checkpoint_latest.pt`,
  `checkpoint_final.pt`, `state.json`, and `summary.json`.

- [x] Write tests for atomic new-run creation, no-overwrite, exact identity-bound resume,
  complete RNG/schedule/EMA/model/optimizer restoration, buffer-empty checkpoint rule,
  compact schema, forbidden full arrays, 0..150000/10000 throughput grid, and terminal
  checkpoints.
- [x] Run focused tests and confirm the new evidence/checkpoint interfaces are absent.
- [x] Integrate update-boundary periodic checkpoint/resume into the owner runner. The
  runner restores the identity-bound live environment/episode snapshot, optimizer, EMA
  and RNG state before taking the next step; checkpoint thresholds are honored at the
  first completed PPO update boundary after each 10k-step threshold.
- [x] Run focused evidence/checkpoint tests and `git diff --check`.

### Task 4: Fixed/RC parity smoke and owner CLI

**Files:**
- Create: `scripts/run_o2_calibration.py`
- Create: `scripts/check_o2_parity.py`
- Test: `tests/test_o2_cli.py`
- Test: `tests/test_o2_parity.py`

**Interfaces:**
- CLI: `check_o2_parity.py --config ... --output ...`.
- CLI: `run_o2_calibration.py --config ... --o1-run ... --output-root ...
  [--run GROUP:SEED] [--resume RUN_DIR] [--smoke-steps N]`.

- [x] Write tests that the parity runner replays the same deterministic states and checks
  sampler selections, Teacher queries, shadow calls, EMA updates and logging counts;
  Fixed/RC may differ only in applied `c_A_reward`.
- [ ] Write CLI tests rejecting absent/mismatched O1 receipt, altered matrix, online LLM,
  O3 topology, invalid resume, multiple simultaneous writes and noncanonical formal
  smoke overrides.
- [x] Run CLI/parity tests and confirm failure for missing entry points.
- [x] Implement the parity runner and owner CLI. `--smoke-steps` marks all artifacts
  `diagnostic_only=true` and can never create a formal O2 Gate result.
- [x] Run CLI help, 64-step parity smoke, focused tests and Flake8.

### Task 5: O2 aggregation and owner handoff

**Files:**
- Create: `scripts/analyze_o2_calibration.py`
- Test: `tests/test_o2_analysis.py`
- Modify: `plan/task-package/o2-calibration-experiment.md`, `CHANGELOG.md`

**Interfaces:**
- CLI: `analyze_o2_calibration.py --config ... --runs-root ... --output ...`.
- Produces: `o2_gate_summary.json` with per-RC-seed coverage, per-seed normalized AUC,
  paired degradation and the median Go/No-Go result.

- [x] Write tests for coverage denominator including all selected-state agent slots,
  each RC seed `>=0.25`, the fixed normalized trapezoidal AUC grid, paired seed
  degradation, median degradation `<=0.10`, missing/corrupt run No-Go and non-finite
  fail-closed behavior.
- [x] Run analysis tests and confirm failure because the analyzer is absent.
- [x] Implement deterministic aggregation without reading held-out seeds or O3 results.
- [x] Run all O2 tests, the relevant O0/O1 regression set, Flake8 and `git diff --check`.
- [ ] Commit the implementation, then provide exact Git bundle transfer commands and one
  owner-only Linux `nohup` command using `/home/lzx/llm-a-mappo`, physical GPU 0 and logs
  under `/home/lzx/`.
