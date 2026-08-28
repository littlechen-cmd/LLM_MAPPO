# P1 Linux Optimization Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, fail-closed Linux execution boundary for optimization-route O1/O2 work on the shared dual-GPU server.

**Architecture:** A standard-library-first runtime module collects and validates machine state without importing Torch before GPU visibility is fixed. A thin owner CLI holds a Linux GPU lease, waits for five clean samples, launches the existing O1 benchmark in a child process, and records atomic, resumable evidence. O2 consumes a hash-bound O1 receipt in its own phase.

**Tech Stack:** Python 3.10.19, PyTorch 2.10.0+cu128, Conda prefix, `nvidia-smi`, psutil, YAML/JSON/CSV, `fcntl`, pytest, Flake8, tmux.

**Spec:** `specs/2026-08-28-p1-linux-optimization-server/requirements.md`

## Global Constraints

- P1 serves only `codex/optimization`; stable-route code and manifests are out of scope.
- Long jobs and all server installation commands are owner-run; Codex runs only local tests and analyzes artifacts.
- Canonical server interpreter is `/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python`.
- O1/O2 canonical physical GPU is index 0, RTX 4090 with at least 48000 MiB.
- Never kill, suspend, reprioritize, or hide another user's process.
- Do not change algorithm, environment, reward, energy, teachers, seeds, workers, thresholds, or budgets.
- Every completed task group updates `CHANGELOG.md` in the same focused commit.

---

## Task group P1-A — Governance and dependency repair

**Files:**
- Modify: `specs/mission.md`
- Modify: `specs/tech-stack.md`
- Modify: `specs/roadmap.md`
- Modify: `TASKS.md`
- Modify: `plan/experiment-protocol.md`
- Modify: `configs/g3_experiment_manifest.yaml`
- Modify: `terminology.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: approved P1 requirements and server inventory.
- Produces: one canonical P1 dependency and vocabulary contract used by every later task.

- [x] Add a governance regression that parses Roadmap/TASKS/manifest and asserts
  `O1 local -> P1 -> O1 CUDA Gate -> O2`, optimization-only scope, owner-run long jobs,
  physical GPU 0 and Linux interpreter.
- [x] Run the regression and confirm it fails on the current A600/Windows assumptions and
  the incorrect O1 Roadmap status text.
- [x] Update the seven governance files, replace optimization-route A600 wording with
  owner-run Linux CUDA wording, add P1, and preserve every frozen experimental value.
- [x] Add terminology entries for Conda prefix, physical/logical GPU index,
  `CUDA_VISIBLE_DEVICES`, preflight, fail-closed, GPU lease, manifest, atomic write,
  explicit resume, and tmux.
- [x] Run the governance regression, YAML parsing, and `git diff --check`; confirm pass.
- [ ] Commit with subject `docs: define P1 Linux optimization server contract`.

## Task group P1-B — Reproducible Python/CUDA environment

**Files:**
- Create: `constraints/linux-py310-cu128.txt`
- Create: `scripts/verify_linux_environment.py`
- Create: `tests/test_linux_environment_contract.py`
- Create: `docs/runbooks/p1-linux-environment-setup.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: `verify_environment(expected: EnvironmentContract) -> EnvironmentReport` and
  an owner-readable installation runbook using only the canonical interpreter.
- The report JSON fields are `python`, `torch`, `torch_cuda`, `packages`,
  `editable_project`, `freeze_sha256`, and `pass`.

- [x] Write failing tests using injected package/version probes for exact Python 3.10.19,
  Torch 2.10.0+cu128, CUDA availability, pinned package versions, and wrong-version
  fail-closed behavior.
- [x] Run `D:\Anaconda3\envs\py310\python.exe -m pytest tests/test_linux_environment_contract.py -q`
  and confirm failure because the verifier and constraint file do not exist.
- [x] Implement the verifier with dependency injection; keep production collection
  read-only and never create or modify an environment.
- [x] Write the exact owner commands for Conda prefix creation, official cu128 Torch
  installation, constrained editable install, import verification and freeze export.
- [x] Run focused tests and Flake8; inspect the runbook for absence of `conda activate`,
  shared-base modification, secrets and unpinned Torch.
- [ ] Commit with subject `build: freeze Linux py310 CUDA environment`.

## Task group P1-C — Machine inventory, preflight wait and GPU lease

**Files:**
- Create: `llm_mappo/linux_server_runtime.py`
- Create: `scripts/check_optimization_server.py`
- Create: `configs/optimization/p1_linux_server.yaml`
- Create: `tests/fixtures/nvidia_smi_p1_inventory.csv`
- Create: `tests/test_linux_server_runtime.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- `collect_machine_snapshot(command_runner, paths) -> MachineSnapshot`
- `evaluate_preflight(snapshot: MachineSnapshot, policy: ServerPolicy) -> PreflightResult`
- `wait_for_resources(sample, policy, clock, sink) -> PreflightResult`
- `gpu_lease(lock_path: Path)` context manager backed by `fcntl.flock` on Linux.
- `write_new_atomic_json(path: Path, payload: Mapping) -> None` performs flush, fsync and
  `os.replace` while refusing an existing final artifact.
- `replace_state_atomic(path: Path, payload: Mapping, identity: RunIdentity) -> None` is
  restricted to `state.json` and rejects an identity change.

- [x] Write failing parser tests for two GPUs, UUID/index mapping, compute PIDs, malformed
  `nvidia-smi`, missing commands, wrong GPU, insufficient RAM/disk, dirty Git, CPU overload,
  five-sample reset, 48-hour timeout, and competing project lease.
- [x] Run the focused tests and confirm failures name the absent interfaces.
- [x] Implement immutable dataclasses and pure parsers first; make all ambiguous or missing
  inventory a failed result with machine-readable reasons.
- [x] Implement polling and Linux lease; keep `nvidia-smi` and `psutil` collection behind
  injected boundaries so Windows tests never require a GPU or `fcntl`.
- [x] Implement CLI modes `--once` and `--wait`, both writing versioned JSON plus JSONL wait
  samples under `artifacts/optimization/p1_linux_server/`.
- [x] Run focused tests, Flake8 and a Windows mocked CLI smoke; confirm no process-control
  calls other than read-only subprocess queries exist.
- [ ] Commit with subject `feat: add Linux optimization server preflight`.

## Task group P1-D — Correct O1 Gate and isolate H4 diagnosis

**Files:**
- Modify: `scripts/benchmark_reward_calibration.py`
- Modify: `tests/test_reward_calibration_benchmark.py`
- Create: `tests/test_o1_linux_gate_contract.py`
- Modify: `configs/optimization/o1_reward_calibration_smoke.yaml`
- Modify: `plan/task-package/o1-role-alignment-implementation.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Normal CLI subcommand: `gate --modes baseline h12`.
- Diagnostic CLI subcommand: `diagnose-h4 --failed-gate-summary <path>`.
- `validate_cuda_binding(expected_physical_index, snapshot, torch_probe) -> DeviceBinding`.
- Normal result contains `runtime_gate_pass`, `memory_gate_pass`, and `gate_pass`;
  diagnostic result contains `diagnostic_only=true` and no Gate decision field.

- [x] Replace parser expectations in tests first: normal baseline/H12 accepted; H4 in normal
  mode, wrong order, wrong workers/windows, CPU config, absent preflight receipt and wrong
  CUDA binding rejected; diagnostic requires a real failed normal summary.
- [x] Run focused tests and confirm the current baseline/H4/H12 parser fails them.
- [x] Split normal and diagnostic argument parsing without changing horizon, worker, repeat,
  window or threshold calculations.
- [x] Validate preflight/config/code/environment hashes before creating the output directory;
  force the benchmark child config to `device: cuda` and logical `cuda:0`.
- [x] Extend manifest rows with physical index, UUID, PCI, GPU memory, driver, CPU/RAM,
  environment-freeze hash and preflight hash.
- [x] Run benchmark tests, optimization smoke tests, Flake8 and CLI help; confirm H4 cannot
  produce or mutate a Go result.
- [ ] Commit with subject `fix: align O1 Linux CUDA gate contract`.

## Task group P1-E — Atomic O1 shards, explicit resume and receipt

**Files:**
- Create: `llm_mappo/run_evidence.py`
- Create: `tests/test_run_evidence.py`
- Modify: `scripts/benchmark_reward_calibration.py`
- Modify: `tests/test_reward_calibration_benchmark.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- `RunIdentity(code_commit, config_sha256, immutable_machine_sha256, environment_sha256)`;
  dynamic utilization/free-capacity values are excluded from identity.
- `load_valid_shard(path, identity, schema) -> Mapping | None` rejects rather than silently
  recomputing a corrupt or mismatched existing shard.
- `classify_failure(exception) -> Literal['infrastructure','algorithm']` accepts only
  owner SIGINT/SIGTERM, exit 130/143, monitored external-GPU interference, and
  `EIO/ENOSPC/EDQUOT/ESTALE` as infrastructure; unknown and CUDA OOM are algorithm failures.
- `write_o1_gate_receipt(summary, identity, path) -> Mapping` writes only for Gate Go.
- `verify_o1_gate_receipt(path, expected_identity) -> Mapping` is the future O2 boundary.

- [x] Write failing tests for atomic no-overwrite, controlled `state.json` replacement,
  partial `.tmp`, interrupted repeat reuse, contaminated-shard exclusion, hash mismatch,
  corrupt shard, the exact infrastructure allowlist, algorithm non-resume and receipt rejection.
- [x] Run focused tests and confirm failure before implementing evidence helpers.
- [x] Implement evidence helpers and migrate each runtime repeat/memory window to an atomic
  shard; aggregation must be deterministic by mode/repeat/window index.
- [x] Add `--resume <run_dir>`; require identical identity and preserve prior failure logs.
- [x] Emit receipt only after both gates pass and all required artifacts validate.
- [x] Run focused tests, an injected interruption/resume smoke, Flake8 and `git diff --check`.
- [ ] Commit with subject `feat: make O1 gate evidence resumable`.

## Task group P1-F — Owner-started wait-to-O1 launcher and O2 handoff

**Files:**
- Create: `scripts/run_o1_when_available.py`
- Create: `tests/test_run_o1_when_available.py`
- Create: `docs/runbooks/p1-o1-tmux.md`
- Modify: `plan/experiment-protocol.md`
- Modify: `configs/g3_experiment_manifest.yaml`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Launcher arguments are `--server-config`, `--gate-config`, `--output-root`, and optional
  `--resume`; GPU index, polling, thresholds, workers and Gate modes come only from frozen
  configs.
- Launcher sets `CUDA_VISIBLE_DEVICES` in the child environment before Python imports Torch,
  holds the lease through Gate completion, and never invokes O2.
- Go output names the next required phase `O2` and the receipt path; No-Go output names `O0`.

- [ ] Write failing subprocess tests for environment ordering, lock lifetime, wait timeout,
  occupied GPU, child nonzero status, Gate Go/No-Go routing and absence of O2 invocation.
- [ ] Run focused tests and confirm the launcher is absent.
- [ ] Implement launcher using `sys.executable` and argument lists, never shell interpolation;
  propagate SIGINT/SIGTERM as infrastructure interruption and retain artifacts.
- [ ] Write exact `tmux new -s p1-o1`, attach/detach, status and resume commands using the
  canonical interpreter and `/home/lzx/llm-a-mappo`.
- [ ] Update protocol/manifest so P1 completion mandates O1 next and O1 Go mandates O2;
  explicitly state that the O2 continuation is implemented in O2, not fabricated by P1.
- [ ] Run focused tests, Flake8, CLI help and static scans for `kill`, shell execution,
  A600 paths and ordinary GPU override flags.
- [ ] Commit with subject `feat: add owner wait-to-O1 launcher`.

## Task group P1-G — Local cross-platform validation and owner command handoff

**Files:**
- Create: `docs/evidence/p1-local-validation.md`
- Modify: `specs/2026-08-28-p1-linux-optimization-server/validation.md`
- Modify: `TASKS.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces a literal owner installation command block and one short CUDA smoke command;
  does not claim either was run.

- [ ] Run all P1 focused tests, full pytest, full Flake8, every affected CLI `--help`, all
  YAML/JSON parsers, package build, static secret/process-control/A600 scans and
  `git diff --check`.
- [ ] Audit the diff against frozen algorithm, environment, reward, energy, teacher, seed,
  worker, threshold and budget files; record exact changed-file classification.
- [ ] Write the validation report with commit, commands, results, known risks and prohibited
  claims; update TASKS/CHANGELOG without marking P1 complete.
- [ ] Commit with subject `test: validate P1 Linux runner locally`.
- [ ] Stop and deliver only the environment-install and P1 smoke commands to the owner.

## Task group P1-H — Owner Linux smoke evidence and P1 decision

**Files:**
- Owner produces: `artifacts/optimization/p1_linux_server/<run_id>/...`
- Modify after review: `docs/evidence/p1-linux-owner-smoke.md`
- Modify after review: `specs/roadmap.md`
- Modify after review: `TASKS.md`
- Modify after review: `CHANGELOG.md`

**Interfaces:**
- Owner evidence must include environment report/freeze hash, machine snapshot, five clean
  samples, lease evidence, CUDA tensor smoke, 128-step optimization smoke and artifact hashes.

- [ ] Owner creates the Conda prefix and runs the installation verification command.
- [ ] Owner waits until GPU 0 becomes eligible and runs the P1 short CUDA smoke in tmux;
  this is not the O1 Gate.
- [ ] Architect verifies exact server identity, versions, CUDA binding, free-sample sequence,
  no external-process interference, finite smoke output and complete hashes.
- [ ] If evidence fails, keep P1 in progress and fix only the identified infrastructure issue;
  never lower thresholds or change experimental contracts.
- [ ] If evidence passes, mark P1 complete and commit `validate P1 Linux optimization server`.
- [ ] Immediately hand off the owner-started wait-to-O1 command. O1 Go routes to O2; O1
  No-Go routes to O0. Do not start or claim either task inside P1.
