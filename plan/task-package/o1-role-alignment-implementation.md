# O1 Role Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Every task group
> stops after its focused commit for architect review; do not begin the next group
> without written approval.

**Goal:** Implement the O0-approved optimization-route architecture without changing
its environment, reward, formal seeds, evidence budget, or mathematical contracts.

**Architecture:** Add a versioned optimization path alongside the legacy Phase 3/4
path. DirectGoal observations, Pure Motion A*, three-score semantics, paired shadow
reward calibration, the new Student, strict checkpointing, and compact evidence logs
are separate components joined only by typed interfaces. Legacy evaluation loaders
remain isolated and are never used to initialize the optimization Student.

**Tech Stack:** Python 3.10, PyTorch, NumPy, Gymnasium, PyYAML, pytest, Flake8. All
Python commands use `D:\Anaconda3\envs\py310\python.exe`; repository searches use
`C:\Users\28016\bin\rg.exe` directly.

**Authoritative spec:**
`docs/architecture/o0-reward-calibrated-heterogeneous-distillation.md` and
`specs/2026-08-24-o0-astar-teacher-redesign/{requirements,plan,validation}.md`.
If this task package conflicts with the canonical architecture, stop and return to
O0-G. Do not choose an alternative locally.

## 1. Global constraints

- Use test-driven development: add one failing contract test, run it, implement only
  enough to pass, then run the focused regression set.
- O1 may modify Python code, add optimization runtime YAML, and add test-only fixtures.
  It may not generate 60/800 labels, run training or long evaluation, alter reward,
  energy `1.10/0.30/0.80`, formal seeds, O2/O3/E1/E2 budgets, or stable-route code.
- Do not retrofit the new Student into `DualHeadActor` or make the Phase 3/4 checkpoint
  loader accept it. New and legacy checkpoint namespaces remain disjoint.
- No online LLM call is allowed in training, evaluation, visualization, or tests.
  A committed semantic fixture must be marked `fixture_only=true` and rejected by a
  formal-data loader.
- Update `CHANGELOG.md` in every completed task-group commit. Do not mark `TASKS.md` or
  Roadmap O1 complete until the architect reviews all evidence.
- After each group, run `git diff --check`, record exact validation results, create one
  focused local commit, and stop for architect approval. Do not push or merge.
- Long A600 work is owner-only. The implementation must prepare the frozen command and
  output schema, but the implementing agent must not run it.

## 2. Frozen module and interface map

The following names and ownership are mandatory. Private helpers may be added within
the owning module only; responsibilities may not move across modules without returning
to O0-G.

| File | Public interface | Sole responsibility |
|---|---|---|
| `llm_mappo/optimization_observation.py` | `ObservationSchema`, `build_direct_goal_observation`, `build_no_goal_hint_observation`, `PlannerQueryCounter` | Build exact 613D physical observations and prove execution planner queries are zero |
| `llm_mappo/pure_motion_teacher.py` | `PureMotionQuery`, `PureMotionResult`, `PureMotionTeacher` | Produce deterministic root-action-conditioned motion preferences and validity |
| `llm_mappo/semantic_v3.py` | `SemanticViewV3`, `SemanticRecordV3`, `SemanticDatasetV3`, `TruncatedExponentialOOD` | Encode 61D semantic views, validate immutable records, retrieve 3D targets and reliability |
| `llm_mappo/optimization_student.py` | `O0StudentActor`, `O0CentralizedCritic`, `O0StudentOutput` | Implement the frozen 613D/61D actor, motion head, detached semantic adapter and critic |
| `llm_mappo/optimization_buffer.py` | `OptimizationRolloutBuffer`, `OptimizationBatch` | Preserve per-agent validity/reliability/calibration masks and exact KD tensors |
| `llm_mappo/shadow_state.py` | `ShadowSnapshotV1`, `ShadowStateAdapter`, `EventAddressedRandomness` | Canonically snapshot, restore, hash, fork and isolate environment/random state |
| `llm_mappo/reward_calibration.py` | `CalibrationSamplerV1`, `DeltaGEMA`, `RewardCalibrator`, `CalibrationResult` | Select 1/16 states, run paired H-step shadows and compute detached team confidence |
| `llm_mappo/optimization_checkpoint.py` | `O0CheckpointV1`, `save_o0_checkpoint`, `load_o0_checkpoint` | Strictly serialize the O0 Student, optimizer, schedule, EMA, schemas and provenance |
| `llm_mappo/optimization_logging.py` | `O0RunLogger`, `validate_o0_log_record` | Emit only the compact O0-F evidence schema and reject non-finite required values |
| `llm_mappo/optimization_training.py` | `OptimizationTrainingConfig`, `OptimizationTrainer`, `train_optimization` | Compose approved components without importing legacy AStarExpert or online LLM code |
| `train/train_optimization.py` | CLI `main` | Validate YAML and start only the optimization training path |
| `eval/evaluate_optimization.py` | CLI `main` | Strict optimization checkpoint evaluation and DirectGoal/NoGoalHint sensitivity |
| `train/collect_optimization_labels.py` | CLI `main` | Owner-only pilot/formal label generation; never imported by runtime training |
| `scripts/benchmark_reward_calibration.py` | CLI `main` | Run the frozen baseline/H4/H12 runtime and memory gate and write its manifest |

`Phase2Warehouse` receives one new keyword-only
`observation_schema: ObservationSchema = ObservationSchema.LEGACY_WAYPOINT_V1`.
The default must preserve byte-for-byte legacy observation construction. Only
`DIRECT_GOAL_V1` and `NO_GEOMETRIC_GOAL_HINT_V1` call the new builder. The optimization
trainer must reject the legacy schema; legacy Phase 3/4 trainers must reject or ignore
optimization-only configuration keys rather than silently switching schema.

## 3. Frozen optimization configuration

Create exactly:

- `configs/optimization/o1_functional_smoke.yaml` — CPU, one environment, seed `901`,
  128 real environment steps, rollout length 32, one optimizer epoch, batch size 32,
  DirectGoal, Fixed-KD mode, synthetic fixture only, `K_motion=12`, `H_reward=12`,
  sampler 1/16, and all frozen O0 parameters;
- `configs/optimization/o1_reward_calibration_smoke.yaml` — the same environment and
  method contract, CUDA-capable, 12 workers, containing benchmark configuration only;
  the benchmark CLI owns warmup/measurement/repeat overrides shown in section 11;
- `tests/fixtures/semantic_v3_smoke.jsonl` and
  `tests/fixtures/semantic_v3_smoke.manifest.json` — deterministic non-LLM records with
  `fixture_only=true`, a distinct fixture schema marker, at least four finite neighbors
  per scenario stratum used by tests, and no API credentials or model claim.

Both YAML files freeze environment ID `llm-mappo-medium-3ag-v1`, `n_agents=5`, dynamic
ingress interval 40, batch size `[4,8]`, queue size 8, task target 50, max steps 1000,
oracle task generation, deadlock threshold 180 and energy `1.10/0.30/0.80`. The
functional smoke is an interface test, not a learning result and may not be reported
in the paper.

## 4. Task group O1-A — DirectGoal observation and planner-query proof

**Files:**

- Create `llm_mappo/optimization_observation.py`.
- Modify `llm_mappo/phase2.py` only at the versioned observation dispatch boundary.
- Create `tests/test_optimization_observation.py`.

**TDD sequence:**

1. Add failing tests that assert exact width 613, exact slot ordering, normalized goal
   deltas, seven zero reserved slots, and unchanged legacy observations under the
   default constructor.
2. Add failing tests for `NO_GEOMETRIC_GOAL_HINT_V1`: exactly all nine former waypoint
   slots are zero while every other physical feature equals DirectGoal.
3. Add a throwing planner double plus `PlannerQueryCounter`; training/evaluation-style
   observation construction and action selection must complete with query count zero.
4. Implement the enum, pure builders and explicit dispatch. Any unknown schema raises
   `ValueError`; no silent legacy fallback.
5. Run:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pytest tests/test_optimization_observation.py tests/test_phase2.py -q
& "D:\Anaconda3\envs\py310\python.exe" -m flake8 llm_mappo/optimization_observation.py llm_mappo/phase2.py
```

**Acceptance:** New builders never instantiate or query `AStarPlanner`; the legacy
default regression is unchanged; all goal-hint variants have width 613.

## 5. Task group O1-B — Student, buffer, losses and strict checkpoint

**Files:**

- Create `llm_mappo/optimization_student.py`,
  `llm_mappo/optimization_buffer.py`, and
  `llm_mappo/optimization_checkpoint.py`.
- Create `tests/test_optimization_student.py`,
  `tests/test_optimization_buffer.py`, and
  `tests/test_optimization_checkpoint.py`.

**TDD sequence:**

1. Assert exact actor/critic tensor shapes, layer widths, five-action output, three-action
   motion output, three-score sigmoid output and 4-head centralized attention.
2. Assert gradient ownership: RL reaches physical/semantic encoders and action head;
   LLM MSE reaches only semantic encoder/head; A* KL reaches only physical encoder and
   motion-prior head. The detached 3→16 adapter blocks RL gradients into semantic
   encoder/head.
3. Assert buffer shapes for per-agent `m_A_valid`, shared `m_calib`, shared
   `c_A_reward`, per-agent semantic validity, shared OOD reliability and three-score
   targets. All-invalid A* and semantic batches return differentiable scalar zero.
4. Implement schedule `linear-env-step-v1` from actual real environment steps only:
   `lambda_A=.05*(1-t/B)` and `lambda_L=.10*(1-t/B)`, clipped at zero. Shadow, reset,
   evaluation and replay steps cannot advance it.
5. Assert `o0-student-checkpoint-v1` strict metadata, tensor schemas, EMA Welford and
   exponential state, sampler/version/provenance restoration, and round-trip equality.
   Missing/extra/incompatible fields fail before any tensor load. Legacy 1D/2D
   checkpoints and weight filling must fail.
6. Implement only enough to pass, then run:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pytest tests/test_optimization_student.py tests/test_optimization_buffer.py tests/test_optimization_checkpoint.py -q
& "D:\Anaconda3\envs\py310\python.exe" -m flake8 llm_mappo/optimization_student.py llm_mappo/optimization_buffer.py llm_mappo/optimization_checkpoint.py
```

**Acceptance:** No optimization class is accepted by the legacy checkpoint loader and
no legacy actor is accepted by `load_o0_checkpoint`.

## 6. Task group O1-C — Pure Motion Teacher

**Files:**

- Create `llm_mappo/pure_motion_teacher.py`.
- Create `tests/test_pure_motion_teacher.py`.

**TDD sequence:**

1. Build fixed tiny-map cases for every input/output field, tie-break level, direction
   order `[UP,RIGHT,DOWN,LEFT]`, root rank `[FORWARD,LEFT,RIGHT]`, shared budget and
   cache-key component.
2. Assert one bounded search preserves root branch identity; instrument OPEN pops and
   fail if total expansions exceed 512. Three independent full searches are forbidden.
3. Assert finite root costs equal window motion cost plus deterministic static
   cost-to-go, state-local min-max normalization, `tau_motion=1.0` Boltzmann transform,
   exact zeros for invalid roots and deterministic root argmax tie-breaking.
4. Assert the full fail-closed priority: dead, picking lock, mandatory toggle, at goal,
   no legal motion, budget exceeded, search failure, no progress, non-finite output.
   Invalid agents output an all-zero five-action vector and `valid_mask=false`; an
   all-invalid team gives scalar A* loss zero.
5. Add metamorphic purity tests: vary IDs, priorities, other tasks/goals, reward,
   reservation/yield/coordinator state, Student/LLM/calibration values and confirm
   identical labels/cache keys. Vary anonymous occupied coordinates or physical mask
   and confirm the query changes.
6. Static-scan this module to reject imports of `AStarExpert`, `rules`,
   `semantic_controls`, Student, reward calibration and reservation/coordinator code.
7. Run:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pytest tests/test_pure_motion_teacher.py -q
& "D:\Anaconda3\envs\py310\python.exe" -m flake8 llm_mappo/pure_motion_teacher.py
```

**Acceptance:** Target rewrite count, reservation/coordinator reads and illegal action
probability mass are all exactly zero; repeated identical queries are bitwise stable.

## 7. Task group O1-D — semantic-view-v3 and offline data boundary

**Files:**

- Create `llm_mappo/semantic_v3.py` and `train/collect_optimization_labels.py`.
- Create the two fixture files from section 3.
- Create `tests/test_semantic_v3.py` and `tests/test_optimization_label_cli.py`.

**TDD sequence:**

1. Assert the exact 61D order, robot-centered relative coordinates, Manhattan neighbor
   selection, frozen tie-break/sort, three-neighbor padding and mask. Anonymization
   removes IDs but retains the frozen task/state semantics.
2. Assert the immutable record schema, three `[0,1]` scores, whole-record validity,
   audit-only reasons, exact version/fingerprint/provenance fields and non-finite
   fail-closed behavior.
3. Reproduce k=3 inverse-square retrieval, exact-match averaging and one shared
   reliability. Fit the frozen truncated-exponential OOD formula from 61D features
   only; any attempt to pass 615D observations must fail.
4. Assert invalid dataset/query/target/OOD gives semantic KD weight zero, never a
   uniform/rule/legacy fallback. Per-robot validity stays separate while OOD reliability
   is shared for the record.
5. Assert a fixture loader accepts only test/smoke calls, while a formal loader rejects
   `fixture_only=true`. Assert runtime modules do not import the label-generation CLI
   or DeepSeek client.
6. Implement CLI dry-run/schema-validation paths only. A real API call requires the
   owner to supply `DEEPSEEK_API_KEY` at process scope, records no key, pins prompt/model/
   temperature/generator/parser, and implements whole-pilot Flash→Pro switching plus
   fingerprint pause; O1 must not invoke it.
7. Run:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pytest tests/test_semantic_v3.py tests/test_optimization_label_cli.py -q
& "D:\Anaconda3\envs\py310\python.exe" -m flake8 llm_mappo/semantic_v3.py train/collect_optimization_labels.py
```

**Acceptance:** No network call occurred; no key exists in repository files or command
history captured by artifacts; test fixtures cannot be mistaken for pilot/formal data.

## 8. Task group O1-E — deterministic shadow state and reward calibration

**Files:**

- Create `llm_mappo/shadow_state.py` and `llm_mappo/reward_calibration.py`.
- Add only the minimal explicit export/import hooks required by the frozen snapshot
  schema to `llm_mappo/phase2.py`, `llm_mappo/environment.py`, and wrappers.
- Create `tests/test_shadow_state.py` and `tests/test_reward_calibration.py`.

**TDD sequence:**

1. Assert canonical snapshot round-trip hash, independent mutable objects, all RNG/
   ingress/rule/metrics/adapter state, cache isolation and identical real continuation
   after restore. Static immutable configuration belongs in the config hash, not the
   mutable payload.
2. Assert two preconstructed shadow environments are restored from the same bytes and
   never share mutable references with the real environment or each other.
3. Assert `calibration-sampler-v1` exact UTF-8 key and SHA-256 modulo 16 behavior.
4. Assert event-addressed randomness depends only on the frozen event key; branch-local
   event counts cannot shift another event. Student actions use deterministic masked
   argmax; teacher-valid robots use Pure Motion argmax; teacher-invalid robots recompute
   Student argmax in the A* shadow current state.
5. Assert H-step discounted team returns, unilateral terminal behavior, no terminal
   bootstrap, detached critic bootstrap at nonterminal H, and no calibration gradient
   into critic/A*/Student action choice.
6. Assert `DeltaGEMA`: first 64 finite samples use Welford and return confidence zero;
   sample 65 uses pre-update state; later updates use decay .99, minimum scale 1e-3,
   `clip(DeltaG/sigma,0,1)`, and compute-before-update. Non-finite values are No-Go.
7. Assert Fixed and RC execute identical sampler/shadow/log/EMA paths. Their only loss
   difference is multiplication by `c_A_reward`; unselected states have A* KD zero in
   both modes.
8. Run:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pytest tests/test_shadow_state.py tests/test_reward_calibration.py tests/test_env.py tests/test_phase2.py -q
& "D:\Anaconda3\envs\py310\python.exe" -m flake8 llm_mappo/shadow_state.py llm_mappo/reward_calibration.py llm_mappo/phase2.py llm_mappo/environment.py
```

**Acceptance:** Real rollout state/hash/metrics are unchanged by calibration; branch
divergence does not change exogenous events; H4 exists only as benchmark diagnostics.

## 9. Task group O1-F — integration, logging and short smoke

**Files:**

- Create `llm_mappo/optimization_logging.py`,
  `llm_mappo/optimization_training.py`, `train/train_optimization.py`,
  `eval/evaluate_optimization.py`, `scripts/benchmark_reward_calibration.py`, and the
  two YAML files from section 3.
- Create `tests/test_optimization_logging.py`,
  `tests/test_optimization_training.py`, and
  `tests/test_reward_calibration_benchmark.py`.

**TDD sequence:**

1. Assert config rejects unknown/missing/frozen-field deviations, legacy observation,
   online LLM, H other than 12 for training, noncanonical schedule/EMA/OOD, and any
   attempt to use a test fixture outside functional smoke.
2. Assert compact logs match O0-F exactly: teacher summaries, effects, coverage buckets,
   failure counters, provenance, pollution counters and DirectGoal query counter.
   Full per-state teacher arrays are forbidden in the normal log sink.
3. Compose trainer dependencies explicitly. `optimization_training.py` must not import
   `AStarExpert`; the Pure Motion result is computed before reward calibration and
   cannot read `DeltaG` or Student disagreement.
4. Assert one rollout/optimizer update for MAPPO-DG, Fixed-KD and RC-KD with identical
   schedules and no NaN/Inf. Assert all-invalid and unselected batches remain valid
   zero-KD updates.
5. Run the frozen functional smoke once:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" train/train_optimization.py --config configs/optimization/o1_functional_smoke.yaml --output artifacts/optimization/o1_functional_smoke
```

6. The smoke must report: exactly 128 real steps, at least one deterministic sampler
   selection, at least one H=12 calibration call, planner queries=0, all pollution
   counters=0, illegal teacher mass=0, finite losses/logs and `fixture_only=true`.
   Failure is O1 No-Go; do not change seed/density/H to make it pass.
7. Run focused integration checks:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pytest tests/test_optimization_logging.py tests/test_optimization_training.py tests/test_reward_calibration_benchmark.py -q
& "D:\Anaconda3\envs\py310\python.exe" train/train_optimization.py --help
& "D:\Anaconda3\envs\py310\python.exe" eval/evaluate_optimization.py --help
& "D:\Anaconda3\envs\py310\python.exe" scripts/benchmark_reward_calibration.py --help
```

**Acceptance:** Short smoke is strictly an interface result; no convergence or method
performance claim may be made from it.

## 10. Task group O1-G — complete local validation and owner handoff

**Files:** Update `CHANGELOG.md`, this task package evidence section, and O1 checklists
only after every prior group is architect-approved. Do not change runtime behavior here.

1. Run the full local regression and static checks:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pytest -q
& "D:\Anaconda3\envs\py310\python.exe" -m flake8 rware llm_mappo eval train scripts figures/core
& "D:\Anaconda3\envs\py310\python.exe" visualize.py --help
& "C:\Users\28016\bin\rg.exe" -n "AStarExpert|reservation|coordinator|yielding" llm_mappo/pure_motion_teacher.py llm_mappo/optimization_training.py
git diff --check
git status --short
```

2. The search command must return no matches. Audit changed files against this packet,
   confirm no stable-route/experiment-budget/reward/formal-seed drift, and verify no
   credential is tracked.
3. Produce the owner handoff: task IDs, changed files, exact commands/results, smoke
   artifact path, unresolved checks, known risks, prohibited claims, commit IDs and
   worktree status.
4. Stop. O1 cannot pass until the owner supplies the A600 benchmark artifact and the
   architect evaluates both runtime and persistent-memory gates.

## 11. Owner-only A600 runtime/memory gate

The owner runs exactly the following after O1 local validation. Do not substitute a
local laptop result or silently change workers, repetitions, horizon or windows:

```powershell
& "D:\Anaconda3\envs\py310\python.exe" scripts/benchmark_reward_calibration.py `
  --config configs/optimization/o1_reward_calibration_smoke.yaml `
  --modes baseline h4 h12 --workers 12 --repeats 5 `
  --warmup-vector-steps 16 --measure-vector-steps 128 `
  --memory-warmup-windows 2 --memory-measure-windows 10 `
  --output artifacts/optimization/o1_reward_calibration_gate
```

Required output is `manifest.json`, `runtime.csv`, `memory.csv`, `branch_objects.csv`
and `summary.json`, all carrying config hash, code commit, device, CUDA/PyTorch/platform,
mode, horizon, worker/repeat/window settings and exit status. `summary.json` must expose
the median H12/baseline runtime ratio and the memory regression inputs.

Go requires median H12/baseline runtime ratio `<=3.0`, no non-finite/interface failure,
no persistent memory growth above `max(64 MiB,5%)` with Spearman `rho>=0.80`, and no
monotonic branch/cache object growth. H4 is diagnostic only. Any failure is O1 No-Go
and returns to O0; H12 may not be shortened.

## 12. Evidence ledger

Fill this table only after executing a group. Evidence paths and commit IDs must be
literal, not “passed locally”.

| Group | Focused tests | Regression/static checks | Commit | Architect approval |
|---|---|---|---|---|
| O1-A | pending | pending | pending | pending |
| O1-B | pending | pending | pending | pending |
| O1-C | pending | pending | pending | pending |
| O1-D | pending | pending | pending | pending |
| O1-E | pending | pending | pending | pending |
| O1-F | pending | pending | pending | pending |
| O1-G local | pending | pending | pending | pending |
| Owner A600 gate | owner-run pending | owner-run pending | artifact hash pending | pending |
