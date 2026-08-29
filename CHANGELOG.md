# Changelog

## 2026-08-29

- Added: O1 shadow restore failures now report the first mismatching component and field
  path, expected/actual overall and component hashes, and compact ndarray dtype/shape/hash
  summaries while preserving fail-closed behavior.
- Added: An H12 CPU stress regression crosses multiple workers, rollout updates, dynamic
  ingress and episode reset boundaries without changing the frozen O1 Gate parameters.

## 2026-08-28

- Fixed: P1 server preflight now parses the Chinese-localized `lscpu` model-name field
  (`型号名称`) emitted by the shared Linux host, while retaining English parsing and all
  fail-closed resource checks.
- Validated: P1-A through P1-G pass local cross-platform regression, Flake8, CLI, YAML,
  package-build and active-runtime safety audits; P1 remains in progress pending owner Linux
  environment and GPU smoke evidence.
- Added: A literal, owner-only P1 short-smoke runbook captures read-only preflight, five-sample
  eligibility, CUDA logical-device binding, a 128-step functional smoke and artifact hashes
  without starting O1 or O2.

This file records meaningful project engineering and experiment-governance
changes. Completing a `TASKS.md` subtask requires a matching entry here.

## 2026-08-28

- Added: P1-F introduces the owner-only tmux wait-to-O1 launcher, which pins GPU 0 before
  the benchmark child, retains the lease through the Gate, and only reports the O2 handoff.
- Added: P1-E makes O1 runtime and memory evidence atomic and identity-bound, permits only
  declared infrastructure recovery, and writes an O2 receipt only after a complete Gate Go.
- Changed: P1-D makes the normal O1 Linux CUDA Gate baseline/H12-only, isolates H4 as a
  failed-Gate diagnostic, and binds evidence to GPU 0, CUDA visibility, preflight and environment reports.
- Added: P1-C provides fail-closed Linux server inventory, five-sample/48-hour resource
  preflight, project-only GPU lease, immutable artifact writes, and read-only CLI modes.
- Added: P1-B freezes the Linux Python 3.10.19/cu128 dependency contract, a read-only
  fail-closed verifier, and the owner-only user-prefix installation runbook.
- Changed: P1-A aligns the experiment protocol and machine-readable manifest with the
  owner-run Linux sequence `P1→O1→O2`, physical GPU 0, and separate O2 launch boundary.
- Added: The approved P1 specification freezes the optimization route's Ubuntu
  Python/CUDA environment, shared-server preflight, GPU binding and lease, baseline/H12
  O1 Gate, diagnostic-only H4 path, resumable evidence, and mandatory P1→O1→O2 order.

## 2026-08-27

- Validated: The project owner approved O3 topology/interface readiness after a fresh
  34-test focused gate, 257-test full regression, Flake8, CLI, and leakage scan; no
  learned-policy performance or cross-topology generalization claim is implied.
- Completed: O3-A through O3-F freeze two evaluation-only unseen topologies and
  pass 34 focused tests, 257 full tests, Flake8, CLI, static leakage, and contract
  audits; learned-policy performance and the owner readiness decision remain pending.
- Added: O3-E central fail-closed guards keep both unseen topology IDs and their
  provenance out of Phase 2/3/4 and optimization training, offline label collection,
  and semantic/OOD reference fitting; nine leakage regressions pass.
- Changed: Optimization resource governance now keeps the 65-run formal matrix,
  reduces O2 from nine to six calibration runs, requires a revised baseline/H12
  O1 Gate as the fail-fast prefix of the same owner A600 job, and reserves H4 for
  failure-only diagnostics; runner implementation remains pending.
- Changed: Canonical-core held-out seeds are now the mandatory fixed-topology
  robustness evidence; O3 learned-policy performance is an optional, precommitted
  exploratory stress test with no cross-topology generalization claim.
- Added: A three-perspective optimization resource replan records the remaining
  `71` A600 learning runs plus one short CUDA Gate and keeps the formal 65-run
  training budget unchanged.
- Added: O3-D verifies byte-deterministic reset and 41-step ingress trajectories for
  seeds `9301/9302`, 613D/61D interfaces, zero planner queries, Pure Motion Teacher
  provenance, shadow restore, and shared collision/charging/load safety behavior.
- Added: O3-C introduces immutable `TopologySpec` metadata, packaged `v2` map
  resources, source/effective hash fail-closed checks, and a temporary-registration
  evaluation factory that leaves held-out IDs unavailable to ordinary `gym.make`.
- Fixed: O3 map dimensions now match the canonical core at `20×24`
  (width×height); the owner-approved `v2` maps are deterministic rotations that
  preserve density and graph certificates without adding aspect ratio as a confound.
- Added: O3-B records the initial `v1` narrow-passage and central-cross previews,
  subsequently superseded by the dimension-corrected `v2` maps, together with
  ASCII maps, graph certificates, explicit charging-station coordinates, deterministic
  PNG previews, and static format/structure regression tests without policy execution.

## 2026-08-26

- Added: Root `terminology.md` aligns planning, topology, hashing, experiment-isolation
  and multi-teacher concepts using plain-language explanations, professional definitions
  and their project-specific implications.
- Changed: Replanned optimization dependencies so O3 topology/interface work can run
  while O1 waits for the owner-only A600 gate; O2 remains blocked and unseen-topology
  performance remains deferred until E2 after D1.
- Added: The unique O3 feature spec freezes two 5-AGV evaluation-only loaded-transport
  topologies, explicit files and IDs, dual hashes, graph certificates, owner map-preview
  approval, code-level leakage guards, and non-performance validation boundaries.
- Added: The O1 owner-only A600 gate now runs the frozen baseline/H4/H12
  comparison with matching rollout updates, fresh repeat processes, persistent-memory
  windows, fail-closed CUDA checks, and complete machine-readable evidence artifacts.
- Fixed: Paired shadow rollouts now use disposable Pure Motion Teacher caches, leaving
  the real rollout Teacher cache unchanged; H4 remains diagnostic-only while H12 is
  the sole formal calibration horizon.
- Changed: O1 local implementation and validation are complete; the phase now waits
  for the research owner to run the frozen A600 runtime and persistent-memory gate.

## 2026-08-25

- Added: O1-F composes the isolated optimization runtime, strict functional-smoke
  configuration, compact evidence logger, semantic fixture boundary, and frozen CLI
  entry points; the 128-step smoke is interface-only and makes no performance claim.
- Added: O1-E introduces canonical JSON `o0-shadow-state-v1` snapshots,
  preconstructed branch restore, stateless `crn-v1` dynamic-ingress events, shared
  `calibration-sampler-v1`, and the frozen Welford/EMA reward-calibration state.
- Added: O1-D adds strict three-score semantic parsing, `semantic-view-v3` 61D
  encoding, validity-only offline retrieval, truncated-exponential OOD reliability,
  and a zero-network owner-only label validation CLI.
- Added: O1-C implements the deterministic `pure-motion-astar-v1` Teacher with a
  shared bounded search, root-conditioned soft motion prior, exact-query cache, and
  fail-closed per-robot results.
- Added: O1-B adds the isolated 613D/61D O0 Student, per-agent optimization buffer,
  real-environment-step schedule, and strict `o0-student-checkpoint-v1` loader.
- Added: O1-A introduces versioned DirectGoal and NoGoalHint physical observations,
  zero planner-query instrumentation, and regression coverage while preserving the
  legacy waypoint schema as the default.
- Changed: The research owner gave final written approval to the complete O0 contract;
  O0 is closed and O1 role-alignment implementation is now the active phase.
- Added: O0-G records the zero-deviation architecture audit and freezes the sole O1
  implementation task package, including module interfaces, TDD order, short smoke,
  strict legacy isolation, and the owner-only A600 runtime/memory gate command.
- Changed: O0-A through O0-F are recorded as individually approved; O0 remains pending
  until the research owner gives final written approval of the complete frozen contract.
- Fixed: Replaced the non-executable WinGet Links ripgrep target with the verified
  `C:\Users\28016\bin\rg.exe` executable path.
- Changed: Windows ripgrep searches must directly invoke the canonical approved
  executable; bare `rg` and repair or diagnosis of the Codex WindowsApps copy are
  prohibited.
- Added: O0-F freezes compact teacher/effect logs, the 9-run O2 gate, the
  65-run E1/E2 evidence matrix, a 74-run total budget, seven seed-level primary
  contrasts, Holm correction, confidence intervals, effect sizes, and claim limits.
- Changed: Optimization semantics now use a continuous three-score DeepSeek rubric;
  deterministic RuleKD-v3 is isolated as a baseline, ShuffleKD-v3 is a stratified
  fixed-point-free derangement, and the old optimization NoWP group is renamed
  NoGoalHint to avoid misrepresenting execution-time A* evidence.
- Changed: Synchronized the Python/tool contract, P0 experiment protocol, governance
  manifest, label-audit protocol, Roadmap, TASKS, figure/table manifests, and
  traceability matrix; mapped and removed the two untracked proposal inputs.
- Added: O0-E freezes the 613D DirectGoal physical observation, 61D three-score
  semantic branch, detached late-fusion Student, motion-only prior loss,
  environment-step linear teacher schedule, strict 3D checkpoint isolation, and
  the evidence gate required before claiming A*-free Student execution.
- Added: O0-D freezes a leakage-resistant 61-dimensional `semantic-view-v3`,
  five-point three-score anchors, truncated-exponential OOD reliability selected
  in the semantic view space, Flash-to-Pro pilot gating, immutable 60/800 label
  datasets, fingerprint isolation, and formal dataset-level No-Go thresholds.
- Changed: All project Python work must directly invoke the existing canonical
  `D:\Anaconda3\envs\py310\python.exe`; PATH lookup, Conda activation, and virtual
  environment creation or modification are prohibited.
- Added: O0-C freezes a canonical branch-state snapshot, H=12 paired Student/A*
  shadows, event-addressed common random numbers, detached return calibration, a
  deterministic shared 1/16 calibration mask for Fixed/RC-KD, EMA state, and the
  A600 3x runtime/persistent-memory No-Go gate.
- Added: O0-B freezes `pure-motion-astar-v1` as an independent, anonymous-occupancy
  bounded A* teacher with a 12-step motion window, 512-expansion budget, motion-only
  support, deterministic tie-break, binary per-agent validity, and fail-closed output.
- Changed: O0-B replaces fixed 0.85/0.15 smoothing with a root-action-conditioned
  short-horizon planning-cost prior, using one provenance-preserving bounded search,
  state-wise min-max normalization, and a fixed `tau_motion=1.0` Boltzmann transform.
- Added: O0-A now records the P0 field-level teacher-to-policy data flow, twelve
  confirmed A* and semantic pollution gaps, the historical label/checkpoint
  provenance, and a claim-by-claim disposition of both research inputs.
- Added: The O0 specification scopes a Pure Motion A* teacher, H=12 paired-shadow
  reward calibration, route-isolated three-dimensional LLM semantics, and an explicit
  owner approval gate before implementation, with mandatory review pauses between
  task groups.

## 2026-08-24

- Added: P0 migration audit and ignored safety backup uniquely classify and protect
  every pre-existing workspace change before baseline integration.
- Changed: Replaced the root Constitution and legacy G2/G3 task entrances with the
  dual-route specs, route-aware protocol/manifest/TASKS, and indexed historical archive.
- Added: Behavior-neutral A* diagnostics now preserve the planner-to-execution action
  pipeline, distinct conflict reasons, explicit legacy-schema availability, and count
  conservation while retaining the default NOOP coordinator behavior.
- Added: Static ASCII layout preview validates geometry, symbols, cell size and Pillow,
  writes deterministic PNG output, and exits before environment or policy setup.
- Removed: Two untracked rejected-layout drafts and the obsolete active 8-AGV planning
  contract; O3 will design unseen topologies from scratch.
- Verified: A clean P0 checkout installs through an isolated editable prefix and passes
  184 tests, focused A*/layout/config checks, 30 YAML parses, CLI smokes, and Flake8.
- Added: The stable-route engineer handoff freezes S1 scope, permissions, evidence,
  artifact isolation, owner-run boundaries, acceptance thresholds, and prohibited claims.
- Completed: P0 governance, shared diagnostics, reproducibility validation, and the
  common baseline for the optimization and stable branches.

## 2026-08-20

- Added: A complete G3 architecture task package now fixes ownership, evidence,
  the provisional/final freeze sequence, and the implementable G3-6 unseen-layout
  contract without starting experiments.
- Added: G3-5 external MARL, semantic-control, NoWP, and heuristic baseline
  configurations and their reproducible implementation boundaries.
- Added: Eight-seed formal analysis, normalized learning-curve AUC, frozen
  confirmatory comparisons, and the blind semantic-label audit protocol.
- Changed: The project experiment contract now targets a realistic CAS Q2 method
  paper evidence chain while retaining Q1 as a stretch objective.
- Fixed: Bounded A* terminal reservations, explicit waiting, and partial-path
  diagnostics reduce false no-path and rotation-livelock risks pending paired audit.
- Fixed: Experiment checkpoints can be reproduced from a clean checkout.
- Changed: Formal architecture governance is split between project owner, core
  architect, and project engineer; future completed task items require changelog
  updates in the same commit.

## 2026-08-19

- Added: Battery-cost scale experimentation for charging-exposure calibration.

## 2026-08-17

- Added: The intermediate `1.10/0.25/0.80` charging candidate and its matched
  retraining configuration.

## 2026-08-16

- Added: The project Constitution, charging calibration workflow, charging
  hysteresis, and charging observability metrics.

## 2026-08-15

- Added: Persistent process-based environment workers and parallel CUDA Phase 4
  training support.
- Added: Offline dual-semantic LLM distillation with zero online LLM calls during
  training and execution.

## 2026-08-11

- Added: Phase 3 R2 robustness workflow.

## 2026-08-10

- Added: Phase 3 dual-head MAPPO baseline and multi-agent yielding recovery.

## 2026-07-31

- Added: Time-reserved multi-agent A*, medium-map warm-start validation, checkpoint
  visualization, and the small-map A*-MAPPO curriculum.

## 2026-07-30

- Added: Staged training policy documentation and Phase 2 waypoint-reward
  comparison support.
