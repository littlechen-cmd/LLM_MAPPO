# Project Task List

Update this file whenever a task begins, completes, or is blocked. Major completed milestones require a Git commit.

## Constitution Chapter 8 Roadmap Tracking

This section mirrors the task IDs, order, and meaning in `CONSTITUTION.md`
Chapter 8 one-for-one. Chapter 7 defines the acceptance criteria. Mark a Gate
passed only after every Chapter 8 task is complete, its evidence is retained,
and the corresponding Chapter 7 criteria are satisfied.

### Gate G0 — Current Training Artifact Audit

- [x] **G0-1** Inspect the 800-episode run's `summary.json`, `episodes.csv`,
  `updates.csv`, `checkpoint_final.pt`, and TensorBoard event file.
- [x] **G0-2** Confirm no NaN/Inf, abnormal exit, missing required artifact, or online
  LLM call. The final 100 episodes reached 1.000 completion and success with
  zero deaths and terminating deadlocks; retain 17.92 collisions per episode,
  A* diagnostic counts, and the copied summary's stale checkpoint path as
  findings for G2 rather than formal method evidence.
- [x] **G0-3** Record the result only as a G0 feasibility pass, not a formal performance
  conclusion.

### Gate G1 — Load, Evaluate, And Replay Loop

- [x] **G1-1** Repair Phase 4 `semantic_dim=2` checkpoint loading by inferring legacy
  tensor shapes and persisting semantic width in new checkpoints.
- [x] **G1-2** Pass Phase 3 single-semantic, Phase 4 dual-semantic, and incompatible
  checkpoint regression tests.
- [x] **G1-3** Complete a held-out seed-0 single-episode evaluation smoke and
  deterministic replay with the real Phase 4 checkpoint; retain the evaluation
  JSON, 447-step trace, replay summary, and machine-readable G1 audit.
- [x] **G1-4** Verify `eval/evaluate_phase3.py` and `visualize.py` process Phase 4
  checkpoints; nested evaluation output creation and visualization-controller
  loading have regression coverage.

### Gate G2 — Environment Risk Review

- [ ] **G2-1** Analyze false no-path results and livelock caused by A* goal reservation
  lasting for the full planning horizon. This item is intentionally postponed
  while charging is handled first; no A* behavior changes are in the G2
  charging patch.
- [x] **G2-2** Review long-run charging returns, congestion, AGV deaths, and task
  interruption using the accepted 800-episode checkpoint and targeted replays.
  The original setting rarely exposes charging, while stronger unmatched
  energy pressure reveals unreliable charger arrival/wait behavior.
- [ ] **G2-3** Add a configurable battery-cost scale and charging hysteresis, then pilot
  scales `1.00/1.25/1.50`, refined scales `1.10/1.15/1.20`, and an earlier
  `0.30` trigger (including refined `1.05/1.10` scales). No setting is frozen
  from this old-checkpoint diagnostic: the original-threshold `1.10` preserves
  zero energy deaths but does not provide sufficient exposure, while every
  higher-exposure candidate exposes energy deaths and/or deadlocks.
  Run three matched 200-episode retraining groups before closing this task:
  control `1.00/0.20/0.80`, early-charge candidate `1.10/0.30/0.80`, and
  high-consumption candidate `1.20/0.20/0.80`, with all other settings, seed
  rotation, teachers, rewards, and PPO parameters held fixed.
- [x] **G2-4** Add pre-freeze metrics for low-battery triggers, charger arrivals,
  charged events, waiting, task recovery, minimum battery, and energy deaths.
- [x] **G2-5** After changing charging behavior, rerun the applicable regression suite
  before protocol freeze (`138 passed` for the full suite on 2026-08-16).
- [x] **G2-6** Record the charging limitation and evidence: the accepted checkpoint was
  trained with insufficient charging exposure, so its stress-test performance
  cannot validate learned charging. Candidate settings require matched pilot
  retraining before G3 freeze and cannot support teacher-effect claims yet.

### Gate G3 — Formal Protocol Freeze

- [ ] **G3-1** Freeze the code commit, four group configurations, environment contract,
  label-dataset hash, training seeds, equal environment-step budget, and held-out
  `10 seeds x 20 episodes` evaluation protocol.
- [ ] **G3-2** Freeze runtime A* waypoint and training A* KL as separate factors, with
  the four core experiment names defined in the Constitution.
- [ ] **G3-3** Freeze battery-cost scale, charging rate, trigger/release hysteresis
  thresholds, task interruption/recovery semantics, and rule-layer safety
  boundaries.
- [ ] **G3-4** Freeze the shared log schema, run manifest, checkpoint-selection rule,
  failure/retry policy, and formal experiment data/figure pipeline.

### Gate G4 — Small-Budget Pilot

- [ ] **G4-1** Run all four core groups end-to-end with the same small environment-step
  budget and matched seeds.
- [ ] **G4-2** Inspect losses, throughput, GPU memory, logs, checkpoints, evaluation
  scripts, and zero-online-LLM compliance for every pilot group.
- [ ] **G4-3** Confirm all four pilots receive sufficient matched low-battery/charging
  exposure and pass the frozen charger-arrival, task-recovery, energy-death,
  and charging-congestion safety checks.
- [ ] **G4-4** Use the matched learning curves to set one formal interaction budget for
  all four groups without group-specific post-hoc tuning.

### Gate G5 — Core Multi-Seed Training

- [ ] **G5-1** Complete all four core groups with at least five training seeds per group.
- [ ] **G5-2** Verify every run follows the frozen protocol and handle failed runs only
  through the preregistered failure/retry rule.
- [ ] **G5-3** Confirm no group received a result-dependent hyperparameter or budget
  change after outcomes were observed.

### Gate G6 — Independent Evaluation And Statistics

- [ ] **G6-1** Complete held-out main evaluation and robustness evaluation for all core
  groups.
- [ ] **G6-2** Generate statistical summaries, confidence intervals, effect sizes,
  priority/task metrics, safety results, and representative failure cases.
- [ ] **G6-3** Map every intended paper claim to supporting evidence or an explicit
  limitation before drafting results.

### Gate G7 — Paper Ready

- [ ] **G7-1** Generate every reported figure and table from frozen raw logs through
  versioned scripts.
- [ ] **G7-2** Audit terminology and values across method text, configurations, code,
  tables, and figures for consistency.
- [ ] **G7-3** Complete the internal review before writing the conclusions into the
  English paper.

## Ongoing Experiment Observability

- [x] Provide deterministic visual replay, GIF capture, diagnostic traces, and
  TensorBoard metrics for training and evaluation artifacts.
- [ ] For every subsequent training or gate run, provide a TensorBoard command
  and at least one deterministic replay command before judging the result.

## Training Throughput Optimization

- [x] Collect synchronous rollouts from configurable independent environments.
- [x] Replace Windows thread workers with 12 persistent spawned environment
  processes and validate Phase 4 throughput and deterministic rollout behavior.
- [x] Cache unchanged A* teacher decisions and streamline reservations/search.
- [x] Build the offline semantic nearest-neighbour index once at load time.
- [x] Append training-update metrics instead of rewriting `updates.csv`.
- [x] Validate compatibility, rollout correctness, and a multi-environment smoke run.
- [x] Prevent base RWARE request renewal from conflicting with dynamic task ingress.
- [x] Support reproducible CPU/CUDA device selection for accelerator training.

## Charging Layout And Coordination

- [x] Expand the dynamic medium warehouse with a two-cell outer highway ring
  (`20x16` to `24x20`) and place two charging stations in each outer corner.
- [x] Permit charging-station capacity above fleet size; publish per-station
  occupancy and reservation metadata for deterministic replay diagnostics.
- [x] Prevent full-battery idle AGVs from targeting charging stations and
  allocate distinct stations to low-battery AGVs by battery urgency.
- [ ] Evaluate the proposed reward-led charging policy separately. Do not
  remove the current low-battery safety target until a controlled local
  ablation demonstrates no energy-safety regression.

## Conditional Runtime Interaction Extensions

- [ ] After the core paper evidence chain is complete, design and evaluate an
  event-triggered natural-language priority adjustment interface for urgent
  cargo requests. Reuse the existing LLM parser and atomic rule layer, add an
  execution-loop user input boundary, deterministic fallback, audit logs, and
  explicit online-call latency/cost metrics; do not treat it as part of the
  zero-online-LLM core method.

## Conditional Learned-Charging Extension

- [ ] After the core evidence chain is complete, add a high-level
  `continue_task/charge` policy decision while retaining an emergency forced-
  charging safety floor; do not remove task consistency, station capacity, or
  other hard safety checks.
- [ ] Compare fixed-threshold charging, learned charging with an emergency
  shield, and an unshielded diagnostic under matched energy pressure and seeds.
  Report learning-curve AUC, time to charging-success threshold, charger
  arrival, charged events, task recovery, congestion, throughput, and deaths.
