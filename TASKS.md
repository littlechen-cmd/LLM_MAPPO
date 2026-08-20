# Project Task List

Update this file whenever a task begins, completes, or is blocked. Major completed milestones require a Git commit.

## Execution Ownership

Training, multi-seed evaluation, long replay, and other clearly time-consuming runs are
executed by the project owner. The coding collaborator prepares implementations,
short deterministic tests, commands, and artifact audits, and must not start a long run
without a new explicit request from the project owner.

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

- [ ] **G2-1** Remove false no-path results and rotation livelock caused by full-horizon
  A* terminal reservations, without weakening vertex, edge-swap, or action-mask safety.
  The implementation is limited to bounded terminal occupancy, an explicit NOOP wait,
  truthful partial-path status, and state-dependent stationary occupancy; CBS and task
  reassignment remain out of scope.
  - [x] **G2-1a** Freeze deterministic reproducers for shared terminals, temporary
    no-path fallback, explicit waiting, persistent occupancy, vertex conflicts, and
    head-on edge swaps. Instrument reached-goal, partial-path, terminal-conflict,
    reservation false-no-path, wait, replan, rotation-livelock, state-deadlock,
    expanded-node, and planning-time metrics.
  - [x] **G2-1b** Return a time-expanded internal plan with its first action and
    `reached_goal`; add a true NOOP successor; default successful terminal occupancy
    to two steps; reserve dead agents to the horizon, picking locks for their remaining
    duration, and toggle/blocked fallbacks for only the next executed step.
  - [x] **G2-1c** Pass the focused planner tests, existing vertex/edge safety tests,
    full regression suite, and a short local smoke. Do not start training or a
    multi-seed evaluation as part of this item. The implementation passes 158 tests,
    Flake8, and the two-step diagnostic smoke on 2026-08-20.
  - [ ] **G2-1d** Project owner runs paired legacy/fixed diagnostics on calibration
    seeds `300–309`, 20 episodes per seed, using identical environment settings.
    Formal held-out seeds `200–209` remain untouched.
  - [ ] **G2-1e** Audit the long-run artifacts. Adopt the fix only if controlled false
    no-path and rotation-wait cases are eliminated, reservation safety violations stay
    at zero, completion drops by at most 1 percentage point, collisions/deadlocks/
    energy deaths do not increase, and p95 planning time is at most 1.20× legacy.
    Otherwise retain legacy behavior and record the risk as a paper limitation.
- [x] **G2-2** Review long-run charging returns, congestion, AGV deaths, and task
  interruption using the accepted 800-episode checkpoint and targeted replays.
  The original setting rarely exposes charging, while stronger unmatched
  energy pressure reveals unreliable charger arrival/wait behavior.
- [x] **G2-3** Add a configurable battery-cost scale and charging hysteresis, then pilot
  scales `1.00/1.25/1.50`, refined scales `1.10/1.15/1.20`, and an earlier
  `0.30` trigger (including refined `1.05/1.10` scales). No setting is frozen
  from this old-checkpoint diagnostic: the original-threshold `1.10` preserves
  zero energy deaths but does not provide sufficient exposure, while every
  higher-exposure candidate exposes energy deaths and/or deadlocks.
  Four matched 200-episode retraining groups were completed:
  control `1.00/0.20/0.80`, early-charge candidate `1.10/0.30/0.80`, and
  high-consumption candidate `1.20/0.20/0.80`, plus intermediate candidate
  `1.10/0.25/0.80`, with all other settings, seed rotation, teachers, rewards,
  and PPO parameters held fixed. The selected core setting is
  `1.10/0.30/0.80`: its diagnostic held-out evaluation reached 1.000 task
  completion, 113.46 completed tasks/1000 steps, 20% episodes with charging,
  and zero energy deaths. `1.10/0.25/0.80` was more efficient (115.47 tasks/1000
  steps) but was rejected because charging exposure fell to 1% and one held-out
  energy death occurred. Seeds `0–9` are calibration evidence and are excluded
  from formal evaluation.
- [x] **G2-4** Add pre-freeze metrics for low-battery triggers, charger arrivals,
  charged events, waiting, task recovery, minimum battery, and energy deaths.
- [x] **G2-5** After changing charging behavior, rerun the applicable regression suite
  before protocol freeze (`138 passed` for the full suite on 2026-08-16).
- [x] **G2-6** Record the charging limitation and evidence: the accepted checkpoint was
  trained with insufficient charging exposure, so its stress-test performance
  cannot validate learned charging. Candidate settings require matched pilot
  retraining before G3 freeze and cannot support teacher-effect claims yet.

### Gate G3 — Formal Protocol Freeze

- [ ] **G3-1** Freeze the code commit, four core group configurations, environment
  contract, label-dataset hash, formal training seeds
  `7/17/27/37/47/57/67/77`, diagnostic seeds `7/17/27`, the matched G4 step
  budget, formal-budget selection rule, and held-out `10 seeds x 20 episodes`
  evaluation protocol.
  The provisional manifest fixes seeds, deterministic evaluation, final-checkpoint
  selection, the dataset hash, a `150,000`-step G4 budget, and untouched formal
  evaluation seeds `200–209`; the final Git commit and G2-1d/e remain open. The
  2026-08-20 pre-freeze audit is recorded in `plan/review/g3-pre-freeze-audit.md`;
  it verifies the former five-seed static inputs but does not remove the listed
  blockers or satisfy the new eight-seed Q2 protocol.
- [x] **G3-2** Freeze runtime A* waypoint and training A* KL as separate factors, with
  the four core experiment names defined in the Constitution. Phase 4 now exposes
  independent A* KL and offline LLM teacher switches; no-LLM groups use fixed-zero
  semantic motion inputs while preserving the same two-dimensional architecture.
- [x] **G3-3** Freeze battery-cost scale, charging rate, trigger/release hysteresis
  thresholds, task interruption/recovery semantics, and rule-layer safety
  boundaries. Core groups use `1.10/0.30/0.80`; charging remains a fixed rule-layer
  safety mechanism, and `1.20/0.20/0.80` remains evaluation-only stress.
- [x] **G3-4** Freeze the shared log schema, run manifest, checkpoint-selection rule,
  failure/retry policy, and formal experiment data/figure pipeline. Formal episode
  logs include cumulative environment steps and the plotting scripts do not fabricate
  data. Manifest v2, the strict aggregator, table schema, figure contract, and tests
  now require the eight-seed core matrix and emit per-seed/Aggregated normalized
  throughput AUC artifacts.
- [x] **G3-5** Implement and freeze the fair-comparison boundary, configuration,
  budget, and allowed claims for `QMIX-WP`, `MAPPO-WP+A*KD+RuleKD`,
  `MAPPO-WP+A*KD+ShuffleKD`, `MAPPO-NoWP`, and `Heuristic-Dispatcher+A*`.
  If QMIX cannot
  share the frozen observation/action/safety contract, preregister `IPPO-WP` or
  `VDN-WP` before G5; do not omit the external MARL baseline.
  The implementation freezes QMIX-WP's shared environment interface, deterministic
  RuleKD/ShuffleKD derivation, zeroed-width-preserving NoWP, and a named heuristic
  dispatcher+A* entry point. G4-5 remains the required end-to-end smoke gate.
- [ ] **G3-6** Implement and freeze two evaluation-only unseen layouts: one
  narrow-aisle layout and one central-bottleneck/cross-aisle layout. Record their
  map files or generators, hashes, environment IDs, and unified observation/action
  compatibility before viewing formal outcomes.
- [x] **G3-7** Freeze completed tasks per 1000 environment steps as the primary
  utility metric, environment-step learning-curve AUC as the primary sample-
  efficiency metric, collision and energy-death rates as safety constraints, the
  five-comparison confirmatory hypothesis family, and a blinded label-audit protocol
  with 100 held-out states and two independent raters. The protocol, fixed quotas,
  sampling seed, scoring scale, and reporting rules are recorded in
  `plan/label-audit-protocol.md` and manifest v2.

### Gate G4 — Small-Budget Pilot

- [ ] **G4-1** Run all four core groups end-to-end with the same small environment-step
  budget and matched seeds.
- [ ] **G4-2** Inspect losses, throughput, GPU memory, logs, checkpoints, evaluation
  scripts, and zero-online-LLM compliance for every pilot group.
- [ ] **G4-3** Confirm all four pilots receive comparable low-battery/charging exposure
  under the core setting and pass the frozen charger-arrival, task-recovery,
  energy-death, and charging-congestion safety checks; use the common energy-stress
  scenario for high-exposure behavior rather than claiming autonomous charging.
- [ ] **G4-4** Use the four core matched learning curves to set one formal interaction
  budget for every formal learning group, record the single pre-G5 protocol
  amendment, and avoid group-specific post-hoc tuning.
- [ ] **G4-5** Smoke-test the required external/planning baselines, semantic controls,
  NoWP diagnostic, both unseen layouts, eight-seed aggregation contract, and label-
  audit data entry end to end. Use only a G3-preregistered fallback if a smoke fails.

### Gate G5 — Formal Multi-Seed Training

- [ ] **G5-1** Project owner completes the four core groups, `QMIX-WP`, and
  `MAPPO-WP+A*KD+RuleKD` with the eight matched formal training seeds per group.
- [ ] **G5-2** Verify every run follows the frozen protocol and handle failed runs only
  through the preregistered failure/retry rule.
- [ ] **G5-3** Confirm no group received a result-dependent hyperparameter, baseline
  selection, or budget change after outcomes were observed.
- [ ] **G5-4** Project owner completes `MAPPO-WP+A*KD+ShuffleKD` and
  `MAPPO-NoWP` with the
  three fixed diagnostic seeds each; restrict these runs to mechanism direction,
  variance, and failure-mode analysis rather than significance claims.

### Gate G6 — Independent Evaluation And Statistics

- [ ] **G6-1** Complete held-out main evaluation for the core groups, `QMIX-WP`,
  `MAPPO-WP+A*KD+RuleKD`, and `Heuristic-Dispatcher+A*`, plus the frozen two-layout
  zero-shot and load/fleet/energy robustness evaluations for every required group.
- [ ] **G6-2** Generate statistical summaries, confidence intervals, effect sizes,
  priority/task metrics, safety results, and representative failure cases.
- [ ] **G6-3** Map every intended paper claim to supporting evidence or an explicit
  limitation before drafting results.
- [ ] **G6-4** Complete the two-rater blind audit of at least 100 frozen states,
  analyze LLM/rule/shuffled semantic labels, and report A* calls, replans, failures,
  partial paths, expanded nodes, cache hits, and planning/control-loop P50/P95/P99
  latency.

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
