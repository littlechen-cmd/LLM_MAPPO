# Plan — Phase 1 MAPPO Baseline

## Task group 1: Map, facility and workload inspection

- [x] Inspect the current RWARE medium and large maps, their usable traversable
  cells and existing workstation/shelf geometry.
- [x] Produce three proposed profiles: `medium-3ag-low-workload`,
  `medium-5ag-low-workload` and `large-5ag-high-workload`; each must state
  exact AGV count, task-ingress parameters, charging-station coordinates and
  picking-station coordinates.
- [x] Enforce chargers = AGVs and picking stations = `floor(AGVs / 2)`.
- [x] Generate one layout preview per profile using a four-class A/B/C/D colour
  legend, without per-item labels or identifiers.
- [x] Owner approved the three previews and profile table. Do not
  create training configurations or launch training before approval.

## Task group 2: Recurrent MAPPO-FP implementation

- [x] Replace the feed-forward MAPPO Actor with the approved structured
  observation encoder, MLP, single-layer 64D GRU and masked policy head.
- [x] Replace the feed-forward FP Critic with an MLP, independent single-layer
  64D GRU and value head over the approved current-only FP state.
- [x] Implement deterministic peer selection, padding and masks exactly as
  specified; prove prohibited teacher/global/future fields cannot enter the
  Actor.
- [x] Extend rollout storage, collection, GAE, PPO updates, checkpoints and
  resume logic to preserve worker/AGV recurrent state, terminal masks and
  contiguous recurrent training sequences.
- [x] Add focused regression tests for hidden-state reset/isolation, sequence
  ordering, recurrent checkpoint resume, structured-observation contracts and
  teacher-information exclusion.

## Task group 3: Local runner, evidence and reward candidate

- [x] Add the three owner-approved environment profiles and the initial
  document reward candidate; record every reward component in CSV and
  TensorBoard.
- [x] Configure local CPU execution for six independent environment workers,
  four learner PyTorch threads and one numerical-library thread per worker.
- [x] Ensure all step budgets count cumulative joint environment transitions
  across workers, not steps per worker.
- [x] Make every completed diagnostic emit final-checkpoint evaluation on seeds
  `9400` and `9401`, a slow replay on seed `9400`, TensorBoard and a concise
  review receipt.
- [x] Record each pre-formal reward or training-configuration revision as a new
  run identity; do not overwrite or compare incompatible runs as paired data.

## Task group 4: Medium 3-AGV DirectGoal diagnostic

- [x] Train `MAPPO-DG` from scratch with training seed `7` for 300k cumulative
  steps on `medium-3ag-low-workload`.
- [x] Produce the required two-episode evaluation, slow replay and TensorBoard
  evidence.
- [x] Owner/architect review judged DG unsuccessful: all 715 completed training
  episodes and both fixed evaluations had zero completed tasks, despite 651
  training episodes reaching successful pickup. A controlled exact-profile
  delivery check completed correctly, so the failure is the learned
  pickup-to-delivery policy rather than task-completion plumbing. Do not infer
  a numerical formal claim from this one-seed run.

## Task group 5: Conditional waypoint diagnostic

- [x] Execute this group only if Task group 4’s joint review judges DirectGoal
  unsuccessful.
- [ ] Train `MAPPO-WP` from the identical initial state for 300k steps with the
  same training/evaluation seeds, reward version, worker settings and PPO
  configuration; substitute only the DirectGoal geometry with current waypoint
  geometry.
- [ ] Compare the two fixed evaluation completion rates. If WP’s mean is at
  least 20 percentage points higher, prepare an explicit recommendation for
  owner decision; otherwise retain DirectGoal.
- [ ] Pause for owner/architect review before selecting a goal interface for
  the 5-AGV diagnostics.

## Task group 6: Medium 5-AGV capability diagnostic

- [ ] Train the owner-selected goal-interface baseline from scratch for 300k
  steps on `medium-5ag-low-workload`, seed `7`.
- [ ] Produce its required evaluation/replay/TensorBoard review package.
- [ ] If double review finds a learning trend but insufficient effect, resume
  the exact same run to a cumulative 500k steps and regenerate the package.
- [ ] Pause for owner/architect review before entering the large-profile task.

## Task group 7: Large 5-AGV high-workload diagnostic

- [ ] Train the owner-selected goal-interface baseline from scratch for 300k
  steps on `large-5ag-high-workload`, seed `7`, with all relevant settings
  recorded.
- [ ] Produce its required evaluation/replay/TensorBoard review package.
- [ ] If double review judges the scratch run insufficient, warm-start exactly
  from the final medium-5ag-low-workload Actor, Critic and ValueNorm state;
  reset optimizer, scheduler and step counters; train a separate 300k-step
  diagnostic run.
- [ ] Record the final double-review findings, all parameter revisions and the
  recommended next project action without making a formal performance claim.
