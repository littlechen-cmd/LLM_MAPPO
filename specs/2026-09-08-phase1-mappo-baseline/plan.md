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
- [x] Revise the pre-formal Reward-v2 candidate after the DG/WP diagnosis:
  every blocked-forward and occupancy-conflict event is `-0.1`, static
  traversable distance replaces Manhattan progress, and `max_steps=800` gives
  dynamically admitted late tasks an executable delivery window. Worker faults
  must leave per-worker evidence before any replacement long run is authorized.

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

## Task group 5a: Medium 3-AGV efficiency-recovery diagnostics

- [x] Register a `+0.05` charge reward that applies only when an AGV remains
  at its active charging target with `NOOP` and its battery actually rises.
- [x] Register three otherwise-identical fresh 300k DirectGoal profiles:
  P1 (`-0.15` blocked-forward, `-0.50` occupancy conflict), P2
  (`-0.50`, `-0.50`) and P3 (`-0.15`, `-1.00`).
- [ ] Owner runs P1/P2/P3 sequentially, retaining the evaluation, replay and
  TensorBoard package for each run.
- [ ] Owner/architect compare completion and execution efficiency before any
  further reward or training-configuration change.

## Task group 5b: DirectGoal deadlock-recovery diagnostic

- [x] Keep the owner-identified executable `-0.1/-0.1` event-penalty pair and
  `-0.005` step cost; do not increase the step cost for this task.
- [x] Add one `-5.0` terminal deadlock cost, with no repeated per-step charge.
- [ ] Owner runs one fresh 300k DirectGoal diagnostic with the required review
  package. If DG completes tasks, stop Phase 1 baseline reward optimization and
  move to the next project task regardless of efficiency or one-seed rate.

## Task group 5c: Training-only RWARE replenishment recovery

- [x] Restore the original RWARE delivery-time fixed-request replacement in
  training workers: one delivered request is replaced by one new requested
  shelf and corresponding project task.
- [x] Keep deterministic evaluation on dynamic ingress, rather than silently
  evaluating under the training-only replenishment mode.
- [ ] Owner runs the next DirectGoal diagnostic under the recorded training /
  evaluation split. Any confirmed task completion ends Phase 1 baseline tuning.

## Task group 6: Medium 5-AGV capability diagnostic

- [x] Select the owner-approved DirectGoal interface after the waypoint review
  and prepare the 5-AGV configuration with 800-step episodes, the executable
  Reward-v2 values, RWARE delivery replenishment in training, and dynamic
  ingress in fixed evaluation.
- [x] Construct and reset both 5-AGV environments without training: training
  begins with five active requests/tasks, while evaluation retains its separate
  dynamic-ingress mode.
- [ ] Train the owner-selected goal-interface baseline from scratch for 300k
  steps on `medium-5ag-low-workload`, seed `7`.
- [ ] Evaluate that one completed checkpoint independently under both approved
  same-topology profiles: Low (120-step ingress, one task/batch, target 8) and
  High (60-step ingress, two tasks/batch, target 12). Both receipts use the
  same medium map, five AGVs, facilities, checkpoint and fixed evaluation
  seeds. Low is the baseline-readiness evidence; High is a recorded stress
  diagnostic only.
- [ ] If double review finds a learning trend but insufficient effect, resume
  the exact same run to a cumulative 500k steps and regenerate the package.
- [ ] Pause for owner/architect review before entering Phase 2 A* KD.

## Task group 7: Medium 5-AGV workload-shift evaluation

- [ ] Retain the initial large-5AG task as not executed for Phase 1. It changes
  both topology and workload, so it cannot answer the adopted workload-only
  question with the same recurrent Actor/Critic checkpoint.
- [ ] Record both medium-5AG evaluation receipts and the owner/architect review
  decision before beginning Phase 2. The High result is not a new training
  gate and cannot silently trigger further baseline optimisation.
