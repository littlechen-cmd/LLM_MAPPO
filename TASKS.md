# Project Task List

Update this file whenever a task begins, completes, or is blocked. Major completed milestones require a Git commit.

## Phase 0: Repository and Development Baseline

- [x] Verify the upstream RWARE worktree and create a recoverable Git bundle.
- [x] Promote the RWARE Git history and source tree to the workspace root.
- [x] Create the root contributor guide and project task list.
- [x] Add package extras, build metadata, experiment ignores, and module directory skeletons.
- [x] Run packaging, lint, and baseline test checks.
- [x] Commit the Phase 0 checkpoint.

## Phase 1: Environment and Interfaces

- [x] Implement dynamic batches, priority tasks, battery/charging, collision accounting, and picking locks.
- [x] Freeze task, priority adjustment, engagement label, task queue, and A* planner interfaces.
- [x] Add FIFO rule-layer validation and the `demo.py` operational view.
- [x] Reach the Phase 1 Go/No-Go threshold: 100 headless episodes / 10,000 steps passed; 120/120 nonblank RGB frames averaged 29.4 ms with a 47 ms maximum.

## Later Phases

- [ ] Phase 2: Oracle-path MAPPO baseline and 10-seed Go/No-Go evaluation.
  - [x] Implement the custom PyTorch CTDE MAPPO baseline, oracle-waypoint adapter, and artifact tracking.
  - [ ] Run and record a local CPU 800-1000 episode feasibility test before any server-scale training.
  - [x] Compare 500-episode local 1-AGV runs with waypoint rewards 0.01 and 0.05; neither completed a task.
  - [ ] Validate collision classification and oracle interaction masking in a local 500-episode 1-AGV run.
  - [x] Validate the small-map A* expert, behavior-cloning warm start, and 800-episode local MAPPO feasibility run.
  - [x] Validate the medium-map A* expert, behavior-cloning warm start, and 800-episode local MAPPO feasibility run.
  - [x] Add deterministic checkpoint visualization with GIF and paced human modes.
  - [ ] If server training is needed, record whether the 4080S run uses Linux or WSL plus CUDA/PyTorch details.
  - [x] Train the 1-AGV waypoint/charging curriculum stage.
  - [ ] Train the 3-AGV avoidance/charging-coordination stage (first local trial No-Go: 0.82 completion, 3.94 collisions/episode, 0.385 deadlock rate).
    - [ ] Pass the 100-episode A* multi-AGV safety/completion gate before collecting demonstrations; rolling-horizon reservations reach 99.0% completion with zero collisions, but three seeds still stall with a loaded AGV.
  - [x] Run the 10-seed Go/No-Go evaluation and record the acceptance result.
    - [ ] KL-protected checkpoint: 0.973 completion, 9.585 collisions/episode,
      0.05 deadlock rate, 0.046 success-rate std; overall No-Go due to collisions.
    - [ ] Collision diagnosis and execution-time coordination correction are
      intentionally deferred. Preserve `medium / 3 AGV` and do not expand the
      map or fleet while Phase 3 is being developed.
- [ ] Phase 3: Dual-head Actor and A* distillation ablation.
  - [x] Phase 3a implementation: rule-labelled engagement head and
    priority-conditioned motion head on `medium / 3 AGV`, without A* KL
    distillation.
  - [x] Verify the Phase 3a entry point with a 2-episode CPU smoke run; this is
    an interface check only and is not the required feasibility result.
  - [x] Complete the Phase 3a training preflight: stabilize orientation-aware
    A* heap ordering, persist live plotting data atomically, and report
    priority-latency evaluation metrics.
  - [x] Run the local CPU 800-episode Phase 3a feasibility test and 10-seed
    evaluation. It reached 0.948 completion, 0.815 collisions/episode, 0.09
    deadlock rate, and A<C priority latency; it is No-Go due to completion and
    deadlock rate.
  - [x] Phase 3b implementation: add time-reserved A* path distillation as the
    only change from the frozen Phase 3a configuration. The A* teacher
    completes seed 4's 20 evaluation instances with 1.0 completion, zero
    collisions, and zero state deadlocks.
  - [x] Supersede the original Phase 3a/3b pair with the r2 semantic ablation;
    do not run the original Phase 3b configuration as a comparison baseline.
  - [x] Implement the r2 semantic engagement architecture: A/B/C/idle labels
    are 0.8/0.5/0.3/0.1, engagement has an independent encoder, and the motion
    head consumes `e.detach()`.
  - [x] Run Phase 3a-r2 for 1,000 local CPU episodes without A* KL, then freeze
    its configuration as the only baseline for Phase 3b-r2. Its 10-seed x
    20-episode result was No-Go: 0.895 completion, 2.960 collisions/episode,
    0.180 deadlock rate, and no aggregate A<C priority latency.
  - [x] Run Phase 3b-r2 for 1,000 local CPU episodes with only 0.05 time-reserved
    A* KL added, then compare both r2 runs over 10 seeds x 20 episodes. It
    improved to 0.993 completion, 2.690 collisions/episode, 0.005 deadlock
    rate, 0.020 success-rate std, and A<C latency, but is No-Go on the
    collision gate because seed 3 recorded 21.15 collisions/episode.
  - [ ] Run the Phase 3b-r2 multi-seed robustness feasibility experiment with
    round-robin training seed groups 100-109 and held-out evaluation seed
    groups 0-9. Record per-episode seed provenance and compare the aggregate
    and per-seed collision rates against the static 3b-r2 baseline.
  - [ ] Phase 3 final acceptance: enable dynamic ingress after the current
    static r2 training is complete. Preserve the static r2 configurations as
    architecture-ablation baselines; do not modify the active run.
    - [ ] Parameterize and pass through `batch_interval`,
      `batch_size_range`, `initial_priority_label`, `request_queue_size`, and
      `priority_schedule`, while preserving static defaults for reproducibility.
    - [ ] Use `max_steps: 1000`, `batch_interval: 100`,
      `batch_size_range: [1, 3]`, `request_queue_size: 4`,
      `initial_priority_label: A`, and no priority-label wraparound.
    - [ ] Replace fixed A/B/C engagement targets with the confirmed
      active-letter rank linear mapping; keep idle or inactive agents at 0.1.
    - [ ] Pass the dynamic-ingress time-reserved A* feasibility gate before
      collecting demonstrations or running MAPPO: completion >= 0.95,
      collisions <= 2.0, and deadlock rate <= 0.05.
    - [ ] Run dynamic-ingress Phase 3a-r2 and Phase 3b-r2 with identical
      ingress streams and seeds, then compare completion, collisions, deadlock,
      A<C latency, and behavior-group metrics.
    - [ ] Add dynamic-ingress behavior-group evaluation: narrow-corridor
      yielding, high/low-priority intersection passage, and low-battery
      charging diversion. Special scenario injection remains deferred to
      Phase 4.
- [ ] Phase 4: DeepSeek label adjustment and engagement distillation.
- [ ] Phase 4b: Large-scale 6-AGV evaluation.
- [ ] Phase 5: 80x120, 10-AGV target-scale evaluation and paper outputs.
- [ ] Phase 6 (optional): Runtime LLM exception handling.
