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
  - [ ] Run the required local CPU 800-1000 episode Phase 3a feasibility test
    and record the configuration, runtime, TensorBoard curves, checkpoint, and
    seed-level evaluation before considering any server run.
  - [ ] Phase 3b: add A* path distillation as the only change from the frozen
    Phase 3a configuration, then run the on/off ablation.
- [ ] Phase 4: DeepSeek label adjustment and engagement distillation.
- [ ] Phase 4b: Large-scale 6-AGV evaluation.
- [ ] Phase 5: 80x120, 10-AGV target-scale evaluation and paper outputs.
- [ ] Phase 6 (optional): Runtime LLM exception handling.
