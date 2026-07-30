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
  - [ ] Train the 1-AGV waypoint/charging curriculum stage.
  - [ ] Train the 3-AGV avoidance/charging-coordination stage.
  - [ ] Run the 10-seed Go/No-Go evaluation and record the acceptance result.
- [ ] Phase 3: Dual-head Actor and A* distillation ablation.
- [ ] Phase 4: DeepSeek label adjustment and engagement distillation.
- [ ] Phase 4b: Large-scale 6-AGV evaluation.
- [ ] Phase 5: 80x120, 10-AGV target-scale evaluation and paper outputs.
- [ ] Phase 6 (optional): Runtime LLM exception handling.
