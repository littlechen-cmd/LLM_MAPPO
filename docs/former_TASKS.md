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
  - [x] Run the Phase 3b-r2 multi-seed robustness feasibility experiment with
    round-robin training seed groups 100-109 and held-out evaluation seed
    groups 0-9. Its 10-seed x 20-episode result passes the static gate: 0.978
    completion, 1.575 collisions/episode, 0.005 deadlock rate, 0.037 success
    std, and A<C latency. Retain seed 2's 9.8 collisions/episode as a
    per-seed risk to monitor during dynamic-ingress validation.
  - [ ] Phase 3 final acceptance: enable dynamic ingress after the current
    static r2 training is complete. Preserve the static r2 configurations as
    architecture-ablation baselines; do not modify the active run.
    - [x] Parameterize and pass through `batch_interval`,
      `batch_size_range`, `initial_priority_label`, `request_queue_size`, and
      `priority_schedule`, while preserving static defaults for reproducibility.
    - [x] Add matched dynamic-ingress 3a-r2 and 3b-r2 configurations using
      `max_steps: 1000`, `batch_interval: 40`,
      `batch_size_range: [1, 3]`, `request_queue_size: 4`,
      `task_completion_target: 9`, initial A/B batches, and no priority-label
      wraparound. Batch arrivals continue while the completion target has not
      been reached, even when more than nine tasks have already arrived.
    - [x] Replace fixed A/B/C engagement targets with the confirmed
      active-letter rank linear mapping; keep idle or inactive agents at 0.1.
    - [ ] Pass the dynamic-ingress time-reserved A* feasibility gate before
      collecting demonstrations or running MAPPO: completion >= 0.95,
      collisions <= 2.0, and deadlock rate <= 0.05.
      - [x] Add the independent `eval/evaluate_dynamic_ingress_astar.py` gate.
        Its prior 1-episode smoke used the superseded 100-step ingress policy;
        rerun the gate with the 40-step, initial-A/B, target-9 contract before
        collecting demonstrations. The required 10-seed x 20-episode gate
        remains pending.
    - [ ] Run dynamic-ingress Phase 3a-r2 and Phase 3b-r2 with identical
      ingress streams and seeds, then compare completion, collisions, deadlock,
      A<C latency, and behavior-group metrics.
      - [x] Complete matched local CPU training for both configurations:
        3a reached 1,000 episodes / 311,834 steps and 3b reached 1,000
        episodes / 235,500 steps. In their final 200 training-distribution
        episodes, both reached 1.0 completion and 0 deadlocks; mean collisions
        were 1.84 (3a) and 1.72 (3b). These are development metrics, not
        held-out acceptance results.
      - [ ] Archive or rerun the matched held-out dynamic-policy 10-seed
        evaluation. No dynamic 3a/3b `evaluation_10x20.json` artifact was
        present when results were reviewed on 2026-08-13, so completion,
        collision, deadlock, and A<C acceptance cannot be claimed.
    - [ ] Add dynamic-ingress behavior-group evaluation: narrow-corridor
      yielding, high/low-priority intersection passage, and low-battery
      charging diversion.
      - [x] Implement and run natural-rollout behavior evaluation over held-out
        seeds 0-9 with 5 episodes each. 3b observed 95 priority-intersection
        decisions with 16.84% high-priority-first actions; 3a observed 466
        with 5.15%. 3a saw one narrow-corridor opportunity and did not yield;
        3b saw none. Neither model naturally covered low-battery loaded-AGV
        charging diversion. Uncovered groups are reported as such, not passed.
      - [ ] Add controlled Phase 4 scenarios for sufficiently sampled
        narrow-corridor yielding and low-battery charging diversion. This is
        scenario injection for evaluation/training and remains deferred from
        Phase 3.
- [ ] Phase 4: DeepSeek label adjustment and engagement distillation.
  - [x] Replace the ambiguous single engagement scalar with the frozen Phase 4
    dual-semantic contract (`task_commitment`, `local_assertiveness`) before
    any local CPU feasibility training. The actor, offline teacher, rollout
    buffer, losses, checkpoint, and audit path now preserve both dimensions.
  - [x] Define an offline-only semantic-teacher contract: strict JSON schemas,
    label-only priority adjustments, cached JSONL labels, and zero API calls in
    MAPPO training/evaluation.
  - [x] Add a deterministic mock teacher plus a Phase 4 configuration. Mock
    labels are limited to interface/smoke validation and are not experimental
    evidence for LLM distillation.
  - [x] Freeze the Phase 4 scale as `medium / 5 AGV / batch [4,8] / N=50`.
    It is a new scale track, not a direct Phase 3b ablation; retain a matched
    rule-engagement + A* KL baseline before attributing effects to LLM labels.
  - [x] Add controlled scenario injection and stratified label sampling:
    120 normal transport (30%), 100 priority conflicts (25%), 80 narrow-corridor
    yields (20%), 60 low-battery diversions (15%), and 40 station/exit conflicts
    (10%). Controlled states are label-only and never replace PPO rollouts.
  - [x] Run a 5-AGV A* safety/completion preflight before DeepSeek collection.
    The local 3-seed x 2-episode run reached 1.0 completion, 0 collisions, and
    0 terminating deadlocks under [4,8] ingress and N=50; it also recorded
    344-362 path-livelock and 1-7 transient state-repeat events per seed, so
    retain them as Phase 4 diagnostics rather than treating the gate as a
    coordination-quality proof.
  - [x] Generate and review the frozen 400-record DeepSeek offline label set.
    Audit at least 10% of records for scenario type, bounded JSON, rationale,
    priority semantics, and absence of action/assignment instructions.
    - [x] Stabilize DeepSeek response parsing, safe empty-response diagnostics,
      and the local tests required before a one-request smoke collection.
    - [x] Run and inspect one real DeepSeek request before starting the full
      400-record collection. The smoke produced one schema-valid 615-dimension
      record with label 0.8 and removed its partial checkpoint.
    - [x] Pass the dual-semantic pilot gate on 25 real DeepSeek records (five per
      scenario type): all automatic direction/schema checks and the full
      rationale review passed after isolating controlled-scene geometry.
    - [x] Resume the user-run formal collection from its validated 79-record
      checkpoint and complete the frozen 120/100/80/60/40 quotas. The final
      dataset has 400 unique v2 IDs. Its SHA-256 is recorded in the formal
      audit report.
    - [x] Close the formal rationale audit: the deterministic 10% review and a
      full text/state scan found 11 rationale-only anomalies, including two
      score/reason contradictions. Two targeted repair rounds corrected all
      affected rationales without changing any numeric training target.
      - [x] Add a non-destructive, resumable targeted re-label utility and a
        frozen 11-ID repair list. The audit now detects all 11 affected records
        automatically instead of relying only on sampled manual review.
      - [x] Re-audit the final `repaired_r2` artifact: all 400 records, fixed
        quotas, 400 unique IDs, and the deterministic 10% sample passed; the
        full automatic issue count is zero.
    - [ ] Optionally run a same-scenario Flash non-thinking versus Flash-high
      versus Pro-high label-quality pilot. This is a provider-quality study and
      does not block the Phase 4 CPU feasibility experiment.
  - [x] Verify the dual-semantic training interface with a one-episode local CPU
    Mock-label smoke: both component losses were non-zero and the checkpoint
    stored a 2-output semantic head feeding a 66-input motion head.
  - [x] Repeat the one-episode local CPU interface smoke with the formal 400
    records: both component losses were non-zero, the checkpoint shapes were
    `(2, 64)` and `(5, 66)`, and training made zero API calls.
  - [ ] Run the required local CPU 800-episode Phase 4 feasibility experiment,
    with TensorBoard and deterministic visual replay, before any server run.
  - [ ] Evaluate the Phase 3b versus Phase 4 comparison on 10 held-out seeds;
    report significance, priority latency, starvation, safety, and zero online
    LLM calls.
- [ ] Phase 4b: Large-scale 6-AGV evaluation.
- [ ] Phase 5: 80x120, 10-AGV target-scale evaluation and paper outputs.
- [ ] Phase 6 (optional): Runtime LLM exception handling.
