# Requirements — Phase 1 MAPPO Baseline

## Scope

Phase 1 rebuilds the project’s teacher-free MAPPO baseline so that it follows
the owner-provided recurrent CTDE MAPPO baseline design and is appropriate for
this warehouse navigation project.

It covers:

- a structured decentralized observation, a feature-pruned (FP) centralized
  critic state, and parameter-shared recurrent Actor/Critic networks;
- local RWARE-derived medium and large layouts with charging and picking
  stations, plus owner-reviewed colour-only layout previews;
- local CPU diagnostic training in the prescribed 3-AGV, 5-AGV medium and
  5-AGV large profiles; and
- per-run evaluation, TensorBoard evidence and a slow replay for joint owner /
  architect review.

## Out of scope

- A*, LLM labels, KL/KD, reward calibration, task assignment learning,
  reservation, coordinator or other teacher information in the MAPPO-DG
  baseline.
- Any server training, multi-seed formal statistic, paper claim, final reward
  freeze, or automatic decision to alter the formal architecture.
- Recreating or editing `specs/roadmap.md`. The owner deliberately removed the
  Roadmap and will manage task ordering independently.

## Decisions

| Decision | Rationale |
| --- | --- |
| Actor and FP centralized Critic each use an independent, single-layer GRU with hidden size 64. | This is the recurrent MAPPO form requested by the owner and is closer to the original MAPPO RNN implementation than the current feed-forward baseline. |
| All homogeneous AGVs share one Actor parameter set and one Critic parameter set. | Parameter sharing preserves the cooperative homogeneous-agent MAPPO setting. |
| The Actor receives only structured execution-time local information: self state, DirectGoal geometry, a robot-centric local map, and three anonymous nearest-neighbour states. | It supports decentralized navigation and local coordination without leaking global, A*, LLM or future information. |
| The Critic receives an FP agent-specific centralized current-state representation, not attention/GNN features or future/oracle data. | It follows the MAPPO centralized-value-function rationale while avoiding redundant flat observations. |
| PPO trains consecutive recurrent sequences; hidden state is maintained per worker and AGV and reset only at that worker/AGV’s episode boundary. | Shuffling individual time steps would invalidate GRU learning. |
| The initial Phase 1 reward candidate is the version in `docs/plan/MAPPO baseline.md`: completion `+5 × priority weight`, pickup `+0.5`, signed progress `0.1Δd`, blocked-forward `−0.15`, collision `−2.0`, step cost `−0.005`. | It gives a symmetric progress signal and retains completion as the dominant objective. |
| Reward values and training hyperparameters are adjustable before formal training, but every run writes its complete configuration and no DG/WP comparison may differ in any setting except its goal interface. | Phase 1 is a diagnostic recovery phase, while preserving interpretable comparisons. |
| Local diagnostics run on CPU with six independent environment workers; the learner uses four PyTorch CPU threads and every worker has one numerical-library thread. | The i5-13500H has 12 physical cores / 16 logical processors; this leaves capacity for the learner and the owner’s normal desktop workload without oversubscription. |
| Each diagnostic run uses training seed `7`, fixed evaluation seeds `9400` and `9401`, and a slow replay of seed `9400`. | These are compact, repeatable review evidence rather than formal statistics. |
| Charging-station count equals AGV count; picking-station count equals `floor(AGV / 2)`. | This is the owner-approved facility contract for all three Phase 1 layouts. |
| Layout priority is shown through a four-class `A/B/C/D` colour legend: A red-orange, B amber, C blue and D purple. Same-priority goods share exactly the same colour and no individual task or shelf identifier is shown. | It represents every project priority class while keeping the preview free of artificial per-item identity. |
| The large-5AG task-ingress interval, batch size and completion target are deliberately deferred until the medium-5AG diagnostic has been reviewed. | It avoids freezing a high-workload demand level before observing what the recovered baseline can handle. |
| A waypoint run is a conditional diagnostic, not part of the pure MAPPO baseline. | It isolates whether execution-time geometric guidance is the learning bottleneck. |
| If the 5-AGV large high-workload scratch run is judged insufficient, its warm start loads the final 5-AGV medium low-workload Actor, Critic and ValueNorm state only; optimizer, scheduler and global step are reset. | It transfers learned navigation representations without falsely continuing the previous optimization trajectory. |

## Context

### Recurrent MAPPO contract

The Actor is `structured local observation → MLP → GRU(64) → masked policy
head`. The Critic is `FP centralized state → MLP → GRU(64) → value head`.
Actor and Critic recurrent states are distinct. The critic is training-only;
deployment uses the Actor and its own recurrent state.

The structured Actor observation contains exactly these semantic blocks:

- **self:** orientation, carrying state, battery, task phase, previous action
  and previous blocked status;
- **goal:** relative DirectGoal coordinate, normalized distance and goal type;
- **map:** robot-centric `5 × 5` channels for static obstacle, shelf,
  workstation, charger and other AGV; and
- **peers:** the three nearest anonymous AGVs, each with relative position,
  orientation, carrying state, battery, task phase, priority and recent motion.

Nearest peers are ordered by Euclidean distance, then relative x, relative y
and orientation rank; missing peers use zero padding plus an explicit validity
mask. The Actor excludes A* paths, waypoints, planner actions, LLM labels,
yield preferences, global queue information and all future information.

The FP Critic state contains the current poses, orientation, battery, carrying
state, active goal, task phase and legal-action state of all AGVs, together
with current shelf/workstation/charger occupancy and current task-queue state.
It includes an agent-specific marker for the evaluated AGV but excludes future
arrivals, A* routes, LLM outputs and reward-calibration data.

Priority weight uses the project’s current task-queue function over active
priority labels; it therefore remains in the documented range `[0.5, 2.0]`.
The concrete priority schedule is an owner-approved component of each workload
profile rather than an implicit training default.

### Environment-design gate

Before creating a training profile, the implementer must inspect the existing
RWARE medium and large maps. For each required profile, it must show that all
stations occupy valid non-overlapping cells, do not replace a required static
object, are reachable from usable traversable cells, and preserve at least one
usable route between every relevant work area. If a stock map cannot satisfy
that contract, the implementer must provide a minimal map amendment and stop
for owner approval instead of silently choosing locations or workload values.

The three profiles are:

| Profile | Base map | AGVs | Workload |
| --- | --- | ---: | --- |
| `medium-3ag-low-workload` | RWARE medium | 3 | low; exact initial/dynamic task parameters approved with its preview |
| `medium-5ag-low-workload` | RWARE medium | 5 | low; exact initial/dynamic task parameters approved with its preview |
| `large-5ag-high-workload` | RWARE large | 5 | station layout approved now; workload intensity determined after medium-5AG review |

### Diagnostic sequence

1. Train `MAPPO-DG` from scratch for 300k cumulative joint environment steps
   on `medium-3ag-low-workload`.
2. The owner and architect inspect its two evaluation episodes, replay and
   TensorBoard before deciding whether it has an acceptable learning trend.
3. Only if they judge DG unsuccessful, train the paired `MAPPO-WP` run for 300k
   steps. It differs only by replacing the DirectGoal geometry block with a
   current online waypoint-geometry block; goal type stays unchanged. It
   receives no planner action, path, reservation or priority/coordinator
   information.
4. If WP exceeds DG’s two-episode mean completion rate by at least 20
   percentage points, record a waypoint-return recommendation for an explicit
   owner architecture decision. It does not automatically change the baseline.
5. Train the selected goal-interface baseline from scratch for 300k steps on
   `medium-5ag-low-workload`. After joint review, continue the exact same run
   to a **cumulative** 500k steps only if it shows a learning trend but its
   effect is still insufficient.
6. Train the selected goal-interface baseline from scratch for 300k steps on
   `large-5ag-high-workload`. After joint review, run the specified 300k-step
   warm-start diagnostic only if the scratch run is insufficient.

Every completed training run writes its final checkpoint, immutable run
configuration, TensorBoard events, training/episode CSV evidence, a two-seed
evaluation receipt and a seed-9400 slow replay. No fixed numerical pass/fail
threshold applies; the owner and architect conduct the stated manual review.

## Open questions

The large-5AG workload intensity is deliberately deferred until the medium-5AG
diagnostic is reviewed. Its station coordinates and all low-workload profile
values are owner approved.
