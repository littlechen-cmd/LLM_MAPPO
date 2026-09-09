# Validation — Phase 1 MAPPO Baseline

## Definition of done

- [ ] The three owner-approved RWARE-derived profiles have valid previews and
  explicit facility configurations; the two low-workload profiles also have
  explicit workload configurations, while the large workload is documented as
  deferred until its stated review gate.
- [ ] MAPPO uses separate 64D single-layer Actor and Critic GRUs with
  parameter sharing across AGVs, correct recurrent rollout/reset semantics and
  contiguous recurrent PPO updates.
- [ ] Actor observation contains no A*, LLM, global queue, future or other
  prohibited information; the Critic receives only the approved current FP
  state.
- [ ] The initial document reward candidate, all revisions and complete run
  identities are recorded without overwriting prior evidence.
- [ ] Each executed diagnostic has its checkpoint, TensorBoard, two fixed
  evaluations and slow replay, and has received owner/architect double review.
- [ ] Conditional WP, 500k continuation and warm-start runs occur only under
  their stated review decisions.

## How to verify

1. Inspect each layout preview and configuration table. Confirm facility counts,
   colour-only priority representation and reachable placement before approving
   Task group 1.
2. Inspect the network, storage and checkpoint contracts. Confirm one separate
   hidden state for every worker/AGV and correct reset at that environment’s
   terminal boundary.
3. Inspect a recorded recurrent batch. Confirm time order remains intact inside
   each sequence and no GAE/hidden state crosses an environment or episode.
4. Inspect the run manifest and log schema. Confirm the CPU worker/thread
   settings, cumulative-step accounting, reward version, goal interface and
   seed are present.
5. For every completed training diagnostic, inspect TensorBoard, both fixed
   evaluation receipts and the slow replay before jointly recording the next
   action.
6. For the WP comparison, calculate `(WP mean completion − DG mean completion)`
   over seeds `9400` and `9401`; only `≥ 0.20` warrants a waypoint-return
   recommendation, not an automatic architecture change.

## Tests to run

- [ ] Canonical Python runs focused structured-observation, FP-state, recurrent
  hidden-state, sequence-storage, GAE and checkpoint-resume regression tests.
- [ ] Canonical Python runs the existing MAPPO-FP and Reward-v2 regression
  tests after the recurrent migration is completed, including threshold-entry
  energy shaping and productive-delivery-versus-stagnation checks.
- [ ] One short local functional run verifies CPU-only workers, cumulative-step
  accounting, evaluation receipt, TensorBoard output and replay generation;
  it is not a performance result.

## Merge criteria

- [ ] All applicable task groups in `plan.md` are complete; conditional groups
  are explicitly marked not applicable when their trigger is absent.
- [ ] Every stated owner approval and double-review decision is recorded.
- [ ] All validation checks pass or any deliberate non-execution is documented
  as a Phase 1 limitation.
- [ ] `CHANGELOG.md` is updated in the same commit as completed work.
