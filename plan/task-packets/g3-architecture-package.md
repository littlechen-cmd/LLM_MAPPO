## Task Packet

- Scope: Produce the persistent architecture and engineering handoff package for
  the whole G3 formal-protocol-freeze stage, with a directly implementable G3-6
  unseen-layout contract. Do not implement layouts or run experiments.
- Files to read: `AGENTS.md`, `CONSTITUTION.md`, `TASKS.md`, `CHANGELOG.md`,
  `configs/g3_experiment_manifest.yaml`, `plan/experiment-protocol.md`, all
  existing G3 task packets/reviews, environment registration, layout parsing,
  Phase2Warehouse, and charging-station code.
- Files allowed to edit: `plan/g3-architecture-task-package.md`, this task packet,
  `plan/progress.md`, `plan/review/g3-architecture-task-package-review.md`, and
  `CHANGELOG.md`.
- Required skills: `using-research-writing`, `paper-orchestration`,
  `experiment-results-planning`, and `verification`.
- Evidence/data inputs: current G3 manifest v2, G3 task states, current Git history,
  existing 24x20 training layout contract, and the custom ASCII-layout interface.
- Required artifacts: one authoritative G3 package containing status/dependencies,
  per-task ownership and acceptance evidence, the two-stage freeze procedure, the
  exact G3-6 engineering contract, prohibited actions/claims, validation commands,
  and the engineer handoff template.
- Rejection checks: do not mark G3-1 or G3-6 complete; do not invent experiment
  results; do not run training/evaluation; do not silently resolve G2-1d/e or the
  null formal budget; do not allow layout selection from trained-policy outcomes.
- Validation commands: verify the main package has one authoritative subsection
  for every G3 ID,
  verify required G3-6 interface/topology/hash/claim clauses, compare status against
  `TASKS.md`, run `git diff --check`, and perform specification and quality reviews.
