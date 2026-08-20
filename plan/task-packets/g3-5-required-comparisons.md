## Task Packet

- Scope: Complete G3-5 preparation only: freeze fair comparison boundaries and
  implement the minimum reproducible configurations and entry points for QMIX-WP,
  RuleKD, ShuffleKD, MAPPO-NoWP, and Heuristic-Dispatcher+A*. Do not start a
  formal or multi-seed training run.
- Files to read: `CONSTITUTION.md`, `TASKS.md`, `plan/experiment-protocol.md`,
  `configs/g3_experiment_manifest.yaml`, Phase 3/4 training and evaluation
  modules, label-record types, and existing tests.
- Files allowed to edit: comparison implementation modules, their focused tests,
  `configs/`, `eval/`, `train/`, `plan/`, `CONSTITUTION.md`, and `TASKS.md`.
- Required skills: `using-research-writing`, `paper-orchestration`,
  `experiment-results-planning`, `statistical-analysis`, and `verification`.
- Evidence/data inputs: the frozen 400-record offline semantic-label dataset;
  the manifest v2 seed, safety, observation/action, and budget contract.
- Required artifacts: machine-readable comparison contract; deterministic rule
  and shuffled-label derivation; NoWP switch that preserves the frozen interface;
  heuristic dispatcher evaluation entry point; executable QMIX-WP entry point;
  configuration files; focused tests; review and capability-use audit.
- Rejection checks: no online LLM calls; no change to rule-layer safety; no
  group-specific environment-step budget; QMIX must share observations, action
  masks, rewards, waypoint availability, and safety contract; diagnostic groups
  must be marked non-confirmatory; no long training is launched.
- Validation commands: focused pytest for new components, manifest/configuration
  validation, relevant existing regression tests, and Flake8.
