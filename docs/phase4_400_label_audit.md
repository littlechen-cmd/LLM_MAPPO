# Phase 4 400-Label Audit

## Frozen artifact

- Dataset:
  `artifacts/phase4_labels/deepseek_medium_5ag_400_v2_repaired_r2.jsonl`
- SHA-256: `9928F5756C1261589946EB3AEDF8DC2C1FA6F73F037CD05C31AFCA0683161797`
- Records / unique scenario IDs: `400 / 400`
- Observation contract: `phase4-semantic-v2`, 615 values per observation
- Frozen quotas: normal 120, priority conflict 100, narrow corridor 80,
  low battery 60, station exit 40
- Provider recorded in the artifact: `deepseek:deepseek-v4-flash`

## Mechanical and score-direction checks

The strict loader accepted all records. Every score is bounded in `[0, 1]`,
there are no duplicate IDs, and the final automatic audit found no schema,
direction, state/rationale, priority-order, score/reason, or forbidden-control
issue. The deterministic 10% sample uses seed `20260814` and is stored in
`artifacts/phase4_labels/deepseek_medium_5ag_400_v2_repaired_r2_review_10pct.csv`.

Score distributions are intentionally concentrated for the controlled anchor
scenarios:

| Scenario | Task commitment | Local assertiveness |
| --- | --- | --- |
| Normal transport | 0.9: 22, 1.0: 98 | 0.5: 43, 0.7: 77 |
| Priority conflict | 0.9: 16, 1.0: 84 | 0.7: 52, 0.8: 16, 0.9: 32 |
| Narrow corridor | 0.9: 80 | 0.2: 62, 0.3: 18 |
| Low battery | 0.2: 60 | 0.5: 60 |
| Station exit | 0.9: 34, 1.0: 6 | 0.3: 40 |

## Rationale review

The assistant pre-review found that all 40 sampled numeric labels agree with
the frozen semantics. The first full-dataset text/state consistency scan found
11 rationale anomalies (2.75%). Two non-destructive targeted repair rounds
re-labelled those records, after which the full automatic issue count became
zero. The repaired artifact preserves the original order and scenarios, and no
numeric target changed across the repair rounds.

Two priority-conflict rationales are internally contradictory: they incorrectly
describe the nearby B agent as loaded and say the focal A agent should yield,
while emitting the state-correct high local-assertiveness score:

- `phase4-semantic-v2-priority_conflict-agv1-26f493f06ee2654b`
- `phase4-semantic-v2-priority_conflict-agv1-850f00112b3396c4`

Seven low-battery rationales incorrectly assert a nearby loaded agent although
all recorded peers are unloaded. Their task-commitment score remains correct:

- `phase4-semantic-v2-low_battery_diversion-agv1-f4438f690198a43e`
- `phase4-semantic-v2-low_battery_diversion-agv1-1d07f7c78217554f`
- `phase4-semantic-v2-low_battery_diversion-agv1-2857f48dcb9a3ddf`
- `phase4-semantic-v2-low_battery_diversion-agv1-0acbfb38138af6b8`
- `phase4-semantic-v2-low_battery_diversion-agv1-c99d41d32c2799c6`
- `phase4-semantic-v2-low_battery_diversion-agv1-354f61ded141658d`
- `phase4-semantic-v2-low_battery_diversion-agv1-6cfe292c405fafd5`

Two station-exit rationales unnecessarily claim that the load rule overrides
priority even though both focal and station agents are unloaded. The congestion
reason and low assertiveness score are otherwise correct:

- `phase4-semantic-v2-station_exit_congestion-agv1-3e7d79f4ed641a56`
- `phase4-semantic-v2-station_exit_congestion-agv1-f5110a75cf6f1b00`

## CPU interface smoke

A one-episode CPU smoke loaded all 400 records with three-neighbour offline
retrieval and made zero API calls. Both semantic component losses were non-zero.
The final checkpoint contains a `(2, 64)` semantic output weight and a `(5, 66)`
motion-head weight, proving that both detached semantic preferences reach the
actor and updater. This is an interface test, not performance evidence.

## Release decision

The final `repaired_r2` dataset is released for the local 800-episode Phase 4
feasibility experiment. The earlier artifacts remain unchanged for provenance.
Because no numeric target changed, the completed real-data CPU interface smoke
also covers the final dataset's training tensors. Do not treat its one-episode
outcome as an experimental result.
