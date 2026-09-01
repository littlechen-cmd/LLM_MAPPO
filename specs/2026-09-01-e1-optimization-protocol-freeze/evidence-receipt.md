# E1 Evidence Receipt — Governance Closeout

## Status

`implementation_complete_governance_closeout_in_progress`

本 receipt 记录截至 2026-09-02 已取得的证据和仍待完成的审计。它不把 raw LLM strict
Gate 的 No-Go 改写为 Go，也不因 E2 已启动而自动宣称 E1 全部 Gate 完成。

## Frozen identities

| Item | Frozen value |
| --- | --- |
| Route | optimization |
| Formal matrix | 65 learning runs |
| Per-run budget | 150000 aggregate joint environment transitions |
| MAPPO execution | 4 learner slots; 16 spawned CPU env workers per learner; rollout length 128; 2048 transitions/update |
| QMIX-DG execution | single-environment trainer; up to four independent runs |
| Training GPU | physical GPU 0, NVIDIA GeForce RTX 4090 |
| RTX 4080 SUPER | training and CUDA smoke prohibited |
| E2 launch implementation | `7de1f04c772ccf49d422a53aa0c1ad01deec9204` |
| E2 artifact root | `artifacts/optimization/e2_formal_vector16_7de1f04` |
| Governance manifest SHA256 before this closeout | `eb293c7a01286ffbb9d79637cfde3140b653b2df30b8da6b99032c2ed06a27f5` |

## Raw semantic evidence

| Item | Recorded value |
| --- | --- |
| Evidence role | immutable exploratory noisy-teacher evidence |
| Request/response model | `deepseek-v4-pro` / `deepseek-v4-pro` |
| Prompt | `semantic-prompt-v5-state-contract` |
| Backend fingerprint | `a307abda487cd1b463329ccb945ce396` |
| Attempts / records / valid | 800 / 800 / 799 |
| Records SHA256 | `a108d35e8398114baec1cf88dbe34446efaefda4c9066725d5e7a57a0fed2512` |
| Strict semantic-quality Gate | No-Go, preserved |
| Manifest anomaly | collection content complete, but source manifest retains stale `status: running`; owner accepted disclosure without editing the immutable source |

## CUDA evidence

- Owner-reported CUDA smoke root:
  `/home/lzx/llm-a-mappo/artifacts/optimization/e1_cuda_smoke/20260901T145230Z`.
- Smoke implementation commit: `5f56f20349acd97b25b1446e89898d75ca9c71ab`.
- Reported aggregate: 8 members, 2048 total environment steps, physical GPU 0 only,
  aggregate pass true.
- Evidence boundary: this validates the E1 functional paths, not the later 16-worker rollout
  introduced by `7de1f04`. The owner explicitly waived a new standalone smoke and accepted early
  formal-run stability as engineering health evidence.

## Allowed claims

- MAPPO formal rollout uses 16 real CPU environment workers and one centralized GPU learner.
- Raw offline LLM supervision is imperfect exploratory noisy-teacher evidence.
- Training and execution make no online LLM calls; optimized Student execution planner queries
  remain subject to the frozen zero-query evaluation audit.
- Throughput and ETA may be reported as engineering monitoring only.

## Prohibited claims

- The raw LLM semantic dataset passed its strict semantic-correctness Gate.
- `validity × OOD reliability` is a semantic-correctness confidence.
- CUDA throughput demonstrates an algorithmic performance advantage.
- E1 is fully closed before the pending matrix identity/resume/duplicate audit and branch merge.

## Pending closeout evidence

- The current machine-readable governance validator still emits the legacy
  `artifacts/optimization/e2_formal` root. Its Go result proves the frozen 65-run
  combinatorics only; it must not be used to locate the running matrix. Backfill the
  manifest and validator to `artifacts/optimization/e2_formal_vector16_7de1f04` only
  after the post-run identity audit, without changing or interrupting the matrix already
  running from `7de1f04`.
- Verify all 65 run identities use `7de1f04`, the frozen raw hash and one configuration family.
- Audit maximum simultaneous learner count, failed/restarted members, resume use, duplicate
  artifacts and final-checkpoint uniqueness.
- Any resumed RC/Fixed run must be reviewed for the known missing EMA restore path and may not be
  admitted silently.
- Fast-forward the approved E1 closeout to `codex/optimization`, then push or provide the exact
  owner command.
