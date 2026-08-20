# 研究进度

## 2026-08-15

- 阶段：S0 范围确认完成，进入 S3 实验规划。
- 已确认论文类型、方法主线、核心四组对照、训练预算、统计协议、教师模型、
  时间窗口与 Constitution 输出形式。
- 当前任务：编写第一版中文 `CONSTITUTION.md`。
- 尚未开始论文正文写作，也未生成或使用模拟实验结果。

### Constitution v0.1 交付记录

- 已生成根目录 `CONSTITUTION.md`。
- 已完成规格符合性审查与质量审查。
- 已同步 `TASKS.md` 的后期实验与论文任务。

### Capability-use audit

- **Required skills**：using-research-writing、brainstorming-research、
  paper-orchestration、experiment-results-planning、verification；涉及 GPT 模型时使用
  openai-docs。
- **Skills actually used**：以上技能均已使用。
- **Inputs consumed**：用户草稿与确认项、两份任务交接、`repomix-core.xml`、
  `AGENTS.md`、`TASKS.md`、`requirement.md`、Phase 4 配置和官方 OpenAI 模型文档。
- **Inputs not used and why**：未使用训练 artifact 的实际数值，因为当前任务不授权
  实验审计，也不得把未核验结果写成证据。
- **Artifacts produced**：`CONSTITUTION.md`、`plan/project-overview.md`、
  `plan/outline.md`、`plan/progress.md`、任务包和两份审查记录。
- **Verification run**：文件存在性、必需章节/术语、字符与行数、Markdown 空白和
  Git 差异检查。
- **Remaining risk**：目标期刊未定；核心实验配置、A* 因子、当前训练产物和教师
  API 可用性仍需在对应阶段门中验证。

## 2026-08-17

- G3 并行准备启动；充电参数仍等待 G2-3，未标记为正式冻结。
- 已确认正式训练 seed `7/17/27/37/47`、评估 seed `0–9 × 20 episodes`、确定性动作和
  final-checkpoint 规则。
- 已将 5-AGV Phase 4 改造为独立 A* KL/LLMKD 因子，并为无 LLM 组定义固定零语义输入。
- G4 pilot 使用精确的 150,000 team-environment steps；正式预算数值等待 G4 后统一写入。
- 按实验规划硬门建立协议、可追溯矩阵、表格 schema 与图数据清单；未生成模拟结果。
## 2026-08-17 — G3 formal evidence pipeline

- Added strict four-group × five-training-seed held-out evaluation aggregation.
- Fixed the statistical unit at training seed and implemented bootstrap confidence
  intervals, exact paired sign-flip tests, paired effect sizes, and Holm correction.
- Added real-data learning-curve and core-comparison plotting scripts; neither script
  fabricates missing runs or fills incomplete matrices.
- Added cumulative environment steps to formal episode logs.

## 2026-08-20 — G2-3 core energy selection

- Selected `1.10/0.30/0.80` for all four core groups after matched 200-episode
  retraining and deterministic 10×20 diagnostic evaluation.
- Rejected `1.10/0.25/0.80` despite its higher throughput because held-out
  charging exposure fell to 1% and one energy death occurred; the selected 0.30
  threshold had 20% charging episodes and zero energy deaths.
- Reserved seeds `200–209` for formal evaluation because calibration consumed
  seeds `0–9`; no formal result is inferred from the calibration comparison.
- Charging remains a fixed rule-layer safety mechanism, not an autonomous MAPPO
  decision or an LLMKD contribution.
