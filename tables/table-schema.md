# 正式结果表格 Schema

所有数值必须来自 D1 后冻结日志。重复训练以训练 seed 为统计单位，报告 mean ± SD 与 95% CI。
同一 checkpoint 的 `10 × 20` 评估回合先按 checkpoint 聚合，不能作为独立统计样本。

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| T1 Core 2×2 | 主比较、两教师主效应与交互 | 优化：四组 × 8 seed；稳定：四组 × 5 seed | 主要任务吞吐、完成率、steps、reward、碰撞、死亡 | selected-route held-out JSON | E1 适配后的聚合器 |
| T2 Required baselines | 规则语义、Reward Calibration、规划及外部 MARL | 优化：Fixed-KD、QMIX-DG、RuleKD-v3、启发式A*；稳定：RuleKD、启发式A* | 主要任务吞吐、安全、95% CI、效应量 | selected-route held-out JSON | E1/E3 聚合脚本 |
| T3 Efficiency | 样本与计算效率 | 所选路线正式学习组 | 标准化吞吐AUC、达阈值步数、env steps/s、墙钟、显存、A*时延、在线调用 | episodes/updates/runtime/summary | E3 |
| T4 Robustness and priority | 优先级；优化路线另含跨拓扑 | 所选组 × 场景/优先级 | 性能退化、优先级时延、充电暴露、恢复、死亡、拥堵 | robustness and task-level JSON | E3 |
| T5 Teacher quality | 标签质量、规则对照与成本 | 优化：连续3D LLM/RuleKD-v3/ShuffleKD-v3/NoOOD/盲审；稳定：LLM/规则 | validity、OOD、覆盖、时延、成本、评价者间一致性 | label audit logs | E1/E3 |

不得创建不能支持方法主张的表；规划阶段不填充模拟数值。

T1–T3 的独立统计单位为训练 seed。优化路线已经预注册七个 primary contrasts，并在该族内执行
Holm 校正；E1 只能在看结果前确认实现映射，不能按结果改族。NoGoalHint/Shuffle/NoOOD 始终为
诊断性；稳定路线另行冻结适用统计族。

当前核心流水线输出：

- `per_training_seed.csv`：每组、每训练 seed、每指标一行；
- `group_summary.csv`：mean、SD、训练 seed bootstrap 95% CI；
- `paired_full_vs_baseline.csv`：完整方法相对 MAPPO-DG 的配对差值、95% CI、配对效应量
  $d_z$、paired t-test 及 Holm 校正结果。
- `paired_factorial_effects.csv`：完整方法对基线、A*KD 主效应、LLMKD 主效应及
  2×2 交互效应；当前核心组内每个对比对所有评估指标进行 Holm 校正。
- `learning_curve_auc_per_training_seed.csv`：每个核心组、每个训练 seed 的标准化吞吐 AUC；
- `learning_curve_auc_summary.csv`：AUC 的 mean、SD 与训练 seed bootstrap 95% CI。

Fixed-KD、QMIX-DG、RuleKD-v3、启发式 A*、O3 未见拓扑和盲审的正式输出路径在 E1 补入，
不得用同图 8-AGV、占位或模拟数值替代真实冻结日志。
