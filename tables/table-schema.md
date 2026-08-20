# 正式结果表格 Schema

所有数值必须来自冻结日志。重复训练以训练 seed 为统计单位，报告 mean ± SD 与 95% CI。
同一 checkpoint 的 `10 × 20` 评估回合先按 checkpoint 聚合，不能作为独立统计样本。

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| T1 Core 2×2 | 主比较、两教师主效应与交互 | 四核心组 × 8 seed | 主要任务吞吐、完成率、steps、reward、碰撞、死亡 | held-out evaluation JSON | `eval/aggregate_formal_results.py` |
| T2 Required baselines | 外部MARL、规则语义与规划比较 | 完整方法、QMIX-WP、RuleKD、启发式A* | 主要任务吞吐、安全、效应量 | held-out evaluation JSON | G3-5/G6 聚合脚本 |
| T3 Efficiency | 样本与计算效率 | 正式学习组 × 8 seed | 标准化吞吐AUC、达阈值步数、env steps/s、墙钟、显存、A*时延、在线调用 | episodes/updates/runtime/summary | `eval/aggregate_formal_results.py` + G6 |
| T4 Robustness and priority | 泛化、能源压力与优先级 | 组别 × 未见布局/场景/优先级 | 性能退化、优先级时延、充电暴露、恢复、死亡、拥堵 | robustness and task-level JSON | G6 聚合脚本 |
| T5 Teacher quality | 标签质量、规则对照与成本 | LLM、规则、扰乱标签、双人盲审 | 合法率、一致率、稳定性、时延、成本、评价者间一致性 | label audit logs | G3-5/G6 |

不得创建不能支持方法主张的表；规划阶段不填充模拟数值。

T1–T3 的独立统计单位为训练 seed。G3 冻结的确认性比较族为：完整方法对 MAPPO-WP、
A*KD 主效应、LLMKD 主效应、完整方法对 QMIX-WP、完整方法对 RuleKD；五项比较使用
同一 Holm 校正族。交互效应、其余指标、NoWP 和ShuffleKD均为探索性或诊断性分析。

当前核心流水线输出：

- `per_training_seed.csv`：每组、每训练 seed、每指标一行；
- `group_summary.csv`：mean、SD、训练 seed bootstrap 95% CI；
- `paired_full_vs_baseline.csv`：完整方法相对 MAPPO-WP 的配对差值、配对效应量
  $d_z$、精确符号翻转检验及 Holm 校正结果。
- `paired_factorial_effects.csv`：完整方法对基线、A*KD 主效应、LLMKD 主效应及
  2×2 交互效应；当前核心组内每个对比对所有评估指标进行 Holm 校正。
- `learning_curve_auc_per_training_seed.csv`：每个核心组、每个训练 seed 的标准化吞吐 AUC；
- `learning_curve_auc_summary.csv`：AUC 的 mean、SD 与训练 seed bootstrap 95% CI。

QMIX-WP、RuleKD、启发式A*、未见布局和盲审的正式输出路径在 G3-5/G3-6 实现后补入，
不得用占位或模拟数值替代真实冻结日志。
