# 正式结果表格 Schema

所有数值必须来自冻结日志。重复训练以训练 seed 为统计单位，报告 mean ± SD 与 95% CI。

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| T1 Core 2×2 | 主比较、两教师主效应与交互 | 四核心组 | 完成率、tasks/1000 steps、steps、reward、碰撞、死亡 | held-out evaluation JSON | `eval/aggregate_formal_results.py` |
| T2 Efficiency | 样本与计算效率 | 四核心组 | AUC、达阈值步数、env steps/s、墙钟、显存、推理时延、在线调用 | episodes/updates/runtime/summary | G6 聚合脚本 |
| T3 Robustness | 泛化和能源压力 | 组别×场景 | 性能退化、充电暴露、恢复、死亡、拥堵 | robustness evaluation JSON | G6 聚合脚本 |
| T4 Priority | 优先级与饥饿 | 组别×优先级 | 完成时延、最大等待、未完成年龄 | task-level evaluation logs | G6 聚合脚本 |
| T5 Teacher quality | 标签质量与成本 | 四 LLM 教师 | 合法率、一致率、稳定性、时延、token、成本 | label audit logs | 条件性教师实验 |

不得创建不能支持方法主张的表；规划阶段不填充模拟数值。

T1 的独立统计单位为训练 seed。脚本输出：

- `per_training_seed.csv`：每组、每训练 seed、每指标一行；
- `group_summary.csv`：mean、SD、训练 seed bootstrap 95% CI；
- `paired_full_vs_baseline.csv`：完整方法相对 MAPPO-WP 的配对差值、配对效应量
  $d_z$、精确符号翻转检验及 Holm 校正结果。
- `paired_factorial_effects.csv`：完整方法对基线、A*KD 主效应、LLMKD 主效应及
  2×2 交互效应；每个对比内部对六项核心指标进行 Holm 校正。
