# R1 收敛问题复盘

## 结论

当前 E1/E2 训练不能作为确认性正式证据。摘要展示确有缺陷，但多来源证据一致表明当前策略在
canonical 环境中只完成约 `2%–4%` 的目标，问题不是单个 JSON 字段造成的假象。应暂停 65-run
矩阵，在其前插入 R1，先恢复可测量性、奖励目标与基础学习能力。

## 三个独立审查视角

### 1. 实现正确性

- vector runner 只把最后一个 worker 的部分 episode 写入 summary；正式 `episodes.csv` 为空。
- checkpoint 保存 calibration state，但正式恢复入口没有恢复 RC EMA。
- 日志追加与 checkpoint 保存不是同一提交边界，中断后可能出现日志领先状态。
- 已完成 run 来自多个 implementation commit，不能合并成一组冻结正式证据。

### 2. 学习算法

- 当前距离塑形是单边正奖励，往返可以保留净正收益；其量级可超过最低任务完成 team reward。
- `16×128` 没有减少每个样本的 PPO epoch，但将策略刷新间隔从约 512 增至 2048 transitions，
  需要与奖励问题分开诊断。
- O2 在单环境旧更新节奏下同样表现很低，因此并行 rollout 不是唯一原因。
- DirectGoal 相比旧 Phase 4 移除了执行期 waypoint，是重要架构差异，但尚不能直接断言为唯一根因。

### 3. 规格与论文证据

- O2 Gate 只约束相对 AUC 退化和教师覆盖，没有基础策略绝对能力门，因此工程 Gate 通过不等于
  学会任务。
- 当前 E2 manifest、artifact 身份和实际执行存在漂移；在能力恢复前继续 65-run 只会扩大无效
  计算量。
- LLM 教师不能合理地承担把约 `2%` 基础完成率提升为可用策略的责任；应先用 RC-AStarKD
  证明无 LLM 的核心链路具备能力。

## 证据摘要

- 已完成 MAPPO-DG checkpoint 的 128 个 worker snapshot：平均约 `0.92/50` 个任务。
- 已完成 RC-AStarKD checkpoint 的 64 个 worker snapshot：平均约 `1.00/50` 个任务。
- 旧 O2 的完整 episode 后 30 回合约为 `2.2%–4.4%`（MAPPO）和 `2.2%–2.7%`（RC）。
- 旧 Phase 4 在约 150k 累计 steps 已达到约 `97.8%`，说明“150k 必然不够”不能单独解释当前
  失败；但旧 Phase 4 含 waypoint/旧教师/旧环境合同，不能作为严格同配置对照。

## 决策

- 暂停 E2；现有结果只保留为 diagnostic/infrastructure evidence。
- 新增 R1-A 至 R1-G，并以 RC-AStarKD `总体≥90%、每 seed≥85%` 为重新进入 E2 的硬门。
- MAPPO-DG 不设完成率硬阈值，只判断完整 episode 是否形成向好趋势。
- 每次测试训练后必须提供 TensorBoard 与可视化 replay，由研究所有者人工检查。
- R1-B 的正式奖励合同待研究所有者提供，规划批准不等于授权按临时公式修改代码。
