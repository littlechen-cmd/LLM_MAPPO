# R1 训练收敛恢复——实施计划

## 状态

`[~] R1-A/R1-B 已完成；下一任务为 R1-C 因果诊断准备。`

## R1-A：修复可观测性与恢复语义

- [x] 为 16 个 worker 分别记录完整 episode，至少包含 worker、episode、seed、终止时累计环境步、
  完成任务数/完成率、reward、碰撞、充电、死亡、死锁和 episode steps。
- [x] `summary.json` 从最近 100 个完整 episode 聚合；不足 100 个时使用全部已有完整 episode，
  不再用最后一个 worker 的部分 episode。
- [x] 非短 smoke 运行若没有产生合法完整 episode 记录，证据检查 fail closed。
- [x] checkpoint 恢复同时恢复 learner、worker、计步、RNG 和 RC calibration/EMA 状态；修正日志与
  checkpoint 的更新边界，避免日志领先于可恢复状态。
- [x] TensorBoard 至少记录完整 episode 完成率、完成任务数、team reward、碰撞、policy/value loss、
  entropy、更新步和吞吐量。

## R1-B：冻结并实现正式奖励方案

- [x] 冻结研究所有者提供的 Reward-v2：团队完成 `+10W`、局部取货 `+2W`、有符号进展
  `0.1(d_t-d_{t+1})`、局部阻塞 `-0.15`、局部碰撞 `-2` 和团队单步成本 `-0.01`。
- [x] 目标身份变化时进展为 0；保留既有电量奖励/惩罚；不增加 NOOP、停滞或教师奖励。
- [x] 实现显式 `legacy-v1/reward-v2` 版本，正式优化入口默认 Reward-v2；运行、shadow 与恢复身份
  隔离，防止新旧奖励混用。
- [x] 完成公式、优先级、取货、阻塞、目标切换、等待和跨模块回归；未使用训练结果调参。

## R1-C：隔离奖励与更新节奏

使用当前失败产物作为旧配置基线，不重复训练。固定诊断 seed `9107`，每项最多 50k 累计环境
steps；优先使用 requirements 中定义的简单环境，否则使用 canonical 环境：

1. 只使用正式新奖励，保持 `16×128`；
2. 只将更新节奏改为 `16×32`，保留旧奖励；
3. 同时使用正式新奖励与 `16×32`。

`16×32` 表示每 512 个累计 joint transitions 更新一次。`16×64` 只在 `16×32` 出现可复现的
数值不稳定时作为备选，不与前述三个主诊断并行搜索。

每一项完成后必须执行 R1-V 可视化评估并暂停，由研究所有者检查后才进入下一项。

## R1-D：MAPPO-DG 基础学习能力

- 采用 R1-C 选定的奖励与更新节奏，在 canonical 5 AGV、目标 50、动态入库环境训练。
- 使用训练 seed `9107`，保存 50k、100k、150k checkpoint。
- 对三个 checkpoint 运行相同确定性评估并生成 TensorBoard/图表/replay。
- 不设置完成率硬门槛；仅在完整 episode 完成曲线总体向好且后期无明显崩溃时，提交研究所有者
  决定是否进入 R1-E。

## R1-E：RC-AStarKD 能力恢复

- 先使用 seed `9107` 运行 150k 并执行 R1-V；人工确认方向正确后，再运行 `9117/9127`。
- 环境、奖励、更新节奏、预算和评估协议与 R1-D 一致；只开启既有 RC-AStarKD 链路，不接入
  LLMKD。
- 最终使用固定评估 seeds `9200–9209`，每个模型每 seed 一个 episode，共 30 个 episode。
- 达到 requirements 的 `总体≥90%且每训练 seed≥85%` 才通过 R1。

## R1-F：DirectGoal 与难度升级树

只有 R1-D/E 未形成向好趋势时才按顺序处理：

1. 检查 A* 训练教师覆盖、观测信息和动作掩码是否足以支持 DirectGoal；
2. 使用简单环境到 canonical 环境的课程训练，但最终仍在 canonical 环境验收；
3. 运行一次旧执行期 A* waypoint 诊断对照，判断 DirectGoal 是否是主要瓶颈；
4. 只有前述步骤失败且研究所有者再次批准，才允许把执行期 waypoint 作为正式回退。

任何环境或 waypoint 回退都不得静默改变论文方法主张。

## R1-V：每次测试后的人工可视化检查

每个 R1-C/D/E 测试训练完成后，统一生成：

- TensorBoard 完整 episode 曲线；
- 固定评估集上的完成率、完成任务数、碰撞、充电与 reward 对比图；
- 固定场景的一段确定性 replay；
- 完成率最低场景的一段失败 replay。

产物必须绑定训练 commit、配置、seed、checkpoint 和评估 seed。研究所有者查看后明确继续、返回
上一任务或启用升级树；Codex 不代替该人工检查点。

## R1-G：收口与重新进入 E2

- 汇总 R1 诊断、人工检查结论和最终 Gate receipt。
- 统一冻结可复现 commit、奖励、rollout、环境、seed、日志与 resume 合同。
- 将旧 E2/E1 训练产物标记为 diagnostic-only，新的 E2 从统一 commit 重新开始。
- R1 Gate 未通过时，E2 保持暂停，不运行 65-run 确认性矩阵或 O3 learned-policy 评估。
