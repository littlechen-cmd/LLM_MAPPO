# R1 训练收敛恢复——需求

## 1. 阶段目标

R1 插入 E1 与 E2 正式训练之间，用于解决当前策略任务完成率极低、训练是否收敛无法被可靠
判断的问题。R1 的核心验收对象是 `RC-AStarKD`；只有其在统一的确定性评估中达到任务完成率
要求，优化路线才允许重新进入 E2 正式矩阵。

当前 `artifacts/optimization/e2_formal_vector16_7de1f04` 及其本地下载副本只保留为诊断和
工程证据，不得作为确认性论文结果。已经中断的 RC run 不得作为正式 run 恢复。

## 2. 已确认问题

1. 当前 E1 vector runner 的 `summary.latest_episode_metrics` 只保留最后一个 worker 的最后一次
   transition，且正式 `episodes.csv` 没有完整 episode 行，因此该摘要不能证明收敛。
2. 已完成 checkpoint 的 worker snapshot 与旧 O2 完整 episode 曲线均显示当前能力约为目标的
   `2%–4%`，不能仅解释为摘要展示错误。
3. 当前距离奖励只奖励“距离减小”，不惩罚等量的“距离增大”；默认单步值为 `1.0`，而最低
   优先级任务完成奖励经 5 个 AGV 的 team mean 后为 `0.5`。这允许移动循环获得高于任务完成的
  回报，是已确认的结构性风险。
4. `16×128` 使策略每 2048 个累计 joint transitions 更新一次；相较旧链路，策略刷新节奏明显
   变慢。R1 必须隔离检查奖励与更新节奏，不能把两项变化混为一个结论。
5. DirectGoal 移除了执行期 A* waypoint。它是待验证的架构风险，但 R1 优先保留 DirectGoal，
   不直接恢复执行期 A*。
6. RC checkpoint 保存 calibration state，但现有正式恢复入口没有完整恢复 EMA。中断的 RC run
   因而不能直接续作确认性证据。

## 3. 冻结边界

- 不改变 A* Pure Motion Teacher、Reward Calibration、LLM 三维语义、动作空间、充耗电配置或
  RC/Fixed-KD 定义。
- R1 诊断 seed 与未来正式训练 seed 隔离；R1 结果不进入最终论文统计。
- 长训练、长评估和可视化 replay 由研究所有者在服务器运行；Codex 负责实现、命令和结果分析。
- `MAPPO-DG` 只承担基础学习能力诊断，不设置任务完成率硬门槛。
- `RC-AStarKD` 是唯一硬性能力 Gate。
- 每个测试训练完成后必须生成 TensorBoard 曲线、确定性评估和可视化 replay，并暂停等待研究
  所有者检查；不得连续跨过人工检查点。

## 4. 奖励修正状态

R1-B 当前只冻结问题和方向，不冻结最终公式：距离塑形必须对靠近与远离目标对称，往返循环净
奖励不得为正，目标切换不得制造奖励跳变，任务完成必须支配移动塑形。研究所有者后续提供正式
奖励方案后，R1-B 才能进入实现；在此之前禁止按临时建议修改奖励代码。

## 5. 快速诊断环境

R1-C 优先复用现有的 3 AGV、任务目标 9、动态入库简单环境进行快速因果诊断。该环境仅用于
比较奖励和更新节奏，不构成 R1 最终验收环境。如果该环境在当前代码中不存在、接口不兼容或
无法产生同口径完整 episode 指标，则直接使用 canonical 5 AGV、任务目标 50、动态入库环境，
不得为此另建第三套环境。

最终 RC-AStarKD 能力验收始终优先使用 canonical 5 AGV、目标 50 环境。若必须退回简单环境
作为最终论文环境，须由研究所有者另行批准并同步缩小论文主张。

## 6. 验收口径

- MAPPO-DG：不设固定完成率门槛；要求完整 episode 的任务完成曲线总体向好，后期没有明显
  崩溃。TensorBoard 的 reward 不能单独作为收敛证据。
- RC-AStarKD：三个训练 seed 的 final checkpoint 分别在固定 10 个确定性评估场景上运行，
  共 30 个 episode；总体平均任务完成率不低于 `90%`，每个训练 seed 的平均值不低于 `85%`。
- 动作使用 deterministic argmax；不得选择最佳 checkpoint、删除失败 seed 或用随机动作提高
  指标。
- 默认训练预算为 150k 累计 joint environment transitions。若 RC 在 150k 未达门槛但曲线仍
  明确向好，可经研究所有者批准延长到 300k；低水平停滞不得仅靠增加预算解决。
