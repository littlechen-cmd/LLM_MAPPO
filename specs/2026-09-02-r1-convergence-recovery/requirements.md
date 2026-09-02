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

## 4. Reward-v2 正式合同

研究所有者已批准 Reward-v2。它只改变奖励，不改变环境动力学、动作/观测、Student、A*、RC 或
LLM。对任意 `N` 个 AGV，一个真实环境步的团队奖励固定为：

`10 × Σ完成任务W + mean_i(0.1Δd_i + 2W_i I_pickup - 0.15I_blocked - 2I_collision + r_i_energy) - 0.01`

- 优先级权重沿用 `W=0.5+1.5(L-r-1)/(L-1)`，范围 `[0.5, 2.0]`；只有一个优先级时 `W=1`。
- 完成任务是团队项 `+10W`，不得再被 AGV 数量平均稀释；成功取货是执行机器人局部项 `+2W`。
- `Δd=d_t-d_{t+1}`：同一 DirectGoal 下靠近一格为 `+0.1`，远离一格为 `-0.1`。任务、取货/送达
  或充电目标发生切换时，该步 `Δd=0`。
- 前进被阻塞为执行机器人局部 `-0.15`；碰撞保持局部 `-2.0`；电量不足与耗尽规则保持原实现。
- 合理 `NOOP` 没有额外局部惩罚，但仍承担统一团队单步成本 `-0.01`。不增加停滞惩罚。
- A*、LLM、Student disagreement 与 Reward Calibration 均不得直接产生环境奖励。
- `legacy-v1` 仅用于 R1-C 因果对照；优化训练入口默认 `reward-v2`，奖励版本必须进入运行身份、
  checkpoint/shadow 配置身份和评估配置，禁止跨奖励版本恢复。

## 5. 快速诊断环境

R1-C 唯一诊断环境为 canonical medium formal topology 上的 `4-AGV LowLoad` profile：

- `n_agents=4`、`batch_size_range=[2,4]`、`queue_size=4`、`task_target=20`；
- `dynamic_ingress_interval=40`、`max_steps=1000`、`deadlock_steps=180`；
- `initial_priority_label=A`，不设置额外 `priority_schedule`；
- 电量配置 `battery_cost_scale=1.10`、`charge_threshold=0.30`、
  `charge_release_threshold=0.80`；
- DirectGoal、动作空间、动作掩码、优先级语义、地图布局与正式 5-AGV 环境一致。

`queue_size=4` 不是活动任务硬上限。两个初始批次各产生 2–4 个任务，因此初始活动任务为
4–8 个；之后每 40 步继续加入 2–4 个任务。R1-C 不新增限流规则或 Gym topology ID。

4 AGV 正好让每台机器人拥有三个真实 peer，填满 semantic-view-v3 的三个匿名邻居槽；相较旧
3-AGV/目标9诊断，它与正式 5-AGV 观测分布更接近。该 profile 仅用于可学习性和条件效应诊断，
不构成 curriculum stage、warm start 或最终论文验收环境。

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

## 7. R1-C 允许主张

R1-C 是单 seed、50k 步的 diagnostic `2×2`，不进入论文正式统计。四个 MAPPO-DG arm 在相同
4-AGV LowLoad、Actor/Critic初始参数hash、训练seed `9107`、worker seed派生规则、初始环境RNG、
预算和评估集下运行。策略行为分化后不要求各环境维持逐步相同随机轨迹：

1. `legacy-r128`：`legacy-v1 + 16×128`；
2. `reward-v2-r128`：`reward-v2 + 16×128`；
3. `legacy-r32`：`legacy-v1 + 16×32`；
4. `reward-v2-r32`：`reward-v2 + 16×32`。

旧 5-AGV 失败产物只作背景证据，不能替代上述任一 arm。四组不得加载彼此 checkpoint，不启用
A*、RC 或 LLM 教师，也不得把 4-AGV checkpoint 迁移到 R1-D/E。R1-C 只允许判断该诊断环境中
reward 与更新节奏的可学习性和条件效应，不允许宣称正式 5-AGV 性能或 curriculum 有效。

R1-C 的唯一配置、runner 和 artifact root 分别为
`configs/optimization/r1_4agv_lowload.yaml`、`scripts/run_r1_diagnostics.py` 与
`artifacts/optimization/r1_convergence/r1c_4agv_lowload/`。每个arm必须在manifest记录profile、
layout、代码、初始参数与评估身份，禁止依靠目录名推断实验条件。
