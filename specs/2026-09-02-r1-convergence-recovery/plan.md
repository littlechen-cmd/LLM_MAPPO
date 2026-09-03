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

### R1-C0：诊断基础设施

- [x] 增加独立配置 `configs/optimization/r1_4agv_lowload.yaml` 与 runner
  `scripts/run_r1_diagnostics.py`；不得修改 E1 formal matrix 或建立新的 Gym topology ID。唯一
  artifact root 为 `artifacts/optimization/r1_convergence/r1c_4agv_lowload/`。
- [x] 冻结 requirements 定义的 4-AGV LowLoad profile，并把完整环境配置、layout hash、奖励版本、
  worker 数和 rollout length 纳入 run/checkpoint/resume identity；跨 profile 恢复必须拒绝。
- [x] 补齐 E1 checkpoint 的确定性评估与 replay 接口，能够复建同一 4-AGV profile。始终输出
  seed `9300` 固定回放和最低完成率回放；若五个评估 episode 中存在成功场景，再输出一段成功
  回放，否则显式记录 `successful_replay=unavailable`；最低完成率或成功场景并列时选择最小
  evaluation seed；不得用旧 Phase2/3 loader 代替。
- [x] 验证 4-AGV 的 613D/61D observation、三邻居无 padding、action mask、Critic、A* query=0、
  16-worker stream/GAE 隔离和 Reward-v2 团队完成奖励不随 `N=4` 稀释。

### R1-C1：严格 `2×2` MAPPO-DG 诊断

四组均从相同随机初始化开始，固定训练 seed `9107`、50k 累计 joint environment transitions、
16 workers、相同 worker seed 派生规则和初始环境 RNG 状态。策略分化、提前终止和 reset 后不要求
保持逐步相同随机轨迹，但必须记录相同的初始 Actor/Critic 参数 hash：

1. `legacy-r128`：`legacy-v1 + 16×128`；
2. `reward-v2-r128`：`reward-v2 + 16×128`；
3. `legacy-r32`：`legacy-v1 + 16×32`；
4. `reward-v2-r32`：`reward-v2 + 16×32`。

`16×128` 每次更新收集 2048 个累计 transitions；`16×32` 收集 512 个。四组只运行 MAPPO-DG，
不启用 A*、RC、LLM 或跨组 checkpoint。旧 5-AGV 失败产物不能充当第四个 control。

每组 final checkpoint 固定使用 evaluation seeds `9300–9304`，deterministic argmax，每 seed 一个
episode。至少产生 40 个完整训练 episode；最近 20 个完整 episode 平均完成率必须高于最早
20 个，才具备“向好趋势”最低证据。最终是否继续仍在每组 R1-V 后由研究所有者明确批准；
Codex 不根据 reward 单独选择方案。

Reward-v2 是后续唯一候选奖励。`16×32` 只有在 Reward-v2 条件下相较 `16×128` 获得更好的完整
episode/固定评估表现且人工回放无明显退化时才进入 R1-D；否则保留 `16×128`。若两个 Reward-v2
arm 均无向好趋势，R1-C No-Go，先返回根因分析，不进入 R1-D。

## R1-D：MAPPO-DG 基础学习能力

- 采用 R1-C 选定的奖励与更新节奏，在 canonical 5 AGV、目标 50、动态入库环境训练。
- 从随机初始化开始，禁止加载 R1-C 的 4-AGV checkpoint；本阶段首先检验 5-AGV Formal scratch。
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

1. 若 4-AGV LowLoad 成功而 5-AGV Formal scratch 失败，增加 5-AGV LowLoad scratch 诊断；
2. 5-AGV LowLoad 失败优先检查协调规模、观测、动作掩码与 DirectGoal；5-AGV LowLoad 成功但
   Formal 失败，才把高负载探索难度和 curriculum 作为候选解释；
3. curriculum 若被提出，必须另行冻结所有正式方法共同的 difficulty schedule，并把各 stage
   交互量计入总预算；不得直接复用 R1-C checkpoint 冒充正式 scratch；
4. 运行一次旧执行期 A* waypoint 诊断对照，判断 DirectGoal 是否是主要瓶颈；
5. 只有前述步骤失败且研究所有者再次批准，才允许把执行期 waypoint 作为正式回退。

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
