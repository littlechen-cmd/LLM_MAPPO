# R1 训练收敛恢复——验证

## 文档与身份

- [x] R1-B Reward-v2 公式、系数、目标切换和充电目标边界已获研究所有者批准并实现。
- [ ] 所有 R1 run 绑定唯一 commit、配置、环境、seed 与 checkpoint。
- [ ] R1 seed 未进入 E2 正式统计；旧 `e2_formal_vector16_7de1f04` 保持 diagnostic-only。

## 日志与恢复

- [x] `episodes.csv` 包含每个 worker 的完整 episode，计数可与 checkpoint worker state 对账。
- [x] summary 的 episode 聚合可由原始 episode 行复算，不读取部分 episode 冒充最终表现。
- [x] TensorBoard 显示完整 episode 完成率、任务数、reward、碰撞和优化指标。
- [x] RC 中断恢复完整保留 calibration EMA、计步、worker、RNG 和 learner 状态。
- [x] worker 异常可传播至主进程且所有子进程退出；失败运行不能标记 complete。

## Reward-v2

- [x] 完成奖励是未除以 AGV 数的团队 `+10W`，成功取货是局部 `+2W`。
- [x] 同目标距离变化为有符号 `0.1(d_t-d_{t+1})`，目标切换当步为 0。
- [x] 阻塞、碰撞、单步成本、NOOP 和既有能源项与正式合同一致。
- [x] `legacy-v1/reward-v2` 可显式选择且进入 shadow/运行身份；优化入口默认 Reward-v2。

## R1-C 因果诊断

- [ ] 四组均绑定同一4-AGV LowLoad profile/layout hash、seed `9107`、50k预算、初始参数hash、
  worker seed派生规则、初始环境RNG和evaluation seeds `9300–9304`。
- [ ] profile 同时冻结 `initial_priority_label=A`、无额外 priority schedule，以及现有充耗电机制；
  不得为LowLoad新增任务限流或修改动作掩码。
- [ ] `legacy/reward-v2 × 16×128/16×32` 四格完整；旧 5-AGV 产物未冒充同环境 control。
- [ ] 四组均仅运行 MAPPO-DG、从零初始化，未启用教师、跨组恢复或 4→5 checkpoint 迁移。
- [ ] run/checkpoint identity 包含完整环境 hash；跨 AGV 数、负载、奖励或 rollout 恢复 fail closed。
- [ ] 每组至少40个完整episode，最后20个平均完成率高于最早20个；否则不得声称向好趋势。
- [ ] 每项结束后均生成TensorBoard、固定评估、seed9300与最低完成率replay；有成功episode时
  另生成成功replay，无成功时显式记录unavailable；每项均取得owner继续结论。

## 能力 Gate

- [ ] MAPPO-DG 的完整 episode 指标总体向好、后期无明显崩溃，owner 已批准进入 RC。
- [ ] RC-AStarKD 三个训练 seed 均使用 final checkpoint 和 seeds `9200–9209` 确定性评估。
- [ ] 30 个 episode 总体平均任务完成率 `≥90%`。
- [ ] 每个训练 seed 的 10-episode 平均任务完成率 `≥85%`。
- [ ] 未选择最佳 checkpoint、删除失败 seed、修改评估 seed 或用 LLMKD 掩盖基础能力问题。

## Go/No-Go

- **Go**：以上全部通过，冻结统一合同并重新进入 E2。
- **No-Go**：RC 未达到硬门槛、日志不能证明完整 episode、恢复身份不完整，或人工可视化发现
  明显行为退化。返回对应 R1 任务，不启动正式矩阵。
