# R1 训练收敛恢复——验证

## 文档与身份

- [ ] R1-B 正式奖励公式、系数、目标切换和充电目标边界已获研究所有者批准。
- [ ] 所有 R1 run 绑定唯一 commit、配置、环境、seed 与 checkpoint。
- [ ] R1 seed 未进入 E2 正式统计；旧 `e2_formal_vector16_7de1f04` 保持 diagnostic-only。

## 日志与恢复

- [x] `episodes.csv` 包含每个 worker 的完整 episode，计数可与 checkpoint worker state 对账。
- [x] summary 的 episode 聚合可由原始 episode 行复算，不读取部分 episode 冒充最终表现。
- [x] TensorBoard 显示完整 episode 完成率、任务数、reward、碰撞和优化指标。
- [x] RC 中断恢复完整保留 calibration EMA、计步、worker、RNG 和 learner 状态。
- [x] worker 异常可传播至主进程且所有子进程退出；失败运行不能标记 complete。

## R1-C 因果诊断

- [ ] 旧失败证据作为基线，未重复消耗一轮相同训练。
- [ ] 新奖励-only、`16×32`-only、组合方案使用相同 seed、预算、环境和评估集。
- [ ] 若使用简单环境，已证明它是现有 3 AGV/目标 9/动态入库环境；不可用时直接使用 canonical。
- [ ] 每项结束后均生成 TensorBoard、定量图和两类 replay，并取得 owner 继续结论。

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
