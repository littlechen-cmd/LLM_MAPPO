# R1-C 4-AGV LowLoad 重规划审查

## 审查范围

审查 `LLM_MAPPO_curriculum_pretraining_plan_v1.md` 是否适合作为 R1-C 环境重规划参考，并对照
当前 Reward-v2、R1 规格和 E1 vector runner 评估算法价值、实现可行性与论文边界。

## 三视角结论

### 算法诊断

4 AGV 恰好填满三个匿名邻居槽，比原 3 AGV/目标9更接近5 AGV Formal观测。保持正式地图并
降低任务到达压力，有助于区分基础任务链不可学与正式负载探索困难。R1-C只运行MAPPO-DG，
避免A*/RC/LLM干扰奖励与更新节奏诊断。

### 实现正确性

底层环境、613D observation、共享Actor、attention Critic、action mask与16-worker rollout均支持
4 AGV。阻断项是正式runner写死5 AGV、run/checkpoint未绑定完整环境身份，以及现有评估/replay
不能加载E1 checkpoint。`queue_size=4`并非活动任务硬上限；候选profile初始产生4–8个任务，
之后每40步加入2–4个。

### 规格与论文边界

环境改变后，旧5 AGV legacy结果不能作为4 AGV control，因此采用完整四格`2×2`。4 AGV仅为
diagnostic，不进入正式统计，不迁移checkpoint，不证明curriculum有效。最终Gate仍是5 AGV
Formal上的RC-AStarKD总体≥90%、每训练seed≥85%。

## 采纳

- 固定4-AGV LowLoad：formal topology、batch 2–4、queue4、target20、interval40、max1000。
- 固定四个MAPPO-DG scratch arm、seed9107、每组50k与评估seeds9300–9304。
- R1-C前补独立runner、环境身份和E1评估/replay。
- R1-D仍先运行5-AGV Formal scratch。

四组共享初始参数hash、worker seed派生规则和初始环境RNG；由于策略、终止与reset随后会分化，
不虚假要求逐step随机轨迹始终相同。

## 拒绝或暂缓

- 拒绝用旧5 AGV产物替代同环境control。
- 拒绝把`queue_size=4`解释为活动任务硬上限并新增限流逻辑。
- 暂缓4→5 curriculum、共享warm start和4 AGV checkpoint迁移。
- 只有5-AGV Formal scratch失败后才增加5-AGV LowLoad诊断。
