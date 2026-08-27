# 方法—实验可追溯矩阵

本矩阵描述 D1 前允许规划的证据链。表格编号在 E1 冻结，结果状态在 E2/E3 更新。

| Contribution / boundary | Method module | Optimization evidence | Stable evidence | Allowed claim before results | Status |
|---|---|---|---|---|---|
| 异构双教师 MAPPO | Pure Motion A* + 离线三维语义教师 | 核心 2×2，8 seed | 历史路径教师+二维语义，5 seed | 框架和公平消融设计已定义 | 待 D1/E1 冻结 |
| A* 教师贡献 | O0 批准的局部教师或 S1 历史路径教师 | O1 纯度、O2 覆盖/AUC、正式主效应 | 历史行为恢复、正式主效应 | 只能描述为路径/运动先验，不称调度 oracle | 待 O0/S1 |
| LLM 语义教师贡献 | `task_persistence/yielding_preference/coordination_risk` | LLMKD 主效应、RuleKD-v3、ShuffleKD-v3、NoOOD、盲审 | 历史 LLMKD 主效应、RuleKD | 离线连续语义监督；在线调用为 0 | 正式结果待定 |
| 外部 MARL 竞争力 | QMIX-DG | 8 seed 匹配比较 | 不实施 | 仅优化路线可形成外部算法比较 | 待 E1 |
| Reward Calibration | RC-KD vs Fixed-KD | 8 seed 配对差异 | 不实施 | 只检验 `c_A_reward` 的增量贡献 | 待 E1/E2 |
| Student 执行期 A* | DirectGoal + planner query instrumentation | query=0、抛错替身、正式性能；NoGoalHint只作敏感性 | 历史 NoWP 诊断 | 优化路线只有证据通过后才允许声称无执行期A* | 待 O1/E1 |
| 固定拓扑未见随机实例鲁棒性 | canonical core topology、held-out seeds `200–209` | 与训练/调参隔离、final checkpoint、确定性动作 | 同环境正式评估 | 优化路线正式必需主张 | 待 E2 |
| O3 探索性压力测试 | 两个真正未见拓扑 | evaluation-only、防泄漏、统一接口；E1 预先选择执行或延期 | 不实施 | 无性能门；执行后完整报告，禁止跨拓扑泛化主张 | 待 O3/E1 |
| 稳定交付下限 | 旧 Phase 3 A* 行为 | 不适用 | S1 `300–309 × 20` 验收 | 只证明冻结旧环境的可复现下限 | 待 S1 |
| 充电边界 | 固定规则层 `1.10/0.30/0.80`（稳定路线） | 后续规格冻结 | 完成率/死亡/充电暴露 | 不声称 MAPPO 或 LLM 自主学习充电时机 | 历史选择已归档 |
| 正式统计 | seed 级聚合、final checkpoint | 七 contrasts、Holm、95% CI、paired dz、bootstrap | 核心/RuleKD 5 seed、NoWP 3 | 不用 episode 伪增样本，不挑 seed/checkpoint | 待 E1–E3 |

## 明确禁止

- 不使用同图 8-AGV 压力结果或主张；
- 稳定路线不主张未见拓扑、规模泛化或中科院二区保证；
- 旧 5-AGV A* 吞吐下降只作历史诊断，不能直接证明稳定路线性能；
- 正式结果出现后不得切换路线或修改冻结合同。
