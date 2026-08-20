# 方法—实验可追溯矩阵

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| 异构双教师 MAPPO 框架 | A* KL + 离线双语义头 | 八 seed 核心 2×2 | T1, F1 | 两类教师可独立组合并公平消融 | 协议/流水线就绪，结果待定 |
| 路径教师贡献 | A* reservation KL | A*KD 主效应 | T1, F1 | A*KD 对主要效用/样本效率的影响 | 待 G4–G6 |
| LLM 语义教师贡献 | task commitment + local assertiveness | LLMKD 主效应、行为评估 | T1, T4, F2 | LLMKD 对优先级与局部协调的影响 | 待 G4–G6 |
| LLM 标签必要性 | LLMKD 与同构规则/扰乱标签 | 完整方法对 RuleKD、ShuffleKD 诊断、双人盲审 | T5, F3 | LLM 状态相关监督不能仅由规则或额外损失解释 | RuleKD/ShuffleKD 待 G3-5 |
| 外部算法竞争力 | MAPPO 完整方法 | 完整方法对 QMIX-WP | T2, F2 | 相同输入/安全边界下的相对竞争力 | QMIX-WP 待 G3-5 |
| 零在线 LLM | 缓存标签与近邻检索 | API 调用审计、效率 | T3 | 训练/评估/执行在线调用为 0 | 代码具备，正式日志待定 |
| 泛化与能源压力鲁棒性 | 规则充电 + 语义协调 | 两未见布局、负载/规模、`1.20/0.20/0.80` | T4, F4 | 预冻结场景下的零样本泛化/受限能源鲁棒性 | 布局与正式结果待 G3-6/G6 |
| 执行期 A* 依赖与成本 | waypoint、规划诊断 | MAPPO-NoWP、调用/重规划/时延审计 | T3, F5 | 当前方法仍依赖A*；量化其独立价值与成本 | NoWP/完整审计待 G3-5/G6 |
| 执行期摆脱 A* | 尚未实现 | 无 | 无 | 不允许形成当前贡献主张 | 明确限制/未来工作 |
