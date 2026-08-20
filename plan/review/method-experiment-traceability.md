# 方法—实验可追溯矩阵

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| 异构双教师 MAPPO 框架 | A* KL + 离线双语义头 | 核心 2×2 | T1, F1 | 两类教师可独立组合并公平消融 | 配置已准备，结果待定 |
| 路径教师贡献 | A* reservation KL | A*KD 主效应 | T1, F1 | A*KD 对样本效率/路径安全的影响 | 待 G4–G6 |
| 语义教师贡献 | task commitment + local assertiveness | LLMKD 主效应、行为评估 | T1, T3, F2 | LLMKD 对优先级与局部协调的影响 | 待 G4–G6 |
| 教师互补性 | 两个监督损失联合 | 2×2 交互效应 | T1 | 联合收益是否超过单教师可加解释 | 待 G6 |
| 零在线 LLM | 缓存标签与近邻检索 | API 调用审计、效率 | T2 | 训练/评估/执行在线调用为 0 | 代码具备，正式日志待定 |
| 能源压力鲁棒性 | 规则充电 + 语义协调 | 统一 `1.20/0.20/0.80` 压力评估 | T3, F3 | LLMKD 是否降低能源压力下性能退化 | 待 G6；不主张自主充电 |
| 执行期摆脱 A* | 尚未实现 | 无 | 无 | 不允许形成当前贡献主张 | 明确限制/未来工作 |
