# 优化路线：探索性 Noisy-Teacher 协议

## 状态与目的

本协议是研究所有者于 2026-09-01 批准的 E1 合同修订。它不撤销
`plan/label-audit-protocol.md` 中的原始严格数据集 Gate：DeepSeek v4-Pro
v5 的原始 800 条数据集在该 Gate 中发现关键语义错误，因而不得被表述为
“已验证的正确 LLM 标签数据集”。

修订后的研究问题是：即使离线 LLM 语义教师含有可观测的语义错误，MAPPO
能否仍从其弱语义监督中获得经验性性能收益。所有使用该标签源的训练、评估和
论文结果均为 exploratory noisy-teacher evidence，而非确认性语义正确性证据。

## 冻结的原始证据

- 原始目录：`artifacts/formal/formal_pro_v5_20260901T072610Z/`。
- 请求模型：`deepseek-v4-pro`；prompt：`semantic-prompt-v5-state-contract`；
  backend fingerprint：`a307abda487cd1b463329ccb945ce396`。
- 必须保留全部 800 条 attempts 和 records 的原始顺序、内容、哈希与响应证据。
  禁止删除、修改、重排、补写或重试任一记录。
- 第 791 条保持原始无效记录；其 LLM KD 权重为零。不得由人工或第二次 API
  请求将其补成有效 LLM 标签。
- `review_pack_blinded_v2.json` 是从同一原始 records 派生的审计包；它只用于
  记录已发现的教师噪声，不用于选择或修补训练样本。

## 训练与证据边界

- 所有代码组名保持冻结的 65-run matrix，不新增 Raw-vs-Curated 运行，也不改变
  训练 seed、预算、环境、奖励、A* 教师、网络、schedule 或 OOD 公式。
- `LLMKD` 和 `RC-AStarKD+LLMKD` 在论文和结果表中分别称为 `Raw-LLMKD` 和
  `RC-AStarKD+Raw-LLMKD`，以表明语义来源是未经人工纠错的原始 LLM 输出。
- `validity × OOD reliability` 仍是记录级训练权重；它只表示结构合法性和输入
  分布可靠性，不得解释为语义正确性置信度或幻觉检测器。
- 训练、评估和执行期的在线 LLM 调用仍必须为零。
- E1-C 及后续工程工作可以继续；E2 的对应训练结果只可作为探索性方法证据。
  在完成新的确认性标签合同前，不得据此完成 E1 的 confirmatory label Gate 或
  声称论文的 LLM 语义教师已被人工/数据集级正确性验证。

## 允许与禁止的论文主张

允许：

> The offline LLM semantic teacher is imperfect and may produce semantic
> errors. We evaluate whether MAPPO can nevertheless benefit empirically from
> its noisy semantic supervision through interaction-driven optimization.

禁止：

- 将 LLM 标签称为 ground truth、人工验证标签或无噪声标签；
- 声称 `validity × OOD reliability` 能识别语义幻觉；
- 将观察到的性能差异归因于 LLM 的每次具体决策正确；
- 将探索性结果报告为确认性、预注册的标签质量结论；
- 用后续人工修订单条或选择性子集反向美化此原始数据集。

## 延后的人类复核

若研究所有者后续决定投入人工时间，必须新建版本化的 human-curated 数据合同，
对全体记录执行同一盲态流程，并与本 raw 数据集严格分开。该未来工作不得改变
本协议下任何已有 raw 标签、训练或结果的身份。
