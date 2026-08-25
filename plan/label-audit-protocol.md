# 优化路线三维语义标签双人盲审协议

## 1. 目的与边界

本协议审计 DeepSeek 基于 `semantic-view-v3` 生成的连续三维标签是否符合冻结语义量表。人工评分
不产生训练监督，不逐条修改标签，也不用于选择较有利的数据。正式 LLM 链路是
`semantic-view-v3 -> semantic-prompt-v4-directional-rubric -> DeepSeek -> [0,1]^3`；RuleKD-v3
是独立对照，不是 LLM 标签的参考答案或生成器。

## 2. 冻结样本

- 从 800 条 formal records 以 seed `20260820` 确定性抽取 100 条；五类 provenance 场景各 20：
  `normal_transport`、`priority_conflict`、`narrow_corridor_yield`、`low_battery_diversion`、
  `station_exit_congestion`；
- reviewer 只看脱敏 `semantic-view-v3`、五点量表和 LLM 三维 score/reason；不看模型身份、训练
  结果、scenario ID、RuleKD label、A*、reward、Student 或另一 reviewer 的答案；
- `scenario_type` 只用于抽样配额，不展示给 reviewer，也不进入 LLM/OOD；
- 抽样清单、record hash、盲审编码表分别保存，只有两人提交后才合并。

## 3. 评分任务

两名 reviewer 独立判断三个连续 score 所在的合理区间，并可记录 `insufficient_context`：

1. `task_persistence`：继续当前运输任务的合理程度；综合任务是否存在、优先级、载货、电量、
   进展、中断代价和局部困难；无当前任务时应接近 0；
2. `yielding_preference`：实际局部交互中主动延迟/让行的倾向；综合载货差异、相对优先级、
   延迟代价和让行是否能解除瓶颈；它不是 NOOP 或通行权裁决；
3. `coordination_risk`：冲突、拥堵、死锁或协作失败风险；综合邻居密度、窄道/站点瓶颈、
   dead/blocking robots、运动约束和开放空间。

量表 anchors 为 `0/0.25/0.50/0.75/1`，仅用于解释从“无依据”到“压倒性依据/近确定风险”的
语义位置。标签可取任意 `[0,1]` 连续值，不要求落在 anchor 上；reviewer 也不得用确定性规则把
场景类型直接换算成分数。reason 只用于核对事实与 score 是否一致，不进入 Student、loss 或 OOD。

## 4. Dataset-level acceptance

盲审开始前冻结以下判据：

- critical error 必须为 0：包括 forbidden output、捏造关键事实、score/reason 明显相反、非法值；
- substantive semantic error 不得超过 `5/100`，且每个场景层不得超过 `2/20`；该错误指任一
  score 与 reviewer 判断相隔两个或以上 anchor intervals，或 reason 与输入事实/score 实质矛盾；
- 800 条 overall parser validity 必须不低于 `784/800=98%`，且每层不低于
  `152/160=95%`；
- `insufficient_context` 单独报告但不另设可覆盖上述门槛的替代规则；无法判断的记录不能被
  选择性删除或换样本。

任一阈值失败时，整个 formal dataset 为 No-Go：返回 O0-D，升级 prompt/schema version，重新执行
完整 60 条 pilot 并生成新的完整 800 条。禁止逐条改分、只重试坏标签、选择性删除、拼接不同
版本/fingerprint 或保留较有利子集。formal 生成中 `system_fingerprint` 变化必须暂停并由 owner
审核；一个正式数据集不得静默混合 backend fingerprint。

## 5. 报告

- 报告每维错误率、anchor-interval 偏差分布、两位 reviewer 一致性、critical error 与
  insufficient-context 比例；
- 同时报告三维 Pearson 与 Spearman 相关；任意 `|rho|>=0.80` 只触发人工复核，不自动修改语义、
  删除标签或影响 OOD 公式；
- RuleKD-v3 可作为独立方法基线比较，但不得称为人工 ground truth；ShuffleKD 只验证分层
  derangement 无 fixed point，不参与人工合理性比较；
- 盲审只支持数据合理性与可审计性主张，不能替代 MAPPO 性能、样本效率或因果贡献证据。
