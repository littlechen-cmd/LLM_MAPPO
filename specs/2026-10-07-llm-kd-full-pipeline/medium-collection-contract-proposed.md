# medium 新环境与采集来源合同建议

日期：2026-10-08。状态：待冻结，不启动正式采集或长训练。
本建议落实已批准的到目标站锁定、0.1/步、满电释放机制，不改写旧 C1 合同。

## 1. 有证据的当前状态

旧 C1 seed 7 Actor 可按原始身份严格加载。新环境短诊断两条自然 episode 各运行
600 步，分别完成 4/5 个任务，各发生一次能量死亡。6,000 次 Agent 决策没有 mask
违规；187 次请求与执行差异均由受阻/碰撞后的 FORWARD 转为 NOOP 解释。

两条轨迹充电目标分别暴露 509/587 个 Agent-step，但到达目标站、站内服务步数及
满电释放次数均为 0。一次普通站点 NOOP 补能不属于新合同的目标站服务证据。
因此只确认加载与有界执行可用；不能确认新数据的充电覆盖，也不能据两个 episode
认定整个旧来源不可复用。首个在第 1 步记录失败的尝试保留，不删除后重新计数。

诊断报告与分析位于：

- `artifacts/llm_kd/phase_20261007/audit/medium5-actor-goal-full-bounded-v2/report.json`
- `artifacts/llm_kd/phase_20261007/audit/medium5-actor-goal-full-bounded-v2/analysis.json`
- 同目录 `transition_diagnostics.jsonl` 为两条完整轨迹的逐步检查结果；
  `executed_script_snapshot.py` 与报告记录的执行脚本哈希绑定。

这些产物不计入正式 dataset，不作为训练效果证据。

## 2. 新旧身份必须分开

当前 provisional YAML 沿用旧 C1 schema，因此被 `validate_medium5_c1` 的环境
精确比较拒绝。这是有效保护，不能把新充电字段添加到旧冻结 ENVIRONMENT 来绕过。
现有 collector 还要求来源 Actor 的 config/environment 哈希等于 rollout 配置，
单改 YAML 名称不能合法使用旧 Actor 在新环境采集。

建议新增独立 `core-mappo-209d-goal-full-v1` schema 与
`medium-5ag-goal-full-v1` profile，使用独立 checkpoint namespace。
medium 完整数值暂沿用当前 C1，包括 5 AGV、原布局/站点、reward-v3-c1、800 步
上限及 600k RL 步数；明确差异仅为充电合同、速率、满电释放及新身份。正式数值
仍以获批 YAML 哈希为准，provisional 不自动改为 approved。

采集入口分别绑定：

| 合同 | 必须记录的内容 |
| --- | --- |
| source_actor_contract | 原 checkpoint/Actor/config/environment 哈希、seed、原代码与 reward 身份 |
| rollout_environment_contract | 新 config/environment 哈希、充电合同、观测、布局、reward、当前代码身份 |
| collection_contract | 采集种子、策略采样方式、episode 分区、窗口规则、数据预算、恢复状态 |

旧 Actor 权重与来源文件均不改写。新输出目录包含新环境及采集合同身份；禁止追加
旧 `main-c6d2df...` 或旧 coordination-v2 目录。恢复时同时核对三类合同。

## 3. 推荐执行顺序

1. 冻结新 medium 配置与独立 schema 后，完成入口和来源/rollout 身份拆分。
2. 第一段正式原始采集兼作覆盖检查，沿同一次采集任务保留、追加和恢复；不另开
   多轮只为观察结果而丢弃的 smoke。episode 先分区，后选择 8 帧窗口。
3. 在任何新 API 请求前，根据原始状态核对结构、竞争、普通场景及真正的目标站
   充电服务候选是否足够满足已冻结的数据质量门。不用 LLM 分数挑选有利窗口。
4. 若自然轨迹候选充足，继续同一来源；若不足，保留已采原始轨迹，停止付费步骤，
   提交所有者选择新 baseline Actor 或独立标记的场景补充方案。
5. 默认不加入 A* 行为混采，不人工设置目标站状态并冒充自然 baseline 轨迹，
   不因旧 Actor 可加载就宣布质量门通过。

当前不自动要求重新训练，也不提供实际上会被旧 loader 拒绝的命令供所有者运行。
若需要新 Actor，先实现独立入口、核验完整配置，再交付可执行的服务器命令；
正式长训练仍由所有者启动。

## 4. 需要保留的边界

- 新环境下的 Actor 适用性与旧环境训练成绩分别记录，不把旧 manifest 解释成新环境训练。
- deterministic 充电机制回归已通过，不能代替真实轨迹中站内服务样本覆盖。
- 生成的 medium 主集仍为 3,000 条，2,100/450/450；失败诊断不充作有效主标签。
- 本建议没有放行 formal 容量管理、tiny/formal 正式采集或新的外发范围。
