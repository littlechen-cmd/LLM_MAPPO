# 新 medium 硬充电 checkpoint 来源核验

日期：2026-10-09。所有者提供该运行目录，并授权通过核验后由本代理直接执行采集。

## 结论与边界

该 checkpoint 可作为新的正式 dataset **候选行为来源**：训练已完成，身份可核对，
确为硬充电环境的 baseline，Actor 维度正确且参数有限，有实际到站和充电记录。
完整数据集是否合格，仍取决于新采集的覆盖、固定分区及后续语义质量验收。
不能把来源通过解释为八 seed 性能 Gate 或 LLM KD 效果结论。

## 原始来源

运行目录：
`artifacts/llm_kd/dataset/new-medium-5ag/seed_0007_20261009T055025617937Z`。

| 项目 | 实际核验值 |
| --- | --- |
| 文件 | checkpoint_final.pt |
| schema / profile | three-scene-hard-charge-v2 / medium-5ag-hard-charge-v2 |
| 方法与 seed | MAPPO-FP-DG，7；teacher_modules_enabled=false |
| 完成状态与步数 | complete；900,000 joint environment transitions |
| Actor | 209D observation，5 actions，64 hidden；严格加载且参数全部有限 |
| 充电 | hard_charge_until_full=true；0.30 触发、0.1/步、1.0 释放 |
| 配置 SHA256 | 6e127306367adcd3204f4e8696ef0efdb2ec1504c36164bc106dcc7ce47999e4 |
| checkpoint SHA256 | f112cb62dc77d4b7d6ccf66b0b90b2d088e8f6a191bcccd905b2dcb0a825ed91 |
| 原代码 | 3798ec6b593f4c92f982685ba133e7acdb184a5f |

checkpoint 与 manifest identity 一致。已从原 Git commit 找到配置原始字节：
`configs/hard_charge_v2/formal_multibudget/seed_0007/medium-5ag-baseline.yaml`，
其 SHA 与训练记录完全一致；不用当前同名近似 YAML 替代。

核验报告保存于
`artifacts/llm_kd/phase_20261007/audit/new-source-20261009/checkpoint-review.json`。

## 现有行为证据

保存的 200 次 deterministic argmax 评估合计：1121 次任务完成、1474 次充电站到达、
11977 个 charged_events、46 次能量死亡、6 次 deadlocked episode。
mean_completed_tasks=5.605。charged_events 是环境记录的充电事件计数，
不能解释为充满释放次数；释放与锁定须从新采集另行核验。
这些评估支持候选来源资格，不能代替新采集的 stochastic 覆盖检查。

## 采集接入决定

当前 core-integration 的旧采集器只严格支持历史 C1 source 和 goal-full-v1 runtime。
新来源的原生环境含充电排队/预留逻辑，任务负载也不同：expanded environment 的
dynamic_ingress_interval=801、task_target=15，而旧采集配置是 150/18。
因此新增独立接入，通过原 commit 不可变运行时快照采集，不把新 Actor 强行运行在
旧 rollout，不修改既有严格 loader，不切换当前工作分支或改写另一工作树。
209D 特征布局须核对兼容；源版本的观测实现对 energy_enabled 做了分支，
本来源 energy_enabled=true，仍须实际核验 feature ledger。

新轨迹进入独立目录，固定 120 个 episode、seed 10000..10119，8 帧窗口及
2,100/450/450 主配额保持不变。首条真实轨迹计入正式采集；通过入口及语义输入
兼容检查后按所有者本次授权继续全量，不再要求所有者运行长采集命令。
原始覆盖不足时保留数据和缺口报告，停止付费步骤，不自动更换来源或降低质量门。

旧 checkpoint 的 `goal-full-v3-bf8fad003f9033f8` 产物保留为历史诊断，不追加或覆写。
旧的 [所有者长采集命令](medium-raw-collection-owner.md) 已被本次来源替换，不能再执行。
本次只采用新来源的实际采集环境；不由此冻结 downstream RL 预算或 formal AGV 数。
新 API 请求为 0；本次授权采集不等于跳过付费标注的原始覆盖、预算及语义质量门。
