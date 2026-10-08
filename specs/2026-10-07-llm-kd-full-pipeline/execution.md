# 执行记录

## 2026-10-08：分阶段语义质量门工程验收

新增纯函数 `semantic_quality_v3.py`，覆盖逐维连续分数与 raw 引用重验、辅助维重算、
训练覆盖及去共同零分冗余诊断、复标/Pro 的逐维 MAE 与 known agreement、
60 条匿名审阅、请求审计。缺失证据为 pending，不推定成功；trial Go 不放行正式 pretrain。
独立审查修复两处错误放行风险：repeat/Pro 必须同一固定 40 条，每条 accepted 标签
必须有同组同 view 的成功请求，不能只靠总体 98% 门掩盖单条失败。
另统一 ordinary 定义，排除正分停滞窗口。

工程师原定向 16 passed，补修 5 passed，生产 lint 通过。架构师复核：

```powershell
& D:/Anaconda3/envs/py310/python.exe -m pytest tests/test_llm_kd_semantic_quality_v3.py -q -k 'repeat_and_pro or accepted_label or stagnant_open or trial_go' -p no:cacheprovider --basetemp artifacts/llm_kd/phase_20261007/audit/pytest-quality-v3-root-20261008-01
```

实际 4 passed、16 deselected，10.21 秒，退出码 0；筛选词 trial_go 未匹配测试，
不把它计作本次验证。既有 pyreadline 析构 OSError 同前。此为工程验收，
未生成或通过真实新标签；API 编排仍在独立审查中。

## 2026-10-08：首条正式原始采集已保存

本地提交 `644ecfa` 后执行真实 collector，退出码 0，命令进程约 9.08 秒。
运行目录 `artifacts/llm_kd/phase_20261007/dataset/goal-full-v3-bf8fad003f9033f8`，
完成 seed 10000 的完整 800 步 truncated episode，数组 `[5,800,209]`，
reset 对齐且数值有限。保存原始压缩数组、完成凭据、意图记录和实际代码身份。
20 份实现源码按身份哈希验证后保存快照；没有改写历史 Actor 或来源文件。

500 个非重叠窗口的规则参考诊断：ordinary=462、structural=33、
stagnation_positive=5，competition=0、charging_service=0。
不能从首条轨迹推断完整采集覆盖，尚未选样或调用 API。
审查记录：`artifacts/llm_kd/phase_20261007/audit/first-real-episode-v3-20261008.json`。
补齐正式 120 个 episode 的恢复及选样命令见
[所有者运行说明](medium-raw-collection-owner.md)；按已批准分工由所有者运行长采集。

## 2026-10-08：原始采集与选样工程验收

新增 `source_actor_v3.py`、`goal_full_dataset_run.py` 及
`scripts/run_goal_full_llm_dataset.py`。旧 C1 checkpoint、归档配置、manifest、
state 和 final Actor 独立核验；新硬充电 rollout 使用另一份配置和身份。
固定 120 个完整 episode，允许先运行前缀并恢复；CPU inference 固定单线程。
窗口先按完整 episode 分区，再去重及分层选样；名单与内容哈希冻结后禁止改样。
覆盖不足明确拒绝进入付费标注，不自动扩大采集或混入其他策略。

工程师初版定向 23 passed；独立审查发现冻结已写入但 state 尚未写入时的恢复缺陷，
已修复。架构师运行恢复及冻结检查：

```powershell
& D:/Anaconda3/envs/py310/python.exe -m pytest tests/test_llm_kd_goal_full_dataset.py -q -k 'resume or frozen_selection' -p no:cacheprovider --basetemp artifacts/llm_kd/phase_20261007/audit/pytest-dataset-v3-root-20261008-01
```

结果 13 passed、14 deselected，16.17 秒，退出码 0；退出时出现既有 pyreadline
析构 OSError，不影响测试结果。工程师生产文件 lint 通过，独立审查未发现阻止真实
首条采集的其他问题。本节仅完成采集工程验收，不代表已获得合格的 3,000 条标签。
将首条真实采集计入正式 120 条 episode，后续长采集由所有者运行；不另做丢弃 smoke。

## 2026-10-08：所有者解除环境合同冲突

发现另一工作树镜像资料改为自主充电 v3、formal 8 AGV、沿用旧 6,000 标签后，
曾向所有者提出对齐问题并按要求暂停。所有者现明确：自主充电诊断失败，继续采用
本对话已批准的硬充电；formal AGV 数非当前决策参数，后续再提供。
该诊断失败为所有者报告，本对话未复跑该诊断，也不追加 smoke。

恢复 medium 优先的已批准数据流水线；到指定充电目标站锁定、0.1/步、100% 释放，
连续前三维 LLM 加规则第四维、medium 新 3,000 及旧数据保留的决定继续有效。
formal 配额仍为 8,000，当前不以旧 8/12 AGV 文档或容量提案决定实际配置。
共同基础设施可继续；formal 真实数据/训练仍等规模、配置及来源冻结后推进。
未改写另一工作树、历史 checkpoint、旧标签或历史费用；本节写入时新 API 请求为 0。

## 2026-10-08：连续评分与分维 pretrain 工程交付

前三维新 prompt、严格 parser、raw evidence 支持检查及第四维规则合并已实现；
前三维任意合法连续分数原样保存，参考锚点不用于量化或精确分数匹配。
第四维完整保留跨帧证据。新 pretrain 支持 rank-3 known mask、逐维有效 MSE
加权、全 validation SSE/count、unknown 零梯度及独立 v3 身份；旧合同继续兼容。
best/latest 与 encoder/GRU-only 导入边界保持一致。

独立审查发现 float32 等距 peer 的排序误拒问题：原绝对容差 1e-9 太小，合法等距
观测会使 yielding 错误 unknown。修复采用 2**-22 相对容差，明显乱序继续拒绝；
真实 209D 回归先得到 1 failed/1 passed，修复后通过。其余审查未发现明确阻塞问题。
训练 loader 的原始路径真实性、view 绑定及第四维重算由上游质量门负责，本轮尚未
完成该质量门实现或任何真实新标签验收。

| 验证 | 实际结果 |
| --- | --- |
| 工程师：semantic 与 temporal 定向回归，排序修复前 | 59 passed，5.40 秒 |
| 工程师：v3 pretrain 加五个 legacy 模块 | 137 passed，29.63 秒 |
| 架构师：semantic_v3 与 pretraining_v3 定向复核，排序修复前 | 69 passed，13.55 秒 |
| 审查修复者：最终 semantic 文件 | 49 passed |
| 架构师：最终等距及明显乱序两项 | 2 passed，47 deselected，5.60 秒 |

上述命令均退出码 0；部分进程退出时有既存 pyreadline 控制台析构 OSError。
工程师报告修改生产文件 flake8 通过。没有运行完整仓库 suite，不据此宣称完整项目验收。

架构师另用旧 medium 的一个真实 **train observation** 作为接口输入，模拟连续
响应 0.37/0.63，经 parse、merge、JSON 往返和新训练 loader 均保持原值。
该检查没有真实 LLM 响应，不能充当语义质量证据；报告位于
`artifacts/llm_kd/phase_20261007/audit/v3-continuous-cross-module-root-20261008.json`。
前三维仍须通过试标、复标、Pro 与 60 条匿名审阅，第四维须 100% 重算一致。

所有者曾要求暂停并记录，随后明确恢复；当前继续执行，无待解除的暂停授权。
真实新采集、v3 标注编排与质量报告继续推进；截至本节记录，新增 API 调用为 0，
未运行真实 pretrain 或长训练。公开服务核对见 [服务准备记录](provider-readiness.md)。

## 2026-10-08：合同获批后实施

所有者批准评分/pretrain 合同及 medium 采集合同，并将前三维修改为 `[0,1]`
连续评分。离散值仅作语义参考，解析、缓存、检查、交付及训练不得量化或因不等于
参考值拒绝标签。第四维仍按冻结规则生成确定值。

已批准配置保存在 `configs/llm_kd/phase_20261007/semantic-scoring-v3.yaml`，
原 proposed 文件迁移到该路径；数值锚点、unknown 条件、质量门、试标数量、loss
权重和 pretrain 基础超参数使用本次批准状态。formal 容量管理及后两环境完整配置
仍按各自记录处理；本批准不扩大既有数据外发授权。

依据已批准任务组实施三份互不重叠的工程包：

1. 新独立 medium goal-full schema/profile、精确环境验证及四方法 checkpoint
   namespace；旧 C1 冻结合同保持兼容，八个正式 seeds 均可验证。
2. 连续前三维 prompt、解析、证据检查及第四维规则合并；采用新合同身份。
3. v3 分维 known mask、分维 loss/validation 聚合、标签加载、严格身份与表征导入。

### medium 独立入口证据

新增 `goal_full_contract.py`，schema 为 `core-mappo-209d-goal-full-v1`，profile
为 `medium-5ag-goal-full-v1`。新配置批准状态、完整环境/reward/训练/evaluation
数值独立校验，包含 0.30 触发、0.1/步、1.0 释放、600k 步及八个 seeds。
四方法使用独立 goal-full checkpoint namespace，A* no-validity 分派保持原设计。
旧 `medium5_contract.py`、旧源 Actor 和精确 C1 环境哈希未改。

工程师关联定向测试 54 passed，13.22 秒；最新入口专项 28 passed，10.63 秒。
架构师审阅独立合同、runner 最小 patch、旧兼容及初始化边界，专项复核命令：

```powershell
& D:\Anaconda3\envs\py310\python.exe -m pytest tests/test_goal_full_contract.py -q -p no:cacheprovider --basetemp artifacts/llm_kd/phase_20261007/audit/pytest-goal-full-root-20261008-01
```

结果 28 passed，10.52 秒，退出码 0；Python 控制台析构存在 ignored OSError。
工程师 flake8 与限定 diff 检查通过。runner 只暂存相对实施起点的专用 patch，
现有所有者未提交 hunks 保留。入口通过不等于已执行正式训练或通过数据覆盖门。

新 artifact 接口约定：dataset schema `llm-dataset-build-run-v3`、label schema
`llm-dataset-label-run-v3`、主分布 `native-209d-goal-full-main-v3`。合并记录的
scores/known_masks/sources/evidence/unknown_reasons 均以四个 canonical 名为键，
receipt 绑定新 semantic_contract_version；旧格式不静默升级。

## 2026-10-08：最优 pretrain 模型与 v3 具体合同

### 最优模型修复

- `checkpoint_best.pt` 按实际最小 validation loss 保存；相同损失保留最早 epoch。
  `min_delta` 仅控制早停，不能阻止保存真实最优权重。
- `checkpoint_latest.pt` 保留最后参数、optimizer、RNG、早停计数和批次位置。
  恢复时先校验 best、再恢复 latest，导出前重新加载 best；不覆盖 latest 的进度。
- manifest 记录 best epoch、实际最优损失、早停参考损失及 best/导出文件哈希。
  `history.json` 保留已知历史，按 latest 已完成 epoch 裁剪；缺失历史明确列出。
- 缺少已发生验证的 best、best 进度领先 latest、历史乱序或非有限损失均拒绝恢复。
- 保持 checkpoint schema 与旧语义身份规则；尚未实现 v3 分维 unknown mask。

工程师执行四模块定向命令：

```powershell
& D:\Anaconda3\envs\py310\python.exe -m pytest tests/test_llm_kd_pretraining_run.py tests/test_llm_kd_pretraining_checkpoint.py tests/test_llm_kd_representation_export.py tests/test_llm_kd_initialization.py -q
```

结果：75 passed，14.51 秒。架构师已检查实际 diff，暂停前单独执行 runner 专项测试
13 passed，6.98 秒。恢复任务后的复核因系统临时目录权限在 fixture setup 阶段失败
13 项，未执行测试逻辑；改用项目内全新 basetemp 并禁用 pytest cache 后，
13 passed，21.10 秒，退出码 0。对应 flake8 与限定 diff 检查通过。
恢复任务后的实际命令为：

```powershell
& D:\Anaconda3\envs\py310\python.exe -m pytest tests/test_llm_kd_pretraining_run.py -q -p no:cacheprovider --basetemp artifacts/llm_kd/phase_20261007/audit/pytest-best-review-20261008-02
```

回归覆盖非单调 validation、min_delta 内的小幅改善、相同最优损失、导出权重、
中断恢复、best/latest 写入边界、终止后零更新、哈希及历史完整性。
这些是合成数据回归，不能作为真实 pretrain 收敛或 RL 性能证据。

保留限制：epoch 中途恢复时，旧 train_loss 字段仅统计剩余批次，若无剩余批次则为 0；
不能把该字段当作完整 epoch 的训练损失比较。validation 计算及模型选择不受此统计
限制影响。v3 的全分区分维 validation 聚合另行实现。

随后修复现行全已知标签的 validation 聚合：按有效监督帧数加权，消除对最后一个
不足 batch 大小的批次过度加权。新增真实模型回归以 3 个样本分别产生 0、0、0.25
的 MSE；旧实现 batch_size=2 错得 0.125，修复后 batch_size=1/2/3 均得 1/12。
工程师 runner 文件定向测试 16 passed，24.78 秒，两文件 flake8 及 diff 检查通过；
架构师审阅实际 diff 并执行以下单风险复核：3 passed，13 deselected，13.80 秒，
退出码 0。Python 控制台析构出现 ignored OSError，不影响本次测试退出状态。

```powershell
& D:\Anaconda3\envs\py310\python.exe -m pytest tests/test_llm_kd_pretraining_run.py -q -k validation_loss_weights_samples_equally -p no:cacheprovider
```

该修复针对现有 rank-2 mask、每个监督帧四维均已知的情况；v3 分维 unknown mask
仍需逐维全分区误差和/有效计数聚合，不能直接继承按帧加权作为新合同的实现。

### v3 草案与可观察对应核验

已形成 [评分、质量门及 pretrain 草案](semantic-scoring-v3.md) 和对应 proposed YAML，
数值合同仍待所有者明确批准；继续任务不自动改写为批准状态。

对旧 medium 的 4,200 个 train 末帧核验仅由 observation 推导的 peer/cell 对应：
1,752 帧包含局部 AGV，2,169 个局部 peer 槽全部与占用格一致，失败 0；
最大整数格重建残差约 1.0003e-6，小于 proposed 的 1e-4 容差。
报告位于 `artifacts/llm_kd/phase_20261007/audit/observable-peer-cell-mapping-v3.json`。
held-out 内容未用于该分析，未读隐藏坐标或真实 AGV ID 进行对应。此核验不证明新
充电合同或 formal 数据已合格，后续采集仍执行逐窗口校验。

本轮尚未执行新增 API 标注、真实 pretrain 或正式长训练；medium 行为 Actor 在新
充电合同下的适用性诊断继续推进，旧源 manifest 不改写。

随后完成有界诊断：两条完整自然轨迹各 600 步，共 6,000 次 Agent 决策，mask 违规
0，187 次执行差异全部由受阻或碰撞后的 FORWARD→NOOP 解释；完成任务分别 4/5，
各一次能量死亡。真实目标站到达和充电锁均为 0，服务覆盖未通过。一次普通站点
补能不算目标站服务。首个在第 1 步记录失败的尝试另存，总计三次 episode 启动，
未为补足数量追加运行。两条轨迹不足以认定来源整体不可复用。
报告及下一步建议见 [medium 采集合同提案](medium-collection-contract-proposed.md)。
旧 C1 loader 精确校验保持不变，新的 provisional YAML 尚不能作为正式训练入口。

## 2026-10-07：充电基础与第一轮语义审查

### 工作区与权限

分支 `core-integration`，本轮起点 `1adcb9b`。研究所有者批准执行已提交规格。
现有 runner、reward diagnostics 及其测试的未提交修改已盘点并保留。
工程师负责充电实现，架构师审核并补充入口检查、配置及语义分析。

### 已完成的实现

- 新充电合同 `goal-full-v1`：目标到站强制 NOOP，0.1/step，1.0 释放；
  容差处理避免 0.30 经过 7 次有效充电后仍因浮点误差多等一步。
- wrapper 的合法动作 mask、直接 step 及底层实际执行均约束充电服务；
  非服务路过不锁，充电事件继续视为有效进展。
- 新合同和速率传入环境工厂及 shadow 配置身份；旧合同默认行为与旧配置 payload
  保持兼容。旧数据、checkpoint 与交付包未修改。
- 新 medium baseline 配置位于
  `configs/llm_kd/phase_20261007/medium-5ag-baseline-goal-full.yaml`。
  它仍为 provisional，其他环境/训练参数继承既有 C1，仅供下一轮冻结审阅，不能视为
  正式训练已放行或行为 Actor 已重新验证。
- runner 的本轮改动仅为充电合同入口校验；既有所有者修改不纳入本轮提交。

### 验证命令与结果

所有命令使用 `D:\Anaconda3\envs\py310\python.exe`，运行目录为 core-integration。

| 验证 | 结果 |
| --- | --- |
| 工程师：charging_service、delivery_contract、reward_v2、mappo_fp_baseline、mappo_fp_runner、core_integration_smoke、core_integration_contracts 定向测试 | 初始版本 71 passed，66.80 秒 |
| 架构师：`-m pytest tests/test_charging_service.py -q` | 最终新增入口检查后 16 passed，2.80 秒 |
| `-m flake8 llm_mappo/environment.py llm_mappo/warehouse_environment.py llm_mappo/mappo_baseline/vector_env.py scripts/analyze_medium5_semantic_redundancy.py` | 通过 |
| 离线分析全部窗口 ID 与 episode 隔离 | 6,000 条精确 join；84/18/18 个 episode |
| 离线语义分析 train | 4,200 条，旧 v7 支持检查失败 0 |

离线分析命令：

```powershell
& D:\Anaconda3\envs\py310\python.exe scripts/analyze_medium5_semantic_redundancy.py --dataset artifacts/llm_kd/dataset/medium-5ag/coordination-v2-c6d2df435f63 --labels artifacts/llm_kd/dataset/medium-5ag/coordination-v2-c6d2df435f63/labels/4053b061b8bb/deepseek-official-flash/labels.jsonl --output artifacts/llm_kd/phase_20261007/audit/medium5-training-semantic-audit-v3.json
```

分析脚本拒绝覆盖已有报告；复现时改用新的输出文件名，统计应一致。
v1/v2 是本轮增补统计项和 lint 修正过程中保留的报告，v3 为最终代码身份的正式引用。
脚本只将 train 语义内容用于设计分析，held-out 仅参与身份与分区检查。

### 下一项决策与未完成工作

- 已提交 [formal 容量合同](formal-charging-capacity.md) 供所有者批准。
- 已提交 [语义审查与方向建议](semantic-audit.md) 供所有者批准。
- 当前未执行新增付费调用、pretrain、正式长训练或长评估。
- 未完成三环境配置冻结、行为 Actor 新合同适用性检查和本 phase 数据 Gate；
  充电基础完成不等于整个任务组或 phase 验收通过。
- 不宣称 LLM 学会充电，不宣称新标签或下游性能已合格，不将旧标签重命名为新语义。
