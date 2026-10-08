# 执行记录

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
