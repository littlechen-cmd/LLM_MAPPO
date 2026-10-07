# 执行记录

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
