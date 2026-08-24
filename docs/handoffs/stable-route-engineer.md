# 稳定路线项目工程师交接

## 1. 交接定位

- 交接任务：P0-F，后续实施范围为 S1/S2 稳定路线。
- 工作分支：`codex/stable`。
- 基准：包含本交接文档的最终 P0 commit。进入分支后用 `git rev-parse HEAD` 记录实际 SHA；
  不以旧 feature 分支名或本文中的手写短 SHA 作为复现依据。
- 唯一活动治理入口：`specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md` 和
  根 `TASKS.md`。归档文档只提供历史证据，不得覆盖当前合同。

稳定路线的目标是保证项目按期形成完整、可复现的论文实验下限。它保留 MAPPO、A* 路径教师
和离线 LLM 语义教师，但不承担未见拓扑泛化、8-AGV 规模泛化或中科院二区保证等主张。

## 2. 冻结合同

S1 的最终验收环境固定为：

- 3 个 AGV；
- episode 任务完成目标为 9；
- 启用动态入库；
- 电量消耗系数、充电阈值和恢复阈值冻结为 `1.10/0.30/0.80`；
- A* 作为路径教师，离线 LLM 教师和现有多教师框架继续保留；
- 当前日志、checkpoint 兼容和安全规则层继续保留。

本合同不再安排能源选择 pilot。S1 只允许兼容恢复旧 Phase 3 A* 行为，不得按结果调整上述
参数、seed 或验收阈值，也不得把整个仓库回退到旧 commit。

## 3. 历史追溯入口

优先从以下入口建立“历史 artifact → 配置 → 代码行为”的证据链：

- `docs/archive/phase3_dynamic_ingress_plan.md`：动态入库历史合同；
- `docs/archive/phase3_handover.md`：Phase 3 checkpoint、评估、绘图和 engagement 链路；
- `docs/archive/g2_charging_core_selection.md`：`1.10/0.30/0.80` 选择证据；
- `docs/archive/planning/g2-1-astar-throughput-diagnosis.md`：A* 吞吐下降归因；
- `docs/archive/README.md`：所有归档的原路径和 artifact/config 索引；
- `configs/phase3a_r2_dynamic_ingress.yaml` 与
  `configs/phase3b_r2_dynamic_ingress_astar_kl.yaml`：历史动态入库配置入口；
- `configs/g2_charging_retrain_candidate.yaml` 及对应 `artifacts/g2_charging_retrain_candidate/`：
  冻结能源机制证据；
- `artifacts/previous_artifacts/phase3_dynamic_ingress_astar_*.json`：历史 A* 验收结果。

归档中的旧数值和方案不是自动有效的新合同。工程师必须先追溯其生成 commit 和当前接口差异，
再提交恢复方案；不得直接复制旧文件覆盖当前实现。

## 4. 工程师权限与职责

项目工程师可以：

- 在 `codex/stable` 上实现经架构师确认的 S1/S2 稳定路线代码、配置和回归测试；
- 运行本地 CPU 单元测试、静态检查和短 smoke；
- 准备研究所有者在 A600 上运行的完整命令、manifest 和产物 schema；
- 分析研究所有者返回的日志和结果，形成证据报告；
- 创建聚焦本地 commit，并按任务粒度同步 `CHANGELOG.md`。

项目工程师不得：

- 修改 `specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md`，或自行标记根
  `TASKS.md` 任务完成/Gate 通过；
- 改变冻结环境、能源参数、seed、阈值、正式证据预算或论文主张；
- 启动长训练、多 seed 长评估或长回放；这些任务全部由研究所有者在 A600 上运行；
- 实现 O0–O3 优化路线、未见拓扑或 8-AGV 压力场景；
- push、merge、强制移动分支，或直接复制覆盖另一分支文件；
- 因单 seed、短 smoke 或实现完成而声称稳定路线已通过 S1。

共享环境、评估和日志修复只能以明确、可追溯、已测试的 commit 在两条路线之间合并。

## 5. 产物隔离

- S1 恢复与验收产物写入 `artifacts/stable/` 下的明确子目录；
- 只有满足 S2 启动条件的决策前预备训练才写入 `artifacts/stable/predecision/`；
- S2 必须使用预声明且与正式实验不同的 seed；其结果不得参与 D1、不得进入最终统计，也不得
  直接成为论文正式实验；
- 若 D1 最终选择稳定路线，E2 必须按冻结协议从头执行正式训练和评估；
- 禁止把优化路线、旧散落 artifact 或本地 smoke 混入稳定路线确认性目录。

## 6. S1 实施与验收

工程师先提交历史行为差异审计和最小恢复方案，经架构师确认后再实现。实现至少需要覆盖：

- 旧 Phase 3 A* 行为的精确来源 commit、入口和当前差异；
- 恢复范围对训练接口、教师标签、动作掩码、协调器、奖励、观测和安全层的影响；
- 确定性回归、配置冻结断言、checkpoint 兼容和短 smoke；
- 诊断关闭时的行为等价，以及现有 A* 动作流水线/停滞诊断可用性；
- owner-run 验收命令和失败分类。

S1 最终验收由研究所有者运行非正式 seed `300–309`，每 seed 20 episodes。全部阈值同时满足
才算通过：

- 任务完成率不低于 95%；
- 碰撞率为 0；
- 能量死亡率为 0；
- 终止死锁率不高于 1%。

不要求精确复现历史 99.72%。若未通过，只能修复已确认的历史行为恢复错误并按完全相同合同
重跑；不得按结果调参或换 seed。Gate 由核心架构师审核证据后更新，不由工程师自行通过。

## 7. Owner-run 命令交付要求

工程师不得代替研究所有者启动长任务。每条交付命令必须：

- 从仓库根目录运行，声明 Conda `py310`、代码 commit、配置文件和配置 SHA-256；
- 明确 seed 集合、每 seed episodes、并行度、设备、输出目录和覆盖/续跑策略；
- 将 stdout/stderr、逐 episode 记录、聚合 JSON、运行元数据和失败状态落入
  `artifacts/stable/` 的唯一 run 目录；
- 不静默覆盖既有结果；基础设施故障只允许同 seed、同配置重跑并保留故障记录；
- 同时提供进度查看、正常终止、结果完整性检查和聚合分析命令；
- 标注哪些是短 smoke、S1 验收、S2 预备训练或未来 E2 正式任务，禁止混用。

## 8. 结果分析与风险

分析报告必须逐 seed 给出任务完成率、碰撞率、能量死亡率、终止死锁率、完成任务数/team
reward，并记录配置、commit、checkpoint、环境 ID 和失败原因。聚合结论不得用平均值掩盖
安全失败或缺失 seed。

已知风险包括：历史 artifact 与当前日志 schema 不一致；旧 A* 高层调度行为与当前“路径教师”
定位存在语义冲突；动态入库、终点预约和协调器改动可能共同影响吞吐；Windows 与 Linux 的
进程/路径差异；恢复旧行为时污染共享接口。遇到这些风险应停止扩大修改，提交最小证据给
架构师裁定。

禁止主张包括：未见拓扑或 8-AGV 泛化、执行期无需 A*、已证明 LLM 蒸馏优于全部基线、单次
训练即可收敛、中科院二区录用保证，以及任何未经冻结多 seed 证据支持的因果结论。

## 9. 每次工程交接的标准字段

每个交接必须包含：任务 ID、实现范围、变更文件、验证命令及原始结果、未完成检查、需要研究
所有者运行的长命令、已知风险、禁止主张、commit ID、分支与工作树状态。工程实现完成不等于
Gate 通过；每个完成的 `TASKS.md` 小任务都必须在同一 commit 更新根 `CHANGELOG.md`，但只有
架构师可以在审查证据后更新 `TASKS.md` 完成状态。
