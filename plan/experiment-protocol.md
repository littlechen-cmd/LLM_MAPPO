# 双路线决策前实验协议（P0）

## 1. 状态与权威来源

当前状态为 `PREDECISION`，不授权正式训练或正式 seed 评估。项目方向、技术边界和阶段顺序
分别以 `specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md` 为准；机器可读入口为
`configs/g3_experiment_manifest.yaml`。旧 G3/G2A 文档仅位于 `docs/archive/`，不构成活动合同。

## 2. 共同冻结边界

- MAPPO 输出最终离散动作；规则层管理任务队列、合法目标、固定阈值充电安全与硬约束；
- 离线 LLM 只提供 `task_commitment`、`local_assertiveness`，训练、评估和执行在线调用为 0；
- A* 提供 waypoint 与训练期路径/运动监督，具体教师语义分别由 O0 或 S1 冻结；
- 当前方法仍依赖执行期 A* waypoint；`MAPPO-NoWP` 只用于量化该依赖；
- 核心效用指标为完成任务数/1000 environment steps，样本效率指标为对应训练曲线标准化 AUC；
- team reward、完成率、成功率、步数、碰撞、死锁、能量死亡和充电暴露作为次要/安全指标；
- 统计单位是训练 seed，禁止把同一 checkpoint 的 episode 当成独立训练样本；
- checkpoint 统一使用 `checkpoint_final.pt`，不得按结果挑选最佳 checkpoint；
- 基础设施失败按同 seed/同配置重跑并留痕；算法、数值和安全失败作为结果保留。

## 3. 路线隔离

| 合同 | 优化路线 | 稳定路线 |
|---|---|---|
| 分支 | `codex/optimization` | `codex/stable` |
| artifact | `artifacts/optimization/` | `artifacts/stable/` |
| A* 前置 | O0 人工设计批准、O1/O2 通过 | S1 恢复旧 Phase 3 行为并验收 |
| 环境 | O3 冻结的核心环境与两个真正未见拓扑 | 3 AGV、目标 9、动态入库 |
| 能源 | 后续规格冻结，当前不得借 P0 调参 | `1.10/0.30/0.80` 直接冻结 |
| 泛化边界 | 仅在 O3 与正式结果支持时主张跨拓扑 | 禁止跨拓扑和规模泛化主张 |

共享环境、评估和日志修复只能以明确、已测试 commit 合并；禁止跨分支复制覆盖文件或混用
artifact。稳定路线预备训练只能写入 `artifacts/stable/predecision/`，不得参与 D1 或最终统计。

## 4. D1 前允许工作

- 优化路线：O0 架构设计、O1 实现与短验证、O2 owner-run 校准及结果审查、O3 未见拓扑冻结；
- 稳定路线：S1 实现与 owner-run 验收；满足 Roadmap 条件时可执行隔离的 S2；
- 不使用正式评估 seed `200–209`；不把旧 5-AGV A* 吞吐诊断直接当成稳定环境证据；
- 不执行同图 8-AGV 压力实验，也不在 P0 预先设计 O3 地图内容。

## 5. D1 与正式实验

D1 只依据预注册前置门选择一条路线。若优化路线完成 O0–O3 则选择优化路线，否则选择已通过
S1 的稳定路线；两者均未通过时停止。正式评估固定 `200–209 × 20 episodes`、确定性动作与
final checkpoint，且只能在 D1 后由 E1 冻结完整组别、交互预算、统计假设和命令。

优化路线的证据预算为核心 2×2、QMIX-WP、RuleKD 各 8 seed，ShuffleKD/NoWP 各 3 诊断
seed；稳定路线为核心 2×2、RuleKD 各 5 seed、NoWP 3 seed。启发式 A* 均不训练。

## 6. 长任务权限

全部长训练、长评估和长回放由研究所有者在 A600 手动启动。Codex 与项目工程师仅准备命令、
执行短验证并分析产物；任何文档或工程实现都不得把“命令已准备”写成“实验已完成”。
