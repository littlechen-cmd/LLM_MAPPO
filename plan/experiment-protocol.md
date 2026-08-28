# 双路线实验治理协议（P0 基线，O0-F 一致性修订）

## 1. 状态与权威来源

P0 已完成；本文件保留双路线共同治理边界，并记录 O0-F 对优化路线的后续冻结修订。项目方向、
技术边界和阶段顺序分别以 `specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md` 为准；
机器可读入口为 `configs/g3_experiment_manifest.yaml`。旧 G3/G2A 文档只在 `docs/archive/` 中作为
历史材料，不构成活动合同。

## 2. 共同冻结边界

- MAPPO 输出最终离散动作；规则层管理任务队列、合法目标、固定阈值充电安全与硬约束；
- 训练、评估和执行期间在线 LLM 调用均为 0；优化路线使用连续三维语义，稳定路线保留历史
  二维语义；
- A* 在优化路线只作训练期 Pure Motion Teacher，Student 执行使用 DirectGoal observation 且
  planner query 必须为 0；稳定路线保留历史执行期 waypoint；
- 核心效用指标为 final-checkpoint 完成任务数/1000 environment steps；样本效率指标为固定
  10k-step 网格的标准化训练吞吐 AUC；team reward 和安全指标保留为 secondary；
- 统计单位是训练初始化 seed，禁止把同一 checkpoint 的 episode 当作独立训练样本；
- checkpoint 统一使用 final checkpoint，不得按结果挑选；基础设施失败可同 seed/同配置重跑并
  留痕，算法、数值和安全失败必须作为结果保留。

## 3. 路线隔离

| 合同 | 优化路线 | 稳定路线 |
|---|---|---|
| 分支 | `codex/optimization` | `codex/stable` |
| artifact | `artifacts/optimization/` | `artifacts/stable/` |
| A* | O0 Pure Motion Teacher；Student planner query=0 | S1 恢复旧 Phase 3 waypoint 行为 |
| LLM | 三维连续标签、validity×OOD | 历史二维标签 |
| 环境 | 既有 canonical core environment + O3 两个 evaluation-only 未见拓扑 | 3 AGV、目标 9、动态入库 |
| 能源 | 保持 G2-3 核心合同 `1.10/0.30/0.80` | `1.10/0.30/0.80` |
| 泛化 | 正式主张仅限 canonical core topology 的未见随机实例；O3 只作可选探索性压力测试 | 禁止跨拓扑/规模泛化主张 |

共享环境、评估和日志修复只能通过明确、已测试 commit 合并；禁止跨分支复制覆盖或混用
artifact。稳定路线预备训练只能写入 `artifacts/stable/predecision/`，不得参与 D1 或最终统计。

## 4. D1 前允许工作

- 优化路线：O0 设计、O1 本地验证、P1 Linux 执行基础设施、owner-run Linux CUDA 门禁、门禁
  通过后的 O2 校准，以及已完成的 O3 拓扑/接口冻结；
- 稳定路线：S1 实现和 owner-run 验收；满足 Roadmap 条件时可执行隔离的 S2；
- 不使用 held-out evaluation seeds `200–209`，不执行同图 8-AGV 压力实验，也不让 O3 布局
  参与训练、标签生成、OOD reference、校准、路线或超参数选择；D1 前禁止加载学习策略查看
  O3 性能。

P1 通过后必须由 owner 在共享 Linux 服务器启动 O1 常规 CUDA Gate，且只运行 baseline/H12；H4
仅在失败后写入独立诊断产物。O1 以独立产物通过后，O2 才可由 owner 在单独阶段启动；设备占用
只能触发等待，不得降低门禁或跳过 P1→O1→O2 顺序。

O2 固定为 `MAPPO-DG/RC-AStarKD × 107/117/127 × 150000 steps`，两组均关闭 LLMKD，共
6 次。Fixed/RC 链路计数等价性由 O2 前 MateBook 确定性短受控 smoke 验证。覆盖率以全部
calibration-selected agent slots 为分母，三个 RC seed 各自必须 `>=25%`；AUC 使用
`0,10000,...,150000` 网格，RC 相对 MAPPO-DG 的 seed 级退化中位数不得超过 10%。

## 5. D1 与正式训练预算

D1 只依据预注册前置门选择一条路线。优化路线完成 O0–O3 的非性能前置门时选择优化路线，否则选择已通过
S1 的稳定路线；两者均未通过时停止。正式评估固定 `200–209 × 20 episodes`、确定性动作与
final checkpoint。

优化路线 E1/E2 保持 65 次学习运行：核心 `2×2` 32、Fixed-KD 8、QMIX-DG 8、RuleKD-v3 8、
ShuffleKD-v3/NoOOD-v1/NoGoalHint-v1 各 3。连同 O2 6 次，优化路线总计 71 次。稳定路线预算
不变：核心 `2×2` 和 RuleKD 各 5 seed，历史 NoWP 3 seed；无 QMIX、Shuffle、未见拓扑或
8-AGV。启发式 A* 均不训练。

## 6. 优化路线统计与主张

正式训练 seed 为 `7/17/27/37/47/57/67/77`。primary endpoint 是 seed 级 held-out
`completed_tasks_per_1000_steps`。七个确认性 contrasts、paired t-test、Holm 校正、95% CI、
paired effect size 与 10000 次 seed-level bootstrap 均以 canonical architecture 第 13.4 节为准。
Shuffle、NoOOD、NoGoalHint 只支持诊断性主张；三维标签干预只能说明策略敏感性，不能单独证明
训练贡献。QMIX-DG 必须共享 DirectGoal、环境步、seed、评估、mask、reward、能源和调参预算。

上述正式统计的环境范围固定为 canonical core topology；`200–209 × 20` 只支持固定拓扑下的
未见随机实例鲁棒性。E1 必须在查看任何 O3 policy performance 前，依据资源把 O3 探索矩阵冻结
为“执行”或“延期”。执行时只含 `MAPPO-DG/RC-AStarKD+LLMKD × 8 training seeds ×
2 topologies × 200–209 × 20 episodes`，无最低性能阈值且必须完整报告；延期时不得运行或报告
任何 O3 policy performance。O3 结果不进入七个确认性 contrasts，也不支持跨拓扑泛化主张。

## 7. 长任务权限

全部长训练、长评估和长回放由研究所有者在批准的服务器手动启动。Codex 与项目工程师只准备命令、
执行短验证并分析产物；任何文档或工程实现都不得把“命令已准备”写成“实验已完成”。
