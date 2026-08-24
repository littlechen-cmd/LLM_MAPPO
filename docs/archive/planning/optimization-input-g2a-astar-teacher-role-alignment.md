# Phase G2A：A* 教师角色对齐任务包

> 状态：`READY_FOR_ENGINEER`
> 唯一性：本文件是 Phase G2A 的唯一任务文档
> 负责人：项目核心架构师
> 执行者：项目工程师（实现与短验证）、项目负责人（长 pilot）、核心架构师（验收）
> 前置：G2-1 归因可并行，但 G2A-5 与 G2-1h 均完成后才能冻结 G3
> 性质：架构对齐与校准，不属于正式论文结果

## 1. 目标与不变项

在不放弃多教师蒸馏的前提下，将 A* 从“端到端动作专家”收缩为“既定目标下的局部运动教师”：

- A*：提供 waypoint、几何可达性和局部动作先验；
- 离线 LLM：提供 `task_commitment`、`local_assertiveness` 语义监督；
- MAPPO：输出最终动作，并通过 team reward 与集中式 critic 学习运动层让行、竞争和长期吞吐；
- 规则层：继续负责任务队列、合法目标、优先级数据、固定阈值充电安全和硬约束。

核心 `2×2` 因子、网络宽度、能源配置、奖励、正式 seed、环境交互预算规则和零在线 LLM 合同
不变。当前实现没有可学习的任务选择/自主充电高层动作，因此禁止声称 MAPPO 已学习完整任务
分配或自主充电调度。

## 2. 已确认的实现事实

训练 worker 的 snapshot 当前调用 `AStarExpert.act(env, masks)[1]` 生成 KL 偏好；该方法先计算
预约 A* 偏好，再执行 mask 与 `_coordinate_actions()`，因此协调器的 `NOOP/RIGHT` 让行会进入
教师分布。与此同时，actor 的 waypoint 观测由环境对规则层既定目标调用几何 A* 生成。

因此，当前问题不是“项目是否仍有 A* 教师”，而是“KL 标签是否混入了端到端协调策略”。
G2A 只修正教师语义，不以当前独立 A* 完成率选择 MAPPO 动作。

## 3. G2A-1：冻结接口与职责（已完成）

工程实现前必须以测试可判定的形式冻结以下接口：

```text
LocalMotionTeacher(env, action_masks)
  -> probabilities[n_agents, action_count]
  -> confidence[n_agents]  # [0, 1]
  -> exclusion_reason[n_agents]
```

教师只消费环境/规则层已经给定的目标。它不得调用任务分配、改变优先级、选择充电意图或修改
环境状态。`AStarExpert.act()` 保留给 `Heuristic-Dispatcher+A*`、G2-1 诊断和可视化基线，不再
作为正式训练 snapshot 的 KL 数据源。

2026-08-21 架构审计已核对 `phase3_training._handle_environment_command("snapshot")`、
`AStarExpert.act()` 和 MAPPO KL 实现并冻结上述边界；下一项执行任务为 G2A-2。

## 4. G2A-2：工程实现

### 4.1 有效样本

只有同时满足以下条件的 agent-step 才能具有正 confidence：

- agent 可执行运动动作，且当前不是强制拾取、picking lock、死亡或其他规则强制状态；
- 目标由环境既有规则产生，教师没有改写目标；
- 规划结果是可审计的完整局部运动结果，不是 partial 或 reservation false-no-path；
- 教师分布经 action mask 后仍有合法概率质量；
- 该标签不是 `_coordinate_actions()` 覆盖后的动作；若端到端协调器会改写该动作，应记录
  `coordinator_conflict` 并令 confidence 为零。

充电路径可以作为几何标签，但“是否开始充电”仍由规则层决定。预约等待可以作为低层运动
约束保留，前提是规划结果完整、合法且未被端到端协调器覆盖。

### 4.2 加权 KL

rollout buffer 必须保存 teacher probabilities、confidence 和 exclusion reason 的可聚合编码。
每个 minibatch 的 A*KL 定义为有效 agent-step KL 的置信加权平均：

```text
sum(confidence_i * KL(teacher_i || policy_i)) / sum(confidence_i)
```

当权重和为零时，A*KL 必须严格返回有限的 `0.0`，不得除零或把无效标签归一化为 NOOP。
现有 `reservation_kl_coefficient=0.05` 及衰减合同保持不变，G2A 不进行系数搜索。

### 4.3 日志

至少记录：总 agent-steps、有效 teacher steps、coverage、各 exclusion reason 计数、action-mask
移除概率质量、教师动作分布、KL 有效样本数、零有效 minibatch 数、规划完整/部分/失败计数和
局部规划时延。计数须能守恒到总 agent-steps。

## 5. G2A-3：工程验收与短 smoke

- [ ] 教师调用前后任务、优先级、充电目标和环境状态完全相同；
- [ ] 有效标签中 coordinator-output contamination 为 0；
- [ ] 强制交互、partial、false-no-path 和非法动作概率质量均被排除并有唯一原因；
- [ ] confidence 为零的样本不影响梯度；全零 minibatch 的 loss/gradient 有限且为零；
- [ ] rollout buffer 单/多环境、终止边界和 checkpoint 恢复形状正确；
- [ ] `use_astar_kl_teacher=false` 的轨迹、损失和日志保持兼容；
- [ ] 旧 checkpoint 可加载，新 checkpoint/summary 记录 teacher mode 与 coverage schema；
- [ ] 焦点测试、完整 pytest、规定范围 Flake8、`git diff --check` 通过；
- [ ] smoke 不超过 `2 seeds × 2 episodes`，不启动长训练。

## 6. G2A-4：项目负责人校准 pilot

三组全部关闭 LLMKD，使用核心 5-AGV、`1.10/0.30/0.80`、动态入库环境和相同 PPO 参数：

| 组别 | A*KL 数据源 | 作用 |
|---|---|---|
| `MAPPO-WP` | 无 | 判断蒸馏是否造成明显负迁移 |
| `Coordinated-A*KD` | 当前 `AStarExpert.act()` | 仅作为旧教师语义诊断 |
| `LocalMotion-A*KD` | 新的有效/置信加权局部运动教师 | G3 候选正式教师 |

固定 policy seed `3/13/23`、环境 seed 轮换 `300–309`，每组每 seed 精确 50,000
team-environment steps，统一使用 final checkpoint 做校准评估。项目工程师只提供命令和预计
产物路径，不得代替项目负责人运行。

## 7. G2A-5：Go/No-Go

局部运动教师进入 G3 必须同时满足：

1. 任务、优先级和充电目标改写数为 0；
2. 有效标签中的 coordinator contamination 与非法动作概率质量均为 0；
3. 三 seed 合并的有效 teacher coverage 不低于 25%；
4. 无 NaN/Inf、接口失败、日志不守恒或 checkpoint 不兼容；
5. 相对 `MAPPO-WP`，三 seed 的标准化吞吐 AUC 中位数退化不超过 10%。

安全指标、coverage 分层和 `Coordinated-A*KD` 差异必须报告，但三 seed 不做显著性检验。
未通过时只允许修正一次已定位的接口/掩码错误，并以完全相同合同重跑；不得调整 KL 系数、
训练预算、seed 或验收阈值。再次失败则 G3 保持阻塞，由架构师决定收缩 A*KD 核心贡献或重新
设计方法，不得只保留有利 seed。

## 8. 允许主张

G2A 通过后，可将 A* 描述为：

> a confidence-gated local geometric motion teacher

不得称为全局最优调度专家、端到端多智能体 oracle 或高层任务分配器。G2A pilot 只证明接口
纯度、训练可用性和无明显负迁移；A*KD 的方法收益仍只能由 G5/G6 八 seed 正式实验支持。

## 9. 工程师权限与交付

工程师可修改 A* teacher 接口、snapshot/buffer/KL、诊断日志、配置 schema、测试和
`CHANGELOG.md`；不得修改 Constitution、勾选 TASKS、调整正式 seed/预算/KL 系数、删除
`Heuristic-Dispatcher+A*`、运行长 pilot、push 或 merge。

交付必须包含任务 ID、实现范围、变更文件、验证命令/结果、owner-run 三组命令、未完成检查、
风险、禁止主张、commit ID 和工作树状态。

## 10. 给项目工程师的提示词

> 执行 `plan/task-package/g2a-astar-teacher-role-alignment.md` 的 G2A-1～G2A-3。先阅读
> `AGENTS.md`、`CONSTITUTION.md`、`TASKS.md`、`CHANGELOG.md`、本任务包、
> `llm_mappo/phase2_expert.py`、`llm_mappo/phase3_training.py`、`llm_mappo/mappo.py`、rollout
> buffer 和相关测试。将正式训练 snapshot 的 A*KL 数据源从端到端协调输出拆为既定目标下的
> 局部运动分布，并增加逐 agent confidence/exclusion reason、加权 KL、覆盖率日志与回归测试；
> 保留 `AStarExpert.act()` 供启发式基线使用。不得改变 KL 系数、任务/优先级/充电规则、奖励、
> 正式 seed 或预算，不得运行 50,000-step pilot。完成工程任务后同步 CHANGELOG，按 AGENTS
> 交接格式创建独立 commit，但不要修改 Constitution 或勾选 TASKS。
