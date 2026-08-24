# G2-1 A* 教师吞吐瓶颈诊断任务包

> 状态：`READY_FOR_ENGINEER`
> 负责人：项目核心架构师
> 执行者：项目工程师（代码与短时验证）、项目负责人（长时间评估）
> 关联任务：`TASKS.md` 的 G2-1d–h、G3-1 以及 G6-4
> 性质：校准集工程诊断，不属于正式论文结果

## 1. 任务目标

本任务包用于回答两个相互关联但必须分开验证的问题：

1. legacy 与 bounded-fixed A* 均只有约 68% 任务完成率时，吞吐损失具体发生在规划器、动作协调器、环境安全约束、充电规则还是回合时限中的哪一层；
2. 旧 Phase 4 MAPPO checkpoint 在与独立 A* 完全相同的环境配置和校准 seed 下，是否仍能保持明显更高的完成率。

本任务不继续修改 A* 搜索策略，不引入 CBS，不重新训练 MAPPO，也不把单个 checkpoint 的诊断结果写成正式论文结论。

本文件是 G2-1 的唯一架构任务包，同时覆盖停滞观测、历史完成率回归归因和 MAPPO 同配置诊断；
不得为 G2-1 再建立并列任务包。

## 2. 已知证据与当前架构判断

### 2.1 输入产物

- `artifacts/g2_astar_reservation_legacy.json`
- `artifacts/g2_astar_reservation_fixed.json`
- `artifacts/phase4_cuda_mp12_800_seed007/phase4_cuda_mp12_800_seed007/seed_007/checkpoint_final.pt`
- `artifacts/phase4_cuda_mp12_800_seed007/phase4_cuda_mp12_800_seed007/seed_007/summary.json`
- `configs/phase4_llm_distillation.yaml`
- `configs/g3_core_mappo_wp_astar_kd.yaml`

### 2.2 已确认结果

| 指标 | legacy | bounded-fixed | fixed 相对变化 |
|---|---:|---:|---:|
| 任务完成率 | 67.67% | 68.21% | +0.54 个百分点 |
| 成功率 | 42.5% | 45.5% | +3.0 个百分点 |
| 死锁率 | 7% | 8% | +1.0 个百分点 |
| 碰撞/回合 | 0 | 0 | 不变 |
| 能量死亡/回合 | 0 | 0 | 不变 |
| 规划时间 P95 | 151.88 ms | 4.67 ms | 约 -96.9% |
| expanded nodes | 827,622,097 | 197,826,062 | 约 -76.1% |
| terminal conflicts | 4,373,285 | 225,787 | 约 -94.8% |

配对 seed 的完成率变化方向并不一致：6 个 seed 上升、4 个 seed 下降。bounded-fixed 显著消除了计算膨胀和终点冲突，但没有解决端到端吞吐问题。因此当前证据不能支持“删除 full-horizon 终点预约导致完成率下降”，也不能支持“通用 A* 算法失效”。更准确的待验证假设是：当前优先级时空 A*、逐步重规划和动作协调的组合过度保守或缺少长期路径承诺。

### 2.3 Phase 4 对照的口径限制

- Phase 4 的 800 回合训练均值为 95.74% 完成率、89.75% 成功率，最后 100 个训练回合均为 100%；这些是训练分布结果。
- G1 的冻结 checkpoint 仅完成 seed 0、1 episode 的流水线 smoke；其产物明确声明不是正式性能结果。
- 旧 Phase 4 训练配置为 `1.00/0.20/0.80`，当前核心配置为 `1.10/0.30/0.80`。
- 因而必须完成同环境、同 seed、同 episode 数和同终止条件的诊断，才允许比较独立 A* 与该 checkpoint。

## 3. 不可变边界

1. 校准 seed 固定为 `300–309`，每 seed 20 episodes；正式评估 seed `200–209` 不得使用。
2. 对照环境固定使用 `configs/g3_core_mappo_wp_astar_kd.yaml` 中的 `1.10/0.30/0.80`、5 AGV、动态入库、`max_steps=1000` 和 `task_completion_target=50`。
3. MAPPO 使用 `checkpoint_final.pt`，确定性动作，不允许按诊断结果选择其他 checkpoint。
4. A* 对照首先使用已经完成的 bounded-fixed 产物；不得为了提高结果临时调整 horizon、hold steps、优先级或协调规则。
5. 项目工程师不得启动 200 回合评估、训练或超参数搜索；长时间任务由项目负责人运行。
6. 新增诊断必须是观测性的，不得改变动作、奖励、任务分配、充电、安全层或终止逻辑。
7. 保留现有 `path_livelocks`、`state_deadlocks` 等字段以兼容旧产物，但论文和新报告不得把它们直接解释为真实 episode livelock/deadlock。
8. G2-1e 的既有验收判据不得根据当前结果事后修改。fixed 的聚合死锁率由 7% 增至 8%，因此它当前不满足“deadlock 不增加”的预注册采用条件；最终模式决定由架构师在审计完成后写入 `TASKS.md`，工程师不得自行决定。

## 4. 工作流 A：停滞归因观测

### A1. 建立动作决策流水线记录

工程师应在不改变行为的前提下，能够对每个 `agent-step` 记录以下三个动作：

- `planner_action`：A* preference 取确定性动作后的结果；
- `coordinated_action`：`AStarExpert._coordinate_actions` 处理后的结果；
- `executed_action`：提交环境并经 action mask/环境规则处理后的实际结果或等价可观测结果。

当三个动作不一致时必须记录覆盖层和原因，不得只记录最终 NOOP。

### A2. 固定停滞原因分类

每个没有产生位置变化的 `agent-step` 必须且只能进入一个主要原因：

| 原因码 | 定义 |
|---|---|
| `planner_noop` | planner 明确选择 NOOP |
| `planner_turn` | planner 选择原地左/右转 |
| `coordinator_vertex_yield` | 顶点冲突被协调器改为等待 |
| `coordinator_edge_swap_yield` | 对向边交换冲突被协调器改为等待 |
| `coordinator_occupied_yield` | 前方当前被其他 AGV 占用而等待 |
| `action_mask_block` | 动作在 mask 阶段被禁止或替换 |
| `environment_blocked_forward` | 合法前进尝试被环境冲突处理阻挡 |
| `interaction_lock` | PICK/DROP 或 picking lock 导致静止 |
| `charging_wait` | 充电规则、充电站等待或充电保持导致静止 |
| `dead_or_inactive` | AGV 已死亡或不可行动 |
| `unknown_stationary` | 无法由以上原因解释；必须保留告警计数 |

`planner_turn` 是正常动作，不得再自动称为 rotation livelock。只有连续重复达到既有 livelock 判定条件时，才能另外记录 `rotation_livelock_event`。

### A3. 增加回合级吞吐诊断

每个 episode 至少输出：

- `termination_reason`：`target_reached`、`deadlock`、`time_limit` 或 `energy_failure`；
- `completed_tasks`、`picked_tasks`、回合结束时仍载货的 AGV 数；
- 各停滞原因的 agent-step 计数及占全部 agent-step 的比例；
- planner、协调器和环境分别产生的等待/覆盖次数；
- `replans`、目标变化次数、首动作变化次数；
- reached-goal、partial、reservation-blocked/horizon-exhausted 的规划结果计数；
- 规划延迟 P50/P95/P99、expanded nodes 和 cache hit/miss；
- 充电途中、排队、充电保持所占步数；
- `unknown_stationary` 数量。

必须同时保留原始计数和归一化比例，避免不同回合长度造成暴露量偏差。

### A4. 计数守恒与回归测试

至少新增以下确定性测试：

1. 每个静止 agent-step 恰好对应一个主要原因码；
2. planner 到 coordinator 的每次动作变化都有唯一覆盖原因；
3. coordinator 到 executed 的每次动作变化都有唯一覆盖原因；
4. `target_reached/deadlock/time_limit/energy_failure` 每回合恰好一个；
5. 原地转向不自动累计为真实 livelock；
6. 关闭诊断输出和开启诊断输出时，同 seed 的动作轨迹与 episode 结果完全一致；
7. 旧 JSON 字段仍可读取，新字段缺失时聚合器明确报告旧 schema，而不是填造数据。

## 5. 工作流 B：短时工程验证

工程师只运行用于验证观测链路的短时 smoke：

- 固定使用校准 seed `300`；
- legacy 与 bounded-fixed 各不超过 2 episodes；
- 必须验证新 schema、计数守恒和行为无变化；
- 不设置完成率门槛，不据此选择模式；
- 产物放入独立 `artifacts/diagnostics/` 子目录，不覆盖现有两个 G2 JSON。

工程师交付一条项目负责人可以直接复制的正式诊断命令，但不得代替项目负责人启动长任务。

## 6. 工作流 C：MAPPO 与 A* 严格同配置诊断

### C1. 必需对照

项目负责人使用旧 Phase 4 `checkpoint_final.pt`，在以下固定合同下运行：

- 环境：`configs/g3_core_mappo_wp_astar_kd.yaml` 的环境段；
- seed：`300–309`；
- episodes：每 seed 20；
- 动作：确定性；
- checkpoint：只允许 final checkpoint；
- 输出：独立 JSON 和逐 episode CSV；
- A* 参照：现有 `artifacts/g2_astar_reservation_fixed.json`。

该比较回答的是“旧 checkpoint 在新核心能量配置下是否仍优于独立 A*”，属于模型转移/工程诊断，不是 Phase 4 原始训练配置下的正式复现。

### C2. 可选敏感性对照

只有当 C1 结果不能区分“控制器差异”与“能量配置迁移失配”时，才追加旧 `1.00/0.20/0.80` 配置下的同 seed 对照。追加前由架构师确认；不得自动扩展实验量。

### C3. 必须报告的对照指标

- task completion rate；
- success rate 与四类 termination reason；
- completed tasks / 1000 environment steps；
- collisions/blocked forwards、deadlock、energy deaths；
- 充电暴露与充电等待；
- 平均 episode steps；
- 对每个 seed 的 MAPPO−A* 配对差值及跨 seed 区间估计；
- A* 的停滞原因分解与 MAPPO 的可比环境阻挡指标。

统计单位是 seed，不是 200 个 episode。该单 checkpoint 诊断不得用于显著性主张，也不得替代 G5/G6 的八训练 seed 正式比较。

## 7. 判断矩阵

| 观察结果 | 允许结论 | 下一步 |
|---|---|---|
| MAPPO 同配置仍明显高于 A*，A* 主要为 time-limit | 当前独立 A* 控制器存在吞吐瓶颈 | 根据停滞归因决定是否在正式阶段前优化协调/路径承诺 |
| MAPPO 同配置也降至约 68% | 核心能量配置或 checkpoint 分布迁移是重要因素 | 先完成 G4 匹配重训练，不修改 A* |
| A* 停滞主要来自协调器覆盖 | 瓶颈位于 prioritized coordination，而非单体 A* 搜索 | 形成独立协调层优化任务；CBS 仍为扩展项 |
| A* 停滞主要来自 planner partial/blocked | 有限时域预约搜索是主要瓶颈 | 审计 horizon、路径承诺和局部重规划，但不得用正式 seed 调参 |
| A* 停滞主要来自充电等待 | 能源规则与站点拥堵耦合是主要瓶颈 | 转入充电协调诊断，不把问题归因于 A* 本体 |
| `unknown_stationary` 非零且占比较高 | 当前证据不足 | 先修复观测缺口，不作算法归因 |

## 8. 论文术语与允许主张

在正式证据形成前，A* 应描述为：

> reservation-aware heuristic path teacher

允许描述其提供局部路径归纳偏置或动作偏好。禁止将其称为：

- 全局最优调度教师；
- 完整多智能体专家上界；
- 能独立优化长期动态吞吐的 oracle；
- MAPPO 必须逐动作复制的行为专家。

如果同配置诊断支持 MAPPO 超过 A*，允许在后续多训练 seed 正式实验确认后主张：奖励驱动的 MAPPO 能选择性吸收启发式路径知识，并缓解顺序优先级规划在长期协作吞吐方面的局限。当前单 checkpoint 结果只能用于形成该假设。

## 9. 历史完成率回归归因（G2-1f–h）

### 9.1 证据口径

- 当前 `artifacts/g2_astar_reservation_fixed.json` 完成率约为 `0.6821`；当前 legacy 产物约为
  `0.6767`；
- 历史 `phase3_dynamic_ingress_astar_gate_40step_n9_10x20.json` 使用 3 AGV、目标 9，
  200 episodes 完成率约为 `0.9972`，只作为更轻合同下的历史线索；
- 历史 Phase 4 preflight 使用 5 AGV、目标 50，6 episodes 完成率为 `1.0`，但样本过小且
  存在大量 path-livelock/state-repeat 诊断，不能证明长期吞吐；
- A* 最小修复提交为 `f6f0650`，修复前基准为
  `119f20dfebdc0e89baedd379ed795b27a750e3a0`；
- 当前 `legacy_terminal_reservation=True` 只恢复持续终点预约，并未恢复旧协调器、旧时间展开、
  旧部分路径语义或旧缓存行为，不能称为“旧算法复现”。

### 9.2 G2-1f：工程实现与短验证

工程师在当前 bounded-fixed 实现中增加诊断专用的协调冲突回退开关，只允许在 `NOOP/RIGHT`
之间切换，生产默认值保持不变。除该显式诊断因子外，不得改变规划、预约、缓存、动作 mask、
奖励、任务分配、充电或终止行为。

必须验证：未配置和 `NOOP` 模式的固定 seed 轨迹与修改前一致；受控冲突场景中 `RIGHT` 只改变
指定回退动作且安全检查仍生效；开启/关闭停滞日志不改变轨迹与 episode 摘要；smoke 不超过
`2 seeds × 2 episodes`，且不得据此选择模式。

工程师还须准备项目负责人可复制的隔离 Git worktree 入口，以 `119f20d` 运行同一当前环境合同。
不得回退主工作树，也不得把当前控制器文件选择性复制到旧快照；若需要配置适配层，必须独立、
最小且逐项记录，不能改变旧控制器行为。

### 9.3 G2-1g：项目负责人长运行

三组固定使用 5 AGV、当前 medium `24×20` 地图、动态入库、`1.10/0.30/0.80`、
`max_steps=1000`、`task_completion_target=50` 和 seed `300–309 × 20 episodes`：

1. 当前 fixed + `NOOP`；已有 fixed 产物仅在 commit、配置哈希和 schema 一致时复用；
2. 当前 fixed + `RIGHT`；
3. 修复前 `119f20d` 整包 + 同一当前环境合同。

三组不得修改 horizon、hold steps、优先级、任务分配或充电规则。每组保存启动 commit、配置
哈希、命令、逐 episode CSV、逐 seed 汇总和聚合 JSON。统计单位为 seed，不得使用正式 seed
`200–209`，也不得把 200 episodes 当成 200 个独立样本。

### 9.4 G2-1h：架构审计

| 结果 | 允许归因 | 后续决策 |
|---|---|---|
| `RIGHT` 明显恢复吞吐且安全不退化 | 当前协调回退是主要因素 | 另立行为变更任务，不直接修改正式默认值 |
| 修复前整包恢复、`RIGHT` 单变量不足 | `f6f0650` 其他耦合变化是主要因素 | 结合停滞日志缩小到时间展开、部分路径、缓存或预约语义 |
| 三组均接近当前约 68% | 当前 5-AGV/目标 50 合同或更早变化是主要候选 | 不声称 A* 代码回归；需要时另行预注册负载分解 |
| 动作、配置或产物不可比 | 证据无效 | 先修复可比性，不作算法归因 |

G2-1e 的原有验收阈值不得事后改变。最终 G3 A* 模式由核心架构师在完整审计后冻结；证据不足
时只能记录限制，G3-1 继续保持未完成。

## 10. 工程师允许修改的范围

允许修改：

- A* expert、动作协调和 evaluation 路径中必要的纯观测埋点；
- 诊断专用的 `NOOP/RIGHT` 协调回退开关，且正式默认值保持不变；
- 诊断 schema、聚合器和独立诊断脚本；
- 对应的确定性回归测试；
- `CHANGELOG.md`；
- 工程师自己的交接记录。

禁止修改：

- `CONSTITUTION.md`；
- `TASKS.md` 的完成状态；
- 除已批准的 `NOOP/RIGHT` 单变量开关外的 planner 搜索、reservation、动作选择和协调行为；
- reward、动作 mask、任务分配、充电和终止规则；
- G3 正式 manifest 的 seed、预算、比较组或结论边界。

若观测埋点无法在不改变行为的情况下实现，工程师必须停止并报告具体调用链，不得自行放宽“行为无变化”要求。

## 11. 工程验收条件

- [ ] 新诊断字段定义、单位和聚合层级有唯一解释；
- [ ] 停滞原因、动作覆盖和 termination reason 通过计数守恒测试；
- [ ] 开关诊断前后的固定 seed 动作轨迹完全一致；
- [ ] `NOOP/RIGHT` 只改变受控协调回退，未配置与 `NOOP` 模式保持当前轨迹；
- [ ] 修复前 `119f20d` 隔离 worktree 入口与配置适配差异已记录；
- [ ] 现有 planner、安全、Phase 2/3/4 回归测试通过；
- [ ] 相关范围 Flake8 和 `git diff --check` 通过；
- [ ] smoke 产物不覆盖既有 G2 长跑结果；
- [ ] 工程师提供 C1 的可复制命令和预计输出路径，但未启动长评估；
- [ ] `CHANGELOG.md` 与代码在同一聚焦 commit 更新；
- [ ] 工作区中的既有用户修改未被覆盖或混入 commit。

## 12. 项目负责人长任务产物验收

- [ ] 记录启动 commit、checkpoint SHA-256、配置 SHA-256 和命令；
- [ ] `300–309 × 20` 全部完成，无缺失、覆盖或静默重试；
- [ ] JSON/CSV 明确记录环境能源配置和确定性动作设置；
- [ ] 与 fixed A* 使用同一完成率、成功、死锁和时限定义；
- [ ] 当前 fixed `NOOP/RIGHT` 与修复前 `119f20d` 三组使用同一当前环境合同；
- [ ] 聚合以 seed 为统计单位并保留逐 seed 值；
- [ ] 报告只形成诊断结论，不触碰正式 seed `200–209`；
- [ ] 由架构师据此更新 G2-1e 决策、G3-1 A* 模式和论文限制描述。

## 13. 给项目工程师的启动提示词

> 执行 `plan/task-package/g2-1-astar-throughput-diagnosis.md` 的工作流 A、B 和第 9.2 节
> G2-1f。先完整阅读
> `AGENTS.md`、`CONSTITUTION.md`、`TASKS.md`、`CHANGELOG.md`、本任务包、
> `llm_mappo/phase2_expert.py`、`llm_mappo/planner.py`、环境 action mask/step 冲突处理以及
> `eval/evaluate_dynamic_ingress_astar.py`。增加无行为变化的动作流水线和停滞归因观测、schema、
> 聚合器、诊断专用 `NOOP/RIGHT` 单变量开关、修复前 `119f20d` 隔离 worktree 入口与确定性
> 回归测试；除获批单变量外不得修改规划、协调、奖励、充电、任务分配或终止行为，不得启动训练或
> 200 回合评估。先用测试证明开启/关闭诊断的固定 seed 动作轨迹完全一致，再运行不超过本任务包
> 上限的 smoke。提供项目负责人可复制的 C1 长评估命令，但不要代为运行。每完成对应 TASKS 子任务
> 时在同一 commit 更新 CHANGELOG；不要修改 CONSTITUTION 或自行勾选 TASKS。交付时列出修改文件、
> schema、计数守恒结果、测试命令与退出状态、smoke 产物、长任务命令、commit 和已知限制。
