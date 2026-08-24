# O0 Reward-Calibrated Heterogeneous Distillation Architecture

## 1. 文档状态与审计基线

本文是优化路线唯一 canonical architecture。当前只纳入已经完成并等待研究所有者审核的
O0-A 现状审计，不把 O0-B 至 O0-F 尚未审核的设计选择提前写成实现合同。

- 代码审计基线：P0 最终 commit `6fcb7d3`；
- 规格分支：`codex/o0-astar-teacher-redesign`；
- 审计范围：规则目标、充电目标、A* waypoint/preference/reservation/coordinator、hard mask、
  buffer、KL、离线 LLM、Actor/Critic 梯度、checkpoint、训练、评估和执行入口；
- O0-A 不修改 Python、配置、环境语义、reward、checkpoint 或实验 seed；
- 本章描述“当前代码实际做了什么”，不表示这些行为已被接受为新方法。

本文中的 `N` 是机器人数量，`A=5` 是
`[NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD]` 动作数，`O` 是运行时 actor observation
宽度，`T` 是 rollout 中的团队时间步数，`D` 是当前语义宽度（历史值为 1 或 2）。

## 2. 当前实现的端到端数据流

```text
TaskQueue / battery hysteresis
  -> assigned task / delivery / charging / idle target
  -> AStarPlanner waypoint features ---------------------> actor observation [N,O]
  -> Manhattan target progress --------------------------> shaped team reward
  -> AStarExpert
       -> stalled-agent parking target rewrite
       -> priority-ordered temporal reservation
       -> raw action preference [N,5]
       -> hard-mask normalization
       -> immediate conflict coordinator rewrite
       -> reservation preference [N,5] ------------------> RolloutBuffer

Structured scenario produced from the same environment state
  -> explicit offline DeepSeek call
  -> historical 2D JSONL labels
  -> inverse-distance kNN lookup [N,2] -------------------> RolloutBuffer

RolloutBuffer [T,N,*]
  -> PPO: final Actor + centralized Critic
  -> reservation KL: final Actor distribution
  -> semantic MSE: Semantic Encoder/Head
  -> checkpoint / update log / episode log

Evaluation or policy replay
  -> deterministic MAPPO action under hard mask
  -> Phase2Warehouse.step

Standalone A* evaluation or expert replay
  -> AStarExpert.act final coordinated action
  -> Phase2Warehouse.step
```

结论：当前训练路径把“路径规划、预约优先权、冲突协调和最终动作分布”合并为一个 A* 标签，
而且把该标签直接施加于最终 Actor。它不符合 O0 所要求的 Pure Motion Teacher 因果边界。

## 3. 字段与符号级链路审计

### 3.1 规则目标、充电目标、观测与 reward

| 字段/对象 | 生产者与准确符号 | shape/取值 | 消费者与梯度 | 启用、失败降级、日志与执行期依赖 |
|---|---|---|---|---|
| task queue 与任务优先级 | `llm_mappo/rules.py::TaskQueue.ordered_tasks/assign_next/priority_weight`；`llm_mappo/environment.py::DynamicWarehouse._assign_available_tasks` | 每任务 `Task`；按字母、编号、到达步排序 | `_target_for_agent`、任务完成 reward、priority observation、A* | 电量 `<0.1` 时不分配新任务；任务状态写入 env info；属于规则层合法高层状态，不应由 A* 改写 |
| 普通任务/配送目标 | `llm_mappo/phase2.py::Phase2Warehouse._target_for_agent` | 每机器人 `(x,y), kind`，kind 为 `task/delivery/idle` | waypoint observation、reward 距离、A* | 无任务或死亡时回退当前位置；执行、训练和标签采集均使用 |
| 充电目标 | `Phase2Warehouse._charging_targets`，由 `charge_threshold`/`charge_release_threshold` 滞回、占用和最近站规则产生 | `agent_id -> (x,y)` | `_target_for_agent`、A*、观测、movement reward、充电指标 | 无空闲站直接抛出 `RuntimeError`；按电量和 agent id 分配；`charging_reservations` 写入 env info；这是规则层目标，不是 A* 决策 |
| waypoint path | `llm_mappo/planner.py::AStarPlanner.plan`，由 `Phase2Warehouse._observations` 调用 | 每机器人可变长坐标元组 | next waypoint、desired direction 与相对关系特征 | `include_waypoint_features=True` 时每次构造观测都规划；失败返回空 path/noop preference，观测回退目标点；当前 policy 执行依赖 A* |
| actor observation | `Phase2Warehouse._observations` | `[N,O]`；每行由 raw RWARE、7 个 own、可选 2 个 priority、4 个朝向、2 个 waypoint 位移、4 个期望朝向、3 个 waypoint 关系、15 个邻居和 3 个全局量拼接 | Actor、Critic、离线语义检索、buffer、标签记录 | waypoint 特征关闭时仍保留相同 waypoint 槽但置零；priority 特征在 Phase 3/4 强制启用；直接决定 checkpoint 的 `actor_observation_dim` |
| hard action mask | `Phase2Warehouse.action_masks` | bool `[N,5]` | MAPPO `_masked_logits`、`AStarExpert.act`、buffer、KL | 死亡/拾取锁仅允许 NOOP；必须拾取时仅允许 TOGGLE；否则 oracle mask 只控制 TOGGLE；每行始终至少一个合法动作；训练、评估和执行均依赖 |
| waypoint progress reward | `Phase2Warehouse._waypoint_distances/_movement_rewards/step` | 每机器人标量，最终均值为 team reward 的一部分 | PPO return/GAE 与 Critic | 目标曼哈顿距离降低时加 `waypoint_reward`；另有每步 `-0.01` 和低电量惩罚；当前 reward 依赖规则目标及 Teacher 派生 waypoint 语义 |
| priority delivery reward | `DynamicWarehouse._complete_delivered_tasks` | 每完成任务 `5.0 * priority_weight(label)` | 环境个体 reward，随后由 Phase2 wrapper 取均值 | 完成事件写入 info；优先级影响长期回报，但不应进入 Pure Motion 标签生成 |

### 3.2 A* preference、预约、协调、buffer 与 KL

| 字段/对象 | 生产者与准确符号 | shape/取值 | 消费者与梯度 | 启用、失败降级、日志与执行期依赖 |
|---|---|---|---|---|
| spatial A* plan | `AStarPlanner.plan/_search` | `PathPlan`，含 waypoints、5 动作偏好、事件 | waypoint observation；单机器人 expert | topology 无路时返回空 path 与 NOOP-heavy preference；没有搜索预算上限或 per-agent validity |
| temporal reserved plan | `AStarPlanner.plan_with_reservations/_temporal_search` | `PathPlan`，附 timed positions、first action、expanded nodes、耗时、失败原因 | `AStarExpert._reserved_action_preferences` | horizon 内到不了目标时可能返回 partial path；失败区分 topology/reservation/horizon；诊断写 expert statistics |
| reservation table | `llm_mappo/planner.py::ReservationTable` | `horizon+1` 个 cell/edge/terminal set | prioritized temporal A* | 当前 horizon 为 `AStarExpert._reservation_horizon` 的 16–64；终点默认额外保留 2 步，legacy 可持续至 horizon |
| planning order | `AStarExpert._priority_key` | `(loaded/task/idle rank, agent_id)` | `_reserved_action_preferences` 的逐机器人顺序 | 先规划机器人占用后规划机器人可用时空；等价于通行权分配，属于 O0 禁止的高层污染 |
| parking/yield target | `AStarExpert._target_for_agent/_yielding_agents` | 改写为最近 charging station，kind=`parking` | A* 目标、cache signature、预约 label | 载货机器人停滞 20 步后选择其他机器人让行；改写规则层既定目标，是明确 purity 违规 |
| raw A* preference | `AStarPlanner._smooth_preferences/_noop_preferences` 与 `AStarExpert.action_preferences` | float `[N,5]` | mask 归一化、expert 数据、training snapshot | smooth preference 给非首选动作非零质量，包含 `TOGGLE_LOAD`；死亡/锁/拾取/失败被编码为 NOOP 或 TOGGLE one-hot，而不是无效标签 |
| masked preference | `phase2_expert.py::_mask_and_normalize` | float `[N,5]`，每行和为 1 | coordinator、expert dataset、training snapshot | 无有效质量时整体抛错；hard mask 通常会清掉非法 TOGGLE，但没有“蒸馏前非法质量必须为零”的原始合同 |
| coordinator output | `AStarExpert._coordinate_actions` 及 occupied/edge-swap/contested helpers | action `[N]`；被改写行变为 one-hot preference `[N,5]` | `AStarExpert.act` 返回的动作与 preference | coordinator 直接分配即时让行；原因写入 `last_action_pipeline`；后者只在独立 A* 诊断入口聚合，训练 buffer 接收的是已改写 preference |
| reservation preference buffer | `phase3_training.py::_handle_environment_command('snapshot')` -> `RolloutBuffer.add` | `[T,N,5]` | `MAPPOUpdater.update` | 仅 `astar_kl_enabled` 时由 expert 产生；关闭时 buffer 填均匀分布；没有逐机器人 valid mask、失败原因或 purity 标志 |
| A* KL | `MAPPOUpdater.update` 的 `reservation_kl` | minibatch 中对全部机器人均值的 KL 标量 | 总 loss，梯度流向 `DualHeadActor.motion_encoder` 与最终 `motion_head` | 仅 coefficient>0 启用；hard mask 后再次归一化 teacher；直接约束最终 action distribution，不存在独立 Motion Prior Head |
| A* schedule | `phase3_training.py::_reservation_coefficient` | episode 分段指数衰减标量 | updater hyperparameters | 从配置初值衰减到 minimum；更新日志含 coefficient 和 KL；当前 checkpoint 只间接保存当时 config，不保存独立 schedule 状态 |

### 3.3 离线 LLM 与语义链路

| 字段/对象 | 生产者与准确符号 | shape/取值 | 消费者与梯度 | 启用、失败降级、日志与执行期依赖 |
|---|---|---|---|---|
| structured scenario | `llm_mappo/llm_teacher.py::build_engagement_scenarios` | 每机器人一个 `EngagementScenario`，包含完整 actor observation、battery/load/priority/target kind 和最近 3 个 peer | DeepSeek prompt、JSONL | scenario state 使用规则目标；normal 样本沿当前 `AStarExpert` rollout 采集，受现有协调 A* 访问分布影响 |
| LLM request | `DeepSeekTeacher.label_semantics/_build_request` | 历史输出 2 scores + 2 reasons | strict parser | 只由显式标签 CLI 调用；训练与执行 API 调用为 0；失败最多 3 次后抛错，不生成替代标签 |
| parser 与历史 label | `parse_semantic_response`、`SemanticPreferenceLabel` | `[task_commitment, local_assertiveness] in [0,1]^2`，整条 JSON 必须正好四个 key | JSONL writer/loader | 缺失、额外 key、非数值、越界或空 reason 均失败；这是严格记录有效性，但没有显式 validity 字段 |
| offline dataset | `write/append/load_labelled_scenarios` | JSONL，每行一个机器人状态和 2D label | `OfflineSemanticTeacher.from_jsonl` | 校验必需字段和统一 observation 宽度；不保存 prompt、request body、原始响应、重试轨迹或不可变模型版本 |
| kNN semantic target | `llm_mappo/phase4.py::OfflineSemanticTeacher.targets` | `[batch*N,2]`；k=3 inverse squared-distance mean | rollout buffer | 仅校验 observation 宽度；无距离截断、OOD reliability、validity weight 或无邻居降级；总能返回一个 target |
| semantic target buffer | `RolloutBuffer.engagement_targets` | 历史 `[T,N]` 或 `[T,N,2]` | semantic MSE | 缺失时填 `-1`；2D 以整机器人两维同时有效；没有 reliability、OOD 或 label provenance 字段 |
| semantic MSE | `MAPPOUpdater.update` 的 `engagement_loss` | 有效机器人/记录上的 MSE | `engagement_encoder` 与 `engagement_head` | coefficient>0 启用；PPO 经 `.detach()` 不回传 semantic encoder/head；当前无三维 Semantic Adapter |
| LLM schedule | `phase3_training.py::_engagement_coefficient` | episode 分段指数衰减标量 | updater hyperparameters | 配置初值衰减至 minimum；日志含 coefficient 和 component loss；checkpoint 未保存独立恢复游标 |

### 3.4 Student、Critic 梯度和 checkpoint

| 组件 | 当前实现与 shape | 梯度所有权和依赖 | checkpoint/兼容性 |
|---|---|---|---|
| Motion Encoder | `DualHeadActor.motion_encoder: O -> hidden -> 64` | PPO 与 A* KL 均可更新；不由 semantic MSE 更新 | `model_state` 保存 |
| Semantic Encoder/Head | `engagement_encoder: O -> hidden -> 64`；head `64 -> D -> sigmoid`，D 仅允许 1/2 | semantic MSE 更新；PPO/A* KL 看到的是 detached semantic 值，不能回传该分支 | 旧 Phase 3 为 D=1，Phase 4 为 D=2 |
| Final Motion Head | 输入 `64 + D`，输出 `[N,5]` logits | PPO 与 A* KL 直接更新；语义值虽 detach，但作为执行输入 | 没有独立 Motion Prior Head 或 Semantic Adapter |
| Centralized Critic | 每机器人 `O -> hidden`，attention pooling 后输出每团队状态一个 value | 仅 value loss 更新；Teacher 当前不直接监督 Critic | 与 Actor 一起存入 `model_state` |
| Rollout/GAE | observations `[T,N,O]`、actions/log-probs/masks/preferences/targets、team reward/done/value/stream id | PPO advantage 按团队时间步复制给 N 个 actor action；Critic用团队 return | buffer 不进入 checkpoint，训练不支持精确中断恢复 |
| Phase 3/4 checkpoint | `_save_checkpoint` 保存 `model_state/config/actor_observation_dim/semantic_dim/semantic_features_enabled/episodes/steps/phase` | loader 依据 metadata 与 tensor shape 交叉推断 D，然后 strict `load_state_dict` | `_checkpoint_semantic_dim` 只允许 1/2；没有三维 schema version、维度名、reliability、EMA、H、observation contract 或 schedule state |
| Phase 2 checkpoint | `phase2_training.py::_save_checkpoint` 保存基础 actor/critic state、config、observation width、episodes/steps | 无 semantic branch | 由 `load_policy` 历史加载，不属于新三维合同 |

### 3.5 训练、评估与执行入口

| 入口 | Controller/Teacher 使用 | 当前 A* 执行期依赖 | 关键输出 |
|---|---|---|---|
| `train/train_phase3.py` | `train_phase3`，Phase 3a/3b | observation 默认含 waypoint；3b snapshot 额外运行 `AStarExpert` | checkpoint、episodes/updates CSV、summary、TensorBoard |
| `train/train_phase4.py` | 同一 `train_phase3`，D=2，可启用 A* KL 与 offline LLM | observation waypoint + 可选每步 A* teacher；无在线 LLM | 同上及 offline dataset metadata |
| `train/collect_phase4_labels.py` | 唯一显式 online provider 入口；normal 数据用 A* rollout，controlled 数据注入状态 | 标签生成阶段使用 A* 到达 normal 状态 | JSONL/partial JSONL 与汇总；不保留原始 provider response |
| `eval/evaluate_phase3.py` / `evaluate_phase3` | deterministic MAPPO | policy observation 默认重新运行 waypoint A*；不运行 KL teacher | seed 聚合、完成/碰撞/死锁/充电/语义诊断 |
| `visualize.py --controller policy` | strict checkpoint loader + deterministic MAPPO | `env._observations()` 使历史 policy 依赖 waypoint A* | trace/GIF/summary |
| `eval/evaluate_dynamic_ingress_astar.py` | `AStarExpert.act` 作为非学习端到端 controller | 完全依赖 A*、预约与 coordinator | A* 完成率和详细规划/停滞流水线诊断 |
| `visualize.py --controller expert` | `AStarExpert` 独立回放 | 完全依赖 A* | 可视化；不得与 Student 执行依赖混为一谈 |

## 4. O0 purity 矩阵与违规登记

### 4.1 信息分类矩阵

| 信息/机制 | Pure Motion 标签允许性 | 当前路径 | O0-A 判定 |
|---|---|---|---|
| 规则层已确定合法目标坐标 | 允许，只读 | `Phase2Warehouse._target_for_agent` | 保留边界；A* 不得更换 |
| 当前 pose、朝向、静态可通行几何 | 允许 | `AStarPlanner` search state | 纯运动输入 |
| 邻近机器人当前物理占用 | 可作为后续 O0-B 的局部动态障碍候选 | spatial/temporal planner | 尚未决定具体语义 |
| 搜索成功、失败、耗时、展开节点、path length | 仅诊断；失败可形成 binary validity | `PathPlan`/expert statistics | 不得组合成连续优化置信度 |
| hard action mask | 允许作为统一硬约束，不能由 Teacher 改写 | policy/expert 共用 | 必须保留 |
| task assignment、priority、waiting time | 禁止参与运动标签生成；可供规则层或 LLM 使用 | target 与 reservation order 间接使用 | 污染 |
| 充电/parking 目标选择 | 禁止由 A* 生成或改写 | charging 是规则层；parking 由 expert 改写 | parking 路径污染 |
| `_yielding_agents`、reservation priority、通行权 | 禁止 | 当前 A* 核心路径 | 污染 |
| final coordinator action | 禁止作为 Teacher label；可作为独立启发式基线行为 | 当前 preference 被 coordinator one-hot 改写 | 污染 |
| reward、future return、Critic、Student disagreement、LLM label | 禁止参与 label 生成 | 当前 A* label 不读取这些量 | 当前未污染；未来 calibration 只能事后评价 |
| search/replan/cache statistics | 仅诊断 | expert summary | 允许记录，不得加权 |

### 4.2 已确认违规和缺口

| ID | 证据 | 影响 |
|---|---|---|
| P-01 | `AStarExpert._yielding_agents/_target_for_agent` 将部分机器人目标改为最近充电站 parking | A* 改写高层目标，标签不再只回答“如何到达既定目标” |
| P-02 | `_priority_key` 以 loaded/task/idle 和 agent id 决定 reservation 顺序 | 教师隐式分配通行权；动作偏好包含调度决策 |
| P-03 | `_coordinate_actions` 改写冲突动作，并把 preference 变成 coordinator action 的 one-hot | buffer 中不是 planner preference，而是端到端协调器输出 |
| P-04 | `_smooth_preferences` 给 `TOGGLE_LOAD` 和其他非首选动作非零原始质量 | 违反非运动/非法动作在蒸馏前质量为零的目标合同 |
| P-05 | dead/locked/pickup/search failure 仍生成 NOOP 或 TOGGLE 有效标签 | 没有 fail-closed invalid robot；“无法教”被误编码为动作监督 |
| P-06 | buffer 只有 `[N,5]` preference，没有 `valid_mask[N]` 或逐机器人失败原因 | 无法保持逐机器人 coverage、降级和计数守恒 |
| P-07 | reservation KL 直接比较 Teacher 与最终 Actor distribution | A* 直接约束最终动作策略，不是独立 Motion Representation/Prior Head |
| P-08 | actor observation 的 waypoint/desired-direction/relation 由 A* 在线生成 | 当前 Student evaluation/execution 尚未摆脱 A* |
| P-09 | movement reward 按规则目标距离变化提供 waypoint progress shaping | reward 与未来 NoWP/无 A* 合同存在耦合，需要后续明确兼容边界 |
| P-10 | normal LLM 样本沿带预约、parking 和 coordinator 的 A* rollout 采集 | 历史语义数据的状态分布带有旧 expert policy provenance |
| P-11 | 历史 2D kNN 对任意查询都返回标签，没有 validity × OOD reliability | 不满足新三维语义可靠性合同 |
| P-12 | Actor、buffer 与 loader 只允许 D=1/2，checkpoint 不含新合同元数据 | 旧 checkpoint 不能安全迁移为三维架构 |

这些条目是 O1 未来实现必须消除或隔离的污染面；O0-A 不选择新算法，也不修改现有行为。

## 5. 历史 LLM 标签生成链路追溯

### 5.1 可精确恢复的合同

- CLI：`train/collect_phase4_labels.py`；默认 provider=`mock`，历史正式生成显式使用
  `--provider deepseek`；
- 基础 model ID 字符串：`deepseek-v4-flash`；数据记录中的 provider 名为
  `deepseek:deepseek-v4-flash`；
- API endpoint 默认值：`https://api.deepseek.com/chat/completions`；
- 非 thinking 模式：`temperature=0.0`、`max_tokens=1024`、`timeout=120s`、最多 3 次请求，
  backoff 为 5s/15s；历史记录名称不含 thinking suffix，故记录声明的是非 thinking 路径；
- system prompt：`You are a JSON-only warehouse semantic teacher.`；完整 user prompt 固定在
  `DeepSeekTeacher.label_semantics`，包含 2D key、A>B>C、battery/load/yield 约束和 forbidden
  action/path/task assignment；
- scenario generator：`collect_stratified_offline_labels` + `_inject_controlled_scenario` +
  `build_engagement_scenarios`；历史配额为 normal 120、priority conflict 100、narrow corridor 80、
  low battery 60、station exit 40；默认 collection seeds 为 100/101/102；
- parser：`parse_semantic_response`，要求恰好
  `task_commitment/task_reason/local_assertiveness/coordination_reason`；
- observation version：`phase4-semantic-v2`；
- historical schema：scenario id/version/type、完整 observation、agent id、battery、loaded、priority、
  target kind、最近 3 peer、2D scores、model、created_at 和两个 reasons；
- 当前训练数据：
  `artifacts/phase4_labels/deepseek_medium_5ag_400_v2_repaired_r2.jsonl`，400 records，SHA-256
  `9928F5756C1261589946EB3AEDF8DC2C1FA6F73F037CD05C31AFCA0683161797`；
- 旧单维 checkpoint：Phase 3 loader 以 `phase=3*` 或 tensor shape 推断 `semantic_dim=1`；
- 旧二维 checkpoint：Phase 4 loader 以 `phase=4`、metadata 和 tensor shape 交叉确认
  `semantic_dim=2`，然后 strict load。

### 5.2 无法从现有产物恢复的事实

“`deepseek-v4-flash`”只是请求时的模型别名。当前 JSONL 没有保存服务端不可变 model revision、
provider response id、响应 model/version、完整 request body、原始 response、HTTP/retry 轨迹、prompt
hash 或 generator commit。因此历史数据可以复现其仓库侧请求合同和内容哈希，但不能证明调用时
服务端精确权重版本。该缺口必须在 O0-D 的新三维 manifest 中 fail-closed 补齐，不能用当前
别名冒充不可变版本。

当前 label loader 也没有验证 `scenario_id` 内容哈希、observation version 一致性、模型集合、
重复记录或文件 manifest；这些都是历史合同限制，不得迁移为新三维默认行为。

## 6. 两份研究输入的哈希、差异与方法主张裁决

### 6.1 输入一致性

| 输入 | SHA-256 | 审计结果 |
|---|---|---|
| `方案文档.md` | `261A33536C2E53EEEBDFF08F49DD2A42217A0AAE7324C154CFA7785918ABD2B0` | 用户指定的较新输入 |
| `基于异构多教师知识蒸馏与MAPPO的多机器人路径规划与协同方案.md` | `12DB077897D25DA999746CF044C28497BDF2C5C5675B2705F6598CCE780B70E7` | 同内容排版版本 |

逐行比较显示，方法章节和文字内容相同。差异是无序列表由 `*` 改为 `-`，以及三处范数从错误的
`\left|...\right|_p` 改为 `\left\|...\right\|_p`。canonical architecture 采用明确的
`\lVert...\rVert_p` 语义。没有只存在于其中一个输入的方法主张。

### 6.2 主张处置矩阵

| 输入主张 | 处置 | 理由与 canonical 落点 |
|---|---|---|
| A* Motion + offline LLM Semantic + MAPPO Student 的异构教师主线 | 接纳 | 与 Mission 和 O0 冻结方法主线一致 |
| A* 不做任务分配、优先级、让行权、长期调度或最终动作 | 接纳为目标边界 | 当前 P-01/P-02/P-03 违反；O0-B 必须冻结可执行合同 |
| A* 采用有限时域滚动规划而非完整 episode 求解 | 延后 | 方向可行，但搜索 horizon、动态障碍和预算属于 O0-B，不在审计阶段替实现者选择 |
| A* 输出 motion preference，不是最终动作 | 接纳并修正 | 输入示例动作维度不含项目的第 5 个动作；正式 shape 必须是 `[N,5]`，且非运动/非法质量为零 |
| 用搜索规模、绕行、冲突、推进组合连续 `c_A` | 拒绝 | 已批准合同只保留 binary validity；搜索质量仅诊断，任务价值由独立 Reward Calibration 评价 |
| A* loss 只监督 Motion Representation/Prior Head | 接纳为目标边界 | 当前实现直接 KL 最终 Actor，见 P-07；具体 head/shape 在 O0-E 冻结 |
| LLM 不输出动作、路径、任务分配或强制让行 | 接纳 | 与现有 prompt 边界和新三维合同一致 |
| 三维 `task_persistence/yielding_preference/coordination_risk` | 接纳并修正术语 | 使用 O0 规格中固定名称和顺序；历史 2D 数据不得迁移填充 |
| 相关系数过高后修改语义或数据 | 修正 | Pearson 与 Spearman 都报告；`|rho|>=0.80` 只触发人工复核，不自动改定义、删标签或重生成 |
| LLM reliability=`consistency*OOD*validity` | 拒绝并替换 | 唯一正式形式为整记录 `validity*shared OOD reliability`，不保留 consistency |
| Semantic Head stop-gradient，PPO 训练 adapter 而不绕回语义 head | 接纳为目标边界 | 当前只有 detach、没有 adapter；精确梯度与 shape 在 O0-E 冻结 |
| Student–Teacher disagreement 提高优化权重 | 拒绝 | disagreement 完全移出权重，只保留诊断日志 |
| `lambda_A(t)`/`lambda_L(t)` 随训练衰减 | 接纳并延后常数 | 所有对照共用预注册 schedule；精确算法和恢复语义在 O0-E 冻结 |
| 两类教师无需 softmax 竞争 | 接纳 | 两个知识空间独立门控，不要求权重和为 1 |
| Teacher 不直接监督 Critic | 接纳并限定 | Critic 只为 PPO 与 detached continuation bootstrap 服务；calibration 不反传 |
| 执行期无需 A* | 延后且禁止现时声称 | 当前 P-08/P-09 明确不成立；只有后续 NoWP/无 A* 门通过后才允许 |
| 方法已提高样本效率、协作或泛化 | 拒绝作为当前事实 | 输入是方案而非实验结果；必须由冻结的后续实验支持 |

上述矩阵已覆盖两份输入中的有效方法内容。两份输入仍保持未跟踪状态，待 O0-F 完成全部章节
映射、研究所有者审核并确认 canonical architecture 无内容丢失后再移除；O0-A 不提前删除。

## 7. O0-A 审计结论与后续边界

1. 当前 A* 链路具有几何规划能力和完善的部分诊断，但其 preference 同时携带目标改写、预约
   优先权和 coordinator 决策，不能直接命名为 Pure Motion label。
2. 当前 KL 监督最终 Actor，没有独立 Motion Prior Head，也没有逐机器人 validity；因此不能只
   通过改名满足新论文方法。
3. 当前 policy 仍在 observation 与 reward 上依赖 waypoint A*，所以执行期无需 A* 仅是后续
   条件主张。
4. 历史 LLM 链路可追溯到仓库侧 model alias、prompt、temperature、generator、parser 和 2D
   schema，但缺少服务端不可变版本及原始请求/响应 manifest；新 3D 数据必须重新生成且严格隔离。
5. O0-B 的唯一边界是比较并冻结 Pure Motion Teacher 候选和标签合同。O0-A 没有选择候选、
   搜索 horizon、概率平滑、budget、replan、cache 或 fail-closed 常数。

## 8. O0-A 验证证据

验证日期为 2026-08-25，运行环境为项目 `py310`（Python 3.10.19）。

- `python -m pytest`：退出码 0，184 tests passed；
- `python -m flake8 rware llm_mappo eval train scripts figures/core`：退出码 0；
- `python visualize.py --help`：退出码 0；
- `python eval/evaluate_dynamic_ingress_astar.py --help`：退出码 0；
- `git diff --check`：退出码 0；
- 相对 P0 `6fcb7d3` 的 Python/YAML/JSON 运行代码与配置差异：0 个文件；
- O0 spec 目录：唯一 1 个；canonical architecture 占位术语命中：0；
- 两份研究输入 SHA-256 均与 O0 requirements 登记值一致；
- 两份研究输入按任务组边界继续保持未跟踪，O0-A 未提前执行 O0-F 的删除动作。
