# O0 Reward-Calibrated Heterogeneous Distillation Architecture

## 1. 文档状态与审计基线

本文是优化路线唯一 canonical architecture。当前纳入已经获批的 O0-A 现状审计，以及已经由
研究所有者给出设计批准、等待任务组交付审核的 O0-B Pure Motion Teacher 冻结合同。不把
O0-C 至 O0-F 尚未审核的设计选择提前写成实现合同。

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
5. O0-B 的唯一边界是比较并冻结 Pure Motion Teacher 候选和标签合同。O0-A 当时没有选择候选、
   搜索 horizon、root-action cost、Boltzmann transform、budget、replan、cache 或 fail-closed
   常数；获批选择见第 9 章。

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

## 9. O0-B Pure Motion Teacher 冻结合同

### 9.1 候选比较与唯一选择

| 候选 | 纯度 | 覆盖与计算 | 可解释性与因果边界 | 结论 |
|---|---|---|---|---|
| A：现有预约/协调 `AStarExpert` | 低；包含 parking、规划顺序、reservation 和 coordinator | 在部分拥堵状态下可给出端到端动作，但计算和失败互相耦合 | 无法区分几何建议与通行权/让行决策 | 拒绝作为 Teacher；只保留为 `Heuristic-Dispatcher+A*` 类启发式基线 |
| B：完全忽略其他机器人的静态几何 A* | 高；不读取其他机器人 | 覆盖高、计算低，但可能反复建议驶向当前已占用位置 | 是纯几何标签，但缺少当前局部拥堵事实 | 不作为正式 Teacher；可保留为 Pure Teacher 诊断消融 |
| C：独立、无通行权的局部动态几何 A* | 高；其他机器人只形成匿名当前占用集合 | 以部分 coverage 换取当前几何可行性；每机器人独立、有界 | 只回答已解析目标下的运动方向，不推断谁应让行 | **唯一正式 Pure Motion Teacher** |

正式 Teacher 版本固定为 `pure-motion-astar-v1`。候选 A、B 都不得由实现者切换成主方法，也不得
与 C 混合生成标签。

### 9.2 职责和输入边界

每台机器人独立执行一次 query。query 只包含：

1. 自身 `(x, y, direction)`；
2. 规则层已解析、已验证且只读的目标坐标 `(goal_x, goal_y)`；
3. canonical static layout；
4. 除自身以外的机器人当前坐标组成的匿名、去重、字典序排序集合；
5. 独立 `pure_motion_mask[5]`；
6. 固定 `K_motion=12`、expansion budget `512` 和 teacher version。

其他机器人在整个本次 `K_motion` 搜索窗口内仅视为占据其当前坐标的匿名静态障碍。Teacher
不接收或推断其 ID、方向、载货状态、任务、优先级、目标、动作、路径或未来轨迹。各机器人
query 不共享 reservation table、搜索 frontier、规划结果或可变协调状态；批处理遍历顺序不得
改变任意机器人的输出。

carrying state 默认不进入 query 或 cache。只有底层环境确实因载货改变 footprint 或物理可通行性
时，才允许把规范化 `footprint_class` 作为物理字段加入 static traversability 和 cache；不得以
carrying state 推导任务优先权、让行权或调度语义。

以下信息禁止直接进入输入，也禁止经 mask、heuristic、tie-break、cache 或 fallback 间接进入：

- task assignment、task label、priority、waiting time；
- 充电意图、充电站选择、parking 或规则目标改写；
- reservation、yield、coordinator、机器人规划顺序或通行权；
- reward、future return、Critic、Student action/disagreement；
- LLM label/reliability、Reward Calibration、`Delta G` 或任何训练结果。

职责问题固定为：“给定自身几何状态和已解析目标，哪个运动方向在几何上合理？”是否等待、
谁应让行、何时行动以及最终动作属于 LLM+MAPPO，不属于 Pure Motion A*。

### 9.3 Static graph、mask 与动作支持

`G_static` 是由 canonical layout 和当前物理 `footprint_class` 唯一构造的无权网格图。它只编码
边界、固定不可通行单元和底层 footprint 可通行性；不编码机器人、任务、优先级或动态规则。
静态目标距离固定为：

$$
h_{static}(q,g)=\operatorname{shortest\_path\_length}_{G_{static}}(q,g).
$$

该距离用确定性 BFS 预计算或等价确定性最短路得到，单位为 cell transition，忽略 orientation
和其他机器人。静态不可达时为正无穷并直接 fail-closed。它既是 bounded A* 的 `h`，也是唯一
progress certificate 距离；不得换成 reward、Manhattan 距离或可学习估计。

`pure_motion_mask` 是独立 bool `[N,5]`，只允许物理/执行层合法性进入：地图边界、静态 footprint
碰撞、匿名当前占用、死亡、物理锁和底层强制交互状态。其动作位固定为项目枚举：

```text
NOOP=0, FORWARD=1, LEFT=2, RIGHT=3, TOGGLE_LOAD=4
```

Teacher 的监督 support 只含 `FORWARD/LEFT/RIGHT`。`NOOP` 和 `TOGGLE_LOAD` 在
`pure_motion_mask` 的 Teacher 视图中始终为 false，输出质量始终为 0。priority、yield、
reservation、coordinator 或 task policy 不得参与 mask。若底层环境要求 mandatory
`TOGGLE_LOAD`，该机器人不是 motion teaching state，按 fail-closed 处理。

搜索 successor 也只包含 `FORWARD/LEFT/RIGHT`，每个动作 cost 为 1；不允许把 NOOP 作为内部
等待动作绕过监督 support。转向改变 orientation 而不改变坐标，FORWARD 按当前 orientation
移动一个 cell。输入的 `pure_motion_mask` 只约束当前 root action；后续 successor 必须使用同一
纯物理 transition validator 在各自模拟状态重新计算合法性，不能错误复用 root mask，也不能
调用包含 priority/yield/reservation/coordinator 的环境 mask。所有 successor 均须满足 static
footprint 和匿名占用约束。

### 9.4 Bounded A*、预算和确定性

搜索状态固定为：

```text
(x, y, direction, depth, root_action)
```

`depth` 从首个 root action 后的 1 开始，最大为 `K_motion=12`。`K_motion` 与 Reward Calibration
的 `H_reward=12` 是两个不同配置字段、日志字段和 checkpoint 字段，禁止别名、联动修改或共享
同一个参数对象。累计 motion cost 固定为 `g=depth`，OPEN 使用
`f=g+h_static((x,y),goal)`；该 `f` 与第 9.5 节窗口末 cost 使用同一单位和定义。

每机器人、每次 query 最多 512 次 node expansion。一次 expansion 严格定义为一个 state 从
OPEN pop 后执行 successor processing；过期 heap entry 不计 expansion，但必须在 processing
之前按 deterministic best-cost table 丢弃。达到 512 后不得再 pop 或生成 successor。

OPEN 的唯一排序键固定为：

```text
f ascending
-> h_static ascending
-> depth descending
-> root_action_rank ascending
-> x ascending
-> y ascending
-> direction_rank ascending
```

其中：

```text
root_action_rank: FORWARD=0, LEFT=1, RIGHT=2
direction_rank: UP=0, RIGHT=1, DOWN=2, LEFT=3
```

direction rank 与 `llm_mappo/planner.py::_DIRECTION_ORDER` 一致，不使用底层 `Direction.value` 的
`UP,DOWN,LEFT,RIGHT` 顺序。successor 生成顺序同样固定为
`FORWARD, LEFT, RIGHT`。相同 state 和 root action 的相同或更高 `g` 路径直接丢弃；相同 `g`
保留由固定 successor 顺序首先插入的路径，不增加随机或时钟 tie-break。

这是每机器人每次 query 的**单次** bounded search。所有通过 root mask 的动作共享同一个 OPEN、
同一个 expansion counter 和同一总 512-expansion budget，但每个搜索状态必须保留 root
provenance。禁止为 `FORWARD/LEFT/RIGHT` 分别启动独立完整 A*。搜索持续到：

- 每个物理合法 root 已获得一个按 OPEN 顺序认证为最小 cost 的有效 continuation，或其 frontier
  已耗尽；或
- 达到 512 expansions；或
- 全部共享 frontier 已耗尽。

某个 root 的有效 continuation 首次按 OPEN 顺序出队并通过第 9.5 节证书时，该 root 的最小 cost
已经确定，后续不再展开该 root。达到共享预算时，尚未获得认证 continuation 的 root 统一记为
`C=+∞`，不得启动补充搜索；已经认证的 root cost 保留。该规则可能产生 root 级
`budget_exceeded`，但只有全部 root 都没有有限 cost 时才产生机器人级失败。

不得因为先找到某一机器人的局部建议而改变其他机器人的搜索输入或顺序。规划时间只记录，
不能作为搜索终止条件；因此相同输入的 label、validity、failure reason 和结构化 diagnostics 必须
完全相同。wall-clock timing 本身允许随机器波动，但仅属日志。

### 9.5 Progress certificate 与 root-action-conditioned planning cost

对状态 `s` 和物理合法 root action `a`，有效 continuation 集 `T_K(s,a)` 中的轨迹必须：

- 首动作固定为 `a`；
- 未提前到达 goal 时长度严格为 `K_motion`；若在窗口内首次到达 goal，可在该步终止；
- 每个动作和中间状态均满足本 query 的纯物理约束；
- 满足窗口整体严格进展：

$$
h_{static}(q_{end},g)<h_{static}(q_{start},g).
$$

不要求每一步降低距离，允许先转向或进行必要的几何绕行；但窗口末态无严格进展时不能成为
Teacher label。达到 goal 的 trajectory 以 `h_static=0` 自然满足证书。motion step cost 固定为
`c_motion=1`，只描述运动长度，不读取 reward。正式 planning cost 定义为：

$$
C_{A^*}^{K}(s,a)=
\min_{\tau\in\mathcal T_K(s,a)}
\left[\sum_{t=0}^{L(\tau)-1}c_{motion}(a_t)
+h_{static}(q_{L(\tau)},g)\right].
$$

这里 `K=K_motion=12`；提前到达 goal 时 `L(τ)<=K`，否则 `L(τ)=K`。非法 root、搜索失败、共享
预算耗尽时尚未认证、或不存在有效 continuation 的 root action 统一定义
`C_A*^K(s,a)=+∞`。`+∞` 是“该 root 不进入 soft prior”的显式 sentinel，不是可学习数值，也不
代表低置信度。论文优先使用“root-action-conditioned short-horizon planning cost”这一术语，
不得把 `C_A*^K(s,a)` 称为或暗示为学习得到的 RL Q-value。

### 9.6 Motion preference 与输出 schema

正式 Motion Prior 删除固定 `0.85/0.15` label smoothing。设有限 cost 的 root action 集为：

$$
\mathcal F(s)=\{a\in\{\mathrm{FORWARD,LEFT,RIGHT}\}:C_{A^*}^{K}(s,a)<+\infty\}.
$$

只在 `F(s)` 内执行状态内 min-max 归一化。令 `C_min`、`C_max` 分别为有限 cost 的最小值和
最大值：

$$
\widetilde C(s,a)=
\begin{cases}
0, & C_{max}=C_{min}\ \land\ a\in\mathcal F(s),\\
\dfrac{C_{A^*}^{K}(s,a)-C_{min}}{C_{max}-C_{min}},
& C_{max}>C_{min}\ \land\ a\in\mathcal F(s).
\end{cases}
$$

Boltzmann temperature 唯一固定为 `tau_motion=1.0`，不得按状态、训练性能、搜索规模或
Reward Calibration 调整：

$$
p_{A^*}(a\mid s)=
\begin{cases}
\dfrac{\exp(-\widetilde C(s,a)/\tau_{motion})}
{\sum_{b\in\mathcal F(s)}\exp(-\widetilde C(s,b)/\tau_{motion})},
& a\in\mathcal F(s),\\
0, & a\notin\mathcal F(s).
\end{cases}
$$

当所有有限 cost 相等时，该公式在有限 root 间产生均匀分布；只有一个有限 root 时自然产生
one-hot。实现必须使用数值稳定 softmax。`NOOP`、`TOGGLE_LOAD`、非法动作及没有有效
continuation 的 root action 质量严格为 0。有效行必须为有限 float32、非负、shape `[5]` 且和为
1（容差 `1e-6`）；否则整行 fail-closed。

正式批量输出固定为：

| 字段 | dtype/shape | 合同 |
|---|---|---|
| `motion_preferences` | float32 `[N,5]` | 有效行按上式归一化；无效行全零 |
| `valid_mask` | bool `[N]` | 每机器人独立，不得提升为团队级 validity |
| `failure_reason` | enum string `[N]` | 有效行为 `ok`，无效行为唯一首个失败原因 |
| `diagnostics` | struct `[N]` | 只记录，不参与标签、validity 的连续权重或 loss 权重 |

`diagnostics[i]` 至少包含：`teacher_version/query_hash/cache_hit/expanded_nodes/
planning_time_ms/root_costs_raw/root_costs_normalized/root_status/root_path_length_actions/
root_h_static_end/root_progress_delta/feasible_root_actions/preferred_root_action/
budget_exhausted/non_finite_detected/layout_hash/K_motion/expansion_budget/
cost_normalization/tau_motion`。三类数组均按项目五动作顺序保存；不在 support 的项使用 schema
定义的 null/status，而不把 `+∞` 序列化为非标准 JSON 数字。`preferred_root_action` 只表示有限
cost argmin，cost 相同按固定 root rank 决定；它不替代完整 soft preference。除
`planning_time_ms` 外，结构化 diagnostics 也必须对相同输入确定。`valid_coverage` 是上层按
`sum(valid_mask)/N` 聚合的日志量，不是单机器人 label 字段。

`root_status[5]` 的唯一枚举固定为：`unsupported_action`（NOOP/TOGGLE）、`physical_illegal`、
`finite`、`budget_exceeded`、`search_exhausted`、`no_progress_continuation`。只有 `finite` 对应
JSON 中非 null 的 raw/normalized cost；其余状态对应 null cost 和零概率。任何意外非有限中间
量都不降级为 root status，而是按第 9.7 节使整台机器人 `non_finite_output` fail-closed。

### 9.7 Fail-closed 与失败原因优先级

以下检查按固定顺序执行，首个命中项成为唯一 `failure_reason[i]`：

1. `dead`；
2. `picking_lock`；
3. `mandatory_toggle_load`；
4. `already_at_goal`；
5. `invalid_goal`（越界或非静态可通行）；
6. `static_unreachable`；
7. `no_physical_root_action`；
8. `non_finite_output`；
9. `budget_exceeded`（512 expansions 后没有任何 root 获得有限 cost）；
10. `search_exhausted`（OPEN 耗尽且无合法 trajectory）；
11. `no_progress_trajectory`（存在合法短轨迹但没有严格 progress certificate）。

成功时为 `ok`。第 8 项覆盖 heuristic、有限 root cost、归一化结果或 preference 出现意外
NaN/Inf；按合同为无效 root 设置的 `C=+∞` sentinel 不属于数值异常，且不得直接写入标准 JSON。
输出校验在搜索后再次执行，因此后验非有限也使用同一原因。若搜索结束时既存在合法非进展轨迹
又 OPEN 耗尽，使用更具体的 `no_progress_trajectory`；完全没有合法轨迹才使用
`search_exhausted`。共享预算耗尽但至少存在一个有限 root cost 时机器人级结果为 `ok`，未认证
root 的 `root_status` 为 `budget_exceeded` 且概率为 0。

任何失败都统一产生：

```text
valid_mask[i] = false
motion_preferences[i, :] = 0
```

禁止回退为 Fixed-KD、uniform distribution、NOOP teacher、coordinator action、旧 reservation
preference 或 previous cached label。全队 `sum(valid_mask)=0` 时，本次 A* 蒸馏分子和分母都不
更新，定义 `L_A=0`，PPO/LLM 分支按自身合同继续。

### 9.8 Cache 合同

cache 只能做 exact-query memoization。key 固定为 canonical JSON 的 SHA-256；JSON 使用 UTF-8、
排序 key、无多余空白、整数坐标和显式枚举字符串，并至少包含：

- `layout_hash`；
- `own_pose=[x,y]` 与 `orientation`；
- `resolved_goal=[x,y]`；
- 匿名、去重、按 `(x,y)` 排序的 `occupied_coordinates`；
- 5 位 `pure_motion_mask`；
- `K_motion=12`；
- `expansion_budget=512`；
- `cost_normalization=minmax-v1`；
- `tau_motion=1.0`；
- `teacher_version=pure-motion-astar-v1`；
- 仅在影响 footprint/legality 时存在的 `footprint_class`。

`layout_hash` 同样是 canonical JSON SHA-256，内容为 layout 宽高、固定不可通行坐标及
footprint-class traversability version。cache key 禁止包含机器人 ID、priority、reward、Student、
LLM、calibration、yield、coordinator、reservation 或 wall-clock。

exact key hit 可以返回此前保存的完整确定性结果，包括无效结果；它不是 fallback。key miss、版本
不符、schema 不符或缓存值未通过 shape/finite/sum 校验时必须重新规划，不能返回相似状态或上一
步 label。cache 不跨 teacher version 复用；cache hit/miss 只进入 diagnostics。

### 9.9 Diagnostics-only 与因果隔离证明

planning time、expanded nodes、root path length、failure reason、valid coverage、cache 命中、
Student disagreement、paired return、`Delta G` 和 Reward Calibration 全部只记录。其中前五项可
描述搜索和 coverage，后四项可描述 Student/教师效果；任何一项都不得改变 A* query、OPEN、
heuristic、root cost、preference 或 validity。不得从 search entropy、path-length confidence、
Student disagreement 或任何其他诊断量构造额外 loss 权重。

Reward Calibration 的时序边界固定为：Pure Motion Teacher 先独立生成并冻结本 query 的
`motion_preferences/valid_mask`，随后 calibration 才能评价该标签。Motion cost 只决定标签内部
动作偏好；Reward Calibration 只决定该标签整体是否值得蒸馏：

$$
w_{A,i}(t)=\lambda_A(t)m_{A,i}^{valid}c_A^{reward}.
$$

Calibration 不能重试搜索、修改 root cost/preference、把 invalid 变 valid 或把 reward 写入 cache
key。Student disagreement 同理只在 label 生成后计算和记录，且不进入权重。

因此候选 C 的因果链只有：

```text
own geometry + resolved goal + static layout
+ anonymous current occupancy + pure physical mask
-> deterministic bounded A*
-> per-agent motion preference + binary validity
```

高层任务、协同、回报和学习状态均不存在通向 label generator 的有效边。

### 9.10 固定边界示例与验收不变量

以下示例使用项目 `[NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD]` 顺序：

| 情形 | 有限 root raw cost | `valid_mask[i]` | `motion_preferences[i]` | 原因 |
|---|---|---:|---|---|
| 三个 root 均有限 | F/L/R=`12/13/14` | true | `[0, 0.506480, 0.307196, 0.186324, 0]` | min-max=`0/0.5/1`，`tau_motion=1.0` |
| RIGHT 无 continuation | F/L=`12/14` | true | `[0, 0.731059, 0.268941, 0, 0]` | RIGHT 为 `+∞`，质量为 0 |
| 两个 root 等 cost | F/L=`12/12` | true | `[0, 0.5, 0.5, 0, 0]` | 等 cost 退化时在有限 root 间均匀 |
| 只有 RIGHT 有限 | R=`14` | true | `[0, 0, 0, 1, 0]` | 唯一有限 root 自然 one-hot |
| mandatory TOGGLE | 不适用 | false | `[0, 0, 0, 0, 0]` | `mandatory_toggle_load` |
| 512 expansions 后没有有限 root | 无 | false | `[0, 0, 0, 0, 0]` | `budget_exceeded` |
| 全队均无效 | 无 | 全 false | 全零矩阵 | 本次 `L_A=0`，其他训练分支不变 |

O1 必须以确定性测试证明：重复 query bitwise 相同；置换其他机器人 ID 不改变结果；改变 priority、
reward、Student、LLM 或 calibration 不改变结果；改变匿名占用、静态地图、目标或纯物理 mask
可以改变结果；任意有效行和为 1，任意无效行和为 0，且 NOOP/TOGGLE/非法动作质量恒为 0。

### 9.11 O0-B 验证证据

验证日期为 2026-08-25，运行环境为项目 `py310`（Python 3.10.19）。

- O0-B 五项 plan checklist 全部完成，O0-C checked item 为 0；
- 候选结论、输入白名单/黑名单、`K_motion/H_reward` 隔离、512-expansion 定义、唯一 tie-break、
  direction rank、progress certificate、root-action cost、min-max normalization、固定 Boltzmann
  transform、输出 schema、失败优先级、cache key 和 diagnostics-only 边界均有无占位精确定义；
- 固定边界表验证不同 cost、等 cost 与单一有限 root 的概率分别归一化为 1，NOOP/TOGGLE 为 0；
- `python -m pytest`：退出码 0，184 tests passed；
- `python -m flake8 rware llm_mappo eval train scripts figures/core`：退出码 0；
- `python visualize.py --help` 与
  `python eval/evaluate_dynamic_ingress_astar.py --help`：退出码均为 0；
- `git diff --check`：退出码 0；相对 P0 `6fcb7d3` 的 Python/YAML/JSON 差异为 0；
- 两份研究输入继续保持未跟踪，未提前执行 O0-C 或 O0-F。
