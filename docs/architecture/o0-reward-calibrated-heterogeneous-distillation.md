# O0 Reward-Calibrated Heterogeneous Distillation Architecture

## 1. 文档状态与审计基线

本文是优化路线唯一 canonical architecture。O0-A 至 O0-F 已逐组获研究所有者批准；第 14 节
记录 O0-G 零偏差复核和 O1 实施交接。当前停在 O0 最终研究所有者书面批准门；批准前不得把
O0 标记完成、进入 O1、生成 pilot/formal 标签或修改运行时代码。

- 代码审计基线：P0 最终 commit `6fcb7d3`；
- 规格分支：`codex/o0-astar-teacher-redesign`；
- 审计范围：规则目标、充电目标、A* waypoint/preference/reservation/coordinator、hard mask、
  buffer、KL、离线 LLM、Actor/Critic 梯度、checkpoint、训练、评估和执行入口；
- O0 不修改 Python/runtime training config、环境语义、reward、checkpoint 或实验 seed；O0-F
  仅按 owner 明确授权修订治理 manifest；
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
| P-09 | movement reward 按规则目标距离变化提供 waypoint progress shaping | reward 与未来 DirectGoal/NoGoalHint 合同存在耦合，需要后续明确兼容边界 |
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
| 执行期无需 A* | 延后且禁止现时声称 | 当前 P-08/P-09 明确不成立；只有后续 DirectGoal 零 planner-query 门通过后才允许 |
| 方法已提高样本效率、协作或泛化 | 拒绝作为当前事实 | 输入是方案而非实验结果；必须由冻结的后续实验支持 |

上述矩阵与第 13.5 节已覆盖两份输入中的有效方法内容。O0-F 已按记录哈希完成映射并移除两份
未跟踪输入；canonical architecture 是唯一正式方案，研究输入不再构成并列合同。

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

具体权重由第 10.3 节统一 sampler 合同定义：Fixed 与 RC 都先乘
`m_calib(t)`，RC 再唯一乘 `c_A_reward(t)`；未选择状态两组权重均为 0。

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

## 10. O0-C 状态分叉、Paired Shadow 与 Reward Calibration 冻结合同

### 10.1 当前能力审计与唯一分叉方案

当前 `phase3_training.py::_handle_environment_command("snapshot")` 只返回 action mask、历史
`AStarExpert` preference 与 engagement target，不是环境状态快照。`Phase2Warehouse` 还持有 raw
observation、`EpisodeMetrics`、deadlock/progress 计数和充电集合；`DynamicWarehouse` 持有任务队列、
动态入库、能量和累计事件；底层 `Warehouse` 持有实体、grid、request queue、计步和
`np_random`。因此现状既不能从同一状态派生两个等价分支，也不能证明 shadow 不污染真实 rollout。

O1 唯一允许方案是显式、版本化 canonical snapshot/restore。拒绝以下替代：

- 对整个 Gym/adapter 对象直接 `deepcopy`：renderer、wrapper、space RNG、cache 与对象引用的复制
  语义不稳定，无法形成跨版本 schema；
- 从 episode 起点 replay 到分叉点：运行开销高，且任何 RNG 消费或实现变更都会破坏等价性；
- 共享一个环境反复 restore：异常中断可能把 shadow 状态留在真实 rollout 对象中。

两个 shadow 必须是按同一冻结配置预构造的 branch-local adapter/environment；snapshot import
不得调用 `reset`、实体构造器或会推进 `Agent.counter/Shelf.counter` 的路径。训练真实环境、Student
shadow 和 A* shadow 是三个不同对象。

### 10.2 `o0-shadow-state-v1` snapshot schema

snapshot 顶层固定包含 `schema_version=o0-shadow-state-v1`、代码 commit、environment config hash、
layout hash、run seed、episode index、episode seed、environment index、real global step、episode
step 和下列 mutable state：

| 区域 | 必须保存的字段 |
|---|---|
| Wrapper/clock | wrapper chain class/version 与全部 step-relevant mutable flag：当前 `OrderEnforcing._has_reset`，以及存在时的 `_elapsed_steps/_has_reset`；warehouse `_cur_steps/_cur_inactive_steps` |
| Agent | `id/x/y/prev_x/prev_y/direction/message/req_action/carrying_shelf_id/canceled_action/has_delivered/battery/dead/picking_lock_steps/task_id/collision_count/blocked_forward_count`；`Agent.counter` 作为真实全局 guard |
| Shelf | `id/x/y/prev_x/prev_y`；agent 与 shelf 关系只以 ID 恢复，禁止复制跨分支对象引用；`Shelf.counter` 作为真实全局 guard |
| Base queue/grid | `request_queue` shelf ID 顺序；grid 从实体重建，并与 snapshot grid hash 比较 |
| TaskQueue | 每个 Task 全字段、`_next_task_index`、按 key 排序的 `_next_label_number` |
| Dynamic ingress | `_batch_index`、`_shelf_home`、当前 active/completed task 状态及 arrival/completed step |
| Energy/rules | charging reservations、picking locks、task assignments，以及所有影响 hard mask/target 的 mutable rule state |
| Environment metrics | `total_collisions`、`total_blocked_forwards`、按原顺序保存的 `last_events` |
| Adapter | `_raw_observations` 深拷贝、`EpisodeMetrics` 全字段、`_last_progress_step/_last_completed/_last_picked`、排序后的 `_low_battery_active/_charged_pending_recovery/_charging_active` |
| RNG | environment `_np_random_seed` 与 `np_random.bit_generator.state`；action/observation space 及递归子 space 的 `_np_random` absent sentinel 或完整 state/path；Python、NumPy global、Torch CPU 与全部 CUDA RNG 的完整 state 及 guard digest |

静态 layout、goals、charging/picking stations、reward/energy/dynamic-ingress 参数、动作/观测 schema、
wrapper 配置和 task completion target 不作为可变 payload 重复写入，但必须进入 environment config
hash；import 时任一 hash/version 不一致立即拒绝。`grid` 由实体重建并比对 hash；`global_image` 是
derived cache，branch import 后统一清空并按需重建，不得成为行为状态。

canonical serialization 固定为 UTF-8 JSON、key 排序、无多余空白、显式 dtype/shape、有限浮点；
NumPy/Torch array 以 dtype、shape、C-order bytes 的 base64 和 SHA-256 同时保存并由 branch-owned
copy 恢复。snapshot hash 是完整 canonical payload 的 SHA-256。`None` 使用 JSON null，禁止
NaN/Inf。global RNG state 只用于零污染 guard，不得导入 shadow；shadow 外部事件只用 CRN。

restore 必须满足：

1. 同一 snapshot 导入两个 branch 后，其 canonical state hash 与 snapshot hash 完全相等；
2. 两分支执行相同 action 与相同 CRN 时，逐步 observation/reward/terminal/info/mask/state hash 相同；
3. shadow 前后真实环境 state hash、environment/space RNG state、global RNG guard、rollout buffer
   长度与内容完全不变；
4. training renderer 必须为 null；非 null 时 calibration fail-closed 并阻塞 O1；
5. 真实 waypoint planner/Pure Teacher cache 不复制、不写入；每个 shadow 使用独立空临时 cache，
   结束后整体丢弃。cache 命中只影响耗时，不能影响动作或 return。

### 10.3 统一 deterministic calibration selection mask

Fixed-KD 与 Reward-Calibrated KD 共用唯一 mask `m_calib(t) in {0,1}`。sampler version 固定为
`calibration-sampler-v1`，canonical key 固定为：

```text
[sampler_version, run_seed, episode_index, episode_seed,
 environment_index, real_global_step, episode_step]
```

取该 JSON 的 SHA-256 digest 前 8 字节，按 big-endian unsigned uint64 解释；仅
`value mod 16 == 0` 时 `m_calib(t)=1`，否则为 0。该选择器无 RNG 状态，不读取 reward、A* cost、
validity 数量、Student action/disagreement、LLM、`Delta G`、EMA 或训练性能。预期密度为 1/16，
不承诺每个短窗口恰好 1/16。

`m_calib=0` 时不创建 snapshot、不运行 shadow，Fixed-KD 与 RC-KD 的 A* KD 权重都严格为 0。
`m_calib=1` 但 `sum(m_A_valid)=0` 时记录 `selected_no_valid`，不运行 shadow，两个权重仍为 0，
也不产生 EMA 样本。Fixed/RC 必须使用相同 key、sampler、selected states、shadow engine、日志、
EMA 更新与 sampling density。

权重合同唯一为：

$$
w_{A,Fixed,i}(t)=\lambda_A(t)m_{A,i}^{valid}m_{calib}(t),
$$

$$
w_{A,RC,i}(t)=\lambda_A(t)m_{A,i}^{valid}m_{calib}(t)c_A^{reward}(t).
$$

未被 sampler 选择的状态，两组所有机器人 A* KD 权重均为 0。唯一优化差异是
`c_A_reward(t)`；Fixed 组也必须计算并记录 `Delta G/c_A_reward/EMA`，但权重公式不得读取它。

### 10.4 H=12 paired shadow 逐步时序

`H_reward=12` 是唯一正式 horizon，与 `K_motion=12` 分属不同字段。对每个 selected 且至少一台
机器人 A* valid 的真实状态，先冻结 policy/Critic 参数引用与 snapshot，再导入两个 shadow：

1. 每个 shadow 根据自身当前状态构建 observation 与同一 production hard-mask 函数；初始 mask
   必须 bitwise 相同，分歧后禁止跨分支复用 mask；
2. Student shadow 在 `eval + inference_mode` 下对 masked logits 作 deterministic argmax；相同最大
   logit 按项目 action index 较小者胜出；
3. A* shadow 每步按其当前状态滚动查询 Pure Motion Teacher；`valid_mask[i]=1` 时执行
   `motion_preferences[i]` argmax，概率并列按 O0-B root rank；无效机器人在 **A* shadow 当前状态**
   重新计算 Student masked argmax，禁止复制 Student shadow action；
4. Teacher argmax 必须通过该 A* branch 当前 production hard mask；不通过说明 purity/mask 合同
   破裂，记 `teacher_mask_mismatch` 并使本 calibration 样本失败，禁止改成 NOOP 或静默 fallback；
5. 两个 joint action 分别 step，各自记录 scalar team reward、terminal、mask、Teacher coverage、
   Student fallback 和 state hash；
6. 某分支出现 `terminated`、`truncated` 或 adapter `deadlocked` 即在该步后独立停止；另一分支继续
   到自身 terminal 或 12 步；停止分支不再生成动作、事件或 reward；
7. 两个 shadow 和临时 cache 在结果提取后销毁，随后验证真实 state/RNG/buffer guard。

Student 与 Critic 使用配对开始时相同的参数 snapshot；shadow 内不得 optimizer step、dropout、
BatchNorm 更新、gradient accumulation 或训练 buffer 写入。A* label 在每一步先独立生成，Reward
Calibration 永远不能修改 query、cost、preference 或 validity。

### 10.5 `crn-v1` 外部随机事件

shadow step 中所有外生随机事件必须通过无状态 `crn-v1` 寻址：

```text
[crn_version, episode_seed, real_global_step,
 shadow_offset, event_type, event_slot]
```

两个分支对相同 key 获得相同 digest；不得从 mutable generator 顺序消费。`event_type` 使用冻结
字符串枚举，`event_slot` 是该 step 内同类事件从 0 开始的稳定索引。动态入库数量由 key 映射至
闭区间；无放回候选选择把 canonical candidate ID 与 event key 联合 SHA-256 后排序，取前 N 个。
即使分支候选集合因动作而不同，也使用同一 key/ranking rule，而不是强迫选择同一实体。当前
`DynamicWarehouse._spawn_scheduled_batch` 的 `np_random.integers/choice` 必须在 O1 shadow 路径
改接此接口；真实 rollout 继续使用其真实 RNG，且不能被 shadow 推进。

### 10.6 Return、terminal 与 handoff-to-Student

设分支 `b in {A,pi}` 实际执行长度为 `L_b<=H_reward`，`z_b=1` 表示在 12 步内未发生
terminated/truncated/deadlock。其团队 return 固定为：

$$
G_b=\sum_{k=0}^{L_b-1}\gamma^k r_k^b
+z_b\gamma^{H_{reward}}\operatorname{stopgrad}
\left[V_\phi(S^b_{t+H_{reward}})\right].
$$

`r_k` 是 production adapter 产生的 scalar `team_reward`，不重新组合环境 per-agent reward；
`gamma` 直接引用同一 PPO hyperparameter/checkpoint 字段，禁止 calibration 独立 gamma。到达完整
12 步且未终止才 bootstrap；任何 terminal 分支 `z_b=0`，不 bootstrap，另一分支不受影响。

Critic 在相同冻结 policy snapshot、`eval + inference_mode` 下读取 branch 自身 12 步末 centralized
state。其语义是：经过 H 步 Student 或 A* 干预后，立刻把控制交回当前 Student policy 的
continuation value；不是 A* 长期 value，也不训练新的 calibration critic。Critic output 必须
detach，calibration 不向 Critic、Actor、A* 或 reward 反传。配对优势固定为：

$$
\Delta G=G_A-G_\pi.
$$

### 10.7 `c_A_reward` 与 EMA 状态机

EMA scope 是每个训练 run 一份团队级状态，不按 agent、worker 或 Teacher validity 分裂。只有
`m_calib=1`、至少一台机器人有效、paired rollout 成功且 `Delta G` 有限的样本进入状态机。向量
环境同一 real step 的结果按 `environment_index` 升序处理；跨步按 `real_global_step`，禁止按
worker 完成/IPC 返回时间更新。

冻结常数：

```text
ema_schema = reward-calibration-ema-v1
initialization_sample_count = 64
ema_decay = 0.99
minimum_scale = 1e-3
weight_clip = [0.0, 1.0]
```

初始化前 64 个有限 `Delta G` 使用确定性 Welford `count/mean/M2`；population variance 为
`M2/count`。第 1 至第 64 个样本的 `c_A_reward=0`，第 64 个样本完成后把
`ema_mean=mean`、`ema_variance=max(M2/64,0)`、`initialized=true`。不允许 RC 分支用 Fixed 权重
预热；Fixed 对照始终只按第 10.3 节自己的公式。

从第 65 个有限样本开始，必须先使用更新前状态：

$$
\sigma_{pre}=\max\left(\sqrt{\max(v_{pre},0)},10^{-3}\right),
$$

$$
c_A^{reward}=\operatorname{clip}
\left(\frac{\max(\Delta G,0)}{\sigma_{pre}},0,1\right).
$$

随后更新：

$$
\delta=\Delta G-\mu_{pre},\qquad
\mu_{post}=0.99\mu_{pre}+0.01\Delta G,
$$

$$
v_{post}=0.99\left(v_{pre}+0.01\delta^2\right).
$$

因此 `Delta G<=0` 严格得到 0，正优势才可能蒸馏。`Delta G`、return、bootstrap、EMA 中间量或
最终权重出现 NaN/Inf 时，该样本 `c_A_reward=0`、不增加 count、不更新 EMA，并记录唯一失败
原因；O1 视任何此类事件为 No-Go，禁止把它当作普通无优势样本继续验收。

### 10.8 Checkpoint 与恢复

每个 optimization-route checkpoint 必须原子保存：`ema_schema/count/mean/M2/variance/
initialized`、sampler/crn version、sampler divisor 16、`H_reward=12`、`gamma` 绑定字段、decay、
minimum scale、clip、last processed `(real_global_step,environment_index)` 及 calibration counters。
Fixed 与 RC 都保存相同字段和实际 EMA 状态。

resume 时字段缺失、非有限、count/initialized 矛盾、常数或 version 不符、训练 step 倒退均严格
拒绝加载。禁止重置 EMA、重放已处理样本、从第 64 个样本重新初始化或把历史一/二维 checkpoint
迁移到该 schema。checkpoint 在某 canonical sample 完整完成并更新 EMA 后写入；恢复从下一个
sample key 开始，保证恰好一次更新。

### 10.9 日志与计数守恒

每个 real state 至少记录：sampler key/hash、`m_calib`、`any_astar_valid`、shadow attempted/success、
两分支实际长度/terminal reason/discounted reward/bootstrap/return、`Delta G`、`c_A_reward`、EMA
pre/post、逐机器人 valid mask、Fixed/RC eligibility weight、fallback 数、snapshot/real/RNG guard
hash、耗时、CPU RSS、CUDA allocated/reserved 和失败原因。Student disagreement 仍仅为诊断。

每个统计窗口必须满足：

```text
real_states = sampler_selected + sampler_not_selected
sampler_selected = selected_no_valid + shadow_attempted
shadow_attempted = shadow_success + shadow_failure
finite_delta_g = ema_updates + initialization_updates
rc_positive_agent_weights <= fixed_eligible_agent_weights
```

`finite_delta_g` 只统计成功且有限样本；初始化 update 与 EMA update 互斥。Fixed/RC 必须使用同一
sampler 算法、1/16 密度和条件分支；由于两组 policy 学习后可访问不同状态，不要求 shadow
success、validity 或 `Delta G` 数值相等，但所有计数都必须各自满足上式。切换 Fixed/RC 公式本身
不得改变同一 real state 的 Teacher label、reward、real action、real state hash、LLM retrieval 或
PPO schedule；唯一允许直接改变优化 loss 的字段是乘子 `c_A_reward`。

以下属于 calibration engine failure：snapshot/config/hash 不符、branch import/step exception、CRN
key/schema 错误、`teacher_mask_mismatch`、Critic/return/EMA 非有限、真实 state/RNG/buffer guard 改变，
以及 branch/cache 活对象未释放。它们与普通 A* per-agent invalid、单边 terminal 或非正优势不同。
engine failure 必须在任何 optimizer step 前中止本次更新、写出原始诊断并使 O1 No-Go；不得让
Fixed 继续 KD、把 RC 置零后继续训练或重试到成功。

### 10.10 H=12 runtime 与 memory No-Go gate

O1 在研究所有者的 A600、冻结 12-worker smoke config 上运行三个 fresh-process 条件：baseline
（selection/log schema 保留但 shadow engine disabled）、H=4 diagnostic、H=12 formal。每个条件：

- 16 个 vector step warm-up，不计时；
- 随后 128 个 vector step measured window，包含正常 real collection、相同 PPO update、snapshot/
  restore、Pure Teacher replan、两个 shadow、Critic bootstrap、EMA 和内存日志；
- 排除进程创建、warm-up 和最终磁盘 flush；计时前后 `torch.cuda.synchronize()`；
- 以同一 seed trace 运行 5 个独立 fresh process，报告全部原始秒数和中位数。

正式 runtime gate 为：

$$
\frac{\operatorname{median}(T_{H12,1:5})}
{\operatorname{median}(T_{baseline,1:5})}\le 3.0.
$$

memory gate 对 H=12 使用相同 128-vector-step window：先运行 2 个不计入判定的 warm-up window，
再连续记录 10 个 window。每个边界执行 Python GC 与 CUDA synchronize，但不得调用
`empty_cache`；记录 post-window CPU RSS、CUDA allocated 和 reserved。CPU RSS 与 CUDA allocated
分别计算：

```text
growth = median(last 3 windows) - median(first 3 windows)
threshold = max(64 MiB, 0.05 * median(first 3 windows))
persistent = growth > threshold and Spearman rho(window_index, memory) >= 0.80
```

任一内存序列 persistent，或 H=12 runtime ratio 超过 3.0，或出现持续 branch/cache 对象计数增长，
O1 必须 No-Go 并返回 O0。不得把 H=4 提升为正式 horizon，不得静默改变 1/16 sampler、workers、
measured window 或排除耗时组件。H=4 只用于定位成本来自 snapshot、Teacher、environment step 还是
Critic。

O1 必须实现并由研究所有者运行以下唯一入口；参数默认值也必须与显式值一致：

```powershell
D:\Anaconda3\envs\py310\python.exe scripts/benchmark_reward_calibration.py `
  --config configs/optimization/o1_reward_calibration_smoke.yaml `
  --modes baseline h4 h12 --workers 12 --repeats 5 `
  --warmup-vector-steps 16 --measure-vector-steps 128 `
  --memory-warmup-windows 2 --memory-measure-windows 10 `
  --output artifacts/optimization/o1_reward_calibration_gate
```

入口必须 fail-closed 校验 CUDA A600、commit/config hash、workers、sampler/crn/EMA version 和所有
数字参数，并原子写出每次 raw timing/memory/counter JSONL、环境 manifest 与唯一 `summary.json`。
Codex/工程 AI 只能准备入口与分析结果，不代替研究所有者运行该 A600 基准。

### 10.11 固定边界示例

设 `lambda_A(t)=0.2`、三台机器人 `m_A_valid=[1,0,1]`：

| 条件 | Fixed 权重 | RC 权重 |
|---|---|---|
| `m_calib=0` | `[0,0,0]` | `[0,0,0]` |
| `m_calib=1, c_A_reward=0.4` | `[0.2,0,0.2]` | `[0.08,0,0.08]` |
| `m_calib=1, c_A_reward=0` | `[0.2,0,0.2]` | `[0,0,0]` |

RC 初始化的第 1–64 个有限样本均为 0；第 65 个样本若 prior `sigma_EMA=0.5`，则
`Delta G=-0.1/0.1/1.0` 分别得到 `c_A_reward=0/0.2/1.0`。Fixed 权重不读取这些数值，但运行同一
shadow 和 EMA 更新。

若 A* shadow 在执行两步后 terminal，reward 为 `1,2`，其 return 为 `1+gamma*2`，不含 Critic；
Student shadow 仍可独立运行至 H=12 并在未终止时 bootstrap。若两分支在初始状态采用完全相同
joint action，则 CRN 与 restore 等价性要求其逐步 transition hash 相同。

### 10.12 O0-C 允许主张与 O1 验收边界

O0-C 完成后只允许主张：paired calibration 的状态、随机性、时序、数学、对照和失败合同已经
预注册。不得声称 snapshot 已实现、H=12 已通过 3x gate、EMA 改善训练、A* label 有正优势或
RC-KD 优于 Fixed-KD。

O1 必须以测试证明 snapshot round-trip、相同行为等价、真实状态/RNG/buffer 零污染、单边
terminal、全 invalid、mask mismatch、非有限、初始化 64/65 边界、checkpoint resume 恰好一次、
sampler determinism/density、Fixed/RC 同 mask 与计数守恒。A600 runtime/memory 命令只由研究
所有者运行；Codex 准备命令并分析结果。

### 10.13 O0-C 验证证据

验证日期为 2026-08-25，运行环境为项目 `py310`（Python 3.10.19）。

- O0-C 六项 plan checklist 全部完成；O0-D checked item 为 0；
- 只读代码审计确认现有 worker `snapshot` 不包含状态，mutable state 分布在
  `Warehouse/DynamicWarehouse/Phase2Warehouse/OrderEnforcing/TaskQueue`，并确认 step-time 随机
  动态入库仍使用 environment `np_random`，因此 O1 必须实现本章 snapshot 与 CRN 边界；
- 合同断言确认 `o0-shadow-state-v1/calibration-sampler-v1/crn-v1`、Fixed/RC 共同 `m_calib`、
  `0.99/1e-3/64` EMA、H=12、严格恢复和 A600 gate 均有唯一无占位定义；
- 对 65,536 个 canonical sampler key 的确定性公式检查选中 4,073 个，观测密度
  `0.062149`，与期望 `1/16=0.0625` 一致；这只验证 hash sampler，不是训练证据；
- 固定权重示例、正/负优势截断与第 64/65 个 EMA 样本边界检查通过；
- `python -m pytest`：退出码 0，184 tests passed；
- `python -m flake8 rware llm_mappo eval train scripts figures/core`：退出码 0；
- `python visualize.py --help` 与
  `python eval/evaluate_dynamic_ingress_astar.py --help`：退出码均为 0；
- `git diff --check`：退出码 0；相对 P0 `6fcb7d3` 的 Python/YAML/JSON 差异为 0；
- 未实现/运行 snapshot、paired shadow、EMA、训练或 A600 runtime/memory gate，未提前执行
  O0-D；两份根目录研究输入继续保持未跟踪。

## 11. O0-D：三维离线语义、数据与 OOD reliability

### 11.1 职责与因果边界

离线 LLM 只回答三个高层语义问题：当前任务是否值得继续坚持、当前局部交互是否倾向让行、
当前局部交互风险有多高。它不输出或暗示具体离散动作、路径、任务分配、目标改写、通行权裁决、
充电站控制或规则层 override。训练和执行期均不调用在线 LLM。

三个正式 score 按以下顺序形成
`z_L=[task_persistence,yielding_preference,coordination_risk] in [0,1]^3`：

1. `task_persistence`：继续当前已分配运输任务的合理程度；不等于立即移动，也不等于任务优先级；
2. `yielding_preference`：在当前局部交互中主动延迟或让行的语义倾向；高值表示更倾向让行，但
   不是 `NOOP` 或任何动作命令；
3. `coordination_risk`：当前局部交互造成冲突、拥堵、死锁或协作失败的风险程度；不是行为建议。

三维相互独立解释，不存在 `yielding=1-persistence`、高 risk 必然高 yielding、高 persistence 必然
低 yielding 或其他代数/逻辑蕴含。高 persistence 与高 yielding 可以同时成立，例如任务仍值得
完成但当前应先让行；高 risk 与低 yielding 也可以同时成立，例如双方均有强任务理由而冲突风险
很高。

### 11.2 固定 score anchors 与输出 schema

三个维度共享五点位置，但每个位置按本维含义解释；模型可在 anchors 之间连续插值：

| score | task persistence | yielding preference | coordination risk |
|---:|---|---|---|
| 0.00 | 没有继续当前任务的合理依据 | 没有主动让行的语义依据 | 没有可识别的局部协作风险 |
| 0.25 | 暂停/转移理由明显强于继续 | 只有较弱的让行理由 | 存在轻微但可忽略的交互风险 |
| 0.50 | 继续与暂停理由大致平衡 | 让行与不让行理由大致平衡 | 存在实质但非高危的交互风险 |
| 0.75 | 继续理由明显强于暂停 | 有强理由主动让行 | 存在高冲突、拥堵或协作失败风险 |
| 1.00 | 在给定语义事实下继续任务具有压倒性理由 | 在给定语义事实下让行具有压倒性理由 | 若无协调极可能立即或持续失败 |

LLM response 必须是单个 JSON object，正好六个 key：

```json
{
  "task_persistence": 0.0,
  "task_persistence_reason": "...",
  "yielding_preference": 0.0,
  "yielding_preference_reason": "...",
  "coordination_risk": 0.0,
  "coordination_risk_reason": "..."
}
```

score 必须是非 bool、有限 JSON number 且位于 `[0,1]`；reason 必须是去除首尾空白后长度
`1..1000` 的 string。缺失或额外 key、markdown fence、JSON 前后文字、重复 key、NaN/Inf、越界、
错误类型或空 reason 均使整条 record `validity=0`。parser 不裁剪分数、不提取部分字段、不做逐维
validity。三个 reason 仅随原始数据保存供人工审计，不进入 Student 输入、semantic target、loss、
OOD feature、reliability、schedule 或 checkpoint tensor。

### 11.3 `semantic-view-v3` 与唯一 61D 数值编码

同一 canonical semantic view 同时产生：供 LLM 阅读的 JSON view，以及供 kNN/OOD 使用的 61D
float64 vector。两者来自同一冻结环境快照和同一 encoder version；禁止从旧 615D actor/full
observation 直接计算正式 OOD 距离。`scenario_type`、scenario/AGV/任务 ID 只存在 dataset
provenance，不进入 view 或 61D vector。

LLM JSON view 的完整 schema 为：

```text
semantic_view_version = semantic-view-v3
layout_hash: string
focal:
  battery_ratio: float [0,1]
  loaded: bool
  priority_present: bool
  priority_rank: float [0,1]
  target_kind: task | delivery | charging | idle
  orientation: up | down | left | right
  on_highway: bool
  at_charging_station: bool
  at_picking_station: bool
  adjacent_highway: {forward: bool, right: bool, backward: bool, left: bool}
neighbors: exactly 3 records
  mask: bool
  relative_forward: float [-1,1]
  relative_right: float [-1,1]
  normalized_manhattan_distance: float [0,1]
  loaded: bool
  battery_ratio: float [0,1]
  dead: bool
  priority_present: bool
  priority_rank: float [0,1]
  target_kind: task | delivery | charging | idle
  at_charging_station: bool
```

`priority_rank=(ord(label[0])-ord('A'))/25`；无任务时 `priority_present=false` 且 rank=0。target kind
one-hot 顺序固定为 `[task,delivery,charging,idle]`，orientation one-hot 顺序固定为
`[up,down,left,right]`。所有 bool 编码为 0/1。

邻居候选是除 focal 外的所有机器人。world delta 为 `dx=x_peer-x_focal`、
`dy=y_peer-y_focal`，选择距离为未归一化 Manhattan `d_M=abs(dx)+abs(dy)`。以 focal 朝向建立右手
语义坐标：

```text
up:    forward=-dy, right= dx
down:  forward= dy, right=-dx
left:  forward=-dx, right=-dy
right: forward= dx, right= dy
```

`relative_forward/right` 除以 `M=max(width-1,height-1)`；Manhattan 距离除以
`(width-1)+(height-1)`。候选按以下 tuple 升序排列并取前三个：

```text
(d_M, forward, right, -loaded, dead, -priority_present,
 priority_rank, target_kind_rank, battery_ratio, -at_charging_station)
```

禁止以 ID tie-break。若 tuple 完全相同，则两个候选在所有保留字段上相同，交换顺序不会改变
61D vector；实现必须通过 permutation test 证明这一点。机器人不足三个时在末尾补全；padding
record 的 14 个数全部为 0，其中 `mask=0`。真实邻居 `mask=1`，即使其他字段恰为 0 也不能与
padding 混淆。匿名化只删除身份，不删除 loaded、battery、dead、priority、target kind 或 station
等判断 yielding/risk 必需的语义。

61D vector 顺序固定为：

```text
focal[19] =
  battery, loaded, priority_present, priority_rank,
  target_kind_onehot[4], orientation_onehot[4],
  on_highway, at_charging_station, at_picking_station,
  adjacent_highway[forward,right,backward,left]

neighbor_0[14], neighbor_1[14], neighbor_2[14] =
  mask, relative_forward, relative_right, normalized_manhattan_distance,
  loaded, battery, dead, priority_present, priority_rank,
  target_kind_onehot[4], at_charging_station
```

任何 schema/version/layout hash 不符、非法类别、非有限值、范围错误或 vector 维数不为 61 均使
该 query fail closed，`c_validity=0`、`c_OOD=0`，不得回退到 615D observation。

### 11.4 OOD 候选证据与正式公式

O0-D 只读取历史
`artifacts/phase4_labels/deepseek_medium_5ag_400_v2_repaired_r2.jsonl` 的场景事实和 observation
字段，没有读取旧 LLM score、训练 reward 或下游性能。历史记录包含 focal 朝向、位置、匿名化所
需邻居相对位置/状态和固定 layout，因此可以重建上述 61D view；重建矩阵为 `400x61`、185 个唯一
向量，按 record 顺序 little-endian float64 的 SHA-256 为
`371ff59c5f9b89fc87a9dbd9a45bd22ca9b023e0cbfbd2dbe369ff0e7fb05caa`。此前 400x615
审计只保留为 preliminary evidence，不能支持正式公式。

五折由 `uint32(SHA256(scenario_id)[0:8]) mod 5` 固定。每折只用 reference fold 计算 population
mean/std，`scale_j=max(std_j,1e-3)`，并将每个向量标准化。两个向量的距离是 61 维 normalized
Euclidean：`sqrt(mean((z_q-z_r)^2))`。query distance `d` 是三个最近 reference vector 距离的
算术平均；reference LOO distance 同样排除自身后取三个最近邻均值。分位数使用 Hyndman-Fan
type 7，即 NumPy `method="linear"`。

两个预注册候选为：

$$
c_{exp}(d)=
\begin{cases}
\exp(-d/q_{95}), & d\le q_{99},\\
0, & d>q_{99},
\end{cases}
$$

$$
c_{linear}(d)=
\begin{cases}
1, & d\le q_{50},\\
(q_{95}-d)/(q_{95}-q_{50}), & q_{50}<d<q_{95},\\
0, & d\ge q_{95}.
\end{cases}
$$

61D 五折结果：

| 候选 | `c>0` coverage | `c>=0.1` coverage | mean weight | Spearman(distance,weight) | max float32/64 diff | boundary flip | finite |
|---|---:|---:|---:|---:|---:|---:|---|
| truncated exponential | 0.9800 | 0.9800 | 0.820221 | -0.999952 | `4.19e-8` | 0 | yes |
| LOO piecewise linear | 0.9325 | 0.9225 | 0.802567 | -0.985732 | `1.12e-7` | 0 | yes |

截断指数同时具有更高正权重覆盖、更严格单调性和不差的数值稳定性，因此是唯一正式候选。
不得依据训练结果重新选择。正式 800 数据冻结后，reference set 只含整记录 validity=1 的 records，
并在该 reference set 上重新计算和持久化 `mu/scale/q50/q95/q99`；这一步实例化公式参数，不重新
选择公式。

正式 retrieval 固定 k=3。三个有效邻居标签以
`alpha_j proportional to (d_j+1e-6)^-2` 归一化形成三维 target；如果存在一个或多个 `d_j=0`，
只平均所有 exact-match labels，忽略非 exact neighbors。`c_validity=1` 仅当至少有三个有效 formal
records、query 与三个标签/距离全部有限且 target 合法；否则为 0。正式共享权重为：

$$
c_L=c_{validity}c_{OOD}.
$$

若 reference 少于 3、`q95<=1e-6`、`q99<q95`、任何统计量非有限或 index/hash 不符，则整个 LLM
分支 fail closed 并使 O1 No-Go；禁止 uniform、旧 2D labels、规则标签或 615D fallback。

### 11.5 Prompt、parser 与模型身份

system prompt 固定为：

```text
You are a JSON-only warehouse semantic teacher. Evaluate only the supplied
semantic state. Never output actions, paths, assignments, right-of-way rulings,
station controls, or changes to task labels.
```

user prompt template 的 UTF-8 文本固定如下；实现只能替换最后一行的 `{semantic_state}`，不得改写
其他字符：

```text
Return one JSON object with exactly these six keys: task_persistence,
task_persistence_reason, yielding_preference, yielding_preference_reason,
coordination_risk, coordination_risk_reason. Each score must be a finite number
in [0,1]. Each reason must be a non-empty string of at most 1000 characters.

task_persistence is how reasonable it is to keep the currently assigned
transport task. It is not an immediate movement instruction and is not the task
priority itself. yielding_preference is the semantic tendency to voluntarily
delay or cede local passage; a higher value means a stronger tendency to yield,
but it is not a NOOP or action command. coordination_risk is the risk that the
current local interaction causes conflict, congestion, deadlock, or cooperative
failure; it is not a behavior command.

For each dimension use these anchors and interpolate continuously when needed.
The output is not restricted to the five anchor values. 0.00 means no
semantic basis for that property. 0.25 means weak persistence or yielding, or
minor coordination risk. 0.50 means balanced persistence/yielding reasons, or
material but non-high coordination risk. 0.75 means strong persistence or
yielding reasons, or high coordination risk. 1.00 means overwhelming persistence
or yielding reasons, or near-certain immediate or sustained coordination failure
without coordination.

Use the following factors as a semantic rubric, not as deterministic scoring
rules. For task_persistence, consider whether an active transport task exists,
its priority, carrying and battery state, progress, interruption cost, and local
difficulty. With no active task, persistence should be near zero. For
yielding_preference, consider whether there is an actual local interaction,
loaded versus empty status, relative task priority, delay cost, and whether
yielding can resolve a bottleneck. For coordination_risk, consider neighbor
density, narrow or station bottlenecks, dead or blocking robots, movement
constraints, and whether the area is open. These factors may conflict; weigh
them jointly and use any finite value in [0,1]. Do not convert scenario types or
fixed rules into target scores.

The three scores are not complements or aliases. High task persistence may
coexist with high yielding preference. High coordination risk does not imply a
particular yielding score. Use only facts in SEMANTIC_STATE. Do not invent facts,
IDs, actions, paths, assignments, priorities, right-of-way rulings, station
controls, or task-label changes. Do not emit markdown or text outside the JSON.

Expected JSON shape:
{"task_persistence":0.0,"task_persistence_reason":"...","yielding_preference":0.0,"yielding_preference_reason":"...","coordination_risk":0.0,"coordination_risk_reason":"..."}

SEMANTIC_STATE={semantic_state}
```

`semantic_state` 是 sorted-key、compact-separator 的 canonical `semantic-view-v3` JSON。

该模板版本固定为 `semantic-prompt-v4-directional-rubric`。五点 anchors 只解释量表语义，
`0.18/0.43/0.72/0.91` 等连续值均合法；禁止把 anchor 或 rubric 实现为场景到分数的查表规则。
prompt 中不得出现 `scenario_type`、任何 ID、场景目标分数、RuleKD 标签、controlled reference
direction 或“某类应高/低于某阈值”。prompt bytes、system/user text、semantic JSON 和 request body分别计算
SHA-256。parser 只接受第 11.2 节严格 JSON；禁止 fence stripping、文本中搜寻 JSON、clamp、字段
补全或 reason 推导 score。

首选 provider/model 是 DeepSeek official Chat Completions
`https://api.deepseek.com/chat/completions` / `deepseek-v4-flash`，配置固定：

```text
stream=false
thinking={type: disabled}
temperature=0.0
response_format={type: json_object}
max_tokens=1024
timeout_seconds=120
max_attempts=3
retry_backoff_seconds=[5,15]
```

每次 attempt 保存去除 Authorization 后的完整 request、HTTP status/headers、原始 response bytes、
response id/model/system_fingerprint/created/finish_reason/usage、错误与时间戳。只有 timeout、连接错误、
HTTP 429 和 5xx 可按同一 request bytes 重试；4xx、`finish_reason!=stop`、空 content 或 schema/content
invalid 不作针对性重试，直接形成 invalid record。成功 content 缺少或返回非字符串
`response.model/system_fingerprint` 同样 invalid，并使 formal fingerprint gate No-Go。API key 只能由 owner 通过 `DEEPSEEK_API_KEY`
注入，禁止写入仓库、artifact、日志、命令参数或 checkpoint。

### 11.6 60 条 pilot 与 Flash→Pro 唯一切换门

`semantic-pilot-v3` 固定五类各 12 条；base seeds 为 `[410,411,412]`，每 seed 每类 4 条。pilot
场景、prompt、raw response、review 和 manifest 全部放在 `artifacts/optimization/labels/pilot/`，
不得进入正式 800、kNN reference、Student、训练、checkpoint 或论文性能统计。

首轮必须使用 Flash，并由两名独立 reviewer 在不知道 score 来源的情况下复核全部 60 条。以下
任一条件定义为 Flash 在三维语义判断上的系统性失败：

- parser-valid 少于 57/60，或任一场景类型少于 11/12；
- substantive semantic error 超过 6/60，或任一类型超过 2/12；
- 出现任一 critical error：输出/建议动作、路径、分配、通行权或规则改写，捏造输入中不存在的
  任务/载货/电量事实，或把任一维固定解释为另一维的补数/同义项；
- 两名 reviewer 对同一维的 anchor 区间判断在超过 12/60 records 上相差两个或以上 anchor
  intervals，表明 prompt/scale 无法稳定解释。

若 Flash 通过，formal 必须继续用 Flash。若且仅若 Flash 触发上述门，保持同一 60 场景、
semantic-view、prompt/schema/parser、temperature 和所有请求参数不变，废弃整组 Flash pilot 并用
`deepseek-v4-pro` 重新生成完整 60 条；禁止只重标失败记录。Pro pilot 使用相同门；仍失败则
O0-D No-Go，必须升级 prompt/schema version，不能生成 formal。模型切换不参考 MAPPO reward、
训练性能或研究者偏好的 score。

### 11.7 确定性 60/800 场景生成合同

`semantic-scenario-v3` 使用 optimization medium topology、5 AGV、动态入库、max_steps=1000、
batch interval 40、batch size `[4,8]`、queue size 8、task target 50 与能源
`battery_cost_scale/charge_threshold/charge_release_threshold=1.10/0.30/0.80`。layout、环境参数和
生成器代码 commit 必须进入 manifest。生成器只构造并验证环境快照，不调用或读取 A*、LLM、
Student、reward、reservation、yield/coordinator 或 calibration，也不以旧协调 A* rollout 产生
normal records。

五个 stratum 固定为 `normal_transport/priority_conflict/narrow_corridor_yield/
low_battery_diversion/station_exit_congestion`；名称只进入 provenance。pilot 按第 11.6 节分配。
formal 每类 160 条，base seeds 为 `[500,501,502,503,504,505,506,507,508,509]`，每 seed 每类
16 条。派生环境 seed 固定为：

```text
derived_seed = base_seed * 100000 + stratum_rank * 1000 + within_seed_index
stratum_rank = 0..4 in the order above
within_seed_index = 0..3 for pilot, 0..15 for formal
```

每个 derived seed 重置环境后，确定性参数化 injector 先按坐标、任务 label、shelf ID 的 canonical
tuple 枚举 highway/station/task 候选，再按
`SHA256("semantic-scenario-v3"|derived_seed|canonical_candidate_tuple)` 升序检查，选择第一个满足
invariant 的组合；最多检查前 128 个 candidates，耗尽即整个数据集 No-Go，不跨 seed 借配额，
也不使用进程 RNG。五类 invariant 分别为：normal 的 focal 有 active task 且所有 peer Manhattan
距离大于 4；priority conflict 的两车相邻且 focal 位于 degree>=3 highway，within-seed index 偶数时
focal 优先级更高、奇数时更低；narrow corridor 的 empty focal 与 loaded peer 相邻且 focal 位于
degree=2 highway；low battery 的 loaded focal battery 固定 0.15；station exit 的一车占据充电站、
focal 占据该站 lexicographic 第一合法 highway 出口。其余机器人放置到使其与已放置机器人最小
Manhattan 距离最大的 highway cell，tie-break 为 `(y,x)` 升序。每条只记录一个 focal；环境
invariant、61D view、JSON view 和 scenario content hash 必须同时通过。scenario ID 是
`SHA256(generator_version|layout_hash|derived_seed|focal_snapshot_hash)`，pilot/formal ID 与 content
hash 必须完全不相交。

### 11.8 Formal dataset-level acceptance 与不可变性

`semantic-formal-v3` 是恰好 800 个预注册 scenario attempts，不是“收集到 800 个成功回答为止”。
失败 response 仍占原 scenario 配额并作为 `validity=0` record 保存。formal 生成前必须已批准
pilot、模型和全部 prompt/schema hashes；生成后禁止逐条改分、修改 reason、针对坏标签重试、
选择性删除、用新 scenario 补洞或根据训练表现重生成。

生成期间第一条成功 response 冻结 `response.model` 与 `system_fingerprint`。任一后续 response 的
二者变化必须立即原子保存 partial manifest 并暂停，由 owner 审核；禁止静默继续。一个获准
formal dataset 只能含一个 `(request_model,response.model,system_fingerprint)` tuple。owner 只能
等待原 fingerprint 恢复或废弃整套 partial 后从 800 条第一条重新启动，不能批准混合 backend。

自动完整性与 validity 门：

```text
records = 800 exactly
each stratum = 160 exactly
unique scenario_id/content_hash = 800
overall parser validity >= 784/800 = 98%
each-stratum parser validity >= 152/160 = 95%
one model/fingerprint tuple
all request/response/manifest hashes and count conservation valid
```

人工复核样本在不含 review 的 canonical 800-record content hash 冻结后，以
`SHA256("20260820"|records_content_hash|scenario_id)` 每个 stratum 升序选前 20 条，共 100 条；两名独立
reviewer 只看 semantic view、anchors、scores/reasons，不看训练结果。critical error 必须为 0；
substantive semantic error 必须不超过 5/100 且每类不超过 2/20。substantive error 是 score 落在
reviewer 依据冻结 anchors 判定区间相隔两个或以上 anchor intervals，或 reason 与输入事实/score
方向实质矛盾；轻微措辞差异不算错误。reviewer 分歧由 owner 在 dataset-level 判定前具名裁决，
裁决只能改 audit verdict，不能改 label。

任一门失败时整个 formal dataset No-Go：返回 O0-D，升级 prompt/schema/parser/generator 中实际
需要改变者的 version，重新运行完整 60 pilot 并生成新的完整 800；旧 dataset 原样归档并禁止
训练。禁止只修复失败 records。

所有 validity=1 formal labels 同时报告三个维度两两 Pearson 与 Spearman。任一
`abs(rho)>=0.80` 只触发 owner 人工复核并记录“接受为真实语义相关”或“启动全新 version”之一，
不自动改定义、删记录、改分或重生成当前 dataset；相关性本身不是上述 acceptance No-Go。

### 11.9 日志、损失与计数守恒

dataset 级记录 scenario/prompt/request/raw response/parser/model/fingerprint/review/content hashes；训练
query 级记录 view/index version、三个 neighbor IDs 的匿名 record hashes、三距离、`q50/q95/q99`、
`c_validity/c_OOD/c_L`、三维 target、semantic loss、coverage bucket、零权重/失败原因及 Student
disagreement。disagreement 只诊断，不影响 retrieval、validity、OOD、loss 或 schedule。

每个 dataset 必须满足：

```text
planned = response_received + request_failed
response_received = parser_valid + parser_invalid
records = parser_valid + parser_invalid + request_failed
records = sum(stratum_records)
review_sample = review_pass + review_substantive_error + review_critical_error
```

每个训练统计窗口必须满足：

```text
semantic_queries = validity_zero + validity_one
validity_one = ood_positive + ood_zero
semantic_loss_active <= ood_positive
fallback_count = 0
```

语义损失使用整记录共享权重；精确 Student head/gradient 和 `lambda_L(t)` schedule 在 O0-E 冻结。
当前窗口无有效或正 OOD query 时 `L_L=0`，不产生 optimizer contribution，仍记录零样本窗口。

### 11.10 O0-D 允许主张与验证证据

O0-D 完成只允许声称：三维语义、61D view、正式截断指数 OOD、模型切换、60/800 数据治理和
dataset-level No-Go 已预注册。不得声称 Flash/Pro 已通过 pilot、800 labels 已生成、三维语义改善
MAPPO、OOD 权重有效或标签具有论文质量。

验证日期为 2026-08-25，使用规范解释器
`D:\Anaconda3\envs\py310\python.exe`；O0-D 未读取或保存 API key，未调用 DeepSeek，未生成
pilot/formal labels，未修改 Python/YAML/JSON 运行代码。61D 重建和候选比较结果见第 11.4 节；
旧 615D 审计明确降级为 preliminary evidence。完整 `pytest` 为 184 passed（38.15 s）；Flake8、
`visualize.py --help`、dynamic-ingress A* evaluation help 与 `git diff --check` 退出码均为 0。

## 12. O0-E Student、schedule、执行依赖与 checkpoint 冻结合同

### 12.1 当前实现审计基线

现有 `DualHeadActor` 使用 `motion_encoder: O -> 128 -> 64`、独立
`engagement_encoder: O -> 128 -> 64`、一维或二维 sigmoid semantic head，以及
`(64+D) -> 5` 最终线性 action head。PPO 和旧 A* KL 都更新 Motion Encoder 与最终 Action Head；
semantic MSE 更新 Semantic Encoder/Head；最终 action path 对 semantic 输出执行 detach。当前没有
独立 Motion Prior Head 或 Semantic Adapter。Centralized Critic 对每台机器人执行
`O -> 128 -> 128`，经 4-head self-attention 和 mean pooling，再以 `128 -> 256 -> 128 -> 1`
产生团队 value。

优化路线当前 5-AGV 配置的 observation 宽度实测为 613：575 维 raw observation，加 7 维 own、
4 维当前 orientation、9 维 waypoint block、15 维 nearby 和 3 维 global。在线 A* 执行依赖来自
9 维 waypoint block：next-waypoint delta 2 维、desired direction 4 维和 waypoint relation 3 维。
`include_waypoint_features=false` 虽保持相同宽度并把这些槽置零，但没有为策略提供规则目标相对
位置。名为 `waypoint_reward` 的 shaping 实际只比较规则目标的 Manhattan distance，未调用
planner；其命名与真正的 A* observation 依赖必须分开处理。

现有 schedule 按 completed episodes 阶梯衰减，受 episode 长短和并行完成顺序影响；checkpoint
只保存 model/config/observation width/semantic width/episodes/steps/phase，没有 optimizer、独立
schedule、EMA、RNG 或新三维 schema。历史 loader 还会从 phase 名和 tensor shape 推断一维/二维
semantic width。上述行为只构成 legacy 合同，不迁移为新架构默认值。

### 12.2 `o0-student-v1` 唯一网络

正式网络接收两个互不替代的输入：无 Teacher 派生量的物理观测
`x_phys[N,613]`，以及 O0-D 冻结的 `semantic-view-v3` 数值表示 `x_sem[N,61]`。shape 固定如下：

| 分支/组件 | 输入 → 输出 | 激活与用途 |
|---|---|---|
| Motion Encoder | `613 -> 128 -> 64` | 每层 Linear 后 ReLU；输出 `z_motion[N,64]` |
| Motion Prior Head | `64 -> 3` | 无末端激活；顺序为 `FORWARD/LEFT/RIGHT`，只服务 A* KD |
| Semantic Encoder | `61 -> 128 -> 64` | 每层 Linear 后 ReLU |
| Semantic Head | `64 -> 3` | sigmoid；顺序为 persistence/yielding/risk |
| Semantic Adapter | `3 -> 16` | Linear 后 ReLU；输入必须是 detached Semantic Head 输出 |
| late fusion | `64 + 16 -> 80` | 只做固定顺序 concatenate |
| MAPPO Action Head | `80 -> 5` | 单一 Linear；输出项目五动作顺序 logits |
| Centralized Critic | `[B,N,613] -> [B]` | `613->128->128`、4-head attention、mean、`128->256->128->1` |

Motion Prior Head 不连接最终动作 logits；它把 A* 教授的局部几何知识写入共享 Motion
Representation。MAPPO Action Head 才是最终五动作策略，并继续使用统一 hard action mask。LLM
三维输出不是最终动作、通行权或强制规则，而是经 Adapter 转为可由 PPO 使用的内部语义特征。

### 12.3 梯度所有权与不可绕过边界

| 梯度来源 | 可更新 | 禁止更新 |
|---|---|---|
| PPO actor | Motion Encoder、Semantic Adapter、Action Head | Motion Prior Head、Semantic Encoder/Head、Critic |
| A* KD | Motion Encoder、Motion Prior Head | Semantic 分支、Adapter、Action Head、Critic |
| LLM semantic KD | Semantic Encoder、Semantic Head | Motion 分支、Adapter、Action Head、Critic |
| PPO value | Centralized Critic | Actor 全部分支 |
| Reward Calibration | 无 | 所有网络与 A*；只读取 detached 值 |

Semantic Head 的三维 tensor 必须在进入 Adapter 之前 stop-gradient。实现测试必须分别反传四类
loss 并检查参数梯度集合；不能只依赖模块命名或 optimizer 分组。PPO 不得通过共享输入、额外
auxiliary head、Critic loss 或 checkpoint restore 绕回 Semantic Encoder/Head。A* loss 不得直接
监督最终 Action Head。Teacher target、reliability、root cost、`Delta G`、EMA、Student
disagreement 和 reason 均不得进入 Critic。

Disagreement 只在 detached Student distribution 与 detached Teacher prior 之间计算并记录；其
精确日志字段在 O0-F 冻结。它不得参与 loss、sample selection、reliability、schedule、EMA、
Teacher query 或 fallback。

### 12.4 A* 与 LLM loss 的归一化

Motion Prior 只在 `[FORWARD,LEFT,RIGHT]` 上做 softmax；Teacher `[N,5]` 输出只取同顺序三列。
令：

$$
Z_A=\sum_i m_{calib}(t)m^{valid}_{A,i}.
$$

当 `Z_A>0` 时：

$$
L_A=\lambda_A(t)\frac{\sum_i m_{calib}(t)m^{valid}_{A,i}c_i
D_{KL}(p_{A^*,i}\Vert p_{motion,i})}{Z_A}.
$$

Fixed-KD 固定 `c_i=1`；Reward-Calibrated KD 固定 `c_i=c_A_reward(t)`。当 `Z_A=0` 时
`L_A=0`，不产生 A* optimizer contribution。分母只计算 sampler 选中的有效机器人，禁止把
`lambda_A` 或 `c_A_reward` 放入分母，否则会抵消二者对总优化强度的正式含义。

三维 semantic loss 先对每个 record 的三个维度取 MSE 均值，再定义：

$$
Z_L=\sum_i c_{validity,i},
$$

$$
L_L=\lambda_L(t)\frac{\sum_i c_{validity,i}c_{OOD,i}
\operatorname{MSE}_3(\hat y_i,y_i)}{Z_L},\quad Z_L>0.
$$

`Z_L=0` 时 `L_L=0`。OOD 是整条三维记录共享的强度，不能逐维分母归一化或让 reliability
从最终梯度幅度中消失。两类 Teacher loss 独立相加，不做教师间 softmax，也不要求权重和为 1。

### 12.5 `linear-env-step-v1` 共同 schedule

schedule 的唯一时间变量是 checkpoint 中严格单调的真实环境 transition 总数
`t=global_env_steps`。shadow transition、evaluation step、label generation 和 wall-clock 均不计入。
每个匹配证据族在 manifest 中冻结相同正整数
`B=schedule_total_env_steps`，并定义：

$$
p(t)=\min(\max(t/B,0),1),
$$

$$
\lambda_A(t)=0.05(1-p(t)),\qquad
\lambda_L(t)=0.10(1-p(t)).
$$

每个 optimizer update 开始前，用该 update 已完成收集后的 `t` 计算一次，并在整个 update 内保持
不变。一个 vector worker 产生一个真实 transition 即计一步，因此改变并行 worker 数不能改变
同一交互预算下的 schedule。没有某类 Teacher 的消融组仍记录相同名义 schedule，只把对应有效
mask 置零。禁止使用旧 episode 阶梯、wall-clock、训练 reward、coverage 或性能结果修改 schedule。

恢复时 loader 必须验证 version、`B`、初值和保存的 `t/p/lambda_A/lambda_L` 一致，再从 `t`
继续。不能从 episodes、CSV 行数或 checkpoint 文件名重建，不能在换机器或换 worker 数后重置。

### 12.6 `direct-goal-observation-v1`

新物理观测保持 613 维，避免 observation width 变化成为额外实验变量。原 9 维 waypoint block
唯一替换为：

```text
[goal_dx / max(width-1,1), goal_dy / max(height-1,1),
 0, 0, 0, 0, 0, 0, 0]
```

goal 只由现有规则层解析任务、配送或固定安全充电目标；Student 只读该坐标，不得要求 A* 决定
或改写目标。已到达时两个 delta 均为 0。后七个 reserved slot 必须为 bitwise float32 zero，不能
承载 waypoint、desired direction、trajectory、reservation、coordinator、Teacher prediction 或
learned path embedding。raw/own/orientation/nearby/global 其余 604 维保持当前语义。

observation builder、policy evaluation 和 policy visualization 不得调用 planner。O3 的两个真正
未见拓扑必须保持该字段顺序与 613 维宽度；不满足时先修复 layout/observation contract，禁止以
padding、截断或另建网络进入训练。

当前 `waypoint_reward` 的真实行为是：规则目标 Manhattan distance 下降时给固定 shaping reward，
没有 planner query。O1 只做行为保持的正式重命名 `direct_goal_progress_reward`，并为旧配置保留
只读 legacy alias；公式、数值和时机不变。优化路线主方法与 NoGoalHint 使用相同 reward，防止在消融
observation 时同时改变优化目标。

### 12.7 Legacy waypoint、NoGoalHint 与执行期主张门

历史 1D/2D checkpoint 通过独立 legacy evaluation path 继续获得旧 waypoint observation；该
路径只为复现历史结果，不能进入新三维训练、评估或可视化入口。新三维 loader 遇到 legacy
observation contract 必须拒绝，而不是把旧 waypoint slot 解释成 DirectGoal。

优化路线停止把 `NoWP` 作为正式实验组名；正式诊断名为 `NoGoalHint-v1`，observation schema
为 `no-geometric-goal-hint-v1`。它保留 613 维但把整个 9-slot geometry block 置零，其余环境、
reward、hard mask、网络和训练预算与对应组匹配。它是三 seed 目标几何提示消融；不要求与主方法
性能等同，也不能单独证明或否定执行期 A* 主张。

“Student 执行期无需 A*”只有全部满足后才允许写入论文：

1. policy evaluation 与 visualization 的 instrumented planner query count 均为 0；
2. 把 planner 替换为任何调用即抛错的测试替身后，DirectGoal 与 NoGoalHint 均完成端到端短运行；
3. DirectGoal 主方法通过 O2、O3 和 E1 对应的冻结性能/接口门；
4. 论文同时说明 Pure Motion A* 仍是训练期 Teacher，启发式 A* baseline 仍独立使用 A*。

O0/O1 只能声称“冻结/实现了面向无 A* Student 执行的 observation contract”，不能声称该方法
已经达到无 A* 执行性能或部署要求。

### 12.8 `o0-student-checkpoint-v1`

可恢复 checkpoint 只在完整 optimizer update 完成且 rollout buffer 为空时保存，必须包含：

- checkpoint/architecture/observation/semantic/schedule/sampler/Pure Teacher 的版本；
- 613/61/3/5 等 shape、有序 semantic names 与项目 action names；
- model 与 optimizer state、`global_env_steps/update_count/completed_episodes`；
- canonical config/manifest/Git commit、环境 ID、layout hash、能源参数、reward schema/hash；
- semantic dataset/manifest/index/prompt/parser/OOD 参数及全部内容 hash；
- 相互独立的 `K_motion=12`、`H_reward=12`、512-expansion budget 和 1/16 sampler；
- EMA 的 schema、count/mean/variance/initialized、0.99 decay、`1e-3` minimum scale、64 initial
  samples 与 `[0,1]` clipping；
- Python/PyTorch/CUDA 版本以及 Python、NumPy、PyTorch CPU/CUDA RNG state。

strict loader 先逐字段验证 required key、version、ordered names、shape、hash 和配置，再执行
strict model/optimizer state load。错误必须指出具体不兼容字段；禁止靠 phase 名或 tensor shape
猜测三维合同。缺少 optimizer、schedule、EMA 或 RNG state 的新文件只能由独立 inference-only
入口读取，并标记 `non_resumable=true`。

旧 1D/2D loader 继续服务历史评估，但不能构造 `o0-student-v1`。禁止把旧 semantic weight 复制、
补零、截断、重复、平均或映射到三维结构；新训练必须从新架构初始化。两类 loader 的 dispatch
只读取明确 checkpoint schema version，不允许“尝试新 loader、失败后静默回退”。

### 12.9 O1 模块边界与依赖顺序

O1 不得由实现者另选类名、网络宽度或公式；实际符号命名可以遵守项目模块风格，但职责必须按
以下顺序实现和验证：

1. DirectGoal/NoGoalHint observation schema、legacy alias 和 planner-zero-call instrumentation；
2. 新 Student、四类梯度所有权测试与 strict checkpoint/legacy dispatcher；
3. Pure Motion query、Motion Prior loss、逐机器人 validity 和诊断输出；
4. semantic-view-v3 encoder、三维 offline retrieval、reliability 和 semantic loss；
5. canonical snapshot、CRN paired shadow、detached Critic bootstrap 与 EMA；
6. 共同 sampler、Fixed/RC 开关、buffer、schedule、恢复和计数日志；
7. H=12 功能 smoke、A600 overhead/memory owner-run 命令与完整 O1 Go/No-Go 证据。

依赖只能从前一层已验证接口流向后一层；不能用临时 waypoint、旧 2D weight、Fixed-KD warm
start、H=4 fallback 或关闭严格检查来推进后续步骤。O1 的长 A600 基准仍由研究所有者运行，
Codex 只准备命令和分析结果。

### 12.10 O0-E 允许主张

O0-E 完成只允许声称：Student shape、梯度边界、loss 归一化、共同 schedule、DirectGoal/NoGoalHint、
checkpoint 隔离和 O1 实现顺序已经预注册。不得声称网络已实现、checkpoint 已可恢复、执行期
已经摆脱 A*、三维语义有效、Reward Calibration 改善性能或训练已经收敛。

### 12.11 O0-E 验证证据

验证日期为 2026-08-25，使用规范解释器
`D:\Anaconda3\envs\py310\python.exe`。现有环境短审计确认 canonical 5-AGV observation 为
`[5,613]`、raw observation 为 575 维；`linear-env-step-v1` 在示例
`B=150000` 的 `t=0/75000/150000/200000` 上分别得到
`lambda_A=0.05/0.025/0/0` 和 `lambda_L=0.10/0.05/0/0`。O0 spec 目录计数为 1。

完整 `pytest` 为 184 passed（42.08 s）；Flake8、`visualize.py --help`、dynamic-ingress A*
evaluation help 与 `git diff --check` 退出码均为 0。相对 O0-D commit `611c981` 的 Python、YAML、
YML、JSON 运行代码/配置差异为 0；仓库中未发现研究所有者提供的 API key。两个根目录研究输入
继续按 O0-F 合同保持未跟踪，本组未提前执行内容映射或删除。

## 13. O0-F：证据预算、日志、消融、统计与论文主张

### 13.1 训练预算与 O2 校准门

优化路线全部学习运行共 `74` 次，不得再简称为 65 次：

- O2 校准：`MAPPO-DG`、`Fixed-AStarKD`、`RC-AStarKD` 均关闭 LLMKD，各使用初始化 seed
  `107/117/127`，`150000` real environment steps，共 `3×3=9` 次；
- E1/E2 核心 `2×2`：`MAPPO-DG`、`RC-A*KD`、`LLMKD`、
  `RC-A*KD+LLMKD` 各 8 seed，共 32 次；
- `Fixed-A*KD+LLMKD`、`QMIX-DG`、`RuleKD-v3` 各 8 seed，共 24 次；
- `ShuffleKD-v3`、`NoOOD-v1`、`NoGoalHint-v1` 各 3 seed，共 9 次；
- E1/E2 合计 65 次，连同 O2 为 `9+65=74` 次。启发式 A* 不训练，评估和人工干预不计入
  training-run 数。

O2 的 Fixed/RC 两组除 `c_A_reward` 外保持相同网络、环境、reward、seed、预算、1/16 sampler、
shadow、EMA、日志和 schedule；MAPPO-DG 提供无 A*KD 退化参照。三组均关闭 LLMKD。Fixed 与
RC 的 `m_calib` 选中状态数、A* query 数和 shadow 数必须逐 seed
完全相同，否则是接口失败。主覆盖率定义为：

$$
coverage_A=\frac{\sum m_{calib}m_A^{valid}}
{N\times\#\{m_{calib}=1\text{ states}\}}.
$$

分母包含 dead、lock、mandatory-toggle、at-goal 等 fail-closed agent slots，以反映教师真实可用性；
另报告排除这些结构性非运动状态的 `motion_eligible_coverage`，但它不能替代 25% gate。三个 RC
seed 的主覆盖率必须分别不低于 25%，不得用 pooled mean 掩盖失败 seed。

吞吐 AUC 固定在 `x=0,10000,...,150000` 的 real-environment-step 网格。每点的纵轴为截至该点的
`1000×累计完成任务数/累计 episode steps`，没有已完成 episode 时为 0；在相邻网格点间线性
插值并以 `x/150000` 归一化后做 trapezoidal AUC。先逐训练 seed 计算，再跨三个匹配 seed 取
RC 相对 MAPPO-DG 的退化率中位数；中位退化不得超过 10%。

### 13.2 分层日志与污染计数

禁止为每个普通状态写入含完整数组的无界 `teacher_steps.jsonl`。正式 schema 固定为：

- `run_manifest.json`：commit/config/layout/dataset/checkpoint hash、seed、预算、版本、能源、reward、
  Python/PyTorch/CUDA、日志采样规则；
- `teacher_step_counts.csv`：每个 real vector step 一行，只含 step/update、real states、
  `m_calib` selected、A* valid/failure-reason counts、LLM validity/OOD bins、fallback、planner query、
  shadow/bootstrap/terminal、non-finite 与组件化污染计数；
- `teacher_events.jsonl`：仅记录全部 calibration-selected states、全部 invalid/failure/non-finite/
  pollution events，以及其余状态的 1/256 确定性审计样本；样本由 canonical state hash
  `uint64(SHA256("teacher-audit-v1"||state_hash)[0:16]) mod 256=0` 选取；
- `updates.csv`：loss、`lambda_A/lambda_L`、有效分母、平均/分位权重、disagreement（诊断）、
  `DeltaG`、EMA、梯度范数、optimizer/schedule counter；
- `episodes.csv`：回报、完成任务、episode steps、碰撞、死锁、能量死亡、充电、优先任务完成；
- `resource_windows.csv`：基线/H4/H12 的计时边界、RSS/CUDA allocated/peak 与 window/repeat。

Fixed 与 RC 必须使用完全相同的日志密度。详细事件中的 reason、路径长度、expanded nodes、
planning time、shadow return 和 bootstrap 只作诊断，不进入 loss。污染计数按组件分开：A* 的
priority/yield/reservation/coordinator/reward/Student/LLM/calibration 读取均禁止；LLM 可以读取
`semantic-view-v3` 中匿名优先级，但禁止读取 A*/reward/Student/calibration/scenario type/ID；
Student 最终动作路径中的 execution planner query 必须为 0。不同组件不得共用一个模糊的
`priority_pollution` 总数。

### 13.3 语义对照与诊断消融

正式 LLM 标签唯一链路是
`semantic-view-v3 -> semantic-prompt-v4-directional-rubric -> DeepSeek -> continuous [0,1]^3`。
五点 anchors 和方向性因素是量表说明，不是确定性标签生成器；reason 仍只作 audit metadata。

`RuleKD-v3` 是独立规则教师基线。它在与 formal LLM 相同的 800 个 semantic views 上生成三维
标签，并复用相同 retrieval、整记录 validity 和 OOD reliability；不得把结果注入 LLM prompt。
其唯一全覆盖优先级如下（schema 非法时整条 invalid）：

1. persistence：`idle -> 0`；否则 `charging or battery<=0.30 -> 0.25`；否则
   `loaded and delivery -> 1`；否则 `task and priority_rank<=2/25 -> 0.75`；否则
   `task -> 0.50`；其余 `0.25`；
2. yielding：close neighbor 定义为 `mask=1` 且 normalized Manhattan distance `<=0.10`；无 close
   neighbor 为 0。focal 空载且任一 close neighbor 载货为 load disadvantage；focal 无优先任务而
   邻居有，或双方有且邻居 rank 更小，为 priority disadvantage；两者同时为 1，任一为 0.75，
   否则 close count `>=2` 为 0.50，其余为 0.25；
3. risk：`r=min(close_count,3)+I(constrained)+I(dead_blocker)`；`constrained` 指 focal 位于充电/
   取货站，或位于 highway 且四向相邻 highway 数 `<=2`；`dead_blocker` 指任一 close neighbor dead。
   `r=0/1/2/3/>=4` 分别映射 `0/0.25/0.50/0.75/1`。

`ShuffleKD-v3` 在五个场景分层内按 scenario hash 排序，以 manifest hash 导出的 `1..n-1` 非零
循环位移分配 donor，必须断言每条 recipient 的 donor scenario ID 与自身不同；它保持三维联合
标签分布和 recipient 的 OOD reliability，不允许固定点或仅逐维独立打乱。

`NoOOD-v1` 仅把有效 LLM record 的 `c_OOD` 固定为 1。`NoGoalHint-v1`（旧兼容 alias
`no-geometric-goal-hint-v1`）把 DirectGoal block 九位归零，其他输入、reward、mask、预算不变；
它只支持“策略对目标几何提示的敏感性”诊断，不能证明训练贡献或执行期 A* 依赖。执行期无需
A* 只由 DirectGoal 主方法的 `execution_planner_queries=0`、抛错替身端到端测试及正式性能门支持。

`QMIX-DG` 与 MAPPO 共享 DirectGoal 613D 输入、环境步预算、训练/evaluation seeds、action mask、
rule safety、reward、能源、final-checkpoint 与 held-out evaluation；禁止结果后单独调参或静默换成
IPPO/VDN。允许功能 smoke，不允许依据 smoke 性能选择超参数。

### 13.4 正式统计合同

独立统计单位是训练初始化 seed。正式学习组使用 `7/17/27/37/47/57/67/77` 并共享 held-out
evaluation seeds `200..209 × 20 episodes`。同 seed 的学习组使用配对分析；启发式无训练基线只作
描述性参照，不参与 seed 级显著性检验。基础设施故障只可同 seed/同配置重跑并保留记录；算法、
数值和安全失败保留为结果。缺失或损坏且无法重建的 seed 使该确认性比较 No-Go，不得换 seed。

唯一 primary endpoint 为 final-checkpoint
`completed_tasks_per_1000_steps=1000×sum(completed_tasks)/sum(episode_steps)`，每个训练 seed 先聚合
全部 200 个 held-out episodes。关键 secondary endpoint 是第 13.1 节 step-normalized throughput
AUC；team reward、completion rate、碰撞、死锁、能量死亡、充电和优先任务完成均为 secondary/
safety endpoints。

七个预注册 primary contrasts 为：full RC vs MAPPO-DG、A*KD factorial main effect、LLMKD
factorial main effect、A*KD×LLMKD interaction、full RC vs Fixed-KD、full RC vs QMIX-DG、full RC
vs RuleKD-v3。每个 contrast 以 seed 内差值或 factorial contrast 计算双侧 paired t-test，并在这
七项上执行 Holm family-wise correction。报告未校正 p、Holm-adjusted p、均值差、95% t confidence
interval 与 paired standardized effect `d_z=mean(diff)/sd(diff)`；零方差时 effect 标为 undefined，
不得伪造无穷值。另以固定 seed `20260825`、10000 次 seed-level paired bootstrap 报告 percentile
95% CI 作为小样本敏感性分析。AUC 和其他 secondary endpoints 只报告 95% CI 与效应量，除非 E1
在看结果前另行冻结检验族。episode 不能被当作训练样本扩大 n。

### 13.5 允许主张、失败降级与输入处置

核心 `2×2` 支持两类教师的主效应与交互；full-vs-Fixed 只支持 Reward Calibration 的增量贡献；
full-vs-QMIX 支持相同 DirectGoal 合同下的外部 MARL 比较；full-vs-Rule 与 Shuffle 支持 LLM 标签
来源和状态对应性的证据；NoOOD 只支持 reliability 敏感性；NoGoalHint 只支持目标提示敏感性。
三维均值替换、置零或干预只能说明 policy sensitivity/reliance，不能单独证明该语义维度在训练中
产生因果贡献。真正未见拓扑只有在 O3 防泄漏与正式统计通过后才能支持跨拓扑可靠性主张。

任何 Teacher 无效均 fail closed 为零 KD 权重；禁止 Fixed、uniform、旧 2D、NOOP、缓存旧标签、
缩短 H 或规则标签 fallback。运行/内存 gate、dataset gate、checkpoint gate 或正式统计完整性失败
时必须按所属阶段 No-Go，不得用较有利结果静默改合同。

两份根目录研究输入的有效内容已映射到第 2–13 节：数据流/现状审计映射第 2–8 节，Pure Motion
Teacher 映射第 9 节，Reward Calibration 映射第 10 节，三维 LLM/OOD 映射第 11 节，Student 与
DirectGoal 映射第 12 节，日志/消融/统计/主张映射本节。canonical architecture 是唯一正式方案；
输入文件不得继续作为并列合同。

### 13.6 O0-F 审核裁决与验证证据

O0-F 对外部建议的裁决为：统计合同、NoGoalHint 更名、74 次总预算、RuleKD 唯一规则、O2
coverage/AUC、紧凑日志、干预主张边界、无 fixed-point Shuffle、组件化污染计数和 QMIX fairness
全部采纳。三维语义建议的核心隔离要求全部采纳；“只保留 0/0.5/1”不采纳，因为它会无必要地
降低已冻结连续量表的分辨率。最终保留五点解释 anchors、允许任意连续值，并加入非确定性的
方向 rubric。

验证日期为 2026-08-25，使用唯一规范解释器。YAML schema/74-run/7-contrast 断言通过；完整
`pytest` 为 184 passed（45.52 s）；Flake8、`visualize.py --help`、dynamic-ingress A* evaluation
help、`git diff --check` 均退出 0；仓库未发现 API key。相对 O0-E commit `17e40e4` 没有 Python
运行代码、runtime training config、环境、reward、checkpoint 或 seed 修改；唯一 `configs/`
变更是 owner 明确授权的治理 manifest。两份研究输入已按第 6.1 节哈希映射后移除。

O0-F 交付后，Windows 文本搜索入口已修复并冻结为
`C:\Users\28016\bin\rg.exe`；后续审计必须直接调用该文件，不使用 bare `rg`。

## 14. O0-G：零偏差复核、O1 交接与最终人工门

### 14.1 零偏差结论

O0-G 从头复核 requirements、plan、validation 和本 canonical architecture。O0-A 至 O0-F 冻结
合同之间不存在公式、量纲、输入所有权或失败策略冲突，也没有遗留候选、实现者自行选择项或
未冻结的正式参数。复核确认以下边界必须原样进入 O1：

- Pure Motion Teacher 只产生逐机器人运动先验；`K_motion=12` 与共享 512-expansion budget
  不得被 Reward Calibration 或训练性能改写；
- Reward Calibration 只缩放整条有效 A* 标签；`H_reward=12`、1/16 sampler、两分支状态语义、
  EMA `0.99/1e-3/64` 和 H=12 的 `3×` runtime/memory No-Go 均保持唯一；
- 三维语义只使用 `semantic-view-v3` 61D、整记录 validity 与共享截断指数 OOD reliability；
  reason、disagreement 和 reward calibration 均不得进入语义 target；
- Student 固定为 DirectGoal 613D 物理分支加 61D 语义分支，三维 checkpoint 从新架构初始化；
  `lambda_A(t)` 与 `lambda_L(t)` 使用同一 `linear-env-step-v1`；
- O2/E1/E2 证据预算、对照、日志、统计和允许主张均以第 13 节为唯一依据，O1 不得新增或
  删除实验组，也不得通过超参数搜索改变合同。

`TBD|TODO|placeholder|实现者自行选择` 扫描只命中 validation/plan 中描述该禁止项的审计文字；
活动优化合同中的历史 1D/2D/NoWP 仅用于现状、禁止 fallback 或稳定路线隔离说明，不构成优化
路线正式语义。相对 P0 基线 `6fcb7d3` 的变更仅包含文档和治理 manifest，没有 Python 运行代码、
runtime training config、环境、reward、checkpoint、训练 seed 或稳定路线实现。治理 manifest 是
`governance_manifest_not_runtime_training_config`，不能由训练入口读取。

字段映射复核覆盖：Pure Motion query/result/cache/diagnostics，calibration snapshot/sampler/return/
EMA，61D semantic record/OOD/reliability，613D+61D Student/checkpoint，紧凑日志与七个正式统计
contrast。所有优化量均有唯一生产者、消费者、序列化位置和 fail-closed 处理；不存在额外
`c_A_search`、consistency、逐维 semantic mask、Fixed-KD warm start 或旧标签迁移通道。

### 14.2 O1 唯一实施交接

O1 的唯一实施任务包为
`plan/task-package/o1-role-alignment-implementation.md`。该任务包冻结模块边界、接口、实现顺序、
测试先行步骤、短 smoke、提交边界和研究所有者运行的 A600 基准命令。若任务包与本文冲突，
以本文为准并返回 O0-G 由研究所有者裁决；实现者不得自行选择替代方案。

O1 本地工作只允许短单元/集成测试和单次短 smoke，不得生成 60/800 标签、启动 O2 训练、执行
长评估、搜索 KL/schedule/EMA/OOD 参数或静默缩短 H。A600 的 12-worker H=4/H=12 开销与内存
基准只由研究所有者运行；Codex 只分析其产物。

### 14.3 最终书面批准门

O0-G 交付不等于 O0 自动完成。研究所有者必须在本次交付后明确书面批准以下整体：唯一
canonical architecture、Pure Motion A* 合同、Reward Calibration 与 EMA、OOD 公式、三维数据
合同、Student/schedule/checkpoint、H=12 runtime/memory gate、证据预算/消融/统计和允许主张。
获得该批准后，才可在单独的收口动作中标记 O0 complete、同步 Roadmap/TASKS/CHANGELOG 并进入
O1；批准前这些状态必须保持 pending。

### 14.4 O0-G 验证证据

验证日期为 2026-08-25。使用 `D:\Anaconda3\envs\py310\python.exe` 完整运行 pytest，结果为
`184 passed in 45.37s`；Flake8、`visualize.py --help`、dynamic-ingress A* evaluation help 和
`git diff --check` 均退出 0。`C:\Users\28016\bin\rg.exe --version` 返回 ripgrep 15.2.0。

治理 manifest 通过 YAML 解析且状态为 `o0_g_final_owner_approval_pending`；canonical/spec/O1 任务
包链接全部存在；相对 P0 基线的运行代码与 runtime training config 差异为零；仓库凭据样式扫描
未发现 API key。最终提交只包含 canonical architecture、O1 文档任务包、feature-spec checklist、
治理 manifest 状态和 CHANGELOG，不包含运行代码、标签、artifact、训练或评估结果。
