# 需求 — medium-5ag 离线 LLM 时序语义数据集

当前合同包含 [2026-10-05 已批准语义与采样修正](observable-coordination-v2.md)。下文保留初版历史，冲突条款以该修正为准。

## 目标与顺序

为 core-integration 的 209D recurrent MAPPO Actor 生成一份可追溯、可用于离线语义表征预训练的 medium-5ag 主数据集。研究所有者指定顺序为 medium-5ag 数据集与训练验证先行；tiny-3ag 和正式 12 AGV 的数据生成只在 medium-5ag 训练完成并审阅后另立规格。当前正式地图为 docs/formal_environment/environment.json 所描述的 50×28 静态布局；其中只有 8 个充电站，不能直接作为 12 AGV 运行合同。

## 本功能范围

- 仅生成 medium-5ag 的 Actor 可观察历史、四维语义标签及质量凭据；四维固定为 mission_urgency、concession_appropriateness、interaction_risk、deadlock_severity，沿用当前代码的 temporal-semantic-view-v1 与 llm-label-prompt-v3 语义。不得从旧三维单帧语义自动迁移标签。
- 数据采集环境逐字段取自 core-integration 提交 2e579fd 的已批准 configs/core_integration/medium-5ag-baseline.yaml 的 environment 段：5 AGV、800 步、rware_delivery、A/B/C 优先级、DirectGoal v2/209D、五个充电站、两个拣选站和 C1 local-mean reward 合同。保留原配置中虽名为 llm-mappo-medium-3ag-v1 的 environment_id 以及 environment.reward_version=reward-v2；C1 版本由独立的 reward_definition.version=reward-v3-c1-energy-death-local-mean 标识，不擅自修正历史字段。冻结该配置的字节哈希、环境构造参数哈希和代码提交；不得使用 evaluation.environment_overrides。
- 行为策略只采用上述批准合同下由研究所有者完成的 MAPPO-FP-DG final Actor。源 checkpoint、训练清单与 Actor 导出物均须有匹配 SHA-256、209D schema、5 AGV、环境身份和 final step 身份。当前该 artifact root 下尚无可用 final Actor；当前 runner 也只对旧采集 schema 自动导出 Actor，须为 C1 正式合同增加安全导出入口。不能用旧 Phase 3 的 500 步/A–D 数据采集 Actor，也不能把旧 C1 诊断 checkpoint 当成已批准的正式来源。采集侧不得以 weights_only=False 绕过安全加载。
- 采集 120 个完整回合：独立环境 seed 10000 至 10119，各运行一次，行为策略使用已冻结 Actor 的带掩码随机采样，策略随机流由同一 episode seed 初始化。若选样约束不满足，只按连续 seed 10120 至 10239 增补，最多 240 回合；不得替换失败回合或使用训练 seed 7/17/27/37/47/57/67/77、诊断 seed 9400–9409、正式评估 seed 200–209。
- 先按完整回合做 train/validation/test = 70%/15%/15% 的确定性切分，salt 固定为 medium-5ag-llm-dataset-v1；沿用 dataset.py 的 SHA-256 排序及最大余数配额算法，同一回合的五台 AGV 不跨分区。选取恰好 6000 个 8 帧历史窗口，分区配额分别为 4200/900/900。只选同一回合内的完整 8 帧窗口；同一回合、同一 AGV 的两个已选终点至少相隔 8 步。按 train、validation、test 顺序分别沿用 dataset.py 的可观察 stratum、字典序轮转和 _window_rank 哈希顺序；候选遇到终点间隔冲突或全数据集范围的相同语义视图哈希即跳过，不消耗配额。先用 120 回合验证选样配额及覆盖，不足时逐个按 seed 10120 至 10239 增补，并在每次增补后对全部回合重新切分和选样；第一个满足全部配额及覆盖的回合集合即冻结。240 回合仍不足则 No-Go。冻结前不得发送任何付费标注请求。
- 每个分区的已选窗口须同时包含 congestion=present 与 blocked_or_no_progress=true：train 各至少 100，validation/test 各至少 30；记录所有可观察 strata、任务阶段、优先级和标签分布。不得伪造或注入环境状态补足配额，缺失时按上述增补和 No-Go 规则处理。
- 完整回合以每 seed 一个压缩数值产物保存五台 AGV 的连续 209D float32 观测、reset mask 和回合元数据；读取时禁止 pickle，逐回合核对形状、有限值和哈希。增补回合只新增该回合产物，不随回合数反复重写已有大文件；窗口扫描按回合进行，内存不保留全部 8 帧候选。数据目录与清单可断点恢复且只允许同一冻结身份。
- 将 train 入选窗口按 `_window_rank(window, "medium-5ag-llm-dataset-v1", window.stratum)` 排序，取前 120 个作为付费试标：第 1 条真实 API 请求同时是 smoke 和试标第 1 条，不另增请求；前 60 条做匿名盲审，前 40 条做同模型复标并用同一 40 条做 Pro 对照。试标通过后继续至 6000 个主模型成功标签；已通过的试标计入 train 的 4200 条，不重复计费。模型、prompt、schema 或环境身份改变时必须新建数据集身份，不得与旧试标拼接。
- 主标注模型使用 DeepSeek 官方 API 的 deepseek-flash，非思考模式、temperature=0、JSON object、每次最多 800 输出 token，API 密钥只从 DEEPSEEK_API_KEY 环境变量读取。DeepSeek 官方文档说明旧 deepseek-v4-flash 名称会路由到 V4.1 Flash；新数据使用当前规范名称，并重新生成配置哈希和审计凭据，不能沿用旧 L1 批准哈希。正式调用前再次核对官方模型、价格和返回身份。官方资料：https://api-docs.deepseek.com/news/news260910/ 与 https://api-docs.deepseek.com/quick_start/pricing/ 。
- 主标签最多 6000 个成功请求，主模型 HTTP 尝试总数最多 6300（含试标和重试），单条最多重试一次；额外审计最多 40 次主模型重复请求及 40 次 deepseek-v4-pro 对照请求。所有请求合计的保守费用上界为 50 美元、prompt token 上限为 1 亿、completion token 上限为 500 万。每次请求前以当前官方峰时、缓存未命中单价、输入 token 预留上界 `2×(system+user prompt 的 UTF-8 字节数)+4096` 和 800 输出 token 预留预算；响应到达后用实际 usage 和同一峰时未命中单价更新保守费用上界，释放未用预留。缓存命中节省只在 API 给出可核对的细分 usage 时单独报告，不用于放宽预算；没有账单证据不得把保守上界称作实际费用。任一上限无法容纳下一请求即停止，禁止自动追加预算。响应无 usage、请求状态不明、价格不可核实，或 Pro 的当前官方型号版本与主模型不能区分时暂停付费调用并留下失败凭据，不自动改选第三方模型。
- 保留逐条 provider 请求型号、返回型号、可用的 system_fingerprint、prompt/view/配置哈希、usage、重试次数和去密钥化响应哈希。若已返回的模型身份或非空 fingerprint 在同一主数据集中改变，停止并新建版本，不合并不同 backend 的标签。若 API 不返回 fingerprint，凭据必须明示无法验证底层权重未变，不得只凭请求别名宣称底模同一。请求已发送但响应未可靠持久化时先暂停核查，不能为满足 6000 条静默重发并声称不会重复计费。

## 质量门与使用边界

- 试标盲审采用固定四维锚点及 evidence-path 核对。critical error 指动作/路径/分配指令、使用不可观察事实、凭空引入交互对象或证据与分数明显相反；substantive error 指非 critical 但足以改变一个维度的高低判断。60 条中 critical 必须为 0，substantive 不超过 3。审阅者记录逐条匿名判定和总数；Codex 审阅不得称为独立人类标注。
- 采用已批准 M1 指标定义：首试解析率至少 98%，重试率至多 10%，禁用内容率 0，evidence path 有效率至少 98%，各维总体标准差至少 0.08，靠近 0/1 的饱和率各不超过 60%；分母、总体标准差、边界 ε=0.05 及配对四维平均绝对差均按 configs/llm_kd/representation-diagnostics.yaml 的 metric_definition_version=llm-kd-diagnostic-metrics-v1 计算。40 条同模型复标的四维平均绝对差至多 0.15，40 条独立 Pro 对照的四维平均绝对差至多 0.20。任一试标门失败即暂停批量标注，不在原身份下试探性改 prompt 或阈值。
- 完整主集必须有 6000 个唯一、完整、合法的标签，分区/回合/视图哈希无交叉或重复，且达到同一组标签指标门槛；每条四维分数在 [0,1]，每维引用 1–4 条真实可见 evidence path。原始失败与重试记录保留，不通过删除失败请求美化解析率。
- 数据集 Go 只表示准许进入离线语义预训练，不表示四维语义有真实世界真值，也不表示 MAPPO 性能提升。预训练与四组 medium-5ag 下游训练、正式评估另按各自已批准合同执行。训练、评估及部署中在线 LLM 调用为零。

## 不在本功能范围

- 不启动 tiny-3ag 或正式 12 AGV 数据采集、标注和训练；不从当前 8 充电站地图直接推导 12 AGV 性能。后续两个场景各需独立环境、充电容量、行为 Actor、采样配额及预算规格。
- 不改 reward、任务分配、A* 教师、Actor 209D 观测、四维 ontology 或正式 MAPPO 训练预算；不复用旧三维标签或旧 500 步 Phase 3 数据作为主集。
- 不依据同一训练 seed 的短诊断、标签可解析性或单个 checkpoint 宣称 LLM KD 有效。

## 决策依据与当前缺口

当前 llm_mappo/mappo_llm_kd/dataset_run.py、collector.py、dataset.py 的主数据路径写死 5 AGV；本规格恰好采用 5 AGV，不为未来场景提前泛化。现有 dataset_run.py 会将每个时间步的窗口全部物化，并在每个回合后重写完整 trajectories.jsonl；dataset_labeling.py 会遍历全部窗口。必须先实现有上限的流式候选扫描和逐回合压缩存储，否则一次 120 回合采集可能产生数十万次标注与大量重复磁盘写入。AtomicLabelStore 每次追加会重写完整 JSONL，也须改为可恢复且随记录数线性增长的持久化。

旧 llm-kd worktree 的 collection_actor_final.pt 使用 500 步/A–D 合同；旧 C1 600k 诊断 checkpoint 与新批准配置身份不同，且当前环境下 weights_only 安全加载未通过。它们只用于比较和来源核查，不是本主数据的默认行为 Actor。

## 开放问题

无。本规格中的数量、模型、预算和门槛是研究所有者授权由架构师提出的方案；研究所有者审阅本规格后才冻结。
