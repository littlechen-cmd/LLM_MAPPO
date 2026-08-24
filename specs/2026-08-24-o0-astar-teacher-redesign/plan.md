# Plan — O0 A* 教师算法与架构重设计

各任务组必须按顺序执行。O0 只产生审计、架构和冻结合同，不实施代码或运行训练。O0-A 至
O0-F 每组完成时必须同步本计划与 `CHANGELOG.md`、运行该组验证、创建一个聚焦 commit，然后
停止并向研究所有者交付变更、证据、风险和下一组边界。只有研究所有者书面批准“继续”后才能
开始下一组；等待期间禁止预做、暂存或提交下一组内容。只有研究所有者批准 O0-G 后，核心
架构师才能在 `TASKS.md` 标记 O0 完成。

## Task group O0-A：现有全链路与污染面审计

- [x] 以 P0 最终 commit 为基准，逐函数追踪规则目标、充电目标、waypoint observation/reward、
  `AStarExpert` preference/reservation/priority/yield/coordinator、hard mask、buffer、KL、离线
  LLM、三类 actor/critic 梯度、checkpoint、训练、评估和执行入口；
- [x] 在 canonical architecture 的审计章节记录每个字段的生产者、消费者、shape、梯度、启用
  条件、失败降级、日志和执行期依赖，并提供准确文件/符号引用；
- [x] 建立“允许 Pure Motion 信息 / 禁止高层污染 / 仅诊断信息”矩阵，明确当前实现中所有违反
  O0 purity 的路径；
- [x] 追溯现有标签生成链路的精确基础 model ID/version、prompt、temperature、scenario
  generator、parser、数据 schema 和历史一维/二维 checkpoint；
- [x] 对照两个已记录哈希的根目录研究输入，区分排版差异与方法差异；逐项审计方法主张，标记
  接纳、修正、延后或拒绝及理由，保证 canonical architecture 建成后无有效内容只存在于未
  跟踪输入。

## Task group O0-B：Pure Motion Teacher 职责、候选与标签合同

- [x] 比较至少三个候选：现有预约/协调 `AStarExpert`、无高层调度的独立几何 A*、有限局部动态
  障碍但无通行权分配的 A*；从纯度、覆盖、计算、可解释性和标签因果边界说明拒绝/接纳理由；
- [x] 冻结唯一 Pure Motion Teacher 的输入、短时域搜索状态、邻居/动态障碍语义、动作集合、
  概率平滑、budget、replan、tie-break、缓存键与确定性规则；
- [x] 冻结 `motion_preferences[N,A]`、`valid_mask[N]`、失败原因和诊断 schema，保证
  `TOGGLE_LOAD`/非法动作质量为零且逐机器人有效性不丢失；
- [x] 明确证明目标、优先级、充电、parking、yield、预约优先顺序、coordinator、reward、Student
  disagreement 和 calibration 均不能参与标签生成；
- [x] 定义搜索失败、预算超限、无合法轨迹、零有效标签和异常数值的 fail-closed 行为。

## Task group O0-C：状态分叉、paired rollout 与 Reward Calibration

- [ ] 冻结完整 fork state schema，覆盖仓库实体、任务队列、动态入库、计步、充电/规则状态、
  adapter metrics、环境/空间 RNG、外部随机状态、planner/cache 隔离和恢复后哈希等价；
- [ ] 冻结 Student/A* 两个 H=12 shadow 的逐步动作、分支局部 hard mask、A* shadow 当前状态的
  Student fallback、common-random-number stream、滚动重规划、分支独立 terminal、异常和绝不
  污染真实 rollout 的时序；
- [ ] 冻结 paired discounted team return、detached Critic bootstrap、handoff-to-Student 解释和团队
  权重×逐机器人有效掩码公式；
- [ ] 冻结正优势门控、`ΔG` EMA decay、minimum scale、initialization sample count、更新时机、
  截断、非有限降级、checkpoint 状态和初始化前零权重；参数不得依据训练性能选择；
- [ ] 冻结 calibration 采样时机/频率、日志计数守恒与 Fixed-KD 对照开关，证明开关不改变标签、
  reward、real rollout 或 LLM 分支；
- [ ] 冻结 H=12/H=4 短基准、3× runtime gate 和持续 memory growth 的可执行测量定义；H=12 不
  通过时 O1 必须 No-Go。

## Task group O0-D：三维 LLM 数据、可靠性与语义边界

- [ ] 冻结 `task_persistence/yielding_preference/coordination_risk` 的输入语义、输出 schema、互斥
  禁令、整记录 validity 和共享 OOD 权重，删除 consistency 与逐维门控；
- [ ] 比较截断指数距离和 leave-one-out 分段线性两个预注册 OOD 候选，只用 coverage、
  monotonicity 与 numerical stability 形成选择证据；证据只用现有 observation corpus 与未标注
  确定性状态，证据不足时阻塞而不下放 O1，并冻结唯一公式及所有边界处理；
- [ ] 冻结 60 条 pilot 与 800 条 formal labels 的场景分层、seed/配额、精确模型版本、prompt、
  temperature、生成器、parser、原始响应、重试、缺失和内容哈希合同；
- [ ] 冻结 pilot 与 formal 的隔离和人工复核流程；二者都只能由研究所有者运行，pilot 不得进入
  训练，formal 生成只能在 prompt/schema 冻结后启动；
- [ ] 冻结 Pearson/Spearman 报告、`|ρ|>=0.80` 人工复核触发和“不自动改语义/删标签/重生成”
  规则；
- [ ] 定义 LLM OOD、validity、覆盖、损失、disagreement-only 日志及零有效样本的计数守恒。

## Task group O0-E：Student、schedule、执行依赖与兼容合同

- [ ] 冻结 Motion Encoder/Prior Head、三维 Semantic Encoder/Head/Adapter、late fusion、MAPPO
  action head 和 centralized critic 的输入、输出、shape 与梯度所有权；
- [ ] 明确 Student–Teacher Disagreement 只记录不加权，语义 stop-gradient 不被 PPO 绕过，
  calibration 不反传 Critic/A*；
- [ ] 冻结所有对照共用的 `λ_A(t)`/`λ_L(t)` 精确 schedule、恢复语义和日志，禁止按结果调整；
- [ ] 冻结不含 waypoint/desired-direction/teacher-trajectory 的物理观测替代合同、旧 waypoint
  兼容边界、NoWP 验证门和“执行期无需 A*”条件主张；
- [ ] 冻结三维 checkpoint metadata、strict loader、旧一维/二维历史加载与禁止权重填充迁移；
- [ ] 列出 O1 所需模块边界和依赖顺序，但不写实现代码、伪造类名或提前修改配置。

## Task group O0-F：日志、消融、论文主张与 canonical architecture

- [ ] 冻结每步/每更新/每 episode 的教师覆盖、validity、OOD、paired return、bootstrap、`ΔG`、
  EMA、权重、disagreement、fallback、耗时、内存、污染计数和失败原因 schema；
- [ ] 冻结 Reward-Calibrated KD 与 Fixed-KD 的唯一差异，登记 O2 三诊断 seed，明确不自动改变
  E1 正式预算；
- [ ] 冻结 Pure Teacher、LLM reliability、三维 semantic、NoWP、Fixed-KD 和失败降级所需消融及
  各组允许/禁止论文主张；
- [ ] 将全部批准合同整合进
  `docs/architecture/o0-reward-calibrated-heterogeneous-distillation.md`，不得含 TBD/TODO、未选
  候选、矛盾公式或依赖未跟踪文件的有效条款；
- [ ] 建立两份研究输入→canonical architecture 内容映射，确认无有效内容丢失后移除两个根目录
  未跟踪输入，保证 canonical architecture 是唯一正式方案；
- [ ] 同步三份宪章、Roadmap、TASKS 和 CHANGELOG，并完成方法—代码现状—后续实验—论文主张
  一致性审计。

## Task group O0-G：零偏差复核与研究所有者人工门

- [ ] 从头复核 requirements、plan、validation 和 canonical architecture，清除所有可由 O1
  实现者自行选择的参数、接口、公式、错误处理、命令或验收解释；
- [ ] 逐项核对 O0 验收矩阵、纯度矩阵、污染计数、数学量纲、checkpoint、日志、消融和正式
  参数预注册；
- [ ] 提供 O1 架构任务包，明确实现顺序、测试先行、短 smoke、owner-run A600 基准命令和禁止
  长任务；
- [ ] 取得研究所有者对 canonical architecture、EMA 参数、OOD 公式、三维数据合同、schedule、
  H=12 runtime/memory gate、消融和允许主张的书面批准；
- [ ] 批准前不得标记 O0 complete、进入 O1、生成正式标签或修改运行代码。
