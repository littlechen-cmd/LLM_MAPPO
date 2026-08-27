# 项目术语表

本文档用于统一研究所有者、核心架构师和项目工程师对实验、算法与工程概念的理解。它不是论文
定义章节，而是项目内部的沟通辅助材料。

阅读方式：

- **通俗解释**优先回答“这是什么”；
- **专业定义**用于精确冻结实现和论文表述；
- **本项目中的作用**说明它为什么会影响当前决策。

后续方案首次引入重要新概念时，必须同步更新本文档。向研究所有者汇报时，不得只使用未解释的
缩写或专有名词；应先给通俗解释，再在需要时补充专业定义。

## 1. 项目规划与实验治理

| 术语 | 通俗解释 | 专业定义 | 本项目中的作用 |
|---|---|---|---|
| Phase（阶段） | 一组目标相同、按顺序推进的工作。 | Roadmap 中具有明确范围、前置条件和验收标准的工作单元。 | 例如 O1 负责角色对齐实现，O3 负责未见拓扑就绪。 |
| Task Group（任务组） | 一个阶段中可以独立检查的一批小任务。 | feature spec 的可提交、可验证实施单元。 | O3-B 完成地图设计并在人工预览门暂停。 |
| Gate（门禁） | 必须通过的检查点；没通过就不能继续依赖它的工作。 | 预注册的阶段转换条件，包含输入证据、阈值与失败处理。 | O1 A600 runtime/memory gate 未通过时，O2 不能启动。 |
| Go/No-Go | “可以继续”或“必须停止/返回修改”。 | 根据预注册验收标准作出的二元阶段决策。 | 避免看到结果后临时降低标准。 |
| Canonical（规范版本） | 当前唯一被认可、大家都应引用的版本。 | 项目中作为权威来源的冻结实现、配置或文档。 | canonical architecture 是方法合同的唯一权威描述。 |
| Contract（合同） | 事先约定不能随意改变的规则。 | 对输入、输出、参数、行为、数据和失败策略的可验证约束集合。 | `H=12`、能源参数、seed 和 Teacher 边界都是合同。 |
| Freeze（冻结） | 确认后不再根据结果修改。 | 在指定 commit/版本上固定配置、数据、代码或协议。 | O3 地图批准后不能因为表现太难而换图。 |
| Preregistration（预注册） | 在看结果前先写清楚怎么做、怎么看。 | 在实验结果生成前固定假设、对照、指标、统计和失败规则。 | 防止按结果挑 seed、阈值或对照组。 |
| Roadmap（路线图） | 项目各阶段和依赖关系的总览。 | 描述 phase 状态、前置依赖与交付顺序的治理文档。 | 当前允许 O3 与等待 A600 的 O1 并行，但 O2 仍被阻塞。 |
| Feature Spec（功能规格） | 实施前写清楚“做什么、怎么做、如何验收”。 | 由 requirements、plan、validation 三个文件组成的阶段级规范。 | O3 唯一规格位于 `specs/2026-08-26-o3-unseen-topologies/`。 |
| Evidence（证据） | 用来证明任务确实完成的文件或结果。 | 可追溯到 commit、配置和命令的测试、日志、哈希、统计或人工批准记录。 | 不能只写“测试通过”，要记录具体命令和结果。 |
| Manifest（清单） | 记录一次实验或环境到底用了什么。 | 机器可读的配置、版本、hash、seed、设备和产物索引。 | O3 manifest 将固定地图 ID、双哈希和结构证书。 |
| Artifact（产物） | 程序运行后生成、供分析的文件。 | 与配置及 commit 绑定的日志、checkpoint、CSV、JSON、图像等输出。 | 优化路线产物统一放在 `artifacts/optimization/`。 |
| Provenance（来源追踪） | 能回答“这个结果从哪里来”。 | 数据、模型、配置、代码与处理链的可审计来源信息。 | 防止把旧 Phase 3 结果误当成 O3 正式证据。 |

## 2. 地图、拓扑与哈希

| 术语 | 通俗解释 | 专业定义 | 本项目中的作用 |
|---|---|---|---|
| Layout（布局） | 地图上每个格子具体放了什么。 | 网格尺寸、货架、通道、目标和站点坐标的具体空间排列。 | O3 的两个文本文件就是两份 layout。 |
| Topology（拓扑） | 不只看格子长什么样，更看哪些区域通过哪些通道连接。 | 从可通行图的节点、边、连通性、瓶颈与割点描述空间结构。 | 两张地图分别验证单通道和中央交叉结构。 |
| TopologySpec | 一张地图的“身份证和使用说明”。 | 只读的版本化记录，包含环境 ID、资源路径、使用范围、站点、预期 hash 和结构证书。 | 环境创建时依据它校验地图，训练入口也依据它拒绝 O3。 |
| Evaluation-only | 只能用于独立评估，不能参与学习或选择。 | 被禁止进入训练、标签生成、超参数调整、模型选择和路线选择的数据或环境。 | O3 地图只能在 D1 后的 E2 查看策略性能。 |
| Unseen Topology（未见拓扑） | 模型训练时从没看过的地图连接结构。 | 在训练、校准、标签/OOD 构建和模型选择中完全隔离，仅用于独立泛化评估的拓扑。 | 用于检验方法能否适应新的载货运输结构。 |
| Highway Graph | 把可载货通行的格子画成点和连接线。 | 以 `.`/`G` 为节点、四邻接为边的 loaded-transport traversability graph。 | O3 的通道和瓶颈证书都在这张图上计算。 |
| Connected Component（连通分量） | 互相能走到的一整片区域。 | 图中任意两点存在路径的最大节点集合。 | O3 highway graph 必须整体只有一个连通分量。 |
| Articulation Point（割点） | 去掉这个点，原本连通的区域就会断开。 | 删除该节点会增加图的连通分量数量的节点。 | 中央交叉地图的中心必须是预注册割点。 |
| Cut Set（割集） | 去掉这一小组位置，地图会被分区。 | 删除后使图不连通的节点或边集合。 | 用于给“瓶颈”提供可重复验证的图论证据。 |
| Bottleneck（瓶颈） | 大量路线不得不经过的狭小区域。 | 限制跨区域路径容量或选择数量的局部拓扑结构。 | O3 研究的是载货运输瓶颈，不是新增墙体。 |
| Graph Certificate（图论证书） | 一份机器可以复查的“这确实是瓶颈”的证明摘要。 | 预期节点、通道、割点、分区和连通性属性及其确定性验证结果。 | 避免只凭 PNG 的视觉印象把地图叫作窄通道。 |
| Source Hash（源文件哈希） | 地图文本的数字指纹。 | 对冻结地图原始字节计算的 SHA-256。 | 任意字符或换行被改动都会导致校验失败。 |
| Effective Layout Hash（有效布局哈希） | 环境实际理解到的地图数字指纹。 | 对解析后的尺寸、highway、goals、charging/picking stations 等语义计算的 SHA-256。 | 能发现“文本相同但运行站点配置不同”等问题。 |
| Dual Hash（双哈希） | 同时检查“文件没变”和“环境解释没变”。 | source hash 与 effective layout hash 的联合校验。 | O3 构造环境前后都必须匹配，任一漂移即停止。 |
| Hash Drift（哈希漂移） | 当前内容和批准时的数字指纹不一样。 | 实际 hash 与 manifest 中预期 hash 不一致。 | 发生时 fail closed，禁止自动接受新值。 |
| Fail Closed | 出现不确定或错误时宁可停用，也不偷偷降级继续。 | 验证失败时输出无效/拒绝执行，而不是使用宽松 fallback。 | 地图 hash、Teacher label 和 OOD 失效都遵循该原则。 |

## 3. 训练、验证与实验隔离

| 术语 | 通俗解释 | 专业定义 | 本项目中的作用 |
|---|---|---|---|
| Training（训练） | 更新神经网络参数。 | 通过优化器和训练数据计算梯度并修改模型权重。 | O3 阶段禁止训练，也禁止创建 optimizer。 |
| Smoke Test（冒烟测试） | 很短地跑一下，确认接口能接通。 | 只验证关键路径能够执行的低预算测试，不证明收敛或性能。 | O1 的 128-step smoke 不能写成算法有效性证据。 |
| Regression Test（回归测试） | 确认新修改没有破坏旧功能。 | 对既有行为合同重复执行的自动化测试。 | 每个 O3 task group 都要运行 focused 和 full regression。 |
| Static Check（静态检查） | 不跑实验，仅检查代码和文件是否合规。 | lint、schema、配置、依赖和禁止字符串等分析。 | 用于发现 O3 ID 是否泄漏进训练配置。 |
| Determinism（确定性） | 同样输入和 seed 必须得到同样结果。 | 相同初态、随机状态和动作序列产生逐字节一致的状态与输出。 | O3 用 test-only seeds `9301/9302` 验证。 |
| Seed（随机种子） | 控制伪随机过程的编号。 | 初始化伪随机数生成器、用于复现实验随机性的整数。 | 训练 seed、测试 seed 和正式评估 seed 必须隔离。 |
| Held-out Seed | 最后独立评估才使用的随机种子。 | 不参与训练、调参和路线选择的评估随机种子。 | `200–209` 只能在 D1 后使用。 |
| Same-topology Held-out Evaluation（同拓扑未见种子评估） | 地图不变，只换训练时没用过的随机任务和初态。 | 在固定 topology 下，以隔离 seed 评估同分布随机实现的鲁棒性；不等于 OOD topology generalization。 | 是优化路线唯一正式必需的泛化证据。 |
| Exploratory Stress Test（探索性压力测试） | 用更陌生、更困难的场景探查模型会在哪里失败，但不拿它决定项目是否通过。 | 预注册、非确认性、无最低性能阈值的补充评估；运行后必须完整报告，不能选择性保留。 | O3 两张未见拓扑降级后的性能角色。 |
| Fail-fast Orchestration（快速失败编排） | 先做短安全检查，只有通过才自动开始昂贵任务。 | 在同一作业中保持独立 Gate/manifest/exit state，并以先行 Gate 结果控制后续阶段是否启动。 | O1 先验收 CUDA 开销和内存，之后才允许 O2 训练。 |
| Data Leakage（数据泄漏） | 本应最后考试的内容提前影响了训练或选择。 | 评估环境或结果进入训练、表示学习、超参数、checkpoint 或方法选择。 | 若用 O3 完成率改地图，O3 就不再是真正未见测试。 |
| Control Variable（控制变量） | 比较时尽量只改变一个因素。 | 在实验组间保持相同、用于减少混杂影响的条件。 | O3 与 core 均保持 20×24、144 shelves、2 goals、8 stations 和 5 AGV。 |
| Confound（混杂因素） | 除研究因素外同时变化、让结果难以解释的东西。 | 与处理因素和结果同时相关、破坏因果归因的额外差异。 | 把 stations 从 8 改成 5 会让拓扑效果与充电资源效果混在一起。 |
| Baseline（基线） | 用来回答“相比什么更好”的参照方法。 | 与候选方法共享核心实验合同的比较组。 | MAPPO-DG 是无教师主基线，Fixed-KD 是 RC-KD 的直接对照。 |
| Runtime Overhead（运行开销） | 新机制让程序慢了多少。 | 相对基线的执行时间增量，常用 ratio 表示。 | O1 要求 H12/baseline 中位运行时间倍数不超过 3。 |
| Persistent Memory Growth（持续内存增长） | 程序越跑越占内存，可能存在对象没释放。 | 跨固定窗口呈持续上升趋势且超过预注册绝对/相对阈值的内存增长。 | O1 A600 gate 检查 CPU/GPU 内存和 branch/cache 对象。 |
| Spearman ρ | 看一个量是否总体持续上升，而不要求线性增长。 | 基于秩的单调相关系数，范围 `[-1,1]`。 | O1 用 `ρ>=0.80` 辅助判定内存是否单调增长。 |
| Checkpoint | 某个训练时刻保存的模型状态。 | 模型、优化器及训练元数据的版本化快照。 | 正式比较统一使用 final checkpoint，不能挑最好的一次。 |
| Final Checkpoint | 训练预算结束时的模型。 | 在预注册最终训练步保存的 checkpoint。 | 避免根据评估结果挑选中间模型。 |
| Paired Comparison（配对比较） | 相同随机条件下比较两种方法。 | 用相同 seed/环境随机性形成一一对应样本的统计设计。 | Fixed-KD 与 RC-KD 使用匹配 seed 和 sampler。 |
| Primary Metric（主要指标） | 最重要、决定核心结论的指标。 | 在看结果前指定、用于主要假设检验的结果变量。 | 本项目核心效用指标是每 1000 步完成任务数。 |
| Secondary Metric（次要指标） | 帮助解释结果和安全性的辅助指标。 | 不替代主要终点、用于补充诊断的指标。 | team reward、碰撞、死锁、能量死亡等属于 secondary。 |
| AUC | 把整段训练曲线的总体表现压缩成一个数。 | Area Under the Curve，对预注册横轴网格上的指标曲线积分。 | O2 用标准化吞吐 AUC 衡量样本效率，不能只比较最后一点。 |
| p-value | 在“实际没有差异”假设下，观察到当前或更极端结果的概率。 | 假设检验中衡量数据与零假设相容程度的统计量。 | 它不是“方法正确的概率”，也不能替代效应量。 |
| Confidence Interval（置信区间） | 结果差异可能落在哪个合理范围。 | 在重复抽样解释下具有指定覆盖率的参数区间估计。 | 本项目报告 95% CI，体现不确定性而非只报均值。 |
| Effect Size（效应量） | 差异到底有多大，而不只是“有没有显著”。 | 对处理效应幅度的标准化或原始尺度估计。 | 配对比较使用 paired `d_z` 辅助解释实际意义。 |
| Holm Correction | 同时做很多检验时，提高判定显著的要求。 | 控制 family-wise error rate 的逐步多重比较校正。 | 七项主要 contrasts 统一使用 Holm，避免偶然显著。 |
| Bootstrap | 从现有 seed 样本反复重抽，检查结论稳不稳。 | 有放回重采样得到统计量经验分布的非参数方法。 | 本项目用固定 seed 的 10000 次 bootstrap 作敏感性分析。 |
| Pseudoreplication（伪重复） | 把同一个模型跑出的很多 episode 错当成很多独立模型。 | 将非独立观测作为独立统计单位导致样本量虚高。 | 统计单位是训练初始化 seed，不是同一 checkpoint 的 episode。 |

## 4. 多教师蒸馏与执行接口

| 术语 | 通俗解释 | 专业定义 | 本项目中的作用 |
|---|---|---|---|
| Knowledge Distillation（知识蒸馏） | 让 Student 模仿 Teacher 提供的有用偏好。 | 使用教师输出构造辅助监督损失，将知识迁移到学生网络。 | A* 提供运动先验，LLM 提供离线语义监督。 |
| MAPPO | 多个机器人共享策略、集中训练的一种强化学习算法。 | Multi-Agent Proximal Policy Optimization，通常采用 decentralized actors 与 centralized critic。 | 本项目由 MAPPO Student 输出最终动作。 |
| Centralized Critic（集中式价值网络） | 训练时综合团队状态估计未来收益。 | 利用集中信息估计 state/team value、辅助 actor 优化的价值函数。 | Reward Calibration 只使用 detached critic bootstrap，不让它反传到 calibration。 |
| KL Divergence | 衡量 Student 动作偏好和 Teacher 偏好相差多远。 | 两个概率分布之间的非对称信息差异度量。 | A* KD loss 用 KL 拉近 Student motion logits 与有效 motion prior。 |
| Team Reward | 整个 AGV 团队共同获得的任务反馈。 | 多智能体环境在一步或一段轨迹上的共享回报。 | Reward Calibration 比较 H-step discounted team return。 |
| Student | 最终真正执行动作的 MAPPO 策略。 | 接收观察并输出离散动作分布的学习策略。 | 优化路线执行期 Student planner query 必须为 0。 |
| Teacher | 训练时提供辅助信息，但不一定在执行期控制动作。 | 产生目标分布或语义标签的监督源。 | Pure Motion A* 和离线 LLM 是两类异构教师。 |
| Pure Motion Teacher | 只回答“几何上向哪边移动更合理”的 A*。 | 不读取任务优先级、其他机器人身份/计划或 reward 的局部 bounded A* prior。 | 它不决定谁让行、是否等待或谁有通行权。 |
| Motion Prior（运动先验） | A* 对几个移动方向的软偏好。 | 根据 root-action-conditioned planning costs 得到的动作概率分布。 | 只监督 FORWARD/LEFT/RIGHT，不监督 NOOP。 |
| Valid Mask（有效性掩码） | 标出哪些机器人当前确实有可信 Teacher 标签。 | 每机器人二值变量，控制对应 KD 项是否进入损失。 | 无效机器人权重为 0，不能用默认标签替代。 |
| Hard Action Mask | 物理上不允许的动作必须彻底屏蔽。 | 在策略和 Teacher 分布中将非法动作概率严格设为 0 的布尔掩码。 | Fixed/RC 和 shadow branches 使用相同 mask。 |
| DirectGoal Observation | 直接告诉 Student 目标相对方向，但不让 A* 算 waypoint。 | 613D observation 中九位 geometry block 包含归一化 goal delta，其余槽保持冻结。 | 支持执行期不查询 planner 的主方法。 |
| NoGoalHint | 把目标几何提示清零的诊断观察。 | 保持 613D 宽度，但九位 geometry block 全零。 | 只用于诊断目标提示敏感性，不能称为 A* 消融。 |
| Semantic View v3 | 给 LLM 和语义 Student 使用的精简状态描述。 | 61D、去 ID、方向化的三维语义输入表示。 | 三维标签和 OOD reliability 都以此空间为准。 |
| OOD（分布外） | 当前状态和标签数据见过的情况差得有多远。 | Out-of-Distribution；查询相对 reference corpus 的分布偏离。 | 越偏离，LLM 标签蒸馏权重通常越低。 |
| OOD Reliability | 对“这个离线标签在当前状态是否可信”的折扣。 | 由冻结 61D 距离公式得到的 `[0,1]` 共享权重。 | 与 validity 相乘；为 0 时 fail closed。 |
| Reward Calibration | 用短期真实任务收益判断 A* 标签整体值不值得学。 | 对同一快照运行 H-step Student/A* paired shadows，依据 return difference 生成权重。 | 只改变标签权重，不反向修改 A* 标签。 |
| Shadow Rollout | 从真实状态复制两个短期“假想未来”作比较。 | 不污染真实轨迹的分支环境模拟，使用共同外部随机状态。 | Student branch 与 Teacher branch 固定 H=12。 |
| EMA | 对连续观测量做平滑，减少单次波动。 | Exponential Moving Average，按固定 decay 更新统计量。 | 用于归一化 detached `ΔG`，参数必须冻结。 |
| Planner Query | 调用 A* 或路径规划器计算 waypoint/路径。 | 执行期间对 planner 接口的一次实际调用。 | 优化路线 Student evaluation/visualization 必须记录为 0。 |

## 5. 常见对照组

| 术语 | 通俗解释 | 专业定义 | 本项目中的作用 |
|---|---|---|---|
| MAPPO-DG | 不使用任何 Teacher 的 DirectGoal MAPPO。 | A* KD=false、LLM KD=false 的主基线。 | 用于判断多教师方法相对纯 MARL 的增量。 |
| Fixed-KD | 只要 A* 标签有效且被采样，就按固定权重学习。 | `c_A_reward=1` 的 A* KD 对照。 | 与 RC-KD 的唯一优化差异应是 reward confidence。 |
| RC-KD | 根据短期 team return 调整 A* 标签权重。 | 使用 detached reward-calibration confidence 的 A* KD。 | 检验 Reward Calibration 是否有增量价值。 |
| RuleKD-v3 | 用冻结规则产生三维语义标签。 | 与 LLM 标签共享状态、retrieval、validity/OOD 合同的确定性语义教师基线。 | 比较语义教师来源，而不是混入规则调度器。 |
| ShuffleKD-v3 | 故意打乱状态和标签对应关系。 | 在场景分层内对三维联合标签做无 fixed-point 确定性置换。 | 检验语义标签与具体状态对应是否重要。 |
| NoOOD-v1 | 不按分布外程度降低 LLM 标签权重。 | 对有效 LLM record 固定 `OOD reliability=1`。 | 诊断 OOD reliability 的作用。 |

## 6. 术语维护规则

1. 方案、任务包或实验协议首次出现研究所有者可能不熟悉的概念时，先更新本文档；
2. 同一概念只保留一个 canonical 名称；旧称必须标注弃用或历史范围；
3. 术语解释不得扩大论文允许主张，例如“拓扑就绪”不能解释成“已证明泛化”；
4. 公式、参数或实现细节仍以 canonical architecture 和对应 feature spec 为准；本文档只帮助理解，
   不覆盖冻结合同；
5. 每次重要术语更新在 `CHANGELOG.md` 留痕。
