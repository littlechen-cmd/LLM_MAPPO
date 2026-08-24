# Requirements — O0 A* 教师算法与架构重设计

## 1. 阶段目标

O0 是优化路线的纯架构阶段。它必须在不修改运行代码、不生成正式标签、不启动训练或长评估的
前提下，审计当前教师—环境—buffer—MAPPO 全链路，并冻结唯一可供 O1 无偏实现的架构合同。

方法主线固定为：

```text
Pure A* Motion Teacher
        +
Offline 3D LLM Semantic Teacher
        +
Reward-Calibrated Heterogeneous Distillation
        +
MAPPO Final Action Policy
```

O0 完成只表示架构、接口、数学定义、日志、消融和失败门获得研究所有者书面批准，不表示实现
完成、性能改善、训练收敛或 Gate O1/O2 通过。

## 2. 输入与治理

- 当前分支从 `codex/optimization` 的 P0 最终基线派生；
- 研究输入有两个根目录未跟踪版本：用户最初指定且更新时间较新的 `方案文档.md`，SHA-256
  `261A33536C2E53EEEBDFF08F49DD2A42217A0AAE7324C154CFA7785918ABD2B0`；以及长文件名排版版本
  `基于异构多教师知识蒸馏与MAPPO的多机器人路径规划与协同方案.md`，SHA-256
  `12DB077897D25DA999746CF044C28497BDF2C5C5675B2705F6598CCE780B70E7`。二者章节结构和方法内容
  相同，差异限于 Markdown 项目符号与三处范数 LaTeX 排版；canonical architecture 使用正确
  `\lVert\cdot\rVert` 语义；
- 输入文档不是活动宪章，也不是可直接实施的规格；O0 必须把已批准内容、审计修正和精确冻结
  参数整理为唯一 canonical architecture；
- canonical architecture 路径固定为
  `docs/architecture/o0-reward-calibrated-heterogeneous-distillation.md`；
- canonical architecture 获批并纳入 Git 后，两份根目录未跟踪输入不得继续作为并列正式方案；
  应在逐项内容映射与哈希审计后移除两个根目录副本，不得留下多个活动版本；
- `specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md`、`TASKS.md` 与
  `CHANGELOG.md` 必须和 O0 最终合同同步。
- O0-A 至 O0-F 每个 Task Group 都是独立人工检查点：完成该组全部任务、验证、计划状态、
  CHANGELOG 和聚焦 commit 后必须停止并向研究所有者交付证据；只有研究所有者书面批准继续
  后才能开始下一组。禁止在等待审核时预做、暂存或提交下一组产物；O0-G 是最终批准门。

## 3. 范围

### 3.1 当前链路审计

必须逐字段审计并绘制以下数据流：规则层目标、A* waypoint、A* preference、时空预约、
`AStarExpert` 的 priority/yield/coordinator、hard action mask、环境 observation、waypoint reward、
rollout buffer、A* KL、离线 LLM 检索、语义 head、集中式 Critic、checkpoint、训练/评估/执行入口。

审计必须区分：

- 真实环境状态与 Teacher 派生特征；
- Pure Motion label 与高层调度/协调输出；
- 训练期教师监督、执行期 actor 输入及非学习启发式 A* 基线；
- 当前一维/二维历史 checkpoint 合同与新三维合同。

### 3.2 Pure A* Motion Teacher

A* 只接收规则层已经确定的合法目标和当前物理状态，只输出每机器人局部运动动作偏好及有效性
掩码。它不得：

- 分配或更换任务、优先级、充电目标或 parking 目标；
- 调用 `_yielding_agents` 或等价高层让行规则；
- 通过机器人顺序、任务优先级或预约顺序分配通行权；
- 使用最终 coordinator 输出作为教师标签；
- 输出逐步最终动作或改写 MAPPO 动作；
- 读取 LLM label、Student disagreement、reward calibration 或未来回报来生成标签。

输出至少包括：`motion_preferences[N,A]`、`valid_mask[N]`、逐机器人失败原因、规划耗时、展开
节点数和规划窗口元数据。`TOGGLE_LOAD` 等非运动交互不得获得运动概率质量；非法动作质量必须
在蒸馏前为零。正式 Motion Prior 不使用固定 label smoothing，而由
root-action-conditioned short-horizon planning cost `C_A*^K(s,a)` 构造。每个合法 root action
在首动作固定为 `a` 后，以窗口内累计 motion cost 加末态 `h_static` 为目标；三个 root branch
必须保留 provenance，并在单次 bounded search 与同一总 512-expansion budget 内求解，禁止分别
启动三个完整 A*。未得到有效 continuation 的 root cost 为正无穷且概率质量为零。

有限 root cost 在每个状态内按 min-max 归一化；所有有限 cost 相等时归一化值全为零。随后使用
固定 `tau_motion=1.0` 的 Boltzmann transform，并仅在有限 root 集上归一化。只要至少一个 root
具有有限 cost，该机器人标签有效；若没有有限 root、标签非有限或输出校验失败，则对应机器人
`valid_mask=0`。共享预算耗尽只会把尚未认证的 root 置为正无穷；只有最终没有任何有限 root 时
才产生机器人级 `budget_exceeded`。

搜索规模、耗时、路径长度、绕行和局部冲突只作为诊断日志，不组合为连续
`c_A_search`。Reward Calibration 只能评价已经独立生成的标签，不得反向影响标签生成过程。
search entropy、path-length confidence、Student disagreement 或任何其他量都不得成为额外权重。
Motion cost 只决定标签内部动作偏好，整体蒸馏权重仍唯一为
`w_A,i(t)=lambda_A(t)m_A,i^valid c_A^reward`。论文必须称其为
“root-action-conditioned short-horizon planning cost”，不得表述为学习得到的 RL Q-value。

### 3.3 Paired Shadow Reward Calibration

正式 horizon 唯一固定为 `H=12`。每次 calibration 从同一完整状态快照和同一外部随机状态
派生两个隔离 shadow：

- Student shadow：每步使用当前 MAPPO 的 deterministic argmax；
- A* shadow：有效机器人使用 Pure Motion Teacher argmax；无效机器人使用 A* shadow 当前状态
  下 MAPPO 计算的 deterministic action，不得复制已分歧 Student shadow 中的动作；
- 两个 shadow 使用完全相同的 hard-mask 生成函数与规则；初始 mask 必须相同，状态分歧后的
  mask 按各自 shadow 当前状态计算，禁止把一个分支的 mask 强加到另一个物理状态；A* 可按
  A* shadow 状态滚动重规划；
- 外部随机事件使用从共同快照派生、按 episode/step/event key 寻址的 common-random-number
  stream，不能依赖两个分支碰巧以相同顺序消费可变长度 RNG；
- shadow 不得修改真实环境、真实 worker RNG、教师 cache、rollout buffer 或实际训练轨迹。

团队级配对优势定义为：

$$
\Delta G = \left(\sum_{k=0}^{H-1}\gamma^k r^A_k +
\gamma^H V_\phi(S^A_{t+H})\right)
- \left(\sum_{k=0}^{H-1}\gamma^k r^\pi_k +
\gamma^H V_\phi(S^\pi_{t+H})\right).
$$

每个 shadow 独立处理 terminal：终止分支从其终止时刻停止累计且不 bootstrap，另一分支继续到
自身 terminal 或 H=12。Critic 只估计未终止分支在 H 步干预后重新交还 Student policy 的
continuation value，必须 detach，不能接收 calibration 梯度或教师监督。

团队级 `c_A_reward` 可共享，但逐机器人 `valid_mask` 必须保留。有效权重为：

$$
w_{A,i}(t)=\lambda_A(t)m^{valid}_{A,i}c_A^{reward}.
$$

`c_A_reward` 使用 detached `ΔG` EMA 标准差归一化和正优势门控。O0 必须冻结 EMA decay、
minimum scale、initialization sample count、更新时机、数值截断和 checkpoint 恢复语义。初始化
样本数未达到前 `c_A_reward=0`，禁止 Fixed-KD warm start。非正优势不得产生 A* 蒸馏权重。

### 3.4 三维离线 LLM Semantic Teacher

新 schema 固定为：

1. `task_persistence`：当前状态下继续当前任务的合理程度；
2. `yielding_preference`：局部冲突中主动等待/让行的语义倾向；
3. `coordination_risk`：局部交互导致冲突、拥堵、死锁或协作失败的风险。

三维值均位于 `[0,1]`。LLM 不输出低层动作、路径、任务分配或强制让行决策。标签使用整记录
validity：任一字段缺失、非有限、越界或不满足 schema 时整条记录无效，不做逐维蒸馏门控。

LLM reliability 唯一形式为：

$$
c_L=c_{validity}\times c_{OOD}.
$$

不得保留、记录或默认为满分的 consistency 乘子。OOD reliability 是整条三维记录共享权重。
O0 只能比较两个预注册候选：

- 标准化特征空间中的截断指数距离；
- 基于标签集 leave-one-out 距离分位数的分段线性权重。

选择依据只允许 coverage、distance-to-weight monotonicity 和 numerical stability；不得依据 MAPPO
训练性能或下游 reward。O0 必须冻结唯一公式、特征标准化、近邻数、参考分位数/截断、零方差
处理、无邻居降级和日志字段。选择证据只能来自现有已存 observation corpus 和由冻结现有场景
生成器产生的未标注确定性状态，不得生成新 LLM label；若证据不足以唯一选择，O0 必须阻塞并
请求研究所有者裁定，不能把选择权下放 O1。

新数据合同固定为 60 条 prompt/schema pilot 与 800 条正式标签。pilot 不进入训练。沿用现有
标签生成链路的基础模型，但 O0 必须追溯并冻结精确 model ID/version、prompt、temperature、
scenario generator、parser、seed/配额、原始响应保留和失败处理。正式数据只能在上述合同冻结
后由研究所有者运行生成。60 条 pilot 和 800 条 formal labels 均由研究所有者运行；Codex 只
准备冻结命令、检查产物和分析结果。

标签审计同时报告 Pearson 与 Spearman 相关系数；任意维度对 `|ρ|>=0.80` 只触发人工复核，
不得自动删除标签、修改语义定义或按训练结果重生成数据。

### 3.5 Student、损失与 schedule

Student 使用 Motion Encoder/Motion Prior Head 与三维 Semantic Encoder/Semantic Head/
Semantic Adapter 的异构分支，在 Student 内 late fusion 后由 MAPPO 输出最终动作。A* loss 只
监督 Motion Representation 与 Motion Prior Head；LLM loss 只监督 Semantic Encoder/Head；
PPO 不得穿过 detached 三维 semantic output，但可训练 Semantic Adapter 和最终策略。

Student–Teacher Disagreement 完全移出优化权重，只保留诊断日志。不得使用
`q=c(1+ρd)` 或等价形式影响梯度。

`λ_A(t)` 与 `λ_L(t)` 保留。O0 必须冻结两者完全确定、可恢复的预注册 schedule；所有相关主
方法与对照使用完全相同 schedule，禁止按训练结果搜索或修改。

Fixed-KD 与 Reward-Calibrated KD 除 reward calibration 外必须完全一致：

$$
w_{A,fixed,i}(t)=\lambda_A(t)m^{valid}_{A,i},
$$

$$
w_{A,RC,i}(t)=\lambda_A(t)m^{valid}_{A,i}c_A^{reward}.
$$

Fixed-KD 不得改变 LLM reliability、network、data、schedule、reward、mask、seed、预算或其他
训练合同。O2 使用 3 个预注册诊断 seed；这不自动改变 E1 的 8-seed 正式预算。

### 3.6 执行期依赖与 checkpoint

新 Motion Branch 不得以 A* waypoint、desired direction 或 trajectory 作为必须输入。O0 必须
冻结无 Teacher 派生量的物理观测替代合同，以及当前 waypoint 输入/reward 的兼容与消融边界。

“执行期无需 A*”是条件主张：只有后续 NoWP/无 A* 证据通过冻结门槛后才能成立；O0/O1 不得
预先宣称已经摆脱 A*。启发式 `Heuristic-Dispatcher+A*` 基线仍可独立使用 A*，不等同于 Student
执行依赖。

三维 checkpoint 与旧一维/二维 checkpoint 严格隔离：

- 旧 checkpoint 继续支持历史评估；
- 禁止复制、填充或映射旧 semantic 权重到三维结构；
- 新三维训练从新架构初始化；
- 新 checkpoint 必须保存 `semantic_schema_version`、有序维度名称、可靠性合同、EMA 状态、
  horizon、schedule 和 observation contract；不兼容加载必须给出明确错误。

### 3.7 O1 runtime gate

O1 必须以 `H=12` 通过确定性功能 smoke。`H=4` 只用于故障定位和 overhead 对照，不能成为正式
fallback。O0 必须冻结 A600 短基准的步数、并行度、重复次数、计时边界、warm-up、内存测量和
基线命令。

若 H=12 的每真实环境步 runtime 超过预注册无 shadow 基线的 `3×`，或满足 O0 冻结定义的
持续 memory growth，O1 判定 No-Go 并返回 O0；禁止静默降低 horizon。

## 4. 不在范围内

- 不修改 Python 运行代码、配置、环境语义、reward 或 checkpoint；
- 不生成 60 条 pilot 或 800 条正式标签，不调用在线 LLM；
- 不启动训练、长评估、长回放或 KL/schedule/horizon 搜索；
- 不实现环境 fork/snapshot、Pure Motion Teacher、三维网络或新 buffer；
- 不实现两个未见拓扑；
- 不修改稳定路线 S1/S2；
- 不根据既有或新训练性能选择 OOD 公式、EMA 参数或论文主张。

## 5. 允许与禁止主张

O0 获批后只允许声称“已冻结可实施的 Reward-Calibrated 异构多教师架构”。不得声称方法已
实现、性能优于基线、标签三维独立、执行期无需 A*、样本效率提升、跨拓扑泛化或达到二区水平。

## 6. 开放问题

无。EMA 数值、OOD 唯一公式、精确模型版本、schedule、memory growth 判据和 A600 短基准是
O0 明确要求产出的设计结果，不是交由 O1 实现者自由决定的开放项。
