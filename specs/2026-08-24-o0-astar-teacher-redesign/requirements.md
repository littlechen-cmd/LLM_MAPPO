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
O0-C 第 3.3 节的 `lambda_A(t) * m_A,i^valid * m_calib(t)`，RC-KD 再唯一乘
`c_A_reward(t)`。论文必须称其为
“root-action-conditioned short-horizon planning cost”，不得表述为学习得到的 RL Q-value。

### 3.3 Paired Shadow Reward Calibration

正式 horizon 唯一固定为 `H_reward=12`。每次 calibration 从版本化 canonical state snapshot 和
同一外部随机状态派生两个预构造、隔离的 shadow。snapshot 必须覆盖实体、任务队列内部状态、
动态入库、计步、能量/规则、adapter metrics、wrapper、环境及递归 space RNG；导入分支不得调用
`reset` 或实体构造器。真实 planner/Teacher cache 只读，shadow 使用各自临时 cache 并在结束后
丢弃。导入前后 canonical hash 必须相等，shadow 完成后真实 state/RNG hash 必须保持不变。

- Student shadow：每步使用当前 MAPPO 的 deterministic argmax；
- A* shadow：有效机器人使用 Pure Motion Teacher argmax；无效机器人使用 A* shadow 当前状态
  下 MAPPO 计算的 deterministic action，不得复制已分歧 Student shadow 中的动作；
- 两个 shadow 使用完全相同的 hard-mask 生成函数与规则；初始 mask 必须相同，状态分歧后的
  mask 按各自 shadow 当前状态计算，禁止把一个分支的 mask 强加到另一个物理状态；A* 可按
  A* shadow 状态滚动重规划；
- 外部随机事件使用 `crn-v1`，按
  `(episode_seed, real_step, shadow_offset, event_type, event_slot)` 寻址；候选集合按事件 key 与
  canonical candidate ID 确定性哈希排序，不能依赖两个分支碰巧以相同顺序消费可变长度 RNG；
- shadow 不得修改真实环境、真实 worker RNG、教师 cache、rollout buffer 或实际训练轨迹。

团队级配对优势定义为：

$$
\Delta G = \left(\sum_{k=0}^{H_{reward}-1}\gamma^k r^A_k +
\gamma^{H_{reward}} V_\phi(S^A_{t+H_{reward}})\right)
- \left(\sum_{k=0}^{H_{reward}-1}\gamma^k r^\pi_k +
\gamma^{H_{reward}} V_\phi(S^\pi_{t+H_{reward}})\right).
$$

每个 shadow 独立处理 `terminated/truncated/deadlock`：终止分支从其终止时刻停止累计且不
bootstrap，另一分支继续到自身 terminal 或 H=12。Critic 只估计未终止分支在 H 步干预后重新
交还 Student policy 的 continuation value，使用相同 policy snapshot、`eval/inference_mode`，必须
detach，不能接收 calibration 梯度或教师监督。`gamma` 绑定同一 PPO 配置字段，不允许独立值。

统一 calibration selection mask `m_calib(t)` 由 `calibration-sampler-v1` 产生：对包含 run/episode/
environment/step 身份的 canonical key 取 SHA-256 前 8 字节大端无符号整数，仅余数模 16 为 0 的
状态被选择。选择不读取 reward、Teacher cost、Student、训练性能或 `c_A_reward`。未选择状态
不运行 shadow，Fixed-KD 与 RC-KD 的 A* KD 权重均为 0。选择状态只有在至少一台机器人
`valid_mask=1` 时运行 paired shadow。两组运行完全相同的 shadow、日志、EMA 与 sampling density。

团队级 `c_A_reward` 可共享，但逐机器人 `valid_mask` 必须保留。Fixed-KD 与 RC-KD 权重分别为：

$$
w_{A,Fixed,i}(t)=\lambda_A(t)m^{valid}_{A,i}m_{calib}(t),
$$

$$
w_{A,RC,i}(t)=\lambda_A(t)m^{valid}_{A,i}m_{calib}(t)c_A^{reward}(t).
$$

唯一优化差异必须是 `c_A_reward`。`c_A_reward` 使用 detached `ΔG`：前 64 个有限样本以
Welford population variance 初始化，期间及第 64 个样本均为 0；从第 65 个样本开始，先用此前
统计计算 `clip(max(ΔG,0)/max(sigma_EMA,1e-3),0,1)`，再以 decay `0.99` 更新 exponential
mean/variance。多环境按 `(real_step, env_index)` 更新，禁止按 worker 返回顺序。非有限样本权重
为 0、不计入初始化、不更新 EMA，并作为 O1 No-Go 故障。checkpoint 必须严格保存并恢复 schema、
count/mean/variance/initialized、sampler、H、decay、minimum scale 和 clipping；禁止缺失时重置。

O1 overhead gate 固定在 A600、12 workers：baseline/H4/H12 各用 16 vector-step warm-up、128 个
measured vector steps、5 个 fresh-process repeats，计入完整 collection/update、snapshot、shadow、
Teacher、Critic、EMA 与内存日志，排除进程启动和磁盘 flush；CUDA 计时边界必须同步。
`median(H12)/median(baseline)<=3.0`。memory gate 在 2 个 warm-up window 后记录 10 个同长度
window；CPU RSS 或 CUDA allocated 的末三次中位数较首三次增长超过 `max(64 MiB,5%)` 且
Spearman `rho>=0.80` 即为持续增长。任一 gate 失败必须返回 O0；H4 只作诊断，不得替代 H12，
也不得静默修改 1/16 sampler。

### 3.4 三维离线 LLM Semantic Teacher

新 schema 固定为：

1. `task_persistence`：当前状态下继续当前任务的合理程度；
2. `yielding_preference`：局部冲突中主动等待/让行的语义倾向；
3. `coordination_risk`：局部交互导致冲突、拥堵、死锁或协作失败的风险。

三维值均位于 `[0,1]`，使用 canonical architecture 第 11.2 节的固定五点量表 anchors。高
`yielding_preference` 表示更倾向让行；三维之间不存在互补、蕴含或代数推导关系。LLM 不输出
低层动作、路径、任务分配、通行权裁决或强制让行决策。输出必须正好包含三个 score 与对应三个
reason；reason 只作 audit metadata。任一字段缺失、额外、非有限、越界或不满足 schema 时整条
记录无效，不做逐维蒸馏门控。

LLM 输入和 OOD 距离共同使用 `semantic-view-v3`。该视图排除 AGV/scenario ID、`scenario_type`、
raw/full observation、A*、reservation、coordinator、reward、Student 与 calibration；保留 focal
任务/资源/朝向/局部静态几何，以及三个匿名邻居的相对位置、载货、电量、死亡、任务优先级、
目标阶段和充电站状态。邻居选择距离、无 ID tie-break、机器人中心坐标、padding、mask、类别顺序
和唯一 61D 数值编码全部以 canonical architecture 第 11.3 节为准。

LLM reliability 唯一形式为：

$$
c_L=c_{validity}\times c_{OOD}.
$$

不得保留、记录或默认为满分的 consistency 乘子。OOD reliability 是整条三维记录共享权重。
O0 只能比较两个预注册候选：

- 标准化特征空间中的截断指数距离；
- 基于标签集 leave-one-out 距离分位数的分段线性权重。

选择依据只允许 coverage、distance-to-weight monotonicity 和 numerical stability；不得依据 MAPPO
训练性能或下游 reward。旧 615D full observation 审计只算 preliminary evidence。O0-D 已在从
历史 400 条 corpus 精确重建的 61D `semantic-view-v3` 上完成五折比较，并选择第 11.4 节的截断
指数公式；正式 reference set 仅包含 validity=1 的 formal records。特征标准化、k=3 距离、
leave-one-out 分位数、零方差、无邻居和非有限降级均不得由 O1 修改。

新数据合同固定为 60 条 prompt/schema pilot 与 800 条正式记录，五类场景分别为 12/160 条。
pilot 不进入训练。首选 `deepseek-v4-flash`；只有 60 条完整 pilot 达到第 11.6 节预注册的系统性
失败条件时，才允许保持场景、prompt/schema 和所有参数不变，将整组 pilot 切换为
`deepseek-v4-pro`。Pro 仍失败则 O0-D No-Go。模型为 non-thinking、temperature=0、JSON object、
max_tokens=1024；prompt、生成器、parser、seed/配额、原始请求/响应、重试、缺失、fingerprint 和
内容哈希合同见第 11.5 至 11.8 节。

60 pilot 和 800 formal 只能由研究所有者运行；Codex 只准备命令、检查产物和分析结果。formal
生成期间 `system_fingerprint` 变化必须暂停；一个 formal dataset 只能含一个 fingerprint。正式
记录禁止逐条人工改分、针对坏标签重试或选择性删除。若 validity 或确定性 100 条分层复核未
达到第 11.8 节阈值，整个 formal dataset No-Go，返回 O0-D、升级 prompt/schema version、重跑
pilot，并重新生成完整 800 条。

标签审计同时报告 Pearson 与 Spearman 相关系数；任意维度对 `|ρ|>=0.80` 只触发人工复核，
不得自动删除标签、修改语义定义或按训练结果重生成数据。相关性本身不属于 dataset No-Go，
只能形成具名 owner 审核结论。

### 3.5 Student、损失与 schedule

Student 固定使用 `o0-student-v1`。canonical 5-AGV 环境的物理观测为
`x_phys[N,613]`，三维语义视图为 `x_sem[N,61]`。网络 shape 固定为：

- Motion Encoder：`613 -> 128 -> 64`，两层后均为 ReLU，输出
  `z_motion[N,64]`；
- Motion Prior Head：`64 -> 3`，顺序固定为
  `[FORWARD,LEFT,RIGHT]`，只用于 A* KD；
- Semantic Encoder：`61 -> 128 -> 64`，两层后均为 ReLU；
- Semantic Head：`64 -> 3 -> sigmoid`，顺序固定为
  `[task_persistence,yielding_preference,coordination_risk]`；
- Semantic Adapter：接收 detached 三维输出，执行 `3 -> 16 -> ReLU`；
- late fusion：拼接 `z_motion[N,64]` 与 adapter 输出 `[N,16]`；
- MAPPO Action Head：线性 `80 -> 5`，输出项目动作枚举顺序的最终 logits；
- Centralized Critic：只接收 `[B,N,613]` 物理观测，逐机器人
  `613 -> 128 -> 128`，4-head attention 后按机器人 mean pooling，再经
  `128 -> 256 -> 128 -> 1` 输出团队 value `[B]`。

梯度所有权固定如下：PPO actor loss 可更新 Motion Encoder、Semantic Adapter 与最终 Action
Head；A* KD 只更新 Motion Encoder 与 Motion Prior Head；LLM semantic loss 只更新 Semantic
Encoder/Head；value loss 只更新 Critic。Motion Prior Head 不进入最终动作路径；Semantic Head 的
输出必须在进入 Adapter 前 stop-gradient，因此 PPO/A* 均不能绕回 Semantic Encoder/Head。
Teacher target、reliability、planning cost、`Delta G`、EMA 或 disagreement 不得作为 Critic 输入。
Calibration 只在 `eval/inference_mode` 读取 detached Critic value，不反传 Critic 或 A*。

A* KD 只在运动三动作 support 上比较 Teacher prior 与 Motion Prior：

$$
Z_A=\sum_i m_{calib}(t)m^{valid}_{A,i},
$$

$$
L_A=\begin{cases}
\lambda_A(t)\dfrac{\sum_i m_{calib}(t)m^{valid}_{A,i}c_i
D_{KL}(p_{A^*,i}\Vert p_{motion,i})}{Z_A},&Z_A>0,\\
0,&Z_A=0,
\end{cases}
$$

其中 Fixed-KD 的 `c_i=1`，RC-KD 的 `c_i=c_A_reward(t)`。分母不得包含 `lambda_A` 或
`c_A_reward`，避免把正式缩放抵消。LLM loss 对三维先取逐记录 mean squared error，再定义：

$$
Z_L=\sum_i c_{validity,i},
$$

$$
L_L=\begin{cases}
\lambda_L(t)\dfrac{\sum_i c_{validity,i}c_{OOD,i}
\operatorname{MSE}_3(\hat y_i,y_i)}{Z_L},&Z_L>0,\\
0,&Z_L=0.
\end{cases}
$$

Student–Teacher Disagreement 完全移出优化权重，只保留诊断日志。不得使用
`q=c(1+ρd)` 或等价形式影响梯度。

共同 schedule 固定为 `linear-env-step-v1`。令 checkpoint 中严格单调的真实环境交互总数为
`t=global_env_steps`，令当前证据族 manifest 中预注册、正整数且所有匹配组相同的交互预算为
`B=schedule_total_env_steps`：

$$
p(t)=\min(\max(t/B,0),1),\qquad
\lambda_A(t)=0.05(1-p(t)),\qquad
\lambda_L(t)=0.10(1-p(t)).
$$

每个 optimizer update 开始前按该 update 已收集完成的 `global_env_steps` 计算一次，整个 update
保持不变。vector worker 每产生一个真实环境 transition 计一步，shadow transition 不计入；因此
并行度和 episode 长短不改变 schedule 语义。所有主方法、Fixed/RC 对照及消融记录相同名义
schedule；缺少某教师的组只把对应有效 mask 置零，不能换 schedule。checkpoint 必须保存
schedule version、两个初值、`B/t/p/lambda_A/lambda_L`，恢复时先逐字段校验再从保存的 `t`
继续。禁止重置、按 wall-clock/episode 推断、按结果搜索或修改。

Fixed-KD 与 Reward-Calibrated KD 除 reward calibration 外必须完全一致：

$$
w_{A,fixed,i}(t)=\lambda_A(t)m^{valid}_{A,i}m_{calib}(t),
$$

$$
w_{A,RC,i}(t)=\lambda_A(t)m^{valid}_{A,i}m_{calib}(t)c_A^{reward}(t).
$$

Fixed-KD 不得改变 calibration sampler、shadow rollout、EMA/logging、LLM reliability、network、
data、schedule、reward、mask、seed、预算或其他训练合同。未被 sampler 选择的状态两组 A* KD
均为 0；唯一优化差异是 RC 公式的 `c_A_reward`。O2 使用 3 个预注册诊断 seed；这不自动改变
E1 的 8-seed 正式预算。

### 3.6 执行期依赖与 checkpoint

新物理观测固定为 `direct-goal-observation-v1`。canonical 5-AGV 环境继续保持 613 维：现有
575 维 raw observation、7 维 own state、4 维自身朝向、15 维 nearby 和 3 维 global 字段不变；
原 waypoint block 的 9 个位置改为：

```text
[normalized_goal_dx, normalized_goal_dy, 0, 0, 0, 0, 0, 0, 0]
```

目标坐标只由规则层既有 `_target_for_agent` 语义解析；位移分别除以 `max(width-1,1)` 与
`max(height-1,1)`，已经在目标点时为 0。后七位是版本化 reserved zeros，不得填入 A* waypoint、
desired direction、path/trajectory、reservation、coordinator 或可学习预测。构造 observation
不得调用 planner。O3 真正未见拓扑必须保持同一 613 维 schema，否则在进入训练前 No-Go。

当前 `waypoint_reward` 实际按规则目标的 Manhattan 距离是否下降计算，并未调用 A*。O1 只允许
行为保持地把正式名称改为 `direct_goal_progress_reward`，数值、时机和公式不变；旧字段仅作为
历史配置 alias。优化路线 NoWP 也保留相同 reward，以便只消融 observation cue，不把 reward
变化混入比较。

历史一维/二维 checkpoint 继续使用隔离的 legacy waypoint observation builder 执行历史评估，
但该 builder 不得被新三维 policy、训练、评估或可视化入口调用。新 loader 遇到 legacy
observation/schema 必须拒绝恢复。优化路线 NoWP 保留配置别名，正式 schema 名为
`no-geometric-goal-hint-v1`：保持 613 维但把上述 9 位全部置零；它是三 seed 诊断消融，不要求
性能等同于主方法。

“执行期无需 A*”是条件主张：只有后续 NoWP/无 A* 证据通过冻结门槛后才能成立；O0/O1 不得
预先宣称已经摆脱 A*。启发式 `Heuristic-Dispatcher+A*` 基线仍可独立使用 A*，不等同于 Student
执行依赖。该主张必须同时满足：Student policy 评估/可视化的 planner query count 为 0；把
planner 替换为任何调用即抛错的测试替身后 DirectGoal 与 NoWP 都能完成端到端短运行；DirectGoal
通过 O2/O3/E1 对应冻结性能门；论文明确 A* 仍在训练期提供 Motion Teacher。NoWP 的性能只作
诊断，不能单独决定该主张。

新 checkpoint schema 固定为 `o0-student-checkpoint-v1`，只允许在完整 optimizer update 后且
rollout buffer 为空时保存可恢复 checkpoint。必须保存：

- architecture/observation/semantic/checkpoint/schedule/sampler/Teacher 版本和有序维度/动作名；
- model 与 optimizer state、`global_env_steps/update_count/completed_episodes`；
- canonical config/manifest/Git commit、环境 ID、layout hash、能源参数与 reward schema/hash；
- semantic dataset/manifest/index/prompt/parser/OOD 参数及内容 hash；
- `K_motion=12`、`H_reward=12`、expansion budget、1/16 sampler；
- EMA schema 与 `count/mean/variance/initialized/decay/minimum_scale/init_count/clipping`；
- Python/PyTorch/CUDA 版本和 Python/NumPy/PyTorch CPU/CUDA RNG state。

strict loader 必须先验证所有必需字段、版本、顺序、shape 和 hash，再执行 strict model/optimizer
state load；不兼容错误必须具名指出字段，禁止根据 tensor shape 猜测后继续。三维 checkpoint 与
旧一维/二维 checkpoint 严格隔离：

- 旧 checkpoint 继续支持历史评估；
- 禁止复制、填充或映射旧 semantic 权重到三维结构；
- 新三维训练从新架构初始化；
- legacy loader 只接受历史 1D/2D schema，不能训练或恢复 `o0-student-v1`；
- 新 loader 只接受明确的 3D metadata，不从 phase 名或 tensor shape 推断 semantic width；
- 缺失 optimizer、schedule、EMA 或 RNG 状态的文件可由独立 inference-only 入口评估，但必须
  标记 `non_resumable=true`，不得恢复训练。

### 3.7 O1 runtime gate

O1 必须以 `H=12` 通过确定性功能 smoke。`H=4` 只用于故障定位和 overhead 对照，不能成为正式
fallback。O0 必须冻结 A600 短基准的步数、并行度、重复次数、计时边界、warm-up、内存测量和
基线命令。

若 H=12 的每真实环境步 runtime 超过预注册无 shadow 基线的 `3×`，或满足 O0 冻结定义的
持续 memory growth，O1 判定 No-Go 并返回 O0；禁止静默降低 horizon 或调整 1/16 sampler。

### 3.8 O1 实现依赖

O1 必须依次完成并验证：DirectGoal/NoWP observation 与零 planner-call instrumentation；新 Student
梯度边界和 strict checkpoint；Pure Motion Prior；三维 semantic retrieval；snapshot/paired shadow/
EMA；共同 sampler/buffer/schedule/log；最后才执行 H=12 smoke 与 owner-run A600 gate。后一步
不得用临时 waypoint、旧二维权重、Fixed-KD warm start、H=4 fallback 或关闭 strict validation
绕过前一步。O0 只冻结职责和依赖，不预先修改 Python/YAML/JSON 或虚构尚不存在的运行符号。

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
