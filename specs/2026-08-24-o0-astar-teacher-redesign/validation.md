# Validation — O0 A* 教师算法与架构重设计

## 1. Definition of done

### 1.1 治理与文档

- [x] canonical architecture 存在于
  `docs/architecture/o0-reward-calibrated-heterogeneous-distillation.md`，已纳入 Git 且是唯一正式
  架构方案；
- [x] 两份根目录研究输入均按记录哈希完成逐项映射并移除，不存在长期未跟踪或并列正式版本；
- [x] canonical architecture、requirements、plan、validation、三份宪章、Roadmap、TASKS 与
  CHANGELOG 对三维语义、Reward Calibration、H=12 和阶段边界完全一致；
- [x] canonical architecture 不含 `TBD`、`TODO`、未决候选、占位参数或“实现者自行选择”；
- [x] O0 没有修改 Python 运行代码、runtime training config、reward、环境、训练 seed 或稳定路线
  合同；O0-F 仅按 owner 明确授权修订治理 manifest；
- [x] O0-A 至 O0-F 各自拥有聚焦 commit、对应 CHANGELOG、验证证据和研究所有者继续批准，
  不存在跨人工检查点预做或混合提交；
- [ ] 研究所有者已书面批准唯一架构，O0 才可标记 complete。

### 1.2 Pure Motion Teacher

- [x] 教师输入/输出、root-action-conditioned cost、min-max 归一化、`tau_motion=1.0` Boltzmann
  prior、共享 budget、确定性、cache、有效掩码和 fail-closed 合同精确冻结；
- [x] purity 矩阵证明任务/优先级/充电/parking/yield/预约优先权/coordinator/reward/LLM/Student/
  calibration 都不能参与标签生成；
- [x] `TOGGLE_LOAD`、非法动作及无有效 continuation 的 root 概率质量必须为零；三个 root 不得
  启动独立完整 A*，必须共享单次 bounded search 的总 512-expansion budget；没有任何有限 root
  cost 时逐机器人 mask=0；
- [x] reward calibration 只评价已经生成的标签，不能改变标签内容、validity 或规划过程；
- [x] 搜索质量指标只记录，不存在连续 `c_A_search`、search entropy、path-length confidence、
  Student disagreement 或其他优化乘子；论文不把 planning cost 表述为 RL Q-value。

### 1.3 Reward Calibration

- [x] 完整 fork schema 覆盖真实状态、动态入库、全部 RNG、adapter 状态、metrics、wrapper、
  cache 隔离、canonical hash round-trip 和真实 rollout 零污染；
- [x] paired shadows 从相同状态/外部随机状态开始，使用 deterministic Student、Pure Teacher
  argmax、A* 分支当前状态的 Student fallback、相同 mask 规则、分支局部 mask 和按事件寻址的
  common-random-number stream；
- [x] H=12、return、terminal、detached bootstrap、handoff-to-Student 和团队权重×逐机器人 mask
  的数学合同唯一且量纲一致；单边 terminal 后另一分支继续且只有未终止 H-step 末态 bootstrap；
- [x] `calibration-sampler-v1` 精确实现 1/16 deterministic selection；Fixed/RC 共用
  `m_calib(t)`、shadow、日志、EMA 与 sampling density，未选择状态两组 A* KD 均为 0，唯一优化
  差异为 `c_A_reward`；
- [x] EMA `0.99/1e-3/64`、Welford 初始化、exponential mean/variance、先计算后更新、顺序、
  `[0,1]` 截断、非有限 No-Go、严格恢复和 RC 初始化前零权重均精确冻结；
- [x] disagreement 不参与权重，calibration 不反传 Critic/A*，shadow 不污染 real rollout；
- [x] H=12 runtime `3×` gate 和持续 memory growth 具有 A600/12-worker、16/128 vector-step、5-repeat、
  2+10-window、`max(64 MiB,5%)` 与 `rho>=0.80` 的可执行判定式；
  H=4 只作诊断，不能通过降 horizon 通过 O1。

### 1.4 三维 LLM 教师

- [x] 三个标签名称、定义、顺序、五点 anchors、范围、整记录 validity、共享 OOD、reason
  audit-only 与 forbidden outputs 精确冻结；
- [x] reliability 公式只包含 validity×OOD，不存在 consistency、逐维门控或 disagreement 权重；
- [x] `semantic-view-v3` 的 61D 顺序、类别、匿名邻居距离/tie-break/排序、机器人中心坐标、
  padding 和 mask 唯一；正式 OOD 不读取旧 615D full observation；
- [x] 两个 OOD 候选已在重建 61D corpus 上仅按 coverage、monotonicity、stability 选择截断指数；
  旧 615D 结果明确仅为 preliminary evidence，未生成新标签或查看训练性能；
- [x] 60 pilot/800 formal 的精确模型切换门、prompt、temperature、scenario generator、parser、
  seed/配额、原始响应、fingerprint、失败和哈希合同完整；
- [x] pilot/formal 隔离、formal dataset-level validity/100-record review No-Go、禁止逐条修复和整组
  version 升级规则完整；Pearson/Spearman 与 `|ρ|>=0.80` 只触发人工复核；
- [x] 数据生成责任明确为研究所有者；O0 未调用 LLM、生成 pilot/formal 标签或使用所提供 key。

### 1.5 Student、兼容、对照与主张

- [x] 新三维网络的 shape、梯度边界、late fusion、semantic detach、Motion Prior loss 与 PPO
  所有权精确冻结；
- [x] `λ_A(t)`/`λ_L(t)` schedule 为所有相关组共同的预注册确定算法；
- [x] Fixed-KD 和 RC-KD 共用 `m_calib`、shadow/EMA/logging 且只在 `c_A_reward` 上有优化差异，
  O2 三诊断 seed 已登记但未运行；
- [x] 三维 checkpoint 从新初始化并与历史一维/二维严格隔离，loader/metadata/EMA 恢复无歧义；
- [x] 无 Teacher 派生物理观测和 waypoint 兼容边界完整；执行期无需 A* 仍是后续证据条件主张；
- [x] 每项 O0 后允许主张均有合同依据，所有性能/泛化/收敛/部署主张保持禁止。
- [x] 总预算 74、O2 覆盖率/AUC 分母与聚合、紧凑日志、RuleKD-v3、无 fixed-point ShuffleKD、
  NoOOD、NoGoalHint、QMIX-DG fairness 和七项正式统计 contrast 均唯一冻结；
- [x] 正式 LLM prompt 使用方向性 rubric 但仍输出连续三维标签；RuleKD 不参与 prompt、formal
  label 或人工 ground truth；

## 2. 审查方法

1. 从 `specs/mission.md` 开始，逐段对照 canonical architecture 与三份 feature-spec 文件；
2. 沿 O0-A 数据流图逐个打开准确代码符号，确认审计没有遗漏生产者、消费者或执行入口；
3. 用 purity 矩阵逐项模拟目标、优先级、充电、yield、coordinator、非法动作和零有效标签；
4. 手工推导 paired rollout 在正常、单边 terminal、双边 terminal、Teacher 部分无效、Critic 非有限
   和 EMA 未初始化状态下的结果；
5. 用固定示例检查 RC-KD/Fixed-KD 权重、正优势门控、逐机器人 mask 和 schedule 一致性；
6. 对两个 OOD 公式执行边界表：零距离、分位点、截断外、零方差、无邻居、NaN/Inf；
7. 检查三维 schema、dataset manifest、checkpoint metadata 和日志字段能否唯一重建实验合同；
8. 执行文档/工作树检查并由研究所有者完成最终人工批准。

## 3. 自动与静态检查

在项目根目录运行并记录原始退出码：

```powershell
git diff --check
git status --short
D:\Anaconda3\envs\py310\python.exe -m pytest
D:\Anaconda3\envs\py310\python.exe -m flake8 rware llm_mappo eval train scripts figures/core
D:\Anaconda3\envs\py310\python.exe visualize.py --help
D:\Anaconda3\envs\py310\python.exe eval/evaluate_dynamic_ingress_astar.py --help
```

此外必须执行并记录：

- O0 阶段/唯一 spec/文档链接一致性检查；
- canonical architecture 的 `TBD|TODO|placeholder` 扫描；
- 旧二维术语与新三维术语的活动文档扫描；
- 运行代码和 runtime training config 相对 O0 基线的零差异检查；治理 manifest 的 owner-authorized
  O0-F 一致性修订单独审计；
- 根目录未跟踪研究输入在 canonical 内容映射完成后的不存在性检查；
- 所有公式字段在日志/checkpoint/dataset schema 中的一一映射检查。

O0-G 最终交付还必须确认唯一 O1 任务包为
`plan/task-package/o1-role-alignment-implementation.md`，其中固定模块接口、测试先行顺序、短
smoke、禁止长任务和 owner-only A600 命令。O0-G 交付时只允许保留“研究所有者最终书面批准”
相关条目为未完成；不得提前更新 Roadmap/TASKS 的 O0 complete 状态。

## 4. Merge criteria

- [ ] `plan.md` 的 O0-A 至 O0-G 全部完成；
- [ ] 所有 validation 条目有证据且无未解释失败；
- [ ] canonical architecture 不依赖未跟踪输入且已由研究所有者批准；
- [ ] `specs/roadmap.md` 的 O0 状态更新为 `[x] complete`；
- [ ] `TASKS.md` 的 O0 只由核心架构师在证据审查后标记完成；
- [ ] `CHANGELOG.md` 已按每个完成任务组同步；
- [ ] O0-A 至 O0-F 的逐组人工审核记录完整，且每组批准时间早于下一组首个 commit；
- [ ] 没有运行代码、runtime training config、标签、训练、长评估、O3 拓扑或稳定路线修改混入
  O0；治理 manifest 仅含文档合同；
- [ ] 分支工作树干净，commit 聚焦且未 push/merge，交由研究所有者发布。
