# Roadmap

状态含义：`[ ] pending`、`[~] in progress`、`[x] complete`。每个阶段后续只生成一个
`specs/YYYY-MM-DD-phase-slug/` feature spec。优化路线与稳定路线可以并行，但只有 D1 选中的路线
能够进入最终正式实验。

## 依赖关系

```text
             ┌──> O1 本地 ──> P1 ──> O1 CUDA Gate ──> O2 ──┐
P0 ──> O0 ───┤                                             ├──> D1 ──> E1 ──> E2 ──> E3
 │           └──> O3 ──────────────────────────────────────┤
 └──> S1 ──> S2（可选预备训练）────────────────────────────┘
```

O3 已在 O1 等待服务器期间完成。P1 是优化路线所有剩余服务器工作的硬前置：P1 通过后必须先
运行 O1 CUDA Gate，O1 Go 后进入 O2；设备长期占用只允许等待，不允许跳过或降低门槛。

## Phase P0：基线整理与双分支建立

- [x] complete
- 目标：盘点当前脏工作区，将有效修改拆成聚焦 commit，完成适用验证并合入 `master`；从同一
  已验证 commit 创建 `codex/optimization` 与 `codex/stable`。
- 文档迁移：审计根 `CONSTITUTION.md` 的有效条款是否已进入 `specs/`，随后删除根宪章；保留并
  重构 `TASKS.md`，保留 `CHANGELOG.md`。
- 验收：两个分支基准 commit 相同，工作树归属明确，无用户修改丢失，无未经验证的跨路线代码。

## Phase P1：优化路线 Linux 服务器执行基础设施

- [x] complete
- 唯一规格：`specs/2026-08-28-p1-linux-optimization-server/`。
- 目标：冻结用户级 Python 3.10/CUDA 环境、物理 GPU 绑定、共享服务器预检、等待型 owner
  launcher、O1 baseline/H12 Gate、H4 独立诊断、原子证据和显式恢复链路。
- 边界：只服务优化路线；不运行 O1/O2，不改变任何算法、环境、奖励、能源、seed、workers、
  阈值或预算，不干预其他用户进程。
- 验收：本地跨平台回归通过，研究所有者完成 Linux 环境安装和短 CUDA smoke，机器/环境/GPU
  provenance 完整。通过后下一动作必须为 O1 Gate；O1 Go 后 O2 为强制下一阶段。

## Phase O0：优化路线 A* 教师算法与架构重设计

- [x] complete
- 目标：在不改代码、不训练的前提下，完整审计 waypoint、A* preference、协调器、rollout
  buffer、KL、规则目标和 MAPPO 动作的数据流；比较候选后冻结 Pure Motion Teacher、H=12
  paired shadow Reward Calibration 与三维离线 LLM 教师。
- 必须冻结：A* 教师职责、输入输出、逐机器人有效掩码、团队 reward confidence、环境状态分叉、
  EMA、三维标签与 validity×OOD reliability、共同 schedule、checkpoint、执行期 A* 条件依赖、
  日志、消融、runtime/memory No-Go 和失败降级。
- 人工门：研究所有者书面批准唯一方案后才能进入 O1。

## Phase O1：优化路线角色对齐实现与静态验证

- [x] complete
- 状态：本地实现、短 smoke、P1 Linux CUDA smoke 与 owner-run `baseline/H12` Gate 均已通过。
  Gate artifact 为 `20260829T075212Z_7c305ea2`，commit 为
  `7c305ea24cdca34467c2e7e8a5a9d66ba1133d1e`，H12/baseline runtime ratio 为 `2.880`，
  memory Gate 通过；H4 未作为正常 Gate 运行。
- 目标：只实现 O0 批准的方案，完成确定性教师纯度、action mask、零有效样本、buffer、KL、
  checkpoint 兼容、完整回归和短 smoke。
- 禁止：训练、长评估、KL 系数搜索、环境/奖励/正式 seed 变更。
- 验收：目标改写、端到端协调器污染和非法动作概率质量均为 0；测试、静态检查和短链路通过。

## Phase O2：优化路线校准训练与 Go/No-Go

- [~] experiment-layer implementation in progress
- 目标：由研究所有者在 Linux 优化服务器上先通过 O1 门禁，再运行匹配校准训练；O1/O2 产物与
  Gate 状态保持逻辑隔离，Codex 分析日志与结果。
- 固定组别：`MAPPO-DG/RC-AStarKD × 107/117/127 × 150000 steps`，两组均关闭 LLMKD，共
  6 次；Fixed/RC 链路计数等价性由 MateBook 确定性短受控 smoke 验证。
- 最低门槛：三个 RC seed 的有效教师覆盖率分别≥25%；无 NaN/Inf 或接口失败；相对
  `MAPPO-DG` 的固定 10k-step 网格吞吐 AUC 中位数退化≤10%。
- 失败规则：只允许修复一次已定位的接口/掩码错误并以相同合同重跑；不得根据结果调 KL 系数、
  seed、预算或阈值。

## Phase O3：优化路线真正未见拓扑

- [x] complete（仅表示拓扑/接口就绪，不表示 learned-policy performance 通过）
- 唯一规格：`specs/2026-08-26-o3-unseen-topologies/`。
- 目标：建立一个窄通道布局和一个中央瓶颈/交叉通道布局，冻结地图文件、显式坐标合同、哈希、
  环境 ID、统一观测/动作接口和防泄漏协议。
- 边界：两个布局不参与训练、G2/O2 校准或参数选择；不再实施同图 8-AGV 压力场景。
- 验收：只构成拓扑与评估协议就绪门；确定性环境测试、安全测试、接口 smoke、地图哈希和
  防泄漏审计通过。O3 不加载学习策略、不产生完成率或难度结论，也不是 D1 性能门。
- 下游边界：canonical core topology 的 `200–209 × 20` 是唯一正式必需 held-out 随机鲁棒性
  证据。E1 在查看 O3 策略性能前按资源冻结 O3 探索矩阵为执行或延期；执行时只含
  `MAPPO-DG`、完整方法、8 个匹配训练 seed、两个拓扑和 `200–209 × 20`，不设性能阈值。

## Phase S1：稳定路线旧 Phase 3 行为恢复

- [ ] pending
- 目标：追溯历史 artifact 对应的 A* 行为，在当前工程接口中兼容恢复，不把整个仓库回退到旧
  commit；保留当前日志、checkpoint 兼容和多教师框架。
- 冻结环境：3 AGV、任务完成目标 9、动态入库、`1.10/0.30/0.80`；不安排能源选择 pilot。
- 最终验收：研究所有者运行 `300–309 × 20 episodes`；完成率≥95%，碰撞率=0，能量死亡率=0，
  终止死锁率≤1%。
- 失败规则：只修复已确认的历史行为恢复错误，不调整能源参数、seed 或验收阈值。

## Phase S2：稳定路线决策前预备训练（可选）

- [ ] pending
- 启动条件：S1 已通过、稳定路线配置已冻结、优化路线仍在 O1–O3、研究所有者批准的服务器有空闲资源。
- 目标：提前验证稳定训练链路和估算资源，以提高项目效率。
- 隔离：使用与正式实验不同的预声明 seed，产物写入 `artifacts/stable/predecision/`。
- 禁止：预备结果不得参与 D1 路线选择、不得写入最终统计、不得直接作为论文正式实验。
- 若 D1 选择稳定路线，必须在决策后从头执行 E2 正式训练和评估。

## Phase D1：唯一路线决策门

- [ ] pending
- 优化路线选择条件：O0 人工批准、P1 通过、O1 本地验证与 owner-run Linux CUDA
  runtime/memory 门通过、O2
  校准门通过、O3 拓扑/接口就绪门通过。
- 决策规则：优化路线满足全部条件时选择优化路线；否则选择已通过 S1 的稳定路线。
- 若优化路线未通过且稳定路线也未通过 S1，D1 状态为 blocked，不得进入 E1；由研究所有者
  重新裁定论文范围，不能静默降低既有门槛。
- 防泄漏：不得使用正式 seed；不得依据 S2 性能结果选择路线；决策、commit、配置和允许主张
  必须写入 `TASKS.md` 与 `CHANGELOG.md`。
- 决策后未选路线冻结为诊断/附录，不再进入确认性实验。

## Phase E1：所选路线协议冻结与链路验证

- [ ] pending
- 目标：冻结唯一代码 commit、环境、教师语义、奖励、seed、交互预算、checkpoint 规则、日志
  schema、失败处理、统计假设和论文主张；完成所有必需组端到端短 smoke。
- 优化路线 E1/E2 正式训练预算保持不变：核心 `2×2` 32 次，Fixed-KD、QMIX-DG、RuleKD-v3 各 8 次，
  ShuffleKD-v3、NoOOD-v1、NoGoalHint-v1 各 3 次，共 65 次；启发式 A* 无训练。资源修订后
  连同 O2 共 71 次；不做 8-AGV 压力实验。
- O3 探索性矩阵不增加训练；是否运行必须在 E1 预先冻结，不能根据 O3 性能触发、取消或删减。
- 稳定路线预算：核心 `2×2` 与 RuleKD 各 5 seed，NoWP 3 诊断 seed，启发式 A* 无训练；删除
  QMIX、ShuffleKD、未见拓扑和 8-AGV 压力实验。

## Phase E2：正式训练与独立评估

- [ ] pending
- 研究所有者在批准的服务器上运行全部长任务；Codex 和项目工程师分别准备所负责路线的命令，并对
  产物进行独立分析和交叉审查。
- 正式必需评估固定在 canonical core topology 使用 held-out evaluation seeds
  `200–209 × 20 episodes` 和 final checkpoint；训练 seed 是学习方法比较的独立统计单位。
- 若 E1 已预先选择执行 O3 探索性矩阵，则在 E2 完整运行并报告；结果不改变 D1、核心配置、
  checkpoint 或正式主张范围，失败只能作为局限性保留。
- 禁止：正式结果生成后切换路线、修改核心配置、静默重跑算法失败或挑选最佳 seed/checkpoint。

## Phase E3：统计、图表与论文就绪

- [ ] pending
- 目标：从冻结原始日志生成 seed 级配对统计、Holm 校正、95% 置信区间、效应量、bootstrap
  敏感性分析、表格、图像和失败案例；完成方法—实验—论文一致性审计。
- 优化路线允许在证据支持时形成二区目标主张；稳定路线只形成完整、可复现的方法实验，不主张
  未见拓扑泛化或二区保证。
- 验收：每项论文主张可追溯到证据或明确限制，图表由版本化脚本生成，最终由研究所有者批准。

## 开放问题

无。每个阶段的实现细节、命令和测试矩阵由该阶段唯一 feature spec 定义。
