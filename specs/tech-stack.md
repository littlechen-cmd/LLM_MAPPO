# Tech Stack

## 语言与运行环境

- Python 3.10；项目必须始终使用已有 `py310` 环境；
- 唯一规范 Python 解释器为 `D:\Anaconda3\envs\py310\python.exe`；所有 Python
  命令、测试、脚本与训练必须直接调用该解释器，不依赖 `python` 已加入 `PATH`，也不要求
  `conda activate py310`；
- 禁止为本项目创建、重建、升级或修改虚拟环境；
- UTF-8、LF、四空格缩进，最大行长 89；
- 本地 Intel i5、无独显 Huawei MateBook 用于文档、代码开发、静态检查、确定性测试和短
  smoke；
- A600 服务器用于长训练和长评估，所有长任务由研究所有者人工启动。

选择理由：保持现有项目和历史 checkpoint 的兼容性，避免在论文后期引入无关运行时迁移。

## Text Search Environment

- Windows 上禁止直接调用裸 `rg`。Codex Desktop 可能将其解析到
  `C:\Program Files\WindowsApps` 下集成 shell 无法执行的 bundled executable；
- 所有基于 ripgrep 的仓库检索必须直接调用：
  `C:\Users\28016\bin\rg.exe`；
- 禁止尝试修复、重装或诊断 Codex bundled `WindowsApps\...\rg.exe`。

## 核心框架

- PyTorch：MAPPO、QMIX/对照网络、优化器与 checkpoint；
- Gymnasium 与项目 `rware/`：动态仓储多智能体环境；
- NumPy：状态、动作偏好、日志聚合与数值处理；
- A*：优化路线只作训练期 Pure Motion Teacher，Student 执行使用 DirectGoal 且 planner query
  必须为 0；稳定路线保留历史运行期 waypoint 与路径教师；
- 离线 JSON/JSONL 标签与近邻检索：优化路线三维、稳定路线历史二维的 LLM 语义教师；
- CSV、JSON、JSONL 与 TensorBoard：训练、评估、诊断和证据追踪；
- Matplotlib/Pillow：论文图表与确定性回放；
- pytest 与 Flake8：回归、安全和静态质量门。

## 方法模块边界

- MAPPO 输出最终离散动作，共享 Actor、集中式 Critic；
- A* 只提供路径或局部运动先验，具体教师语义由优化路线 O0 人工设计门或稳定路线历史行为
  合同分别冻结；
- 优化路线离线 LLM 只提供 `task_persistence`、`yielding_preference` 与
  `coordination_risk`；稳定路线保留历史 `task_commitment` 与 `local_assertiveness`；
- 优化路线 LLM reliability 仅为整记录 validity×共享 OOD reliability；A* reward
  calibration 只评价独立生成的 Pure Motion label，不能参与标签生成；
- 规则层负责任务队列、合法目标、固定阈值充电安全、动作合法性和硬安全约束；
- `Heuristic-Dispatcher+A*` 是非学习端到端基线，不等同于训练期 A* 教师；
- 训练与执行期间不得调用在线 LLM。

## Git 与双路线隔离

Phase P0 已完成，双路线从同一已验证 P0 commit 建立。历史落位步骤为：

1. 盘点并归属当前脏工作区修改；
2. 形成聚焦 commit；
3. 运行适用测试和静态检查；
4. 将验证后的当前分支合入 `master`；
5. 从同一个已验证 `master` commit 创建：
   - `codex/optimization`
   - `codex/stable`

共享环境、评估和日志修复只能通过明确、可追溯、已测试的 commit 合并。禁止在两个分支之间
直接复制文件覆盖。任何改变环境语义、A* 教师语义、奖励、观测、指标或正式数据合同的 commit
必须标明适用路线。

## 产物与数据隔离

- 优化路线：`artifacts/optimization/`；
- 稳定路线：`artifacts/stable/`；
- 稳定路线决策前预备训练：`artifacts/stable/predecision/`；
- 正式实验必须使用所选路线独立的 manifest、配置哈希、代码 commit 和 checkpoint；
- 校准、稳定验收和正式评估 seed 必须分离；
- 基础设施故障允许同 seed、同配置重跑并保留故障记录；算法失败、安全失败、NaN、死锁和
  能量死亡必须作为结果保留，不能静默替换 seed。
- 优化路线三维 checkpoint 与历史一维/二维 checkpoint 严格隔离，禁止语义权重填充迁移；
  新三维训练从新架构初始化并持久化 semantic schema、reliability、EMA、horizon 与 schedule。

## 正式实验技术合同

优化路线对照术语固定为：

- RuleKD-v3：在同一 800 条 semantic views 上用冻结规则独立生成三维标签，并复用相同 retrieval、
  validity 与 OOD；规则不得进入正式 LLM prompt 或标签链路；
- ShuffleKD-v3：在预注册场景分层内对三维联合标签做确定性无 fixed-point derangement，只作
  状态—标签对应性负对照；
- NoOOD-v1：只将有效 LLM label 的 OOD reliability 置 1；
- NoGoalHint-v1：保持 613D 输入但把 DirectGoal geometry block 九位清零，只诊断目标几何提示
  敏感性；不得称为执行期 A* 消融。

### 优化路线

- 核心 `2×2`：每组 8 个匹配训练 seed，共 32 次；
- O2 校准：`MAPPO-DG/Fixed-AStarKD/RC-AStarKD × 107/117/127`，三组均关闭 LLMKD，共 9 次；
- QMIX-DG、RuleKD-v3、Fixed-A*KD+LLMKD：各 8 次；
- ShuffleKD-v3、NoOOD-v1、NoGoalHint-v1：各 3 次；
- 优化路线全部学习运行总计 74 次，其中 E1/E2 正式/诊断预算为 65 次；
- `Heuristic-Dispatcher+A*`：无训练；
- 包含两个真正未见拓扑；
- 不包含同图 8-AGV 压力场景；
- 两个未见拓扑的具体评估组、episode 数、聚合与统计规则由 O0 和后续协议 feature spec
  冻结，但不得删除上述必需对照。
- 正式训练 seed 固定为 `7/17/27/37/47/57/67/77`；三个诊断组使用其中
  `7/17/27` 三个诊断 seed。

### 稳定路线

- 核心 `2×2`：5 个匹配训练 seed；
- RuleKD：5 个匹配训练 seed；
- NoWP：3 个诊断训练 seed（稳定路线历史 waypoint 合同下保留旧名称）；
- `Heuristic-Dispatcher+A*`：无训练；
- 不包含 QMIX、ShuffleKD、未见拓扑或 8-AGV 压力场景；
- 稳定验收使用 `300–309 × 20 episodes`；
- 正式评估使用 `200–209 × 20 episodes`；
- 统一使用 final checkpoint，不允许按结果挑选最佳 checkpoint。
- 正式训练 seed 固定为 `7/17/27/37/47`；NoWP 使用其中 `7/17/27` 三个诊断 seed。

## 角色与权限

- 研究所有者（用户）：最终批准架构、路线选择、正式协议和论文主张；在 A600 上运行所有长
  训练与长评估；发布 Git 变更；
- 执行项目负责人兼核心架构师（Codex）：维护总体推进方向，负责优化路线实现、跨路线一致性
  审查、实验命令准备和结果分析；不代替研究所有者启动长任务；
- 项目工程师 AI：负责稳定路线实现、短验证、实验命令准备和结果分析；不得自行修改宪章、
  切换正式路线或启动长任务。

## 文档治理

- `specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md` 是新的唯一项目宪章；
- 根 `CONSTITUTION.md` 在 P0 条款迁移审计完成后删除；
- 根 `TASKS.md` 保留为当前实施任务清单；
- 根 `CHANGELOG.md` 保留，并随每个完成任务同步更新；
- 每个 Roadmap phase 后续只建立一个 feature spec 目录，避免同一阶段出现多个并列任务文档。

## 开放问题

无。具体类名、配置字段、测试用例和命令属于各阶段 feature spec，不在宪章中预设。
