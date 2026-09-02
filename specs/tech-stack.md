# Tech Stack

## 语言与运行环境

- Python 3.10；Windows 与 Linux 分别使用冻结的项目专属环境；
- Windows canonical interpreter 为 `D:\Anaconda3\envs\py310\python.exe`；Linux 优化路线
  canonical interpreter 为 `/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python`；所有命令直接
  调用对应绝对路径，不依赖 `python` 在 `PATH`，也不要求 `conda activate`；
- Windows 已有 `py310` 禁止修改；Linux 只允许研究所有者按 P1 runbook 在用户目录一次性创建
  Python 3.10.19 prefix，禁止修改 `/opt/miniconda3` 共享 base；
- UTF-8、LF、四空格缩进，最大行长 89；
- 本地 Intel i5、无独显 Huawei MateBook 用于文档、代码开发、静态检查、确定性测试和短
  smoke；
- 优化路线服务器为 Ubuntu 22.04.5、双路 EPYC 7542、128 GB RAM、RTX 4090 49140 MiB
  与 RTX 4080 SUPER 16376 MiB；physical GPU 0 的 RTX 4090 是 O1/O2/E1/E2 唯一训练
  GPU，physical GPU 1 的 RTX 4080 SUPER 只保留在硬件清单中，不参与训练或 CUDA smoke；
- E1/E2 在 RTX 4090 上固定最多四个独立 learner 进程。MAPPO 系列每个 learner 固定使用
  `spawn` 启动 16 个 CPU-only 仓库环境 worker，由主进程集中完成 `16×5=80` 个智能体观测的
  GPU Actor/Critic 推理与 PPO 更新；QMIX-DG 保留单环境 trainer，但允许四个独立 seed run 并发；
- MAPPO 并行架构固定为 `num_env_workers=16`。旧 E1/E2 诊断产物使用
  `rollout_length=128`（每次更新 2048 个累计 transitions）；R1 将以 `rollout_length=32`
  （每次更新 512 个累计 transitions）作为主要候选，并只在可复现数值不稳定时考虑 64。R1
  通过后由统一冻结合同确定新的 E2 正式值。`formal_environment_steps=150000`
  始终是所有 worker 的累计预算，一次 vector step 使全局计数增加 16，而不是每个 worker
  分别运行 150000 steps；
- 环境 worker 不初始化 CUDA context，并限制 `OMP_NUM_THREADS/MKL_NUM_THREADS/
  OPENBLAS_NUM_THREADS=1`；RTX 4090 只由 learner 的 Actor、Critic、optimizer 和 PPO update 使用；
- P1 只服务优化路线。目标 GPU 必须通过共享服务器预检与项目 GPU lease；不得抢占、终止或
  隐藏其他用户进程。O1 Gate、O2、正式长训练和长评估仍只由研究所有者人工启动；短诊断直接
  使用当前 SSH，会持续较久的任务使用 `nohup`，日志写入 `/home/lzx/`；不依赖 `tmux`；
- O3 接口验证、E1 本地短验证、标签处理和统计绘图仍在 MateBook/CPU 完成；E1 CUDA smoke
  由研究所有者在 RTX 4090 上运行。

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
- O2 校准：`MAPPO-DG/RC-AStarKD × 107/117/127`，两组均关闭 LLMKD，共 6 次；
- Fixed/RC 的 sampler、shadow、EMA、日志和计数等价性改由 MateBook 上的确定性短受控 smoke
  验证；O2 不再运行 3 次 Fixed-AStarKD 长校准；
- QMIX-DG、RuleKD-v3、Fixed-A*KD+LLMKD：各 8 次；
- ShuffleKD-v3、NoOOD-v1、NoGoalHint-v1：各 3 次；
- 优化路线全部学习运行总计 71 次，其中 O2 为 6 次，E1/E2 正式/诊断预算保持 65 次；
- `Heuristic-Dispatcher+A*`：无训练；
- 包含两个真正未见拓扑作为 evaluation-only 探索性压力测试资产；它们不构成正式性能门；
- 不包含同图 8-AGV 压力场景；
- canonical core topology 上的 final-checkpoint held-out-seed 评估是正式必需证据。E1 必须在
  查看任何 O3 策略性能前，依据资源而非结果冻结 O3 探索性矩阵为“执行”或“延期”；若执行，
  只运行 `MAPPO-DG` 与 `RC-AStarKD+LLMKD` 的 8 个匹配训练 seed、两个拓扑、
  `200–209 × 20 episodes`，无最低性能阈值且必须完整报告。
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

- 研究所有者（用户）：最终批准架构、路线选择、正式协议和论文主张；在批准的服务器上运行
  所有长训练与长评估；发布 Git 变更；
- 执行项目负责人兼核心架构师（Codex）：维护总体推进方向，负责优化路线实现、跨路线一致性
  审查、实验命令准备和结果分析；不代替研究所有者启动长任务；
- 项目工程师 AI：负责稳定路线实现、短验证、实验命令准备和结果分析；不得自行修改宪章、
  切换正式路线或启动长任务。

## 文档治理

- `specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md` 是新的唯一项目宪章；
- 根 `terminology.md` 是研究所有者与实施者的 canonical 概念对齐入口；新方案首次引入重要术语时
  必须同步补充“通俗解释、专业定义、本项目作用”，汇报时优先使用通俗解释；
- 根 `CONSTITUTION.md` 在 P0 条款迁移审计完成后删除；
- 根 `TASKS.md` 保留为当前实施任务清单；
- 根 `CHANGELOG.md` 保留，并随每个完成任务同步更新；
- 每个 Roadmap phase 后续只建立一个 feature spec 目录，避免同一阶段出现多个并列任务文档。

## 开放问题

无。具体类名、配置字段、测试用例和命令属于各阶段 feature spec，不在宪章中预设。
