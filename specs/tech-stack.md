# 技术栈

## 语言与运行环境

- Python 3.10；Windows 与 Linux 分别使用冻结的项目专属环境；
- Windows canonical interpreter 为 `D:\Anaconda3\envs\py310\python.exe`；Linux 训练服务器
  canonical interpreter 为 `/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python`；所有命令直接
  调用对应绝对路径，不依赖 `python` 在 `PATH`，也不要求 `conda activate`；
- Windows 已有 `py310` 禁止修改；Linux 只允许研究所有者按 P1 runbook 在用户目录一次性创建
  Python 3.10.19 prefix，禁止修改 `/opt/miniconda3` 共享 base；
- UTF-8、LF、四空格缩进，最大行长 89；
- 本地 Intel i5、无独显 Huawei MateBook 用于文档、代码开发、静态检查、确定性测试和短
  smoke；
- 训练服务器硬件为 Ubuntu 22.04.5、双路 EPYC 7542、128 GB RAM、RTX 4090 49140 MiB
  与 RTX 4080 SUPER 16376 MiB；长训练和长评估由研究所有者安排；
- MAPPO 每个 learner 使用 `spawn` 启动 16 个 CPU-only 环境 worker，主进程集中完成
  Actor/Critic 推理与 PPO 更新。三个场景对应每个 vector step 的 48、80、128 个 AGV
  观测；并发 learner 数由资源预检决定，不以旧合同的固定四进程数作为新实验保证；
- 新三场景的临时配置为 `num_env_workers=16`、`rollout_length=64`，每次完整更新采集
  1,024 个累计环境 transition。`environment_steps` 始终是所有 worker 的累计预算，
  一次 vector step 使全局计数增加 16；各场景最终长训练步数须经短测后由研究所有者批准；
- 环境 worker 不初始化 CUDA context，并限制 `OMP_NUM_THREADS/MKL_NUM_THREADS/
  OPENBLAS_NUM_THREADS=1`；RTX 4090 只由 learner 的 Actor、Critic、optimizer 和 PPO update 使用；
- 目标 GPU 必须通过共享服务器预检与项目 GPU lease；不得抢占、终止或隐藏其他用户进程。
  正式长训练、耐久评估及长压力评估由研究所有者人工启动；本地 CPU 负责接口验证、短测、
  标签处理和统计绘图。

选择理由：保持现有项目和历史 checkpoint 的兼容性，避免在论文后期引入无关运行时迁移。

## 文本检索环境

- Windows 上禁止直接调用裸 `rg`。Codex Desktop 可能将其解析到
  `C:\Program Files\WindowsApps` 下集成 shell 无法执行的 bundled executable；
- 所有基于 ripgrep 的仓库检索必须直接调用：
  `C:\Users\28016\bin\rg.exe`；
- 禁止尝试修复、重装或诊断 Codex bundled `WindowsApps\...\rg.exe`。

## 核心框架

- PyTorch：MAPPO、QMIX/对照网络、优化器与 checkpoint；
- Gymnasium 与项目 `rware/`：动态仓储多智能体环境；
- NumPy：状态、动作偏好、日志聚合与数值处理；
- A*：核心四组的 A* KD 只作训练期运动教师，Student 执行时 planner query 必须为 0；
- 离线 JSON/JSONL 标签与近邻检索：LLM KD 使用经批准、版本化的语义定义；
- CSV、JSON、JSONL 与 TensorBoard：训练、评估、诊断和证据追踪；
- Matplotlib/Pillow：论文图表与确定性回放；
- pytest 与 Flake8：回归、安全和静态质量门。

## 方法模块边界

- MAPPO 输出最终离散动作，共享 Actor、集中式 Critic；
- A* 只提供路径或局部运动先验；正式四组使用同一经批准的教师定义，不能沿用历史
  checkpoint 的未注明语义；
- 离线 LLM 语义版本与历史二维、候选三维 checkpoint 隔离；具体维度和 reliability
  规则须在正式实验前作为同一四组合同批准，不得静默混用；
- A* reward calibration 若用于评价运动标签，不得反向影响标签生成；
- 规则层负责任务队列、合法目标、固定阈值充电安全、动作合法性和硬安全约束；
- `Heuristic-Dispatcher+A*` 是非学习端到端基线，不等同于训练期 A* 教师；
- 训练与执行期间不得调用在线 LLM。

### 空载→载货状态转换与 A*KD 掩码契约

- 空载→载货是任务语义、A* footprint/goal 和循环策略隐状态同时切换的边界，任何涉及
  `supervision_scope`、`validity`、动作掩码或拾取逻辑的改动都必须将其作为独立回归场景；
- `supervision_scope` 只表示样本是否属于仍需导航的活跃任务：无任务、死亡或已到达当前
  教师目标的状态必须排除。它不得以“当前能否执行运动动作”为条件；
- picking lock 与强制 `TOGGLE_LOAD` 拾取边界虽暂时不能移动，但仍属于活跃任务的过渡样本。
  它们的即时动作语义由 `validity` 处理，而不是从 `supervision_scope` 删除；
- 因而 full-validity A*KD 可通过 `validity` 跳过暂时锁定的教师动作，no-validity A*KD
  必须保留这些样本，以维持空载导航到载货配送的递归状态衔接。

## Git 与实验隔离

历史 `codex/optimization` 和 `codex/stable` 分支及其产物保留作追溯；新的三场景正式合同
由独立功能分支实现，不依赖旧的双路线选择门。共享环境、评估和日志修复只能通过明确、
可追溯、已测试的 commit 整合。任何改变环境、教师、奖励、观测、指标或正式数据合同的
commit 必须标明适用实验版本，不能覆盖历史 checkpoint 的含义。

## 产物与数据隔离

- 三场景各自使用独立的环境 ID、manifest、配置哈希、代码 commit、产物目录和 checkpoint；
- 50k 诊断、正式主评估 `200–209 × 20`、耐久评估 `400–409` 的实例表相互隔离；
- 基础设施故障允许同 seed、同配置重跑并保留故障记录；算法失败、安全失败、NaN、死锁和
  能量死亡必须作为结果保留，不能静默替换 seed。
- 新语义 checkpoint 与历史一维/二维 checkpoint 严格隔离，禁止权重填充、截断或部分迁移；
  新训练须持久化其语义 schema、reward、环境和优化日程。

## 正式实验技术合同

| 场景 | AGV/充电站/拣选站 | 训练与主评估 `max_steps` | 活跃任务 |
| --- | --- | ---: | ---: |
| RWARE tiny | 3/3/2 | 800 | 3 |
| medium | 5/5/2 | 800 | 5 |
| formal 50×28 | 8/8/10 | 1200 | 8 |

- 核心四组为 `baseline`、`A* KD`、`LLM KD`、`A* KD + LLM KD`，每组在三个场景
  各训练 8 个配对 seed `7/17/27/37/47/57/67/77`，正式主矩阵共 96 次训练；
- 正式主评估采用 `200–209` 各派生 20 个实例，每个训练 checkpoint 在同场景的
  200 个实例上评估；固定最终 checkpoint，不按结果选择最佳模型；
- formal-8AGV 另设 5,000 步、`400–409` 共 10 个实例的耐久评估，全部四组与 8 个
  训练 seed 复用实例表；12/16 台同图压力评估需另行批准充电容量与 Actor-only 入口；
- 未见拓扑、QMIX 和旧路线消融不在本轮四组主矩阵内，需要时另订探索性协议；
- `rollout_length=64`、16 worker 和继承 baseline 的 PPO 值是短测起点；最终各场景
  `environment_steps`、reward、学习率日程、checkpoint 里程碑须经研究所有者批准，
  才能把 `approval_status` 改为 `approved`。

## 角色与权限

- 研究所有者（用户）：最终批准架构、正式协议和论文主张；在批准的服务器上运行
  所有长训练与长评估；发布 Git 变更；
- 执行项目负责人兼核心架构师（Codex）：维护总体推进方向，负责跨场景、跨方法一致性
  审查、实验命令准备和结果分析；不代替研究所有者启动长任务；
- 项目工程师 AI：负责批准任务包的代码实现、短验证和实验命令准备；不得自行修改宪章、
  正式实验合同或启动长任务。

## 文档治理

- 凡需提交研究所有者审阅的 Markdown（`.md`）文档，正文、标题、表格说明和结论均使用中文；
  代码标识符、配置字段、文件路径、方法名及难以准确翻译的专有名词或术语可以保留英文；
- `specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md` 是新的唯一项目宪章；
- 根 `terminology.md` 是研究所有者与实施者的 canonical 概念对齐入口；新方案首次引入重要术语时
  必须同步补充“通俗解释、专业定义、本项目作用”，汇报时优先使用通俗解释；
- 根 `CONSTITUTION.md` 在 P0 条款迁移审计完成后删除；
- 根 `TASKS.md` 保留为当前实施任务清单；
- 根 `CHANGELOG.md` 保留，并随每个完成任务同步更新；
- 每个 Roadmap phase 后续只建立一个 feature spec 目录，避免同一阶段出现多个并列任务文档。

## 开放问题

无。具体类名、配置字段、测试用例和命令属于各阶段 feature spec，不在宪章中预设。
