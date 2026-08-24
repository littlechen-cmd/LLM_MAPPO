# Tech Stack

## 语言与运行环境

- Python 3.10；项目统一使用现有 `py310` Conda 环境；
- UTF-8、LF、四空格缩进，最大行长 89；
- 本地 Intel i5、无独显 Huawei MateBook 用于文档、代码开发、静态检查、确定性测试和短
  smoke；
- A600 服务器用于长训练和长评估，所有长任务由研究所有者人工启动。

选择理由：保持现有项目和历史 checkpoint 的兼容性，避免在论文后期引入无关运行时迁移。

## 核心框架

- PyTorch：MAPPO、QMIX/对照网络、优化器与 checkpoint；
- Gymnasium 与项目 `rware/`：动态仓储多智能体环境；
- NumPy：状态、动作偏好、日志聚合与数值处理；
- A*：运行期 waypoint 与训练期路径/运动教师；
- 离线 JSON/JSONL 标签与近邻检索：LLM 双语义教师；
- CSV、JSON、JSONL 与 TensorBoard：训练、评估、诊断和证据追踪；
- Matplotlib/Pillow：论文图表与确定性回放；
- pytest 与 Flake8：回归、安全和静态质量门。

## 方法模块边界

- MAPPO 输出最终离散动作，共享 Actor、集中式 Critic；
- A* 只提供路径或局部运动先验，具体教师语义由优化路线 O0 人工设计门或稳定路线历史行为
  合同分别冻结；
- 离线 LLM 只提供 `task_commitment` 与 `local_assertiveness` 标签；
- 规则层负责任务队列、合法目标、固定阈值充电安全、动作合法性和硬安全约束；
- `Heuristic-Dispatcher+A*` 是非学习端到端基线，不等同于训练期 A* 教师；
- 训练与执行期间不得调用在线 LLM。

## Git 与双路线隔离

Phase P0 完成前不创建正式双路线分支。P0 必须先：

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

## 正式实验技术合同

对照术语固定为：

- RuleKD：以冻结的确定性规则生成与 LLM 相同维度的语义标签，检验 LLM 教师是否优于廉价
  规则教师；
- ShuffleKD：保持语义标签边际分布但打乱状态—标签对应，只作状态相关性负对照；
- NoWP：保持 actor 输入宽度但将 waypoint 槽固定为零，同时关闭 waypoint reward 与 A*KL，
  诊断执行期 A* waypoint 依赖。

### 优化路线

- 核心 `2×2`：8 个匹配训练 seed；
- QMIX-WP：8 个匹配训练 seed；
- RuleKD：8 个匹配训练 seed；
- ShuffleKD：3 个诊断训练 seed；
- NoWP：3 个诊断训练 seed；
- `Heuristic-Dispatcher+A*`：无训练；
- 包含两个真正未见拓扑；
- 不包含同图 8-AGV 压力场景；
- 两个未见拓扑的具体评估组、episode 数、聚合与统计规则由 O0 和后续协议 feature spec
  冻结，但不得删除上述必需对照。
- 正式训练 seed 固定为 `7/17/27/37/47/57/67/77`；ShuffleKD 与 NoWP 使用其中
  `7/17/27` 三个诊断 seed。

### 稳定路线

- 核心 `2×2`：5 个匹配训练 seed；
- RuleKD：5 个匹配训练 seed；
- NoWP：3 个诊断训练 seed；
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
