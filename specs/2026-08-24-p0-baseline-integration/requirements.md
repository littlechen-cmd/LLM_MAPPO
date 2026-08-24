# Requirements — P0 基线整理与双分支建立

## 背景

当前 `feature/phase4-parallel-cuda` 比 `master` 超前，且工作区同时包含代码修改、文档迁移、
旧 G3/8-AGV 规划、重复任务包、静态布局预览工具和未提交的 rejected layout 草案。P0 必须先
把这些内容按来源和用途拆分，形成可验证的共享基线，再建立优化与稳定两条正式分支。

本规格只定义 P0。不得在 P0 中实施优化路线 O0/O1/O3 的算法或环境设计，也不得实施稳定路线
S1 的旧 Phase 3 行为恢复。

## 范围

### 1. 修改盘点与可恢复保护

- 盘点所有 tracked 修改、删除、未跟踪文件、当前分支、HEAD、`master` 和远端引用；
- 在任何删除、移动或拆分提交前，保存当前 dirty diff、未跟踪文件清单和必要哈希到不进入 Git
  提交的 `artifacts/p0_safety_backup/`；
- 安全备份包含 tracked dirty patch、删除项清单，以及除两个 rejected layout 草案外所有未跟踪
  文件的内容副本与 SHA-256；rejected layout 按研究所有者决定不进入任何备份或归档；
- 为每个文件记录唯一处置：共享基线保留、历史证据归档、由新 specs 替代后删除、路线特有内容
  延后、未提交 rejected layout 删除；
- 无法确认归属的修改必须暂停并交由研究所有者决定，禁止猜测。

### 2. 文档治理迁移

- `specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md` 保持唯一项目宪章；
- 将仍能支撑旧 Phase 3/4 artifact、配置或复现口径的历史实验文档迁入 `docs/archive/`；
- `docs/archive/README.md` 必须记录每份归档文档的原路径、新路径和所支持的 artifact/配置；
- 删除已被 specs 覆盖的根 `CONSTITUTION.md`、旧 Constitution/TASKS 交接包和重复任务包；
- 保留并重构根 `TASKS.md`，使阶段和顺序一一对应 P0、O0–O3、S1–S2、D1、E1–E3；
- 保留 `CHANGELOG.md`，P0 每个完成的聚焦提交均写入对应变更；
- 每个 Roadmap phase 后续只允许一个 feature spec 目录和一个当前任务入口。

### 3. 共享代码与工具

- 当前 A* 协调回退开关、动作流水线和停滞诊断属于共享诊断基础设施；通过行为等价、计数守恒、
  回归和静态检查后进入共享基线；
- `visualize.py --layout-preview` 属于共享静态审查工具；通过输入校验、输出和 CLI 测试后进入
  共享基线；
- 两个未提交的 `rejected_draft` 布局文件不保留、不归档、不提交，P0 中删除；O3 必须从零
  重新设计真正未见拓扑；
- 旧 G3 的活动 8-AGV 压力合同必须从 manifest、协议和任务中移除；其中仅具历史意义的设计可
  归档为非活动参考，不得进入活动配置或代码合同；
- P0 不改变 A* 教师语义、奖励、观测、能源参数、正式 seed 或训练算法。

### 4. Git 集成与分支

- P0 规格和实施均在当前 `feature/phase4-parallel-cuda` 分支完成，不额外创建 worktree；
- 修改按 P0-A 至 P0-F 拆成可独立审查的聚焦 commit；
- 所有验证通过后，将当前分支 fast-forward 合入本地 `master`；若不再满足 fast-forward，必须
  停止并由研究所有者确认新的合并方案；
- 从同一个最终 `master` SHA 创建 `codex/optimization` 与 `codex/stable`；
- 研究所有者负责 push；Codex 不在 P0 中向远端发布。

### 5. 稳定路线交接

P0 最后一个任务组必须创建 `docs/handoffs/stable-route-engineer.md`，至少包含：

- 宪章、Roadmap、S1 目标和稳定路线允许/禁止主张；
- `codex/stable` 分支及“包含本交接文档的最终 P0 commit”为基准的定位方式；
- 3 AGV、目标 9、动态入库、`1.10/0.30/0.80` 的冻结合同；
- 旧 Phase 3 artifact/配置/代码历史入口；
- 项目工程师允许修改范围、禁止修改范围、短验证责任和长任务禁令；
- `artifacts/stable/`、`artifacts/stable/predecision/` 数据边界；
- S1 验收阈值、owner-run 命令交付要求、结果分析要求、已知风险和标准交接字段。

## 不在范围内

- 不运行训练、多 seed 评估或长回放；
- 不实现 O0/O1 的 A* 教师重设计；
- 不实现 O3 的未见拓扑；
- 不恢复 S1 的旧 Phase 3 A* 行为；
- 不创建 `artifacts/optimization/` 或 `artifacts/stable/` 的实验结果；
- 不 push、不开 PR、不删除远端分支；
- 不把 rejected layout 草案转为归档或候选配置。

## 决策

- 当前 A* 诊断和静态布局预览可作为共享基线，但必须分别通过焦点测试；
- rejected layout 草案全部删除，O3 从零开始；
- 活动 8-AGV 合同退出新 Roadmap；
- 历史结果文档按“支持 artifact/配置复现”标准归档，重复治理文档删除；
- 本地 `master` 合并和两个正式分支创建由 Codex 执行，研究所有者负责最终审核与 push。

## 开放问题

无。遇到未列明且无法按上述处置规则唯一分类的文件时，P0 实施必须暂停并请求研究所有者决定。
