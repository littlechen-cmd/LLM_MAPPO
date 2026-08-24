# Plan — P0 基线整理与双分支建立

各任务组必须按顺序执行。每组完成后形成一个聚焦 commit，并同步 `CHANGELOG.md`。任一组发现
无法归属的用户修改时立即停止，不得进入下一组。

## Task group P0-A：盘点、保护与迁移矩阵

- [x] 记录当前分支、HEAD、`master`、远端引用、ahead/behind、tracked diff、删除项和未跟踪文件；
- [x] 在 `artifacts/p0_safety_backup/` 保存不进入 Git 的 dirty diff、未跟踪文件清单与哈希，
  并验证备份可读；
- [x] 备份除 rejected layout 草案外的全部未跟踪文件内容并记录 SHA-256；两个 rejected layout
  不复制到安全备份；
- [x] 创建 `docs/p0-migration-audit.md`，逐文件记录来源、处置、目标位置、责任任务组和理由；
- [x] 将所有文件唯一分类为共享保留、历史归档、替代后删除、路线延后或 rejected layout 删除；
- [x] 对任何无法唯一分类的内容暂停并取得研究所有者决定（本次不存在无法唯一分类项）；
- [x] 仅提交迁移审计、对应 CHANGELOG 和本计划的状态同步，不提交安全备份。

## Task group P0-B：宪章、任务与历史文档迁移

- [x] 核对根 `CONSTITUTION.md` 中仍有效的条款均已进入 `specs/mission.md`、
  `specs/tech-stack.md` 或 `specs/roadmap.md`；
- [x] 将能支撑旧 Phase 3/4 artifact、配置或复现口径的历史文档迁入 `docs/archive/`，保持内容
  不被改写并记录原路径；
- [x] 创建 `docs/archive/README.md`，记录原路径、新路径及对应 artifact/配置；
- [x] 删除根 `CONSTITUTION.md`、已被 specs 替代的旧 Constitution/TASKS 交接包和重复任务包；
- [x] 重写根 `TASKS.md`，与 P0、O0–O3、S1–S2、D1、E1–E3 一一对应，保留已完成工作作为
  “历史基线证据”而不是旧活动 Gate；
- [x] 更新活动协议、manifest 和追溯文档，移除 8-AGV 压力合同，且不提前写入 O3 新布局设计；
- [x] 更新 CHANGELOG，运行文档引用、阶段 ID、重复任务入口和 Markdown 静态检查；
- [x] 创建聚焦文档迁移 commit（随本状态同步提交）。

## Task group P0-C：共享 A* 诊断基础设施

- [x] 审查 `AStarExpert` 的 `NOOP/RIGHT` 诊断开关、动作流水线和 stall reason，确认默认行为保持
  当前 `NOOP` 且关闭诊断时无额外行为变化；
- [x] 补齐 planner→mask→coordinator→executed 计数守恒、开关轨迹等价、reason 分类、CLI 和
  JSON 聚合回归测试；
- [x] 确保诊断字段与旧产物兼容，未知停滞保留为显式 `unknown_stationary`，不静默归因；
- [x] 运行焦点 pytest、完整 pytest 和规定范围 Flake8；
- [x] 更新 CHANGELOG，创建聚焦 A* 诊断 commit（随本状态同步提交）。

## Task group P0-D：共享静态布局预览与废弃内容清理

- [x] 审查 `visualize.py --layout-preview`，补齐矩形、字符集、最小 cell size、默认/指定输出、
  Pillow 缺失和 CLI 早退出测试；
- [x] 确认静态预览不创建环境、不加载 checkpoint、不运行策略；
- [x] 删除两个未提交的 `configs/layouts/candidates/*rejected_draft.txt`，不归档、不提交替代文件；
- [x] 删除或归档其余仅服务旧 8-AGV 合同的非活动内容，确保活动配置和任务无 8-AGV 引用；
- [x] 运行焦点 pytest、完整 pytest、规定范围 Flake8 和 `git diff --check`；
- [x] 更新 CHANGELOG，创建聚焦静态工具与清理 commit（随本状态同步提交）。

## Task group P0-E：干净检出与集成候选验证

- [ ] 确认工作树中没有未归属的 tracked/untracked 修改，`artifacts/p0_safety_backup/` 仍未进入
  暂存区；
- [ ] 从干净 checkout 验证 `python -m pip install -e ".[dev,train]"` 或现有等价安装合同；
- [ ] 在 `py310` 运行完整 pytest、规定范围 Flake8、核心 YAML/config 解析、关键 CLI `--help`、
  A* 诊断 smoke 和 layout preview smoke；
- [ ] 运行阶段 ID/文档引用/唯一任务入口检查与 `git diff --check`；
- [ ] 生成 `docs/p0-validation-report.md`，记录命令、退出码、测试计数、环境、commit 和未执行项；
- [ ] 更新 CHANGELOG，创建集成候选验证 commit；任何失败均阻塞 P0-F。

## Task group P0-F：稳定路线工程师交接与双分支落位

- [ ] 创建 `docs/handoffs/stable-route-engineer.md`，完整覆盖 requirements 中的交接字段；
- [ ] 交接文档明确项目工程师只负责 S1/S2 稳定路线实现、短验证、命令准备和结果分析，所有长
  训练/评估由研究所有者在 A600 运行；
- [ ] 将 `specs/roadmap.md` 的 P0 标记为 `[x] complete`，更新 CHANGELOG，并重新运行
  `git diff --check`、规格结构检查和 `git status --short`；
- [ ] 将交接文档、Roadmap 完成状态和 CHANGELOG 作为当前 feature 分支最后一个聚焦 commit，
  此后 P0 不再修改任何文件；
- [ ] 验证 `master` 是当前分支祖先且可 fast-forward；不满足时停止并请求研究所有者决定；
- [ ] fast-forward 本地 `master` 到当前分支最终 HEAD；
- [ ] 从该同一 SHA 创建 `codex/optimization` 与 `codex/stable`；若分支已存在且不指向该 SHA，
  停止并请求研究所有者决定，禁止强制移动；
- [ ] 验证 `master`、`codex/optimization`、`codex/stable` 指向同一 SHA，并向研究所有者提供
  commit 列表、分支状态和 push 命令；
- [ ] 若分支落位后发现仍需修改 P0 文件，按 validation 中的无循环落位规则撤销尚未发布的
  新分支落位并重新形成最终 commit，禁止直接让三个分支停留在不同 SHA。
