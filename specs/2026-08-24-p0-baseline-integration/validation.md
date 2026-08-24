# Validation — P0 基线整理与双分支建立

## 验收标准

### 内容与治理

- [ ] `specs/mission.md`、`specs/tech-stack.md`、`specs/roadmap.md` 是唯一活动宪章；
- [ ] 根 `CONSTITUTION.md` 已删除，且迁移审计证明没有有效条款丢失；
- [ ] `TASKS.md` 与 P0、O0–O3、S1–S2、D1、E1–E3 一一对应；
- [ ] 每个阶段只有一个 feature spec/当前任务入口，无重复 G2/G3 任务包；
- [ ] 支持旧 Phase 3/4 artifact 和配置复现的历史文档位于 `docs/archive/` 并保留原路径索引；
- [ ] 活动 manifest、协议、TASKS 和代码合同中不存在 8-AGV 压力实验；
- [ ] 两个未提交 rejected layout 草案已删除，仓库中不存在其归档、复制或活动引用；
- [ ] `docs/handoffs/stable-route-engineer.md` 字段完整且与新宪章一致。

### 代码与行为

- [ ] A* 诊断默认关闭时动作轨迹与变更前一致；
- [ ] `NOOP/RIGHT` 只改变明确的 coordinator yield 因子；
- [ ] planner→mask→coordinator→executed 计数守恒，stall reason 可聚合且未知项不被误分类；
- [ ] layout preview 不创建环境、不加载 checkpoint、不运行策略；
- [ ] layout preview 的有效输入、无效输入、输出路径和 CLI 行为都有回归测试；
- [ ] P0 未改变训练算法、教师语义、奖励、观测、能源参数或正式 seed。

### Git 与可复现性

- [ ] 所有 P0 修改按 P0-A 至 P0-F 形成聚焦 commit，并在相应 commit 更新 CHANGELOG；
- [ ] `artifacts/p0_safety_backup/` 未被 Git 跟踪；
- [ ] 当前工作树无未归属修改和意外未跟踪文件；
- [ ] 干净 checkout 可安装并通过关键配置/CLI 验证；
- [ ] 本地 `master` 通过 fast-forward 获得 P0 最终 commit，无 merge commit 或历史重写；
- [ ] `master`、`codex/optimization`、`codex/stable` 最终指向完全相同的 SHA；
- [ ] 未 push、未创建 PR、未强制移动既有分支。

## 自动验证

在项目根目录和 `py310` 环境运行：

```powershell
python -m pytest
python -m flake8 rware llm_mappo eval train scripts figures/core
python visualize.py --help
python eval/evaluate_dynamic_ingress_astar.py --help
git diff --check
git status --short
```

此外必须运行项目现有的核心 YAML/config 解析测试、A* 诊断焦点测试、layout preview 焦点测试，
并在 `docs/p0-validation-report.md` 记录准确命令、退出码和测试计数。命令名称由实际测试文件确定，
不得在报告中用“相关测试已通过”替代原始命令。

## 人工验证

1. 对照 `docs/p0-migration-audit.md` 和安全备份，确认每个初始 dirty 文件都有唯一处置；
2. 检查归档文档能从原历史 artifact/配置追溯到新路径；
3. 检查活动文档和配置无 8-AGV 合同或 rejected layout 引用；
4. 运行一个确定性 A* 诊断 smoke，比较关闭诊断前后动作轨迹；
5. 运行一个 layout preview smoke，确认未实例化环境或策略；
6. 审阅稳定路线工程师交接，确认 S1 范围、冻结环境、权限和长任务边界无歧义；
7. 在创建分支前记录 P0 最终候选 SHA，并确认 `master` 可 fast-forward；
8. 创建分支后比较三个引用的 SHA，必须完全相同。

## 无循环分支落位规则

`specs/roadmap.md` 的 P0 完成状态、P0 验证报告、CHANGELOG 和稳定路线交接必须在当前 feature
分支的最终 P0 commit 中完成。随后只执行以下只改变 Git 引用、不再修改文件的操作：

1. fast-forward 本地 `master` 到最终 P0 commit；
2. 从该 SHA 创建 `codex/optimization` 和 `codex/stable`；
3. 验证三个引用一致。

分支创建后不得再为 P0 修改文件，否则必须删除尚未发布的两个新分支、补充新的最终 commit，
并从新的 `master` SHA 重新创建；禁止让三个分支以不同 SHA 结束 P0。

## 合并标准

- [ ] `plan.md` 所有任务完成；
- [ ] 所有自动与人工验证通过，报告无未解释失败；
- [ ] P0 Roadmap 状态为 `[x] complete`；
- [ ] CHANGELOG 已按任务组更新；
- [ ] 研究所有者审核迁移审计、验证报告、交接文档和最终 commit 列表；
- [ ] 三个目标分支引用一致；
- [ ] 无训练、长评估或路线特有实现混入 P0。
