# P0 工作区迁移审计

## 1. 盘点快照

- 盘点时间：2026-08-24（Asia/Shanghai）
- 当前分支：`feature/phase4-parallel-cuda`
- 当前 HEAD：`14b31d6d65c7cdc96d87ce6b867effb5be1ad2d5`
- 本地及远端 `master`：`0697791dc23add168bd348ad8207d0f3171628fd`
- 远端当前 feature：`f6f06507f9d7c0e9d89873e0a7990934bde6fe65`
- 当前 HEAD 相对 `master`：ahead 14、behind 0，可保持 fast-forward 路径；
- 当前分支相对远端同名分支：ahead 7、behind 0；
- 初始工作区：30 个有实际内容差异的 tracked 条目、1 个内容哈希未变化的
  `rware/__init__.py` 状态条目、7 个未跟踪文件；
- 完整 porcelain 状态、tracked patch、删除清单和未跟踪清单位于不进入 Git 的
  `artifacts/p0_safety_backup/`。

## 2. 安全备份核验

- `dirty.patch` 已生成，并能对盘点时工作树通过反向应用检查；
- 除两份 rejected layout 外，5 个未跟踪文档均按原相对路径复制到
  `artifacts/p0_safety_backup/untracked/`；
- `sha256-manifest.txt` 记录上述 5 个副本、patch、盘点和删除清单的 SHA-256；
- 两份 rejected layout 在 manifest 中显式标记为 `EXCLUDED_NO_BACKUP`，未进入任何
  安全备份；
- `.gitignore` 的 `/artifacts/` 规则覆盖整个安全备份，P0 不提交该目录。

## 3. 唯一处置矩阵

“来源”均指盘点时路径和状态；“目标”是对应任务组完成后的唯一落点。归档项保持历史正文
不被改写，并由 P0-B 的归档索引标明原路径及其支持的 artifact/config。标为“路线延后”的
材料不是已批准方案，也不得绕过 O0 人工设计门。

| 来源 | 状态 | 唯一分类 | 目标或最终处置 | 责任组 | 理由 |
|---|---|---|---|---|---|
| `CHANGELOG.md` | M | 共享保留 | 根目录保留并按 P0 组清理/追加 | P0-A–F | 唯一变更日志；旧 8-AGV 表述在 P0-B 清理 |
| `CONSTITUTION.md` | M | 替代后删除 | 有效条款核对进入 `specs/*.md` 后删除 | P0-B | 新 specs 是唯一活动宪章 |
| `TASKS.md` | M | 共享保留 | 重写为 P0/O0–O3/S1–S2/D1/E1–E3 | P0-B | 根任务清单继续作为唯一执行入口 |
| `configs/g3_experiment_manifest.yaml` | M | 共享保留 | 清理为双路线共享/正式合同，不保留 8-AGV | P0-B | manifest 仍是实验冻结入口 |
| `docs/former_TASKS.md` | D | 替代后删除 | 保持删除 | P0-B | 旧任务入口由新 `TASKS.md` 和 specs 替代 |
| `docs/g2_charging_core_selection.md` | D | 历史归档 | `docs/archive/g2_charging_core_selection.md` | P0-B | 支撑 `1.10/0.30/0.80` 冻结证据 |
| `docs/phase2_medium_3ag_expert_baseline.md` | D | 历史归档 | `docs/archive/phase2_medium_3ag_expert_baseline.md` | P0-B | 支撑旧专家基线复现 |
| `docs/phase2_medium_astar_bc_800.md` | D | 历史归档 | `docs/archive/phase2_medium_astar_bc_800.md` | P0-B | 支撑旧 A* 行为克隆 artifact |
| `docs/phase2_small_astar_bc_800.md` | D | 历史归档 | `docs/archive/phase2_small_astar_bc_800.md` | P0-B | 支撑旧 A* 行为克隆 artifact |
| `docs/phase2_waypoint_reward_500.md` | D | 历史归档 | `docs/archive/phase2_waypoint_reward_500.md` | P0-B | 支撑 waypoint reward 历史结论 |
| `docs/phase3_dynamic_ingress_plan.md` | D | 历史归档 | `docs/archive/phase3_dynamic_ingress_plan.md` | P0-B | 支撑旧 Phase 3 动态入库合同 |
| `docs/phase3_handover.md` | D | 历史归档 | `docs/archive/phase3_handover.md` | P0-B | 支撑旧 Phase 3 代码/绘图复现路径 |
| `docs/phase4_400_label_audit.md` | D | 历史归档 | `docs/archive/phase4_400_label_audit.md` | P0-B | 支撑旧离线标签审计 artifact |
| `docs/phase4_5ag_astar_preflight.md` | D | 历史归档 | `docs/archive/phase4_5ag_astar_preflight.md` | P0-B | 支撑旧 Phase 4 A* preflight |
| `docs/phase4_dual_semantic_spec.md` | D | 历史归档 | `docs/archive/phase4_dual_semantic_spec.md` | P0-B | 支撑双语义 checkpoint/标签合同 |
| `docs/大规模动态仓储LLM-MAPPO总体方案.md` | D | 历史归档 | `docs/archive/大规模动态仓储LLM-MAPPO总体方案.md` | P0-B | 保留旧总体方案的配置与设计来源 |
| `eval/evaluate_dynamic_ingress_astar.py` | M | 共享保留 | 诊断链路经回归后保留 | P0-C | 双路线共享、行为中性的 A* 诊断 |
| `llm_mappo/phase2_expert.py` | M | 共享保留 | 诊断开关与动作流水线经回归后保留 | P0-C | 双路线共享的 A* 专家基础设施 |
| `plan/experiment-protocol.md` | M | 共享保留 | 对齐新 Roadmap，移除活动 8-AGV 合同 | P0-B | 继续作为活动实验协议 |
| `plan/g3-architecture-task-package.md` | D | 替代后删除 | 保持删除 | P0-B | 旧 G3 单体任务入口被 P0/O/S specs 替代 |
| `plan/progress.md` | M | 共享保留 | 改写为新 Roadmap 进度索引 | P0-B | 继续提供全局进度摘要 |
| `plan/review/g3-architecture-task-package-review.md` | M | 历史归档 | `docs/archive/planning/g3-architecture-task-package-review.md` | P0-B | 旧 G3 审核，仅作非活动历史证据 |
| `plan/review/method-experiment-traceability.md` | M | 共享保留 | 对齐双路线主张—证据矩阵 | P0-B | 继续承担活动追溯职责 |
| `plan/review/q2-experiment-evidence-sync-review.md` | M | 历史归档 | `docs/archive/planning/q2-experiment-evidence-sync-review.md` | P0-B | 旧 G3/Q2 同步审核，不再作为活动合同 |
| `plan/task-packets/constitution-v1.md` | D | 替代后删除 | 保持删除 | P0-B | 旧宪章交接由 specs 替代 |
| `plan/task-packets/g3-4-g3-7-eight-seed-statistics.md` | D | 替代后删除 | 保持删除 | P0-B | 旧 Gate 任务包由新路线规格替代 |
| `plan/task-packets/g3-5-required-comparisons.md` | D | 替代后删除 | 保持删除 | P0-B | 旧 Gate 任务包由新路线规格替代 |
| `plan/task-packets/g3-architecture-package.md` | D | 替代后删除 | 保持删除 | P0-B | 重复 G3 入口 |
| `plan/task-packets/q2-experiment-evidence-sync.md` | D | 替代后删除 | 保持删除 | P0-B | 旧 Q2 任务入口由新 Roadmap 替代 |
| `rware/__init__.py` | M/同哈希 | 共享保留 | 内容不改，刷新索引状态 | P0-B | 工作树与 HEAD 哈希均为 `1828114…905` |
| `visualize.py` | M | 共享保留 | 静态布局预览经回归后保留 | P0-D | 双路线共享的无环境预览工具 |
| `configs/layouts/candidates/g3_unseen_central_bottleneck_rejected_draft.txt` | ?? | rejected layout 删除 | 永久删除且不备份/归档 | P0-D | 研究所有者要求 O3 从零设计 |
| `configs/layouts/candidates/g3_unseen_narrow_aisle_rejected_draft.txt` | ?? | rejected layout 删除 | 永久删除且不备份/归档 | P0-D | 研究所有者要求 O3 从零设计 |
| `plan/review/g3-6-fleet-load-amendment.md` | ?? | 历史归档 | `docs/archive/planning/g3-6-fleet-load-amendment.md` | P0-B | 仅记录已放弃 8-AGV 决策，不进入活动合同 |
| `plan/task-package/g2-1-astar-throughput-diagnosis.md` | ?? | 历史归档 | `docs/archive/planning/g2-1-astar-throughput-diagnosis.md` | P0-B | 支撑 A* 吞吐诊断来源与历史 artifact |
| `plan/task-package/g2a-astar-teacher-role-alignment.md` | ?? | 路线延后 | `docs/archive/planning/optimization-input-g2a-astar-teacher-role-alignment.md` | P0-B/O0 | 仅作 O0 未批准输入，O0 必须重新设计并人工批准 |
| `plan/task-package/g3-6-fleet-load-stress.md` | ?? | 替代后删除 | 删除；安全备份保留原副本 | P0-B | 已放弃的活动 8-AGV 任务包 |
| `plan/task-package/g3-architecture-task-package.md` | ?? | 替代后删除 | 删除；安全备份保留原副本 | P0-B | 与已删除 tracked G3 包重复且被 specs 替代 |

## 4. 实施边界

- P0-B 只能执行上表已经冻结的文档处置，不引入 O0/O3/S1 的算法或环境实现；
- P0-C、P0-D 只有在对应回归验证通过后才能接纳现有代码；若验证表明行为改变，必须停止并
  回到规格决策，不得凭结果重分类；
- P0-E 前必须消除全部非 artifacts 工作区条目，并核对每个初始条目恰好有一个最终处置；
- 本审计不将任何旧 8-AGV 或 G2A 文档升级为活动实验合同。
