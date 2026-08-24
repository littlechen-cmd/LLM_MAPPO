# S1 历史行为差异审计与最小恢复方案

- 任务 ID：S1（稳定路线旧 Phase 3 A* 行为兼容恢复）
- 工作分支：`codex/stable`（隔离 worktree `.wt-stable`，HEAD `6fcb7d3`）
- 定位：本文件是 S1 的第一步交付物，提交核心架构师确认；确认前不实现、不训练、不评估。
- 证据方法：全程只读（`git log/show/diff` 跨提交追溯），未改动任何代码、未启动任何训练/评估。

## 1. 精确来源 commit、入口与当前差异

### 1.1 来源 commit

- 历史 99.72% 门禁结果对应 `artifacts/previous_artifacts/phase3_dynamic_ingress_astar_gate_40step_n9_10x20.json`
  （3 AGV、目标 9、动态入库、200 episodes、完成率 0.9972、碰撞 0、死锁率 0.005）。
- `docs/archive/planning/g2-1-astar-throughput-diagnosis.md` 记载 A* 最小修复提交为
  `f6f0650`，修复前基准为 `119f20d`。因此旧 Phase 3 A* 行为的代码锚点是 `119f20d`
  （`repair experiment checkpoint reproducibility`，2026-08-20），即 `f6f0650` 之前的
  最后一个 A* 代码状态。
- 注意：该 99.72% 是在历史默认能源配置下取得（`configs/phase3a_r2_dynamic_ingress.yaml`
  未写能源字段）。S1 冻结环境改用 `1.10/0.30/0.80`，因此 S1 只要求 ≥95%，不要求复现
  99.72%。锚点的精确产物归属仍需以 artifact 元数据二次确认，不以手写短 SHA 作为复现依据。

### 1.2 入口

- 评估入口：`eval/evaluate_dynamic_ingress_astar.py`（CLI `main()` → `evaluate_dynamic_astar()`）。
- 控制器入口：`llm_mappo/phase2_expert.py` 的 `AStarExpert.act()` / `action_preferences()` /
  `_reserved_action_preferences()` / `_coordinate_actions()`。
- 规划入口：`llm_mappo/planner.py` 的 `AStarPlanner` 与 `ReservationTable`。
- 环境入口：`llm_mappo/phase2.py` 的 `Phase2Warehouse`。

### 1.3 当前差异（`119f20d` → `6fcb7d3`）

共 3 个提交改动 A* 相关代码：`f6f0650`（bounded 终点预约修复）、`965a4c3`
（冻结对照基线）、`ddf96d9`（共享 A* 诊断）。合计 6 文件、656 insertions / 72 deletions：

| 文件 | 变化 | 性质 |
|---|---|---|
| `llm_mappo/planner.py` | +202 | 行为：`ReservationTable.reserve` 增加 `terminal_hold_steps/persistent`；新增 `TemporalSearchResult` |
| `llm_mappo/phase2_expert.py` | +167 | 行为 + 观测：协调器 yield 动作、缓存签名、统计计数器 |
| `eval/evaluate_dynamic_ingress_astar.py` | +304 | 观测：门禁 CLI 增加 `--reservation-mode`、`--coordinator-yield-action`、`--stall-diagnostics` |
| `llm_mappo/phase3_training.py` | +38 | 观测：`include_waypoint_features` 透传、`_expert_statistics` 重构 |
| `llm_mappo/phase2.py` | +10 | 观测：`include_waypoint_features` 开关（默认 True 等价） |
| `llm_mappo/types.py` | +7 | 观测：`PathPlan` 新增字段 |

## 2. 行为差异逐条归因（旧 → 当前）

| # | 差异点 | 旧行为（119f20d） | 当前行为（6fcb7d3） | 是否有恢复开关 |
|---|---|---|---|---|
| D1 | 协调器 yield 动作 | `Action.RIGHT`（让行时原地右转） | 默认 `Action.NOOP`（让行时等待） | 有：`coordinator_yield_action="right"` |
| D2 | 终点预约 | 终点格整 horizon 持续预约 | 默认 `terminal_hold_steps=2` 有界预约 | 有：`legacy_terminal_reservation=True` |
| D3 | 缓存签名 | 只含 agent 状态 + 目标 | 追加 `_cur_steps`，跨步缓存失效 | 无开关；**输出中性**（miss 即重算），smoke 实证 `cache_hits=0` |
| D4 | dead/pickup/picking 预约 | 一律整 horizon 预约 | dead→`persistent=True`（等价整 horizon）；pickup→`terminal_hold_steps=1`；picking→`picking_lock_steps` | 无开关；**残余小差异**（静止期预约时长），smoke 显示可忽略 |
| D5 | 偏好路径 | `_preferences_for_timed_path(dir, waypoints)` | `_preferences_for_action(result.first_action)` | **代码比对已验证等价**（`_preferences_for_action` 对 NOOP 走 `_noop_preferences`，与旧 `_preferences_for_timed_path` 的 NOOP 分支一致） |

补充：`_edge_swap_yields` 是 `ddf96d9` 从旧 `_occupied_target_yields` 的 `elif` 分支拆出的
独立函数，二者 yield 集合等价（yield 对象相同，仅用于区分停滞原因码），判定为行为中性。
综上：D1/D2 有开关可恢复且为**主要行为差异**；D3 输出中性、D4 为静止期预约时长的小差异、
D5 已验证等价。即「第一步纯开关」可恢复全部主要行为，D3–D5 无需代码改动。

## 3. 恢复范围对七个子系统的影响

1. **训练接口**：`phase3_training.py` 仅新增 `include_waypoint_features` 透传与统计重构，
   不改 A* 教师调用语义；恢复 A* 行为不改训练入口。
2. **教师标签**：`AStarExpert.action_preferences` 输出即蒸馏软标签；D1–D5 都改变该输出的
   首选动作分布，因此恢复必须落在同一 `AStarExpert` 内。
3. **动作掩码**：`_mask_and_normalize` 与 `Phase2Warehouse.action_masks()` 未被 A* 差异改动；
   恢复不触碰。
4. **协调器**：D1 是唯一明确的行为差异（RIGHT→NOOP）；恢复用 `coordinator_yield_action="right"`。
5. **奖励**：奖励由环境规则层决定，A* 差异未改奖励；恢复不触碰。
6. **观测**：`include_waypoint_features` 默认 `True` 等价旧观测；S1 不启用 `False`（NoWP），
   恢复不触碰。
7. **安全层**：规则层硬约束未被 A* 差异改动；恢复不触碰。

结论：恢复范围收敛到 `AStarExpert` 与 `ReservationTable` 的 D1–D5，其余七个子系统的其它部分
无需改动。

## 4. 最小恢复方案（分两步，待架构师确认）

### 4.1 第一步：纯开关恢复（零代码改动，先验证）

用现有 CLI 开关恢复 D1、D2：

```powershell
python eval/evaluate_dynamic_ingress_astar.py `
  --config configs/phase3a_r2_dynamic_ingress.yaml `
  --seeds 300 --episodes-per-seed 2 `
  --reservation-mode legacy `
  --coordinator-yield-action right `
  --output artifacts/stable/s1_smoke_flag_recovery.json
```

判定：若该 smoke 的动作轨迹与 `119f20d` 同 seed 轨迹一致（或与历史门禁指标同量级），
则 D1–D2 开关已足够，D3–D5 判为行为中性/可忽略；否则进入第二步。

**已执行 smoke（2026-08-25，`codex/stable` @ `6fcb7d3`，seed 300 × 2 episodes，默认能源）**：

| 模式 | 完成率 | 碰撞 | 死锁率 | 能量死亡 | terminal_conflicts | expanded_nodes |
|---|---|---|---|---|---|---|
| legacy + right（恢复） | 1.0 | 0 | 0 | 0 | 2071 | 698462 |
| fixed + noop（当前默认） | 1.0 | 0 | 0 | 0 | 162 | 356021 |

两点结论：
1. D1–D2 纯开关（legacy+right）可复现旧行为量级，`gate passed`；`cache_hits=0`、
   `cache_misses=309` 实证 D3（签名含 `_cur_steps`）使跨步缓存失效但输出中性。
2. 在 3 AGV/目标 9 合同下，当前默认（fixed+noop）与恢复模式的头条指标一致（均通过 smoke），
   差异仅体现在内部诊断计数；即 3-AGV 合同上恢复的意义主要是**行为保真**（对齐历史 99.72%
   证据链），而非抢救未达标门禁。这是 2-episode 单 seed 结论，正式 300–309 × 20 仍可能
   出现逐 seed 差异，不得据此声称 S1 已通过。

### 4.2 第二步：最小兼容代码改动（仅当第一步不足时）

- D3：在 `_planning_signature` 中去掉 `_cur_steps`，恢复旧缓存命中语义（或在签名之外保持
  现有观测计数器不动）。
- D4：dead/picking-lock 统一回整 horizon 预约。
- D5：若回归证明非等价，回退到 `expand_for_orientation`/`_preferences_for_timed_path` 旧路径。
- 全部改动只限 `AStarExpert`/`ReservationTable`，带回归测试，不删现有诊断埋点。

## 5. 回归、配置冻结断言、checkpoint 兼容与短 smoke

- 确定性回归：固定 seed 下 `--reservation-mode legacy --coordinator-yield-action right`
  与 `--stall-diagnostics` 开/关的动作轨迹逐步等价（计数守恒）。
- 配置冻结断言：S1 环境固定 3 AGV、目标 9、动态入库、`1.10/0.30/0.80`。冻结环境已由
  `configs/g3_experiment_manifest.yaml` 的 `route_profiles.stable.environment`（`n_agents: 3`、
  `task_completion_target: 9`、`dynamic_ingress: true`）与 `route_profiles.stable.energy`
  （`1.1/0.30/0.8`）唯一定义，`astar_teacher_contract.stable_contract` 标为
  `pending_s1_historical_behavior_restoration`；config 校验 `batch_interval>=2` 等既有约束不破坏。
- checkpoint 兼容：本恢复不改观测/动作维度、不改 `MAPPOPolicy`/checkpoint schema，只改
  A* 教师输出，checkpoint 兼容性不受影响（需在短 smoke 中确认 load/save）。
- 短 smoke 上限：`2 seeds × 2 episodes`，产物写入 `artifacts/stable/` 隔离子目录，不覆盖
  历史 JSON。

## 6. 诊断关闭行为等价与 A* 流水线/停滞诊断可用性

- 现有 `last_action_pipeline`、`statistics()`、停滞原因码与三段流水线守恒（P0-C 已验收）
  全部保留；诊断关闭时默认路径与打开时轨迹等价。
- S1 恢复不得删除 `--stall-diagnostics` 及其 schema；owner-run 正式验收可关诊断、故障复现
  可开诊断。

## 7. owner-run 验收命令与失败分类（占位，待实现后细化）

```powershell
python eval/evaluate_dynamic_ingress_astar.py `
  --config configs/phase3a_r2_dynamic_ingress.yaml `
  --seeds 300 301 302 303 304 305 306 307 308 309 `
  --episodes-per-seed 20 `
  --reservation-mode legacy --coordinator-yield-action right `
  --output artifacts/stable/s1_acceptance_300_309_20.json
```

- 配置缺口：上述命令用的 `configs/phase3a_r2_dynamic_ingress.yaml` 只定义动态入库环境（能源
  走默认值），不含冻结 `1.10/0.30/0.80`；`configs/g2_charging_retrain_candidate.yaml` 是
  5 AGV/目标 50，不匹配 S1。当前没有「3 AGV + 目标 9 + 动态入库 + `1.10/0.30/0.80`」的
  `Phase3TrainingConfig` YAML，需在实现阶段新建（或对 phase3a_r2 增补能源字段），并记录其
  SHA-256 与冻结断言。
- 阈值：完成率≥95%、碰撞=0、能量死亡=0、终止死锁≤1%。
- 失败分类：`target_reached` / `deadlock` / `time_limit` / `energy_failure` 四类，
  逐 seed 记录完成率、碰撞率、能量死亡率、终止死锁率、完成任务数/team reward。
- 聚合以 seed 为统计单位，不得用平均值掩盖安全失败或缺失 seed。

## 8. 已知风险与禁止主张

- 风险：D3–D5 若非行为中性，则纯开关恢复不足；历史 artifact 与当前日志 schema 不一致；
  Windows/Linux 进程与路径差异；恢复旧行为时污染共享接口（需以可追溯、已测试 commit 合并）。
- 禁止主张：不声称已通过 S1、不声称复现 99.72%、不声称未见拓扑或 8-AGV 泛化、不声称
  执行期无需 A*、不依据单 seed 或短 smoke 下因果结论。

## 9. 工作树状态

- 主工作树：`codex/o0-astar-teacher-redesign`（架构师在途，未触碰）。
- 隔离 worktree：`.wt-stable` → `codex/stable` @ `6fcb7d3`，工作树干净。
- 本审计未创建任何 commit；待架构师确认后再实现并创建聚焦 commit。
