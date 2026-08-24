# S1 Owner-run 命令、manifest 与产物 schema

- 任务：S1 稳定路线旧 Phase 3 A* 行为兼容恢复验收。
- 状态：草案。配置冻结、最终 commit 与正式命令待架构师确认恢复方案后定稿。
- 交付对象：研究所有者（在 A600 上运行全部长任务；工程师不代跑长评估）。

## 1. 运行环境声明

- Conda：`py310`（本地 `D:\anaconda3\envs\py310\python.exe`；A600 用 `conda activate py310`）。
- 代码 commit：`codex/stable`，正式运行前用 `git rev-parse HEAD` 记录实际 SHA（审计基准
  `f8af34a`，实现后会前移）。
- 配置文件：`configs/s1_phase3_dynamic_ingress.yaml`（待创建，见 §2），运行前记录其 SHA-256。

## 2. 冻结配置（提案，待批准后创建）

基于 `configs/phase3a_r2_dynamic_ingress.yaml` 增补冻结能源字段（其余照抄）：

```yaml
environment:
  id: llm-mappo-medium-3ag-v1
  n_agents: 3
  max_steps: 1000
  battery_cost_scale: 1.1        # 冻结 1.10
  charge_threshold: 0.3          # 冻结 0.30
  charge_release_threshold: 0.8  # 冻结 0.80
  waypoint_reward: 0.01
  oracle_interaction_mask: true
  deadlock_steps: 180
  batch_interval: 40
  batch_size_range: [1, 3]
  initial_priority_label: A
  priority_schedule: null
  request_queue_size: 4
  task_completion_target: 9
# training / ppo 段照抄 phase3a_r2_dynamic_ingress.yaml（评估只读 environment 段）
```

冻结断言（配置校验需逐项断言）：`n_agents==3`、`task_completion_target==9`、
`batch_interval==40`、`batch_size_range==(1,3)`、`battery_cost_scale==1.1`、
`charge_threshold==0.3`、`charge_release_threshold==0.8`。

注意：历史 99.72% 在默认 `1.0/0.2/0.8` 下取得（`phase3a_r2_dynamic_ingress.yaml` 未写能源
字段）；S1 冻结 `1.10/0.30/0.80`，故验收阈值只要求 ≥95%，不要求复现 99.72%。

## 3. 命令

### 3.1 短 smoke（工程师本地验证，非验收）

```powershell
python eval/evaluate_dynamic_ingress_astar.py `
  --config configs/s1_phase3_dynamic_ingress.yaml `
  --seeds 300 --episodes-per-seed 2 `
  --reservation-mode legacy --coordinator-yield-action right `
  --output artifacts/stable/smoke/s1_smoke_flag_recovery.json
```

### 3.2 S1 验收（研究所有者运行）

```powershell
conda activate py310
python eval/evaluate_dynamic_ingress_astar.py `
  --config configs/s1_phase3_dynamic_ingress.yaml `
  --seeds 300 301 302 303 304 305 306 307 308 309 `
  --episodes-per-seed 20 `
  --reservation-mode legacy --coordinator-yield-action right `
  --output artifacts/stable/s1_acceptance/run_<ts>/aggregate.json 2>&1 | Tee-Object `
  artifacts/stable/s1_acceptance/run_<ts>/stdout_stderr.log
```

- 设备：CPU；并行度：单进程（脚本内串行）。
- 覆盖/续跑：输出目录已存在时先重命名备份，不静默覆盖。
- 阈值：完成率 ≥95%、碰撞 =0、能量死亡 =0、终止死锁率 ≤1%（四项同时满足才算通过）。

## 4. 产物 schema 与 run 目录

```
artifacts/stable/s1_acceptance/run_<ts>/
├─ manifest.json          # 运行元数据（命令、commit、config SHA-256、时间戳、退出码）
├─ stdout_stderr.log      # stdout/stderr 全量
└─ aggregate.json         # 聚合结果（eval CLI 原生 schema）
```

`aggregate.json`（eval CLI 原生）关键字段：

- 顶层：`reservation_mode`、`coordinator_yield_action`、`seeds[]`、`episodes`、
  `task_completion_rate`、`mean_collisions_per_episode`、`deadlock_rate`、
  `mean_energy_deaths_per_episode`、`reservation_teacher{}`、`gate{passed}`。
- 逐 seed：`seed`、`episodes`、`task_completion_rate`、`mean_collisions`、`deadlock_rate`、
  `success_rate`、`mean_energy_deaths`、`path_livelocks`、`state_deadlocks`、
  `reservation_teacher{}`、`stall_diagnostics{}`。

schema 缺口（待实现阶段确认）：eval CLI 当前只输出「逐 seed 聚合」，不输出「逐 episode 记录」；
交接 §7 要求逐 episode 记录。实现阶段需决定：接受逐 seed 聚合（现状），或在脚本里补逐 episode
落盘。正式结论以架构师裁定为准。

## 5. 进度查看 / 正常终止 / 完整性检查 / 聚合分析

```powershell
# 进度查看（验收是单进程前台，观察 stdout_stderr.log 或 tee 输出）
Get-Content artifacts/stable/s1_acceptance/run_<ts>/stdout_stderr.log -Tail 20

# 正常终止判定：aggregate.json 存在且含 gate 段、退出码 0
python -c "import json;d=json.load(open('artifacts/stable/s1_acceptance/run_<ts>/aggregate.json',encoding='utf-8'));print(d['gate'])"

# 完整性检查：seeds 覆盖 300-309 各 20 episodes、无缺失
python -c "import json;d=json.load(open('.../aggregate.json',encoding='utf-8'));print(sorted(s['seed'] for s in d['seeds']), [s['episodes'] for s in d['seeds']])"

# 聚合分析（逐 seed 汇总）
python eval/aggregate_formal_results.py artifacts/stable/s1_acceptance/run_<ts>/aggregate.json
```

## 6. 失败分类与处理

| 原因 | 判定 | 处理 |
|---|---|---|
| `target_reached` | 正常达成 9 件 | 计入完成率 |
| `deadlock` | 终止死锁 | 计入死锁率；>1% 即失败 |
| `time_limit` | 1000 步未达成 | 计入未完成 |
| `energy_failure` | 能量死亡 | 计入能量死亡率；>0 即失败 |
| 基础设施故障 | 进程/IO/环境异常 | 允许同 seed、同配置重跑，保留故障记录 |
| 数值/安全失败 | NaN/碰撞/能量死亡 | 保留为结果，不得静默重试 |

## 7. 任务标注（禁止混用）

- `artifacts/stable/smoke/`：短 smoke（工程师验证，非验收）。
- `artifacts/stable/s1_acceptance/`：S1 验收（研究所有者 300–309 × 20）。
- `artifacts/stable/predecision/`：S2 预备训练（仅 S1 通过后、A600 空闲时，用预声明非正式 seed）。
- E2 正式评估（`200–209`）在 D1 决策后另行冻结协议，与本命令隔离。

## 8. 禁止主张

不依据 smoke 或单次验收声称 S1 通过；不主张复现 99.72%；不主张未见拓扑或 8-AGV 泛化；不把
S2 预备结果计入最终统计。
