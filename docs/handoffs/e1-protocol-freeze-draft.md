# D1 决策记录 + E1 协议冻结清单（草案）

- 状态：草案，供研究所有者审核；审核通过后由所有者正式记录到 `TASKS.md` 与 `CHANGELOG.md`。
- 前置事实：优化方向已宣告崩溃，完全采用稳定方向执行 S1/S2。S1 验收已通过。

## 1. D1 唯一路线决策（草案）

- **决策**：选择**稳定路线**。
- **理由**：优化方向已宣告崩溃（其 O0–O3 前置门无法满足），按 Roadmap「优化路线满足全部条件时选择
  优化路线，否则选择已通过 S1 的稳定路线」，稳定路线已通过 S1，故选中。
- **后果**：
  - 未选路线（优化路线）冻结为诊断/附录，不进入确认性实验，其结果不参与统计。
  - 稳定路线只形成「完整、可复现的方法实验」，不主张中科院二区保证、未见拓扑泛化或 8-AGV 规模泛化。
  - 若最终选择稳定路线，E2 必须按冻结协议从头执行正式训练与评估（S2 预备结果不得作为正式实验）。

## 2. E1 冻结清单

| # | 冻结项 | 冻结内容（草案） | 状态 |
|---|---|---|---|
| 1 | 代码 commit | `codex/stable` 在 E1 结束时的 `git rev-parse HEAD`（当前 `bf08c5f`，S2 后前移） | 待 E1 定格 |
| 2 | 环境 | 3 AGV、任务目标 9、动态入库（`batch_interval 40`、`batch_size_range [1,3]`）、能源 `1.10/0.30/0.80` | 已冻结（S1） |
| 3 | A* 教师语义 | legacy 行为：`reservation_mode=legacy` + `coordinator_yield_action=right`；只提供路径/运动先验，不做任务分配/优先级/充电决策 | 已冻结（S1） |
| 4 | 离线 LLM 教师语义 | 二维 `task_commitment`/`local_assertiveness`，整记录 validity×共享 OOD reliability；3-AGV 数据集已生成（`deepseek_medium_3ag_400_v2.jsonl`，400 条） | 已解决 |
| 5 | 规则层 | 任务队列、合法目标、固定阈值充电安全、动作合法性与硬约束；不输出逐步动作/不分配 AGV | 已冻结 |
| 6 | 奖励 | 环境代码 + `waypoint_reward 0.01`、优先级奖励 `5.0*priority_weight`、team reward；训练/执行零在线 LLM | 由代码 commit 冻结 |
| 7 | 训练 seed | 核心 2×2 与 RuleKD 各 `7/17/27/37/47`；NoWP 诊断 `7/17/27` | 已定 |
| 8 | 评估 seed | 正式评估 `200–209 × 20 episodes`，确定性动作 | 已定 |
| 9 | 交互预算 | 正式预算 **150000 env steps/seed**（S2 实测 ~75 steps/s，单 seed ≈33 min） | 已定 |
| 10 | checkpoint 规则 | 统一 `checkpoint_final.pt`，禁止按结果选最佳 checkpoint | 已定 |
| 11 | 日志 schema | `episodes.csv`/`updates.csv`/`summary.json`/`config.json`/`runtime.json` + 评估 `aggregate.json` | 已定 |
| 12 | 失败处理 | 基础设施故障→同 seed 同配置重跑并保留记录；数值/安全失败（NaN/碰撞/能量死亡/死锁）→保留为结果，不静默重试 | 已定 |
| 13 | 统计假设 | 统计单位=训练 seed；配对比较；Holm 校正（稳定路线比较族见下）；不把 200 episodes 当独立样本 | 已定 |
| 14 | 论文主张 | 完整可复现方法实验；不主张二区、未见拓扑、8-AGV、执行期无需 A*、LLM 蒸馏优于全部基线 | 已定 |

### 2.1 稳定路线实验预算（对照矩阵）

| 组 | 训练 seed | 说明 |
|---|---|---|
| MAPPO-WP（核心基线） | 5（`7/17/27/37/47`） | 无 A*KD、无 LLMKD |
| MAPPO-WP+A*KD | 5 | `reservation_kl>0` |
| MAPPO-WP+LLMKD | 5 | 二维离线 LLM |
| MAPPO-WP+A*KD+LLMKD（完整方法） | 5 | 二者皆开 |
| RuleKD | 5 | 规则生成同维标签 |
| NoWP | 3（`7/17/27`） | 诊断：waypoint 槽置零 + 关 waypoint reward/A*KL |
| Heuristic-Dispatcher+A* | 无训练 | 非学习端到端基线 |

不包含：QMIX、ShuffleKD、未见拓扑、同图 8-AGV。

### 2.2 确认性比较族（稳定路线，草案）

- 完整方法 vs MAPPO-WP
- A*KD 主效应
- LLMKD 主效应
- 完整方法 vs RuleKD

（QMIX 比较删除；Holm 校正按上述 4 项比较族，具体以你最终裁定为准。）

## 3. 缺口状态（已解决 2 / 待办 1）

### 缺口 #1：离线 LLM 二维教师的 3-AGV 数据集 —— ✅ 已解决

- 已按方案 B 离线生成 3-AGV 二维标签：`artifacts/stable/labels/deepseek_medium_3ag_400_v2.jsonl`
  （400 条，5 类配额 120/100/80/60/40，观测维度 615，provider `deepseek:deepseek-v4-flash`）。
- 已派生 RuleKD 控制集：`artifacts/stable/labels/rule_kd_3ag_v1.jsonl`（400 条）。
- LLM 标签 SHA-256：`a480360c981be11cc5390f899cd97e9e9b8ca4ee0007bb909479c94f81f0668f`。

### 缺口 #2：正式实验冻结配置（3 AGV / 目标 9）—— ✅ 已解决

- 已生成 6 个 3-AGV 冻结配置（`configs/stable_*.yaml`）：`mappo_wp`、`mappo_wp_astar_kd`、
  `mappo_wp_llm_kd`、`mappo_wp_astar_llm_kd`、`rule_kd`、`mappo_no_wp`，均带
  `environment_step_budget: 150000` 与冻结能源，并配冻结断言测试
  （`tests/test_s1_stable_route.py::test_stable_formal_configs_freeze_the_contract`，5 项通过）。

### 缺口 #3：服务器 GitHub SSH —— 待办

- 现状：服务器 `git fetch` 报 `Permission denied (publickey)`。
- 方案：服务器生成 ed25519 密钥 + 公钥加到 GitHub（步骤已给），或改用 HTTPS + token。

## 4. 下一步

1. 你正式记录 D1 决策（选稳定路线）与 E1 冻结结果到 `TASKS.md`/`CHANGELOG.md`（我可先备好文字）。
2. 你同步 `codex/stable` 到服务器并跑正式训练（E2 核心 2×2 + RuleKD 各 `7/17/27/37/47`、NoWP
   `7/17/27`，GPU 0，`nohup`，产物落 `artifacts/stable/formal_training/<slug>/seed_<NNN>/`）。
3. 正式评估 `200–209 × 20`（确定性动作，`checkpoint_final.pt`），我逐 seed 分析。
