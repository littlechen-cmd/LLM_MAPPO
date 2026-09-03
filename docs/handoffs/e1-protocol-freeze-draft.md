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
| 4 | 离线 LLM 教师语义 | 二维 `task_commitment`/`local_assertiveness`，整记录 validity×共享 OOD reliability | **缺口 #1（3-AGV 数据集）** |
| 5 | 规则层 | 任务队列、合法目标、固定阈值充电安全、动作合法性与硬约束；不输出逐步动作/不分配 AGV | 已冻结 |
| 6 | 奖励 | 环境代码 + `waypoint_reward 0.01`、优先级奖励 `5.0*priority_weight`、team reward；训练/执行零在线 LLM | 由代码 commit 冻结 |
| 7 | 训练 seed | 核心 2×2 与 RuleKD 各 `7/17/27/37/47`；NoWP 诊断 `7/17/27` | 已定 |
| 8 | 评估 seed | 正式评估 `200–209 × 20 episodes`，确定性动作 | 已定 |
| 9 | 交互预算 | 正式环境 step/训练预算：**待定**（由 S2 资源估算 + 你决策；候选 `150000 steps` 见 manifest） | **待定** |
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

## 3. 三个缺口与可选方案（需你决策）

### 缺口 #1：离线 LLM 二维教师的 3-AGV 数据集

- 现状：现有离线标签是 5-AGV Phase 4 数据集（`deepseek_medium_5ag_400_v2_repaired_r2.jsonl`，400 条）。
- 方案 A（推荐先探）：确认 `OfflineSemanticTeacher` 的近邻检索是否 AGV 数无关；若观测特征与 AGV 数
  解耦，可复用 5-AGV 标签（需审计其 observation 维度/特征语义）。
- 方案 B：离线重新生成 3-AGV 二维标签集（一次性离线生成，训练/执行期仍零在线 LLM；需你的 LLM API 预算）。
- 方案 C：从 5-AGV 标签中按场景类型抽取/降采样适配到 3-AGV（需验证标签一致性）。
- 影响：MAPPO-WP+LLMKD、MAPPO-WP+A*KD+LLMKD、RuleKD 三个组的正式训练都依赖此数据集。

### 缺口 #2：正式实验冻结配置（3 AGV / 目标 9）

- 现状：`g3_core_*` / `g3_q2_*` 是 5 AGV/目标 50，与稳定路线不符。
- 需为 §2.1 的 7 个组各建 3-AGV 冻结配置（基于 `s1_phase3_dynamic_ingress.yaml`（3a）与
  `s2_phase3b_dynamic_ingress_astar_kl.yaml`（3b）扩展），并各配冻结断言测试。
- 待 S2 验证训练链路后，由我逐个生成并提交你审核。

### 缺口 #3：服务器 GitHub SSH

- 现状：服务器 `git fetch` 报 `Permission denied (publickey)`。
- 方案：服务器生成 ed25519 密钥 + 公钥加到 GitHub（步骤已给），或改用 HTTPS + token。

## 4. 下一步

1. 你运行 S2 预备训练（GPU 0，seed `1/11/21` × 200），返回 `summary.json`/`episodes.csv`。
2. 我基于 S2 结果出「训练链路 + 资源估算」结论，并定稿 §2 中「交互预算」。
3. 你裁定三个缺口方案；我据此生成 §2.1 的 7 个冻结配置 + 冻结断言。
4. 你正式记录 D1 决策与 E1 冻结结果到 `TASKS.md`/`CHANGELOG.md`（我可先备好文字）。
