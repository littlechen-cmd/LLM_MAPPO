# Project Task List

本文件是当前实施任务的唯一入口，阶段 ID、顺序和状态与 `specs/roadmap.md` 一一对应。
阶段的详细要求与验收以其唯一 feature spec 为准；工程实现完成不等于 Gate 通过，必须先由
核心架构师审查证据。每个完成的小任务须在同一 commit 更新 `CHANGELOG.md`。

## 执行边界

- 研究所有者：批准架构、路线决策与正式协议；在批准的服务器启动全部长训练、长评估和长回放；push；
- Codex：总体推进与核心架构、优化路线实现、跨路线一致性、命令准备和结果分析；
- 项目工程师 AI：稳定路线 S1/S2 实现、短验证、命令准备和结果分析；不得修改宪章或自行通过 Gate；
- `codex/optimization` 与 `codex/stable` 共享修复只能通过明确、可追溯、已测试的 commit 合并，
  禁止直接复制覆盖；
- 长任务不得由 Codex 或项目工程师启动。

## P0 — 基线整理与双分支建立（完成）

唯一规格：`specs/2026-08-24-p0-baseline-integration/`

- [x] **P0-A** 盘点全部脏工作区内容，建立忽略的安全备份和逐文件迁移矩阵；证据见
  `docs/p0-migration-audit.md`。
- [x] **P0-B** 迁移唯一宪章、任务入口和历史复现文档，移除活动 8-AGV 合同；配置解析、
  阶段一一映射、归档哈希和活动合同检查已通过。
- [x] **P0-C** 验证并接纳行为中性的 A* 动作流水线与停滞诊断；默认/显式 NOOP 与诊断开关
  轨迹等价，三段计数守恒，焦点回归和完整测试通过。
- [x] **P0-D** 验证静态布局预览并删除 rejected layout 与废弃 8-AGV 内容；8 项焦点回归、
  184 项完整测试、Flake8、CLI 与实际 PNG smoke 通过。
- [x] **P0-E** 完成干净 checkout、完整测试、静态检查、配置/CLI/smoke 和验证报告；证据见
  `docs/p0-validation-report.md`。
- [x] **P0-F** 完成稳定路线工程师交接；交接见
  `docs/handoffs/stable-route-engineer.md`。本地 `master`、`codex/optimization` 与
  `codex/stable` 按 P0 最终 commit 同 SHA 落位。

## P1 — 优化路线 Linux 服务器执行基础设施（规格已批准）

唯一规格：`specs/2026-08-28-p1-linux-optimization-server/`

- [x] **P1-A** 同步宪章、Roadmap、协议、manifest、TASKS 与术语，修复 O1 状态误写；
- [x] **P1-B** 冻结并验证用户级 Python 3.10.19、Torch 2.10.0+cu128 与依赖约束；
- [x] **P1-C** 实现机器 inventory、五次连续空闲预检、48 小时等待与 GPU lease；
- [x] **P1-D** 将正常 O1 Gate 修正为 baseline/H12，并把 H4 隔离为失败后诊断；
- [x] **P1-E** 实现原子 shard、显式 resume、失败分类与 O1 Go receipt；
- [x] **P1-F** 实现 owner-started wait-to-O1 launcher，不启动或伪造 O2；
- [x] **P1-G** 完成本地全回归并交付唯一 Linux 安装和短 CUDA smoke 命令；
- [x] **P1-H** 研究所有者已完成 Linux 短 smoke；Codex 审核确认 P1 环境、GPU 绑定与
  provenance 通过。O1 Gate 结果见 `20260829T075212Z_7c305ea2`。

## O0 — 优化路线 A* 教师算法与架构重设计（已完成）

唯一规格：`specs/2026-08-24-o0-astar-teacher-redesign/`

- [x] 审计 waypoint、A* preference、协调器、buffer、KL、规则目标和 MAPPO 动作全链路；
- [x] 比较候选并冻结 Pure Motion Teacher、逐机器人有效掩码、H=12 paired shadow rollout、
  团队 reward confidence、EMA 和 runtime/memory No-Go；
- [x] 冻结三维离线 LLM schema、60/800 数据合同、validity×OOD reliability、共同 schedule、
  checkpoint、执行期 A* 条件依赖、日志、消融和允许主张；
- [x] 建立并版本化唯一 canonical architecture，完成输入映射与跨文档一致性审计；
- [x] 取得研究所有者书面批准；批准前未实施代码或训练。

## O1 — 优化路线角色对齐实现与静态验证（已完成）

- [x] 仅实现 O0 批准方案，遵循测试先行；
- [x] 通过教师纯度、action mask、零有效样本、buffer/KL、checkpoint 与完整回归；
- [x] 通过短 smoke，确认目标/优先级/充电改写、协调器污染和非法动作概率质量均为 0；
- [x] 不启动训练、长评估或 KL 超参数搜索；
- [x] Codex 已冻结旧 owner-only CUDA runtime/memory gate runner 与产物 schema；P1 负责 Linux 修订；
- [x] P1 Linux 环境、资源预检、GPU 绑定与短 CUDA smoke 通过；
- [x] 按资源修订将常规门禁改为 `baseline/H12`，保留 5 repeats、10 memory windows 与原阈值；
  H4 仅在 H12 失败后作为诊断，不参与正常 Go；
- [x] 将 O1 门禁作为 O2 owner job 的 fail-fast 前缀，产物和 Gate 状态仍保持独立；
- [x] 研究所有者已在 Linux 优化服务器完成 Gate，Codex 审核 artifact
  `artifacts/optimization/o1_cuda_gate/20260829T075212Z_7c305ea2`：commit
  `7c305ea24cdca34467c2e7e8a5a9d66ba1133d1e`、runtime ratio `2.880`、memory Gate
  通过。现允许进入 O2。

## O2 — 优化路线校准训练与 Go/No-Go（已完成）

- [x] Codex 已交付 receipt-bound O2 runner、紧凑产物 schema、parity smoke 与聚合器；
  正式运行仍须由研究所有者在服务器执行。
- [x] 已交付单命令 O2 matrix runner：一次 `nohup` 顺序完成 6 个正式 run，按严格身份自动跳过
  已完成项或恢复唯一合法中断项，全部完成后自动聚合 Gate；不再增加服务器性能测试。
- [x] 研究所有者已完成 `MAPPO-DG/RC-AStarKD × 107/117/127 × 150000 steps` 六个正式
  run；全部为非 diagnostic、状态 complete 且具有 final checkpoint，证据见
  `docs/evidence/o2-calibration-gate-v1.json`；
- [x] 在 MateBook 完成 Fixed/RC sampler、query、shadow、EMA、日志与计数等价性的确定性短
  受控 smoke；64 步中命中 5 次采样且所有链路计数一致，不运行 Fixed-AStarKD 长校准；
- [x] 三个 RC seed 的有效教师覆盖率为 `91.19%/89.27%/92.32%`，分别≥25%，无数值或
  接口失败；
- [x] 相对 `MAPPO-DG` 的配对 AUC 退化为 `24.01%/1.89%/-11.39%`，中位数 `1.89%≤10%`；
- [x] 正式矩阵前仅修复一次已定位的动态优先级起始标签配置错误；未按结果调整 KL、seed、
  预算或阈值。O2 只证明校准链路通过，不构成普遍性能提升主张。

## O3 — 优化路线真正未见拓扑

- [x] 重排依赖：O3 结构工作曾与 O1 服务器门禁等待并行，O2 仍被 O1 阻塞；
- [x] 按唯一规格 `specs/2026-08-26-o3-unseen-topologies/` 从零设计窄通道和中央
  瓶颈/交叉通道两个 evaluation-only 拓扑；
- [x] 冻结显式地图/坐标合同、哈希、环境 ID、接口和防泄漏协议；
- [x] 通过确定性环境、安全、接口 smoke 和哈希审计；
- [x] 不实施同图 8-AGV 压力场景，不使用布局参与训练、标签、OOD、路线或参数选择；
- [x] O3 不加载学习策略或报告性能；canonical core held-out seeds 是正式必需证据，O3 性能只可
  按 E1 预先冻结的非门槛探索矩阵在 E2 执行。
- [x] 研究所有者于 2026-08-27 批准 O3“拓扑/接口就绪”验收。

## S1 — 稳定路线旧 Phase 3 行为恢复

- [ ] 在 `codex/stable` 追溯并兼容恢复旧 Phase 3 A* 行为，不回退整个仓库；
- [ ] 冻结 3 AGV、任务目标 9、动态入库、`1.10/0.30/0.80`，保留多教师与当前日志；
- [ ] 项目工程师完成实现、回归和短 smoke，并交付 owner-run 命令；
- [ ] 研究所有者运行 `300–309 × 20`；验收完成率≥95%、碰撞=0、能量死亡=0、终止死锁≤1%。

## S2 — 稳定路线决策前预备训练（可选）

- [ ] 仅在 S1 通过、优化路线仍处 O1–O3 且研究所有者批准的服务器空闲时启动；
- [ ] 使用预声明非正式 seed，产物写入 `artifacts/stable/predecision/`；
- [ ] 结果不参与 D1、不进入最终统计；若最终选择稳定路线，E2 必须从头正式运行。

## D1 — 唯一路线决策门

- [x] O0、P1、O1、O2 与 O3 拓扑/接口门全部通过，按预注册规则选择优化路线进入 E1；
- [x] blocked 条件未触发，未降低任何既有门槛；
- [x] 决策只依据门禁证据，不使用正式 seed、O3 策略性能或 S2 结果；决策基线为 commit
  `11b0c8391514f42880a6ebd8c61100ba4643a46e` 与 O2 config hash
  `c9a5316a25befa57c53d5351a69e02d7d9bea359a6fa48d94101a5935406d78d`；
- [x] 稳定路线冻结为隔离的下限备份/诊断路线，不进入本轮优化路线确认性实验，也不得用其后续
  结果反向改变 D1。允许主张仅为优化路线已满足进入 E1 的工程与校准门禁。

## E1 — 所选路线协议冻结与链路验证

- [x] E1-A：冻结优化路线 65-run 正式矩阵、150000-step 预算、final-checkpoint 规则及 O3 6400-episode exploratory evaluation 合同，并以机器可读 validator 审计；
- [ ] 冻结代码、环境、教师、奖励、seed、预算、checkpoint、日志、失败和统计合同；
- [ ] 完成所选路线全部必需组的端到端短 smoke；
- [ ] 优化 E1/E2 预算为 65 次：核心 2×2 32，Fixed-KD/QMIX-DG/RuleKD-v3 各 8，
  ShuffleKD-v3/NoOOD-v1/NoGoalHint-v1 各 3；正式训练规模不变，连同 O2 共 71 次；
- [ ] 在查看任何 O3 策略性能前冻结探索矩阵为“执行”或“延期”；若执行，唯一矩阵为
  `MAPPO-DG/RC-AStarKD+LLMKD × 8 training seeds × 2 topologies × 200–209 × 20 episodes`；
- [ ] 稳定预算：核心 2×2、RuleKD 各 5 seed，NoWP 3；不含 QMIX/Shuffle/未见拓扑/8-AGV。

## E2 — 正式训练与独立评估

- [ ] 研究所有者在批准的服务器运行全部长任务，所选路线产物写入其隔离目录；
- [ ] 正式必需评估固定在 canonical core topology 使用 `200–209 × 20`、确定性动作和 final
  checkpoint；
- [ ] 仅当 E1 已预先选择执行时，完整运行并报告 O3 探索矩阵；不设最低性能阈值，不得按结果
  取消、删图、删 seed 或改 checkpoint；
- [ ] 不在正式结果后切换路线、改配置、替换 seed 或选择最佳 checkpoint；
- [ ] Codex 与项目工程师按职责分析并交叉审查结果。

## E3 — 统计、图表与论文就绪

- [ ] 生成 seed 级配对统计、Holm 校正、95% 置信区间、效应量、bootstrap 敏感性分析、表格、
  图像和代表性失败案例；
- [ ] 完成方法—代码—配置—实验—论文主张一致性审计；
- [ ] 每项主张映射到冻结证据或明确限制，图表可由版本化脚本复现；
- [ ] 研究所有者批准最终英文论文证据包。

## 历史基线证据（非活动 Gate）

- Phase 4 双语义 checkpoint 加载、Phase 3/4 兼容回归、评估与可视化链路已完成；
- `1.10/0.30/0.80` 已通过既有匹配重训练与诊断选择，充电仍是固定规则层安全机制；
- bounded terminal reservation 修复降低规划成本但未恢复独立 A* 吞吐，不能证明 A* 是高层
  调度 oracle；
- 旧 800-episode 结果只证明可行性，不是正式多 seed 方法证据；
- 可复现历史文档及 artifact/config 入口见 `docs/archive/README.md`。
