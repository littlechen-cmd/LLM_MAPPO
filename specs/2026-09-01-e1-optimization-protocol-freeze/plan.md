# Plan — E1 Optimization Protocol Freeze

本计划原本只完成 E1，不在 E1 启动长训练。2026-09-02，研究所有者明确授权在治理文档
收口前提前启动 E2 全部 65-run 正式矩阵，并批准 MAPPO 执行合同修订为四个 learner slot、
每 learner 16 个 spawned CPU environment workers、`rollout_length=128`。本文件如实保留该
阶段越序；不得把它改写成原计划顺序。QMIX-DG 保持单环境 trainer。

## 2026-09-02 重规划裁决

- 当前 E1 状态为 `implementation complete / governance closeout in progress`；E2 已进入运行中。
- raw v5 Pro 数据保持 strict Gate No-Go，并作为 immutable exploratory noisy-teacher evidence：
  800 attempts、799 valid、单一 fingerprint，records SHA256 为
  `a108d35e8398114baec1cf88dbe34446efaefda4c9066725d5e7a57a0fed2512`。原 manifest 的
  `status: running` 是已接受但不得静默改写的元数据异常。
- `5f56f20` CUDA smoke 已通过 8 个功能成员、2048 总 steps、GPU 0 only；它不覆盖后续
  `7de1f04` 16-worker 改造。owner 决定不补新 smoke，而以 E2 早期稳定运行作为工程健康证据。
- E2 canonical artifact root 为 `artifacts/optimization/e2_formal_vector16_7de1f04`；当前矩阵
  不因本轮文档收口中断，也不得在运行期间混入新实现 commit。
- 矩阵完成后必须审计：是否发生 resume、failed/restarted run、超过四 learner、重复 artifact、
  code/config/data identity 漂移。仅实际受影响的成员进入重跑裁决。
- E1-G 在 receipt、治理清单、上述运行审计和 `codex/optimization` 合并完成前保持未关闭。

## Task group E1-A：治理清单与正式矩阵冻结

- [x] 将 `configs/g3_experiment_manifest.yaml` 更新为 D1 后的优化路线 E1 governance manifest，写入 O1/O2/O3/D1 状态、O2 evidence identity、`formal_environment_steps=150000` 和 O3 matrix=execute。
- [x] 建立机器可读的 65-run matrix；逐项记录 group、seed、教师开关、semantic control、observation schema、预算、final checkpoint 规则和 artifact path。
- [x] 添加 matrix validator，证明运行总数为 65、组别计数为 `32/8/8/8/3/3/3`、seed 精确且没有重复 run identity。
- [x] 冻结 canonical core evaluation 与 O3 6400-episode exploratory evaluation manifest；两者必须使用 final checkpoint，且 O3 标注 non-confirmatory。
- [x] 更新 `terminology.md`，解释 formal run matrix、seed block、GPU provenance/blocking factor、dataset-level Gate、system fingerprint 和 resume identity。

## Task group E1-B：三维正式标签链路（strict Gate 已执行并 No-Go；探索性替代合同生效）

> 状态修订（2026-09-01）：原 v5 Pro 800-record strict label Gate 已 No-Go，
> 不得勾选为 confirmatory data Gate 通过。研究所有者批准保留其为 immutable
> exploratory noisy-teacher evidence，因此 E1-B 的 raw 证据归档不再阻塞 E1-C；
> 具体边界见 `plan/noisy-teacher-exploratory-protocol.md`。

- [x] 完成 semantic-view-v3 generator、prompt-v5 state contract、严格 parser、原始请求/响应、retry 和 manifest 链路；prompt 不接收规则标签、A*、reward、Student 或目标分数。
- [x] 标签 CLI 仅从 `DEEPSEEK_API_KEY` 读取凭据，密钥不进入版本化产物。
- [x] 完成 60-record pilot、模型升级裁决与五层各 160 条、共 800 attempts 的 Pro formal 采集。
- [x] 完成 deterministic 100-record blind-review 流程；strict dataset-level Gate 结论为 No-Go，
  不进行逐条修补、选择性删除或失败标签重试。
- [x] 原 Gate-Go freeze 任务由 owner 裁决替代：原始 800 attempts 原样归档为 exploratory
  noisy-teacher evidence；SHA256、799/800 validity、model/fingerprint 与 metadata anomaly 写入 receipt。
- [~] API key 未进入仓库；最终销毁仍由 owner 在项目不再需要 LLM 接入时执行，E1 不记录实际值。

## Task group E1-C：正式三维 Student 与训练核心（实现完成）

- [x] 建立优化路线唯一正式三维 trainer，并接入 613D physical、61D semantic、三维 head、
  detached late fusion、centralized critic 和整记录 `validity×OOD` semantic loss。
- [x] 接入 `linear-env-step-v1`、Fixed/RC A* KD、1/16 sampler、H=12 paired shadow、EMA 与
  fail-closed zero-validity；MAPPO 正式 rollout 修订为 16 workers × 128 vector ticks。
- [~] checkpoint 已保存模型、optimizer、schedule、RNG、worker snapshot 和 provenance；若正式
  run 发生 resume，仍须审计并修复当前 calibration EMA 未恢复的缺口。
- [~] compact updates/teacher count 与性能字段已写入；episodes/resource windows/events 的完整性
  及 resume 后累计 wall-time 仍列入 E1-G/E2 产物审计。
- [~] latest checkpoint 原子写入已实现；同 identity 自动恢复、失败 attempt 唯一定位和完整
  calibration state 仍不得在运行后审计前宣称通过。

## Task group E1-D：正式方法、基线与消融（实现完成；启发式长评估留在 E2）

- [x] 核心 2×2、Fixed、RuleKD-v3、ShuffleKD-v3、NoOOD-v1 与 NoGoalHint-v1 通过冻结 matrix
  进入正式入口；共享 trainer 的差异由教师 mask、semantic control 和 RC multiplier 决定。
- [x] QMIX-DG 使用 DirectGoal、共同环境/奖励/mask/预算/seed，保持单环境 trainer，不进入
  MAPPO 16-worker 路径，也不允许 waypoint fallback。
- [~] Heuristic-Dispatcher+AStar 仍为 evaluation-only、无 Student checkpoint、65-run 外基线；
  canonical/O3 长评估入口和最终 contract diff 归 E2 运行前审计。

## Task group E1-E：RTX 4090 单卡四槽 owner runner 与评估入口（运行中审计）

- [x] 单命令 E2 dispatcher 已固定 repository、canonical Python、artifact root、`nohup` 日志和
  GPU 0，并自动维持最多四个当前子进程、顺序提交完整 65-run matrix。
- [x] owner 已明确取消重复正式 preflight/性能 benchmark；P1/O1 Gate 身份保持有效，dispatcher
  仅保留启动时实时 free-memory admission，不切换到 GPU 1 或降低 `M_slot`。
- [~] formal slot lock、持续 heartbeat、dispatcher 重启时存活子进程计数和同 identity resume
  尚有缺口。当前矩阵保持 `7de1f04` 不变；完成后依据实际是否触发这些路径决定受影响 run。
- [ ] 保持 P1/O1/O2 的单卡独占 lease 与 evidence 不变；formal slot locks 使用不同路径/schema，防止旧 Gate runner 与 E1 正式 runner 同时占用同一 GPU。
- [~] matrix state、PID、启动 heartbeat、run identity、checkpoint 和失败停止已接入；完成后必须
  审计重复 attempt、失败分类、resume 和 final checkpoint 唯一性。
- [x] 已提供单一状态摘要/ETA 命令；日志写入 `/home/lzx/`，正式 artifact 写入已冻结的
  `artifacts/optimization/e2_formal_vector16_7de1f04`。
- [ ] 提供 canonical core、启发式和 O3 exploratory evaluation runners；E1 只验证命令展开，不执行长评估。

## Task group E1-F：最小充分验证（owner CUDA smoke 已通过，16-worker 偏差已记录）

- [ ] 运行新增与受影响的 deterministic unit/integration tests，覆盖三维 shape/梯度、标签 Gate、方法合同、checkpoint resume、65-run expansion、GPU scheduler 和零 planner/online-LLM 调用。
- [ ] 对每个独立实现路径运行最短 CPU smoke；共享同一代码路径的方法不得因组数重复做性能测试。
- [x] owner-run RTX 4090 CUDA smoke 已完成两波、8 members、128→256、2048 总 steps、GPU 0
  only；证据 commit 为 `5f56f20`。
- [x] owner 批准不为 `7de1f04` 16-worker 改造补独立 smoke，以 E2 early-run stability 替代；
  该偏差只作工程健康证据，不作方法性能证据。
- [ ] Codex 审核 CUDA smoke 产物的 GPU 0 四进程上限、device binding、seed blocking、128→256 resume、三维非零 LLM loss、Fixed/RC parity、QMIX 身份、planner query=0、online LLM=0 和无 NaN/Inf，并据所有家族的峰值冻结 `M_slot`。
- [ ] 运行 manifest/config/schema/hash 审计，确认没有密钥、pilot 数据、O3 数据或旧二维 checkpoint 进入正式 800 reference 或训练路径。

## Task group E1-G：协议冻结、合并与发布（当前收口任务）

- [x] raw label identity、strict No-Go、实现 commit、matrix/config hash、CUDA smoke 边界及允许/
  禁止主张已写入 `evidence-receipt.md`；governance machine-readable manifest 的最终回填留待运行后身份审计。
- [x] `TASKS.md`、`CHANGELOG.md`、Roadmap、Tech Stack 和 terminology 已同步；E1 在运行后审计
  与 `codex/optimization` 合并前保持 closeout in progress。
- [ ] 检查工作树并保留用户未跟踪文件；创建聚焦的 E1 freeze commit，不提交 API key、bundle 或下载压缩包。
- [ ] fast-forward 合并当前实现分支到 `codex/optimization`，运行合并后验证并尝试 push `codex/optimization` 到 GitHub。
- [ ] 若 push 因网络或权限失败，输出含实际旧/新 commit 的本机 push 命令；同时按固定 bundle→scp→fetch→`merge --ff-only` 流程交付服务器同步命令。
- [x] 已交付并由 owner 启动唯一 E2 65-run 命令与监控/ETA 说明；这是明确记录的阶段越序授权。
