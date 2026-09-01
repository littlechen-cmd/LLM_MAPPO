# Validation — E1 Optimization Protocol Freeze

## Definition of done

2026-09-02 状态说明：E1 实现链路与 owner CUDA smoke 已完成，但治理 receipt、运行后调度/
恢复审计以及 `codex/optimization` 合并尚未完成，因此 E1 仍为 closeout in progress。E2 已由
owner 提前启动，不得据此反向伪造 E1 原计划顺序。

- [ ] O0/P1/O1/O2/O3/D1 identities与 E1 frozen implementation/data/config hashes 可追溯，governance manifest 不再含旧 blocker 或 null formal budget。
- [~] 65-run manifest 已精确展开为 32 core、8 Fixed、8 QMIX、8 RuleKD、9 diagnostic runs，
  combinatorics validator 已返回 Go；但当前 machine-readable governance manifest 仍输出旧
  `artifacts/optimization/e2_formal` 路径，不能用于定位正在运行的正式矩阵。实际运行根目录已冻结为
  `artifacts/optimization/e2_formal_vector16_7de1f04`，最终 manifest/validator 路径回填须等待运行后
  identity 审计，以免向 commit `7de1f04` 的在途矩阵混入新合同。
- [ ] 优化路线正式标签、网络输出、loss、日志和 checkpoint 均为有序三维语义，且正式入口无法加载旧 1D/2D checkpoint 或进入旧二维 trainer。
- [ ] 原始 60 pilot 与 v5 Pro 800-record raw 数据均可追溯；pilot 不进入训练。原 strict dataset-level Gate 的 No-Go、799/800 validity、单一 fingerprint 与关键语义错误必须保留为审计证据；任何使用 raw LLM 标签的 E1/E2 结果均标记 exploratory noisy-teacher evidence，而非 confirmatory label-Gate 通过。
- [ ] 仓库、Git history、配置、日志、artifact manifest 和异常文本中没有 API key；训练/评估在线 LLM 调用为 0。
- [ ] Fixed/RC 除 `c_A_reward` 外具有相同 sampler、shadow、EMA、schedule、数据、网络、reward、mask 和计数链路。
- [ ] QMIX、RuleKD、Shuffle、NoOOD、NoGoalHint 和启发式基线满足各自冻结合同，没有 fallback 或二维污染。
- [ ] DirectGoal/NoGoalHint 在 throwing planner stub 下完成短链路，执行期 planner query 为 0。
- [ ] checkpoint 可从相同 identity 恢复模型、optimizer、schedule、EMA 和 RNG；损坏/跨 schema/跨数据恢复 fail closed。
- [~] RTX 4090 单卡最大并发目标为 4、RTX 4080 SUPER 无训练 slot、seed block 固定；当前
  `7de1f04` 矩阵完成后必须核验实际最大 learner 数、dispatcher restart、重复 attempt 与恢复记录，
  在审计前不得宣称重启/恢复 Gate 已完全通过。
- [x] `5f56f20` owner-run CUDA smoke 完成 8 members、2048 steps、GPU 0 only；该证据只覆盖
  功能链路。owner 已批准 16-worker 改造不补独立 smoke，以正式运行早期稳定性作为工程健康证据。
- [~] E2 65-run matrix 已从 `7de1f04` 提前启动；这属于明确记录的 owner 阶段越序授权，训练
  开始不等于 E1 Gate 自动完成。
- [ ] O3 exploratory matrix 已冻结为执行，精确展开为 6400 episodes，并标记 non-confirmatory/no-threshold/no-selective-reporting。
- [ ] E1 evidence receipt、`TASKS.md`、`CHANGELOG.md`、Roadmap 和 terminology 同步；E1 commit fast-forward 合并到 `codex/optimization`。

## 16-worker 与运行后审计

- [x] MAPPO 运行合同冻结为 16 CPU environment workers、128 vector ticks、2048 累计
  transitions/update；150000 是跨 worker 的累计 global-step budget。
- [x] Actor/Critic/optimizer/PPO update 仅存在于 learner；worker 清除 CUDA 可见性并限制内部线程。
- [x] per-stream GAE 与累计 global-step 算术已接入正式 trainer；QMIX-DG 明确保留单环境路径。
- [ ] 矩阵完成后核验所有 run 均使用 `7de1f04`、相同 raw-label hash/config、最多四 learner，
  且不存在未披露的 resume、failed restart 或重复 final artifact。
- [ ] 若发生 resume，核验 Reward Calibration EMA、模型、optimizer、schedule、RNG 和 worker
  snapshot 均连续；当前实现的 EMA restore 缺口意味着发生过 resume 的 RC/Fixed run 必须进入
  单独重跑裁决，不能直接接纳。
- [x] ETA 与吞吐只作为工程监控；不得作为算法效率或方法性能证据。

## How to verify

1. 用 canonical Windows Python 运行 E1 manifest validator；预期输出 `65` 个唯一学习 run 和每组/seed 精确计数。
2. 对 60 pilot manifest 运行 model-selection validator；只有冻结的三种状态之一，且 pilot path 不可被 formal trainer 接受。
3. 对 800-record raw dataset 运行 integrity、fingerprint、validity、盲审与 OOD 审计；预期保留原 strict Gate 的 No-Go 及 content SHA256，并生成 exploratory-noisy-teacher identity，而不是伪造 Go receipt。
4. 构造三维 batch，验证 Student 输出 `[batch,N,3]`；reason/disagreement 改变不影响 tensor、loss 或 retrieval。
5. 对四个 core group 与 Fixed control 比较 machine-readable contract diff；预期差异只出现在允许的 teacher mask 和 RC multiplier。
6. 用 throwing online-LLM client 和 throwing execution planner 运行所有适用短链路；预期无调用且 run 完成。
7. 保存、恢复并继续一个短 run；比较 update counter、schedule、EMA、RNG 和下一步输出；再用旧二维/错误 dataset hash 验证拒绝恢复。
8. 用 fake GPU inventory/lease 测试 GPU 0 四 slot 调度、E1 专用 preflight 对外部 PID 的记录但不拒绝、精确 `4×M_slot` 显存门、busy wait、seed pinning、旧 lease 冲突、crash recovery 和 complete-run skip；预期 GPU 0 无第五个进程、GPU 1 无训练进程、无跨 GPU seed 漂移。
9. 由 owner 在服务器运行唯一 CUDA smoke；确认 8 个指定 run 均完成 128→256 恢复、两波分别在 GPU 0 出现四个并发项目进程，并从 peak reserved memory 按固定公式生成 `M_slot`。Codex 读取产物并执行 E1 evidence aggregator，不进行性能比较。
10. 扫描 tracked files、Git staged diff 与 smoke/formal manifests，确认不存在 secret pattern、pilot-as-training、O3-as-training 或 legacy 2D schema。
11. 完成所有 Gate 后，在合并前后分别运行 validation suite；确认 `codex/optimization` 指向 E1 freeze commit 后才允许 push。

## Tests to run

以下是规格要求的测试类别；实现阶段根据新增入口补全精确文件名，并使用冻结解释器直接调用，禁止改成 bare `python`：

- [ ] `D:\Anaconda3\envs\py310\python.exe -m pytest` passes。
- [ ] E1 manifest/label/checkpoint/scheduler focused test selection passes。
- [ ] `D:\Anaconda3\envs\py310\python.exe -m flake8 rware llm_mappo eval train scripts figures/core` passes，或每个既有失败均有与 E1 无关的基线证据。
- [~] E1 65-run manifest validator returns Go for run combinatorics；artifact-root governance
  backfill remains pending after the running matrix audit。
- [ ] E1 raw-label integrity validator reports the immutable `799/800` dataset,
  single fingerprint and exploratory-noisy-teacher status without claiming a
  confirmatory label Gate Go。
- [ ] E1 local end-to-end smoke aggregator returns Go。
- [ ] owner-run Linux CUDA smoke aggregator returns Go，且精确包含 seeds 9001..9004、8 个指定组、2048 总环境步、128→256 resume、GPU 0 每波四进程、冻结 `M_slot`，provenance 只指向 RTX 4090、canonical Python 和 E1 commit。
- [ ] secret scan、online-LLM zero-call、execution-planner zero-call 和 legacy-schema isolation checks return Go。

## Merge criteria

- [ ] `plan.md` 所有任务完成，且每个完成的 `TASKS.md` 子任务在同一 commit 更新 `CHANGELOG.md`。
- [ ] 所有适用 validation checks 通过；raw LLM label 的历史 strict No-Go 必须保留且
  所有包含该教师的结果必须为 exploratory。CUDA smoke 或任何工程安全 No-Go 仍禁止合并。
- [ ] 没有未解决的算法、安全、数据身份、schema、恢复或单 GPU 四进程调度问题。
- [ ] 只存在一个 E1 feature spec 目录，canonical architecture 与三份 E1 spec 没有矛盾。
- [ ] 研究所有者批准 E1 evidence receipt 和允许/禁止论文主张。
- [ ] 合并使用 fast-forward，不覆盖 `codex/optimization` 的非祖先变更，不删除用户未跟踪文件。
- [ ] 合并后测试仍通过；成功 push 或向 owner 交付精确 push 命令。
