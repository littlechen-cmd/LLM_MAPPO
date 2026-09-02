# Requirements — E1 Optimization Protocol Freeze

> 2026-09-02 R1 amendment: E1 的 65-run 合同保留为历史冻结记录，但旧矩阵因绝对任务能力
> 不足和证据缺陷被暂停并降级为 diagnostic-only。任何新的确认性训练必须先通过
> `specs/2026-09-02-r1-convergence-recovery/`，再由统一合同重新冻结；本文件中“不修改奖励/
> rollout”的限制不阻止经 owner 批准的 R1 纠偏。

## Scope

E1 将已经通过 O0/P1/O1/O2/O3/D1 的优化路线整理为唯一、可恢复、可审计的正式实验实现，供研究所有者在 E2 启动长训练与长评估。范围包括：

- 冻结 `150000` real environment steps 的共同训练预算与 65 次学习运行矩阵；
- 生成并验收 semantic-view-v3 的 60 条 pilot 与 800 条 formal 三维离线 LLM 标签；
- 将三维 Student、LLM KD、A* KD、Reward Calibration、buffer、schedule、日志和 checkpoint 从短 smoke 链路升级为正式链路；
- 实现或修正核心 `2×2`、`Fixed-AStarKD+LLMKD`、`QMIX-DG`、`RuleKD-v3`、`ShuffleKD-v3`、`NoOOD-v1`、`NoGoalHint-v1` 与无训练启发式基线；
- 提供 RTX 4090 单卡、最多四个独立任务的 owner-run 调度器，以及恢复、失败停止、聚合和身份校验；RTX 4080 SUPER 不参与训练；
- 冻结 canonical core topology 的正式评估和两张 O3 未见拓扑的探索性评估协议；
- 完成本地测试、一次最小化整合 CUDA smoke、证据收口、分支合并与发布。

## Out of scope

- E1 原范围不运行 65 次正式训练；研究所有者于 2026-09-02 明确授权在 E1 治理收口前提前
  启动 E2。该偏差只改变阶段执行顺序，不改变 65-run matrix、seed、环境、奖励、教师或评估合同。
- 不修改冻结的环境、奖励、电量、动作 mask、安全规则、Pure Motion Teacher、`K_motion=12`、`H_reward=12`、512 expansion budget、1/16 sampler、EMA、seed 或统计假设。
- 不搜索超参数，不根据短 smoke 或正式结果调整 schedule、训练预算、模型、seed、数据子集或阈值。
- 不恢复旧二维 Phase 4 语义，不做 1D/2D→3D 权重迁移，不混合稳定路线代码或产物。
- 不运行同图 8-AGV 压力测试，不把 O3 探索性结果表述为跨拓扑泛化证据。
- 不使用在线 LLM 进行训练、评估或执行期决策。

## Decisions

| Decision | Frozen requirement and rationale |
| --- | --- |
| Route | D1 已选择优化路线；E1 的目标分支为 `codex/optimization`。 |
| Formal budget | 每个学习运行恰好 `150000` real environment steps；与 O2 和 `linear-env-step-v1` 的 `B=150000` 对齐。 |
| Run matrix | 核心 2×2 为 32 次；Fixed、QMIX、RuleKD 各 8 次；Shuffle、NoOOD、NoGoalHint 各 3 次；总计 65 次，启发式基线无训练。 |
| Seeds | 正式组用 `7/17/27/37/47/57/67/77`；诊断组只用 `7/17/27`；评估用 `200..209 × 20 episodes`。 |
| Semantic contract | 唯一正式语义为 `task_persistence/yielding_preference/coordination_risk`，Student 输出 `[N,3]`；整记录 validity 与共享 OOD reliability 共同加权。 |
| Label generation | 先运行 60 条 Flash pilot；只有 canonical architecture 第 11.6 节的系统性失败门触发时，才废弃整套 Flash pilot 并用 Pro 重做完整 60 条。通过的唯一模型用于完整 800 条 formal。 |
| Raw LLM label evidence | 原始 strict label Gate 及其 No-Go 保留在 `plan/label-audit-protocol.md`。研究所有者于 2026-09-01 批准将已完成的 v5 Pro 800-record raw 数据集作为 exploratory noisy-teacher evidence：它必须保持 800 attempts、唯一 ID/hash、overall validity `799/800`、每层至少 `159/160`、单一 model/fingerprint，但不得称为通过语义正确性 Gate 的 formal label dataset。 |
| Exploratory LLM boundary | `LLMKD` 与完整方法继续使用相同 raw 数据、整记录 `validity × OOD reliability`、网络和 schedule；训练/评估结果均标记 exploratory。该权重不是语义正确性置信度，且不允许以人工改单条、选择性删除或新 API 请求修补 raw 数据。 |
| Secret handling | API 凭据只从进程环境变量 `DEEPSEEK_API_KEY` 读取。程序和文档不得记录、回显、序列化、hash、提交或复制密钥；标签任务结束后由 owner 从 shell 环境清除。 |
| Formal model identity | 第一条成功响应冻结 request model、response model 与 system fingerprint；正式生成中变化立即暂停，禁止静默混合 backend。 |
| Schedule | 所有方法记录同一 `linear-env-step-v1`：`lambda_A=0.05(1-p)`、`lambda_L=0.10(1-p)`、`p=min(t/150000,1)`；缺少教师时仅将对应 mask 置零。 |
| A* comparison | Fixed 与 RC 使用相同 1/16 sampler、shadow rollout、日志与调用密度；唯一优化差异是 RC 额外乘 `c_A_reward`。 |
| MAPPO rollout execution | 每个 MAPPO learner 固定 `num_env_workers=16`、`rollout_length=128`；16 个 CPU-only spawned workers 并行执行 `env.step()`，主 learner 对 `16×5=80` 个智能体观测集中执行一次 GPU Actor/Critic inference。一次 PPO update 使用 `16×128=2048` 个累计 joint transitions；GAE 按 worker stream 和 episode 独立，不跨环境。 |
| Global-step accounting | `formal_environment_steps=150000` 是所有 worker 累计的 joint environment transitions；一次完整 vector step 增加 16。checkpoint、schedule、日志和 resume 均使用该累计计数，任何 worker 都不得各自运行 150000。 |
| Single-GPU scheduling | physical GPU 0 的 RTX 4090 是唯一训练设备，固定最大并发 4 个 learner。MAPPO learner 各持有 16 个 CPU 环境 worker；QMIX-DG 保持单环境 trainer，但允许四个独立 QMIX run 并发。所有 seed block 绑定 GPU 0，不存在跨卡迁移或 GPU 型号混杂。 |
| Formal GPU slots | E1 在 physical GPU 0 建立 `slot-0..slot-3` 四个项目锁；physical GPU 1 的 RTX 4080 SUPER 不建立训练 slot。只统计本项目正式进程，严禁复用或放宽 P1/O1/O2 的单卡独占 lease。每个 slot 同时最多持有一个子进程。 |
| Slot memory admission | 并发 CUDA smoke 对每个训练家族记录 `torch.cuda.max_memory_reserved`；定义 `M_slot=ceil(1.5×max_family_peak_mib+1024 MiB)`。启动任一新 slot 前，目标 GPU 的实时 free memory 必须不少于 `M_slot`，该值写入 E1 receipt 并在 E2 冻结，禁止自动降低。 |
| Formal launch admission | P1/O1 已确认 OS、Python、Git、RAM、disk 与 GPU 身份。owner 于正式训练前明确取消重复 benchmark/smoke/preflight；E2 dispatcher 每次启动 learner 只执行 GPU 0 实时 free-memory admission，外部 PID 不被终止或隐藏，也不使用 P1 的 95% 独占空闲阈值。 |
| Shared-server conduct | 不抢占、终止或隐藏其他用户进程。RTX 4090 不可用时 worker 等待，不回退到 RTX 4080 SUPER。长任务用 `nohup`，日志写入 `/home/lzx/`。 |
| Hardware claims | GPU 型号是 provenance/blocking factor，不把吞吐或训练时间作为方法性能优势。 |
| O3 matrix | E2 完整执行 `MAPPO-DG` 与 `RC-AStarKD+LLMKD` × 8 training seeds × 2 topologies × `200..209` × 20 episodes，共 6400 episodes；无性能阈值、不得选择性报告。 |
| E2 artifact root | 本轮正式矩阵唯一根目录为 `artifacts/optimization/e2_formal_vector16_7de1f04`；每个 run 的实际 attempt directory、PID、log、state、manifest 与 final checkpoint 必须由 matrix state 唯一定位。 |
| Running-commit rule | 已启动矩阵固定使用 `7de1f04c772ccf49d422a53aa0c1ad01deec9204`，运行期间不得混入后续修复 commit。完成后审计中断、恢复、重复和并发；仅被实际缺陷触发的成员进入重跑裁决。 |
| Checkpoints | 仅 `checkpoint_final.pt` 进入正式评估；训练 checkpoint 必须可严格恢复 optimizer、schedule、EMA、RNG 和 provenance；旧 1D/2D checkpoint fail closed。 |
| Implementation flow | Codex 可连续完成任务组；只在 owner-run API/CUDA 操作、冻结合同冲突、数据 Gate 失败或需要扩大范围时暂停。 |
| Integration | 所有 E1 Gate 通过后，将当前实现 fast-forward 合并至 `codex/optimization` 并优先直接 push；网络/权限阻塞时交付无占位符命令。 |

## Method-specific contracts

- `MAPPO-DG`：A* KD=0，LLM KD=0。
- `RC-AStarKD`：Reward-Calibrated A* KD，LLM KD=0。
- `LLMKD`：A* KD=0，semantic-view-v3 raw LLM KD 开启；论文中称为 `Raw-LLMKD`。
- `RC-AStarKD+LLMKD`：完整方法，两类 KD 均开启；论文中称为
  `RC-AStarKD+Raw-LLMKD`。
- `Fixed-AStarKD+LLMKD`：除 `c_A_reward` 外与完整方法完全一致。
- `QMIX-DG`：共享 DirectGoal 613D observation、环境、奖励、动作 mask、训练预算和 seed；不得回退为 MAPPO 或 waypoint 输入。
- `RuleKD-v3`：在同一 800 semantic views 上按 canonical architecture 第 13.3 节独立生成三维规则标签，复用 retrieval/validity/OOD；规则不得进入 LLM prompt 或改写正式标签。
- `ShuffleKD-v3`：在五个预注册场景层内对三维联合标签做确定性、无 fixed-point derangement；只使用诊断 seeds。
- `NoOOD-v1`：仅把有效 LLM record 的 OOD reliability 置 1；其余合同不变。
- `NoGoalHint-v1`：613D DirectGoal 输入仅将九位 goal geometry block 清零；执行期 planner query 仍为 0；只支持目标几何提示敏感性解释。
- `Heuristic-Dispatcher+AStar`：无训练的规划基线，与 Pure Motion Teacher 不等同。

## Context

- Canonical architecture：`docs/architecture/o0-reward-calibrated-heterogeneous-distillation.md`。
- 原始严格标签审计：`plan/label-audit-protocol.md`；探索性 raw LLM 边界：
  `plan/noisy-teacher-exploratory-protocol.md`。
- O2 Gate evidence：`docs/evidence/o2-calibration-gate-v1.json`。
- O2 已证明 Reward Calibration 链路满足 coverage 与中位 AUC 门，但没有启用 LLMKD，不能作为三维 LLM 蒸馏证据。
- 当前 `semantic_v3.py`、优化 Student 与 checkpoint 已包含三维组件；旧 `mappo.py`、Phase 4 teacher 和 `semantic_controls.py` 仍有二维实现，E1 必须阻止正式优化路线误入旧链路。
- P1/O1/O2 的 physical GPU 0 独占 preflight/lease 保持原样；E1 formal slot policy 是新的下游调度合同，不得反向改写既有 Gate evidence。

## Open questions

无。实现不得自行改变以上冻结值；若发现合同无法同时满足，必须暂停并交由研究所有者裁决。
