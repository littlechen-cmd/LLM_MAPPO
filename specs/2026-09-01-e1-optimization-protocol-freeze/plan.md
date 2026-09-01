# Plan — E1 Optimization Protocol Freeze

本计划只完成 E1。每个任务组完成后更新 `CHANGELOG.md`；只有出现 owner-run 操作、Gate 失败或冻结合同冲突时暂停。任何长训练或长评估均不得在 E1 启动。

## Task group E1-A：治理清单与正式矩阵冻结

- [ ] 将 `configs/g3_experiment_manifest.yaml` 更新为 D1 后的优化路线 E1 governance manifest，写入 O1/O2/O3/D1 状态、O2 evidence identity、`formal_environment_steps=150000` 和 O3 matrix=execute。
- [ ] 建立机器可读的 65-run matrix；逐项记录 group、seed、教师开关、semantic control、observation schema、预算、final checkpoint 规则和 artifact path。
- [ ] 添加 matrix validator，证明运行总数为 65、组别计数为 `32/8/8/8/3/3/3`、seed 精确且没有重复 run identity。
- [ ] 冻结 canonical core evaluation 与 O3 6400-episode exploratory evaluation manifest；两者必须使用 final checkpoint，且 O3 标注 non-confirmatory。
- [ ] 更新 `terminology.md`，解释 formal run matrix、seed block、GPU provenance/blocking factor、dataset-level Gate、system fingerprint 和 resume identity。

## Task group E1-B：三维正式标签链路

- [ ] 审计并修正 semantic-view-v3 scenario generator、prompt-v4、严格 parser、原始请求/响应记录、retry 分类和 manifest；禁止 prompt 接收 scenario type、规则标签、A*、reward、Student 或目标分数。
- [ ] 让标签 CLI 只从 `DEEPSEEK_API_KEY` 读取凭据，并添加密钥不落盘、不回显、不进入异常文本和 Git 的回归测试。
- [ ] 提供唯一 owner-run pilot 命令：精确生成五层各 12 条、共 60 条 Flash pilot，并输出 model/fingerprint、validity 和人工复核包；pilot 永不进入训练或 OOD reference。
- [ ] 实现 Flash→Pro 判定器，只按 canonical 第 11.6 节冻结条件给出 `FLASH_GO`、`REGENERATE_FULL_PILOT_WITH_PRO` 或 `DATASET_NO_GO`，禁止依据训练性能选择模型。
- [ ] 在 pilot Go 后提供唯一 owner-run formal 命令：精确发起五层各 160 条、共 800 attempts；第一条成功响应冻结单一 model/fingerprint，变化时原子暂停。
- [ ] 生成 deterministic 100-record blind-review pack（seed `20260820`，每层 20），合并两位 reviewer 结果并执行 validity/critical/substantive dataset Gate；禁止逐条改分、选择性删除或坏标签重试。
- [ ] Gate Go 后冻结 canonical 800-record content、SHA256、61D OOD reference 与 manifest path；No-Go 时原样归档并停止 E1，不生成可训练数据。
- [ ] 输出 owner 清除临时 `DEEPSEEK_API_KEY` 的收尾检查，但不得把实际值写入任何产物。

## Task group E1-C：正式三维 Student 与训练核心

- [ ] 建立优化路线唯一正式 trainer；不得调用只支持 1/2 维的 legacy Phase 3/4 actor、teacher、buffer 或 checkpoint loader。
- [ ] 将 O0 Student 的 613D physical branch、61D semantic branch、三维 semantic head、detached late fusion 和 centralized critic 接入完整 MAPPO rollout/update，而非 fixture-only smoke。
- [ ] 实现三维整记录 semantic loss：同一 record 的三个分量共享 validity×OOD reliability，reason 和 disagreement 只记录诊断，不进入网络、loss 或 OOD。
- [ ] 实现 `linear-env-step-v1` 的 `B=150000` schedule，并确保所有组共享名义 `lambda_A/lambda_L`；从 checkpoint 严格恢复 `t/p/lambda`。
- [ ] 接入 Fixed/RC A* KD、1/16 deterministic calibration sampler、H=12 paired shadow、EMA 和 fail-closed zero-validity；验证 Fixed/RC 唯一优化差异为 `c_A_reward`。
- [ ] 实现正式 checkpoint 的模型、optimizer、schedule、EMA、Python/NumPy/Torch CPU/CUDA RNG、配置/数据/代码/GPU provenance 和原子写入；旧 1D/2D 与缺失训练状态的文件禁止恢复训练。
- [ ] 完成训练日志 schema：run manifest、episodes、updates、teacher counts/events、resource windows；记录三维 loss、有效分母、schedule、calibration 和 planner query，禁止普通状态全数组日志。
- [ ] 保留原子 latest checkpoint 恢复能力；基础设施失败只允许同 run identity 恢复，算法 NaN/安全失败必须保留为结果。

## Task group E1-D：正式方法、基线与消融

- [ ] 用同一 trainer 配置核心 `MAPPO-DG/RC-AStarKD/LLMKD/RC-AStarKD+LLMKD`，仅通过教师 mask/RC 开关形成 2×2，禁止复制漂移的训练实现。
- [ ] 实现 `Fixed-AStarKD+LLMKD`，并添加配置及运行时等价性审计，证明与完整方法仅差 `c_A_reward`。
- [ ] 将 QMIX-DG 接入同一 DirectGoal、环境、奖励、mask、预算、seed、日志和评估协议；检测到 MAPPO/waypoint fallback 立即失败。
- [ ] 将旧二维 `semantic_controls.py` 替换或隔离为正式 `RuleKD-v3`、`ShuffleKD-v3` 和 `NoOOD-v1` 三维控制；稳定路线旧二维行为不得被修改。
- [ ] 实现 `NoGoalHint-v1` 九位 geometry block 清零与 schema/checkpoint 隔离，并证明执行 planner stub 一旦被调用即抛错时仍可端到端运行。
- [ ] 接入 `Heuristic-Dispatcher+AStar` evaluation-only 基线，明确其不使用 Student checkpoint，也不计入 65 次训练。
- [ ] 为每个方法输出 machine-readable contract diff；出现超出允许字段的差异时拒绝创建正式 run。

## Task group E1-E：双 GPU 四槽 owner runner 与评估入口

- [ ] 实现单命令 E2 launcher：固定 repository、canonical Python、artifact root 与 `nohup` 日志约定；默认不依赖 `conda activate`、`tmux` 或服务器 GitHub。
- [ ] 为 physical GPU 0/1 各实现两个独立项目 slot lock，并实现最多四个 worker、每卡最多两个进程的 seed-block scheduler；seed 首次分配后，将所有配对方法和诊断方法固定到同一 GPU provenance，同一 seed 的两个 run 可占用该卡的两个 slot。
- [ ] launcher 启动时执行 P1-compatible host preflight：RAM `>=64 GiB`、disk `>=200 GiB`、CPU `<=50%`、Git clean、60 秒轮询、连续 5 次、48 小时 timeout；按 physical GPU 0/1 分别冻结名称/总显存，不沿用 P1 的 external-PID 独占判据。
- [ ] 每次领取 seed block 或启动第二 slot 前，每 60 秒重新采样一次只读 formal lease/PID/GPU identity/free-memory；按 `M_slot=ceil(1.5×max_family_peak_mib+1024 MiB)` 检查实时显存。外部 PID 只记录且不得干预；无可用 slot或显存不足时等待，其他 GPU/slot 可继续尚未开始的完整 seed block，禁止杀进程、借用第三 slot 或降低 `M_slot`。
- [ ] 保持 P1/O1/O2 的单卡独占 lease 与 evidence 不变；formal slot locks 使用不同路径/schema，防止旧 Gate runner 与 E1 正式 runner 同时占用同一 GPU。
- [ ] 实现原子 matrix state、PID/lease、heartbeat、run identity、latest/final checkpoint 和失败原因；重启只恢复相同 identity，complete run 不重复执行。
- [ ] 任一算法失败使对应 run 保留 failed 并停止新的正式调度；基础设施失败允许 owner 以相同 identity 恢复，不能自动换 seed/配置/GPU 后掩盖记录。
- [ ] 提供单一状态摘要和最终聚合命令，研究所有者无需逐 run 输入或检查；日志写入 `/home/lzx/`，正式 artifact 只写入 `artifacts/optimization/`。
- [ ] 提供 canonical core、启发式和 O3 exploratory evaluation runners；E1 只验证命令展开，不执行长评估。

## Task group E1-F：最小充分验证

- [ ] 运行新增与受影响的 deterministic unit/integration tests，覆盖三维 shape/梯度、标签 Gate、方法合同、checkpoint resume、65-run expansion、GPU scheduler 和零 planner/online-LLM 调用。
- [ ] 对每个独立实现路径运行最短 CPU smoke；共享同一代码路径的方法不得因组数重复做性能测试。
- [ ] 生成一次 owner-run 双 GPU 四槽 CUDA 功能 smoke 命令，固定总计 8 个 run、每个 256 real environment steps，并在第 128 步主动停止/保存后从 checkpoint 恢复至 256；不得采集或比较性能。
- [ ] smoke 固定两波：wave 1 在 GPU0 并行 seed9001 `MAPPO-DG` 与 seed9002 `Fixed-AStarKD+LLMKD`，GPU1 并行 seed9003 `RuleKD-v3` 与 seed9004 `NoOOD-v1`；wave 2 在 GPU0 并行 seed9001 `RC-AStarKD+LLMKD` 与 seed9002 `QMIX-DG`，GPU1 并行 seed9003 `ShuffleKD-v3` 与 seed9004 `NoGoalHint-v1`。每个 run 256 steps、128→256 resume，总计 2048 steps。
- [ ] Codex 审核 CUDA smoke 产物的四槽上限、每卡双进程、device binding、seed blocking、128→256 resume、三维非零 LLM loss、Fixed/RC parity、QMIX 身份、planner query=0、online LLM=0 和无 NaN/Inf，并据所有家族的峰值冻结 `M_slot`。
- [ ] 运行 manifest/config/schema/hash 审计，确认没有密钥、pilot 数据、O3 数据或旧二维 checkpoint 进入正式 800 reference 或训练路径。

## Task group E1-G：协议冻结、合并与发布

- [ ] 将通过 Gate 的 formal label path/hash、实现 commit、65-run manifest hash、环境/config hash、CUDA smoke evidence 和允许/禁止主张写入 governance manifest 与 E1 evidence receipt。
- [ ] 更新 `TASKS.md`、`CHANGELOG.md`、`specs/roadmap.md` 和 `terminology.md`；E1 只有在 owner-run 标签 Gate 与 CUDA smoke 均 Go 后才可标记 complete。
- [ ] 检查工作树并保留用户未跟踪文件；创建聚焦的 E1 freeze commit，不提交 API key、bundle 或下载压缩包。
- [ ] fast-forward 合并当前实现分支到 `codex/optimization`，运行合并后验证并尝试 push `codex/optimization` 到 GitHub。
- [ ] 若 push 因网络或权限失败，输出含实际旧/新 commit 的本机 push 命令；同时按固定 bundle→scp→fetch→`merge --ff-only` 流程交付服务器同步命令。
- [ ] 交付唯一 E2 owner-run 命令与监控/恢复说明，但不得在 E1 自动启动 65 次正式训练。
