# 三场景四组训练初版配置（待负责人审阅）

日期：2026-09-30。状态：**负责人已确定训练预算；完整长训练配置仍待准入核验**。
负责人本次明确采用 tiny/medium/formal 的 `450k/900k/1.8M` 预算。本版利用现有
`core-integration` 工作树下的 `artifacts/reward_diagnostics/group2/` 和
`docs/handoff/medium5_c1_v3_execution.md` 执行交接，
给出可审阅的具体数值。已停止的 tiny 50k 诊断最后持久记录为 11,264 个累计环境步，
`state.json` 仍为 `running`；进程已结束，没有 final checkpoint 或固定评估，
本版不把它计为完成的短测证据。

## 已有证据与适用边界

| 来源 | 训练与评估条件 | 固定评估交付 | 主要问题 |
| --- | --- | ---: | --- |
| 旧 reward-v2，medium-5AG | 300k、16 worker、训练 seed 7/17/27；各评估 9400–9409 | 2/0/0 单 | 三个 seed 中两个零交付；不能以 300k 断言该场景可学成 |
| 旧 C1 local-mean，medium-5AG | 300k、16 worker、训练 seed 7；评估 9400–9409 | 2 单 | 仅单 seed；评估有 5/10 回合死锁 |
| 旧 C1 local-mean，medium-5AG | 600k、16 worker、训练 seed 7；评估 9400–9409 | 43 单、7,456 实际步，约 5.77 单/千步 | 42 次能量死亡、4/10 回合死锁、0 次抵达充电站、0 次完成充电 |
| C1-v3 新合同 | 600k、16 worker、rollout 64、训练 800 步 | 尚无新合同结果 | 执行交接不是训练结果；两份 LLM 配置仍无真实表示产物 |

以上数字来自各运行目录的 `run_manifest.json`、`summary.json` 和
`evaluation/evaluation.json`。旧 C1 的 300k 与 600k 运行在代码提交和学习率日程上
不同，不能把交付差异单独归因于训练步数。旧评估还使用不同于新合同的任务负载及
`task_target` 语义；`success` 与 `task_completion_rate` 不作为本版依据。旧 600k
运行目录为 `reward-v3-c1-energy-death-local-mean-horizon600k/seed_0007`，其运行清单
记载训练 `max_steps=800`；同名本地旧 YAML 的 `1000` 和 SHA 不一致，不能代替清单。

## 初版训练矩阵

| 场景 | 地图与站点：AGV/充电/拣选 | 方法 | 已确定累计 `environment_steps` | 训练及主评估 `max_steps` | 固定里程碑 checkpoint | 方法专有条件 |
| --- | --- | --- | ---: | ---: | --- | --- |
| tiny-3ag | 官方 tiny 10×11；3/3/2 | baseline | 450,000 | 800 | 225k、450k | 无教师 |
| tiny-3ag | 同上 | A* KD | 450,000 | 800 | 225k、450k | A* 教师版本待批准 |
| tiny-3ag | 同上 | LLM KD | 450,000 | 800 | 225k、450k | 离线表示产物待批准 |
| tiny-3ag | 同上 | A* KD + LLM KD | 450,000 | 800 | 225k、450k | 两项身份均待批准 |
| medium-5ag | 现有 medium；5/5/2 | baseline | 900,000 | 800 | 450k、900k | 无教师 |
| medium-5ag | 同上 | A* KD | 900,000 | 800 | 450k、900k | A* 教师版本待批准 |
| medium-5ag | 同上 | LLM KD | 900,000 | 800 | 450k、900k | 离线表示产物待批准 |
| medium-5ag | 同上 | A* KD + LLM KD | 900,000 | 800 | 450k、900k | 两项身份均待批准 |
| formal-8ag | 正式图 50×28；8/8/10 | baseline | 1,800,000 | 1,200 | 900k、1.8M | 无教师 |
| formal-8ag | 同上 | A* KD | 1,800,000 | 1,200 | 900k、1.8M | A* 教师版本待批准 |
| formal-8ag | 同上 | LLM KD | 1,800,000 | 1,200 | 900k、1.8M | 离线表示产物待批准 |
| formal-8ag | 同上 | A* KD + LLM KD | 1,800,000 | 1,200 | 900k、1.8M | 两项身份均待批准 |

这里的 `environment_steps` 为 16 个 worker 的**累计**环境 transition。三个数值
来自负责人本次决定，取代此前 `600k/600k/1.2M` 提案；现有产物仍不能证明这些预算
足够收敛，tiny 和正式图也没有可比长训练实测。四组在同一场景使用相同预算，不按
某组早期交付或某个 seed 选择预算。核心矩阵共 96 次长训练，合计 **10,080 万**
累计环境 transition；tiny/medium/formal 分别占 1,440/2,880/5,760 万步。
按 1,024 步完整 rollout，每次运行分别计划 440/879/1,758 次 PPO 更新，
末批按实际剩余步数采样。这些是资源计数，不是运行时间保证。

## 三场景共享超参数和任务合同

| 项目 | 初版数值或规则 |
| --- | --- |
| 采样 | `num_env_workers=16`，每 worker `rollout_length=64`；常规一次 PPO 更新采集 1,024 个累计 transition，末尾不足整批按实际步数计 |
| PPO | `ppo_epochs=5`，`num_minibatches=1`，`gamma=0.99`，`gae_lambda=0.95`，`clip_param=0.20` |
| 优化 | Actor/Critic 初始学习率均 `3e-4`；随各场景累计环境步线性降至 0；`optimization_schedule_steps=environment_steps`，`post_schedule_learning_rate=0` |
| 网络与损失 | recurrent 隐层 `64`；熵系数 `0.01`，value loss 系数 `0.50`，Huber delta `10`，梯度范数上限 `0.50` |
| 保存 | `checkpoint_interval_steps=25,000`；另保留 50%/100% 预算里程碑及 final；中途越过里程碑后保存，身份以 checkpoint 内实际步数为准；正式主评估固定使用 final |
| Reward 候选 | C1 energy-death local-mean：交付 `+5`、取货 `+0.5`、进度 `+0.1`、前进受阻 `−0.1`、碰撞 `−0.1`、每步 `−0.005`、死锁 `−5`、能量死亡 `−20`（local-mean）、充电事件 `0`；A/B/C 权重 `2/1.5/1` |
| 电量与安全 | `battery_cost_scale=1.10`，充电触发/释放阈值 `0.30/0.80`，`deadlock_steps=180`；规则层保持硬安全约束 |
| 任务流 | `balanced-random-v1`、`rware_delivery`；初始活跃任务数分别为 `3/5/8`，每交付一单立即补入一单；训练与主评估同负载、同一场景同 `max_steps`；不使用 `task_target` 终止或作成功率分母 |
| 随机性 | 四组共享训练 seed `7/17/27/37/47/57/67/77`；正式评估按 `200–209` 各派生 20 个实例，共 200 个；固定带掩码 argmax |

这些超参数沿用 medium baseline 的已运行或 C1-v3 候选值，没有从一个 seed 推导新的
PPO 调参结论。C1 reward 的完整定义哈希候选为
`3b570b7d25f2fc264c113027def4291872db95825fce1c6d23f1eec389c688a6`；
生成最终配置时必须由实际加载器重算并核对，不得只复制名称或哈希。

## 四组唯一允许的区别

- baseline：随机初始化 Actor，不接入教师。
- A* KD：仅训练期提供动作偏好监督。沿当前 C1-v3 候选日程，`astar_lambda`
  从 `0.05` 在前 70% 优化日程线性降到 `0.01`，随后保持 `0.01`，预算终点为
  `0`；教师版本、有效性处理和缓存身份须先获批。旧 `NO-VALIDITY` 配置不能自动
  当作本矩阵的已批准教师。
- LLM KD：仅将经审核的离线表示导入 Actor Encoder+GRU；不导入 Critic，执行期
  不调用在线 LLM。目前 `approved.pt` 及真实 SHA 缺失，不能用占位文件启动。
- 联合组：同时使用上述两项，经批准后才可生成可运行配置。四组执行期都由 MAPPO
  选择逐步动作；评估中 A* 查询和在线 LLM 调用均应为 0。

## 评估与批准前检查

主评估为同场景 `800/800/1200` 步、每个 final checkpoint 共 200 个固定实例；
以每训练 seed 的实例均值为单位比较 8 个配对训练 seed。主指标为完成任务数、
每千实际步交付数、取货到交付转化，另列碰撞、前进受阻、死锁、充电站抵达、完成充电、
充电后继续交付和能量死亡。正式图另用 `400–409` 做 10 个 5,000 步耐久实例，
每 1,000 步分段报告持续吞吐和安全事件，独立于主评估。12/16-AGV 零微调压力
评估在充电容量和 Actor-only 加载合同获批后另做。

正式长训练准入前，仍需：核对 C1-v3 多 seed 新结果（若届时已产生）；完成
三个场景原计划的单 seed、最多 50k 短测与吞吐测量；确认 reward 及学习率日程、
教师身份和最终配置哈希。50k 短测不能证明已确定预算足够，也不能用零交付直接
取消正式图。当前 12 份 YAML 已写入 `450k/900k/1.8M`，优化日程终点与各自预算
相同；`approval_status` 保持 `provisional`，用于表示完整运行合同尚未通过准入。
训练预算无需再次询问批准；reward、教师产物及其余未完成检查仍按原合同处理。

## 本次配置核验

- 三场景合同回归：29 项通过，覆盖 12 份预算及日程、跨组偏移拒绝、临时配置
  禁止长训练、诊断 50k 上限、三场景完整取货配送循环。
- baseline runner 与旧核心合同回归：15 项通过；静态检查与差异格式检查通过。
- 这轮修改不改变环境机制或 PPO 参数；三场景 baseline 的完整 50k 诊断仍待完成。
