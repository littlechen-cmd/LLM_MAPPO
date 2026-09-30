# 三场景四组训练初版配置（待负责人审阅）

日期：2026-09-30。状态：**候选，不构成正式长训练批准**。本版优先利用现有
`core-integration` 工作树下的 `artifacts/reward_diagnostics/group2/` 和
`docs/handoff/medium5_c1_v3_execution.md` 执行交接，
给出可审阅的具体数值。已停止的 tiny 50k 诊断只有 11,264 个累计环境步，
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

| 场景 | 地图与站点：AGV/充电/拣选 | 方法 | 候选累计 `environment_steps` | 训练及主评估 `max_steps` | 固定里程碑 checkpoint | 方法专有条件 |
| --- | --- | --- | ---: | ---: | --- | --- |
| tiny-3ag | 官方 tiny 10×11；3/3/2 | baseline | 600,000 | 800 | 300k、600k | 无教师 |
| tiny-3ag | 同上 | A* KD | 600,000 | 800 | 300k、600k | A* 教师版本待批准 |
| tiny-3ag | 同上 | LLM KD | 600,000 | 800 | 300k、600k | 离线表示产物待批准 |
| tiny-3ag | 同上 | A* KD + LLM KD | 600,000 | 800 | 300k、600k | 两项身份均待批准 |
| medium-5ag | 现有 medium；5/5/2 | baseline | 600,000 | 800 | 300k、600k | 无教师 |
| medium-5ag | 同上 | A* KD | 600,000 | 800 | 300k、600k | A* 教师版本待批准 |
| medium-5ag | 同上 | LLM KD | 600,000 | 800 | 300k、600k | 离线表示产物待批准 |
| medium-5ag | 同上 | A* KD + LLM KD | 600,000 | 800 | 300k、600k | 两项身份均待批准 |
| formal-8ag | 正式图 50×28；8/8/10 | baseline | 1,200,000 | 1,200 | 600k、1.2M | 无教师 |
| formal-8ag | 同上 | A* KD | 1,200,000 | 1,200 | 600k、1.2M | A* 教师版本待批准 |
| formal-8ag | 同上 | LLM KD | 1,200,000 | 1,200 | 600k、1.2M | 离线表示产物待批准 |
| formal-8ag | 同上 | A* KD + LLM KD | 1,200,000 | 1,200 | 600k、1.2M | 两项身份均待批准 |

这里的 `environment_steps` 为 16 个 worker 的**累计**环境 transition。tiny 没有
可比的长训练产物，暂给它与 medium 相同的 600k，以免仅凭地图小就把预算压到旧
medium 已显露不足的 300k；这是保守候选。formal 的 1.2M 是根据地图面积、路径
长度及 1,200 步 episode 提出的 2× medium 探索预算，**没有正式图训练实测支持**。
四组在同一场景使用完全相同的预算，不按某组早期交付或某个 seed 选择预算。若本版
直接获批，核心矩阵为 96 次长训练、合计 7,680 万累计环境 transition；这只是资源
计数，不是运行时间保证。

## 三场景共享超参数和任务合同

| 项目 | 初版数值或规则 |
| --- | --- |
| 采样 | `num_env_workers=16`，每 worker `rollout_length=64`；常规一次 PPO 更新采集 1,024 个累计 transition，末尾不足整批按实际步数计 |
| PPO | `ppo_epochs=5`，`num_minibatches=1`，`gamma=0.99`，`gae_lambda=0.95`，`clip_param=0.20` |
| 优化 | Actor/Critic 初始学习率均 `3e-4`；随各场景累计环境步线性降至 0；`optimization_schedule_steps=environment_steps`，`post_schedule_learning_rate=0` |
| 网络与损失 | recurrent 隐层 `64`；熵系数 `0.01`，value loss 系数 `0.50`，Huber delta `10`，梯度范数上限 `0.50` |
| 保存 | `checkpoint_interval_steps=25,000`；另保留表中里程碑及 final；正式主评估固定使用 final，不据评估挑选最佳 checkpoint |
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

负责人批准长训练数值前，仍需：核对 C1-v3 多 seed 新结果（若届时已产生）；完成
三个场景原计划的单 seed、最多 50k 短测与吞吐测量；确认 reward 及学习率日程、
教师身份和最终配置哈希。50k 短测不能证明 600k/1.2M 足够，也不能用零交付直接
取消正式图。当前 12 份 YAML 的 `environment_steps` 等预算字段保持 `null`，
`approval_status` 保持 `provisional`；本文是待审阅的数值草案。
