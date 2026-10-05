# 三场景正式实验结果记录模板

状态：初版协议已按负责人授权确定，正式结果尚未运行。所有运行，包括零交付、能量死亡、死锁和基础设施故障，都保留原始记录。

## 每个运行的身份

| 必填字段 | 含义 |
| --- | --- |
| `scene_id`、`method_id`、`train_seed` | 场景、四组方法之一、配对训练 seed |
| `code_commit`、`config_sha256`、`layout_sha256` | 代码、配置和地图身份 |
| `reward_definition_sha256`、`teacher_schema`、`representation_sha256` | 奖励和教师身份；不适用时显式写 `none` |
| `checkpoint_step`、`checkpoint_sha256` | 固定 final checkpoint 身份，不按评估结果挑选 |
| `evaluation_protocol`、`episode_seed` | 主评估、耐久评估或后续压力评估及对应实例 |
| `planned_steps`、`actual_steps`、`failure_reason` | 计划与实际时长、提前终止或运行失败原因 |

## 每集原始指标

| 指标 | 记录口径 |
| --- | --- |
| `completed_tasks`、`picked_tasks`、`created_tasks` | 任务事件原始计数；补货任务计入创建数 |
| `throughput_per_1000_steps` | `1000 × completed_tasks / actual_steps`；`actual_steps=0` 时标记无定义 |
| `charging_target_steps`、`charger_arrivals`、`charge_events`、`post_charge_deliveries` | 充电意图、抵达、实际充电及恢复工作分别记录 |
| `energy_deaths`、`collisions`、`blocked_forward`、`deadlock_events` | 安全事件原始计数；同时报告各类暴露步数 |
| `planner_queries`、`online_llm_calls` | 正式执行期目标均为 0 |

持续补货下旧 `success` 和 `task_completion_rate` 不作为主指标，也不把任务完成数除以
`task_target`。主评估按场景分别汇总每组 8 个训练 seed；每个训练 seed 先对共同的
200 个评估实例求均值，再在相同训练 seed 上求 `A* KD − baseline`、
`LLM KD − baseline` 和 `联合组 − 单教师组` 等预声明配对差异及区间。报告每组
8 个 seed 的分布与全部失败运行，不把 200 个实例当成独立训练重复。

formal-8ag 的 5,000 步耐久评估另表记录 0–999、1000–1999、2000–2999、
3000–3999、4000–4999 各窗口的交付和实际步数，同时列出完整 5,000 步总计；
提前终止的窗口保留实际步数和失败原因。有限时长内没有死亡或死锁，只能表述为
“在 N 个预先指定实例中未观察到事件”，不能表述为风险为零。

后续 12/16 台同图压力评估应与 8 台主比较分开成表，标注充电容量方案、
Actor-only checkpoint 来源、AGV 数与任务池规模；只有所有方法使用相同条件时才比较。

计量边界：当前 `task_recoveries` 指充电后电量达到释放阈值并重新持有任务，
`charged_events` 是充电事件数，不等于独立充电循环数。`post_charge_deliveries`
需要同AGV事件轨迹关联；当前汇总未直接记录该字段，不能用上述两个计数代替。
