# Phase 2 MAPPO baseline 与训练评估说明

本文记录 Phase 2 合入 MAPPO baseline 主线后，共享训练与评估基础设施的
实际行为、冻结配置和实验解释边界。它面向后续 LLM KD 开发、实验复现和结果
审核，不替代具体 YAML、run manifest、checkpoint 或原始评估产物。

适用基线为集成提交 `b55c817` 及其包含的 Phase 2 提交。若本文与运行产物中的
有效配置不一致，应先停止实验并核查代码与配置来源，不能直接用本文覆盖产物证据。

## 1. MAPPO baseline 的共享改动

### 1.1 配置和方法身份

MAPPO runner 现在同时识别 Phase 1 baseline、Phase 2 baseline 和 Phase 2
A*KD 配置，但仍按 schema、method、goal interface 和 checkpoint schema 严格
校验身份。不同方法的 checkpoint 不能作为训练恢复状态混用；评估也必须使用与
checkpoint 兼容的配置。

Phase 2 baseline 的正式对照方法仍为 `MAPPO-FP-DG`。Actor 使用 DirectGoal
接口，FP Critic 仅在训练时使用集中状态。checkpoint 校验和加载仍覆盖完整 trainer
状态，但评估循环只调用 Actor，不构造评估地图对应的 FP Critic 输入；因此可以在
Actor 接口兼容的前提下进行跨地图压力评估。

### 1.2 持续吞吐训练

训练不再以完成固定任务数作为 episode 终止条件。当前 YAML 中仍保留
`task_target: 18`，用于环境身份、记录和兼容性，但同时设置
`terminate_on_task_target: false`；因此真正的训练截断条件是
`max_steps: 500`。

训练采用 `rware_delivery` 补货：请求货架完成配送后立即补入新任务，以持续
提供工作负载。配置中与动态入库有关的 interval 和 initial batch 字段不会改变
这一训练语义。不要因为看到 `task_target: 18` 就把训练解释为“完成 18 个任务即
成功”或提前终止 episode。

这种修改的目的不是提高表面完成率，而是减少策略通过等待、停滞或规避冲突来结束
有限任务集的可能性，使训练信号更接近单位时间持续吞吐。

### 1.3 独立 checkpoint 评估

评估入口以 evaluation-only 模式加载 checkpoint，不创建训练 worker、不恢复
rollout 或 worker 状态，也不继续训练。完整 trainer state 会通过兼容性校验后加载，
但 optimizer 不会被执行。评估 manifest 会记录源 checkpoint 路径及哈希、源身份和
最终生效的环境配置，后续汇报应以这些产物为准。

评估采用确定性策略 `argmax`。A* teacher 不参与评估查询，MAPPO 仍是唯一动作
策略；因此评估结果不会因为 teacher cache、validity 或运行时 A* 调用而改变。

### 1.4 计数与诊断修复

多进程 rollout 现在使用每一步 worker 返回的最新 payload，并正确处理训练预算
末尾不足完整 worker batch 的情况。这修复了 teacher query、cache build/hit 等
诊断曾重复使用 reset payload 的问题。该修复主要服务 A*KD，但共享 runner 的
step 计数、partial batch 和产物一致性也因此得到统一。

episode 记录新增持续吞吐和行为诊断，包括 `steps_per_task`、各目标阶段动作比例、
direction reversal、motion steps/task，以及原有的 collision、blocked forward、
deadlock、energy death、奖励分量和 PPO 稳定性指标。零完成任务时
`steps_per_task` 为未定义值，而不是零；聚合与绘图不得把它填成“最佳表现”。

## 2. 当前冻结的训练与评估合同

### 2.1 正式训练

| 项目 | 当前约定 |
| --- | --- |
| 正式预算 | `450000` cumulative environment steps |
| checkpoint 间隔 | 每 `25000` environment steps |
| 训练 episode 上限 | `max_steps: 500` |
| 训练补货 | `rware_delivery`，完成配送后持续补货 |
| 任务目标终止 | 关闭；`task_target: 18` 不触发终止 |
| 吞吐主指标 | `steps_per_task`，越低表示单位任务所需步数越少 |
| 当前训练 seed | 沿用配置中的 `7` |
| 初始化 | 从头训练，不迁移或续接旧 300k checkpoint |

450k 是累计环境步数，不是每个 worker 各跑 450k。学习率和 A*KD 辅助权重日程
均按总预算的相对进度计算；提高预算后不要再手工按旧 300k 时间点解释衰减阶段。

### 2.2 固定评估

| 项目 | 3 AGV | 5 AGV |
| --- | ---: | ---: |
| 初始待处理任务 | `2 × AGV = 6` | `2 × AGV = 10` |
| 后续入库 | 每 100 steps 固定新增 3 个 | 每 100 steps 固定新增 5 个 |
| 评估完成目标 | `5 × AGV = 15` | `5 × AGV = 25` |
| episode 上限 | 800 steps | 800 steps |
| 策略 | deterministic argmax | deterministic argmax |
| seeds | 9400、9401 | 9400、9401 |

评估不再区分 high/low workload，也不随机抽取入库批量。两批初始任务用于降低
待拣选任务少于 AGV 数量造成的 idle；之后固定每 100 steps 入库一个 AGV-sized
batch，以降低跨 episode 方差。

评估的 `task_target` 与训练中的 `task_target: 18` 含义不同：评估会在达到
`5 × AGV` 后正常终止，未达到时最多运行 800 steps。不能把训练与评估的成功率、
episode 长度或 task target 直接横向混算。

large-5AG 只作为探索性压力评估。它用于观察地图放大后的绕路、拥堵、协作和
泛化失效，不是正式训练场景，也不构成自动淘汰方法或选择超参数的性能门槛。

## 3. 指标的使用方式

### 3.1 训练期

训练以 `steps_per_task` 作为持续吞吐主指标，同时联合查看：

- completed tasks 随 cumulative environment steps 的变化；
- collision、blocked forward、deadlock 和 energy death；
- PPO policy/value loss、entropy、clip fraction、explained variance 和梯度范数；
- 空载取货、载货配送、充电、idle 各阶段的动作比例；
- direction reversal、motion steps/task 和代表性 replay。

不能只看最后一个 checkpoint 或最后一个 logging window。若 450k 末尾仍有明显
振荡，应同时检查移动平均、多个 checkpoint、行为 replay 与 PPO 稳定性，再判断
是尚未收敛、探索仍强、奖励冲突，还是环境中存在持续负载瓶颈。仅增加训练量不是
默认答案。

### 3.2 评估期

评估可以报告完成任务数、是否达到 `5 × AGV`、steps/task、安全事件和行为轨迹，
但必须逐 seed 保留结果。两个固定 seed 主要用于可重复的配对诊断，不足以单独证明
普遍性能提升。

对方法进行比较时，应使用相同 checkpoint 规则、相同地图、相同 seeds、相同入库
机制和相同 max steps。不得针对某一方法改用最佳 checkpoint、删除失败 seed，或在
看到评估结果后修改 workload。

## 4. 验收与论文主张边界

Phase 2 不设置单一固定数值作为自动验收标准。验收采用人工与 AI 双审核，并综合
考虑训练曲线、固定评估、行为 replay、稳定性、安全事件和产物完整性。任何单项
指标都不能单独宣布通过或失败。

尤其需要遵守以下限制：

- `steps_per_task` 是训练吞吐主指标，但不是唯一 Gate；较低数值必须排除少任务、
  统计缺失或异常终止造成的假改善。
- 完成率达到 100% 不代表路径高效；仍须检查绕路、反复转向、阻塞和移动成本。
- collision 或 blocked-forward 较低也不必然更好；策略可能通过 idle 或放弃任务来
  回避惩罚。
- 单 seed、单 checkpoint、单 replay 或探索性 large-5AG 结果只能作为诊断证据。
- 不得把 smoke test、接口测试或 cache 命中测试描述为学习效果证据。
- 正式结论必须保留代码 commit、配置、checkpoint、seed 级结果和有效环境 manifest，
  并经过人工与 AI 共同复核。

本文中的“正式配置”表示实验协议已经冻结，不表示方法已经达到固定性能阈值，也不
表示可以跳过双审核。

## 5. 对后续 LLM KD 的影响

后续 LLM KD 应继承本轮已经统一的 MAPPO baseline、持续吞吐训练和确定性评估合同，
以保证对照实验只改变明确声明的 KD 因素。特别是训练/评估补货方式、max steps、
task target、seed、checkpoint 选择规则和吞吐指标不能因加入 LLM KD 而隐式变化。

A*KD 的大部分实现与 LLM KD 解耦，只需注意以下共享边界：

- A*KD 是训练期辅助监督，不是执行期控制器；评估不查询 A* teacher。LLM KD 若
  采用不同运行时机制，必须另行声明，不能沿用 A*KD 的名义掩盖差异。
- A*KD 的 `validity` 和 `supervision_scope` 是 A* motion teacher 的语义。LLM
  标签需要自己的可靠性、有效性和监督范围定义，不能直接复用这些布尔量。
- 当前正式 A*KD 方向选择 no-validity 变体，但它仍受 `supervision_scope` 限制。
  该选择不构成取消 LLM 可靠性控制的依据。
- A*KD checkpoint 保留 MotionPriorHead 供诊断；策略执行仍由 MAPPO policy head
  决定。后续 checkpoint loader 不应误把 MotionPriorHead 当作动作输出。
- teacher 专属诊断应保持可选；纯 baseline 和 LLM KD 不应被迫构造 A* 模块或产生
  A* 查询。

## 6. 运行和审核检查表

训练前：

1. 记录代码 commit、完整 YAML 和输出目录，确认没有续接旧 300k 产物。
2. 确认正式预算为 450k、训练 max steps 为 500、RWARE 补货开启且任务目标终止关闭。
3. 确认方法 schema、goal interface、Actor/Critic 维度和 checkpoint identity 匹配。

评估前：

1. 使用独立 checkpoint 评估入口，而不是恢复训练。
2. 检查最终有效配置：100-step 固定入库、两批初始任务、目标 `5 × AGV`、上限 800。
3. 确认 seeds 为 9400/9401、策略为 argmax，且评估没有 teacher query。
4. 对 large-5AG 结果标记为 exploratory pressure evaluation。

结果审核时：

1. 同时检查 seed 级表格、汇总、训练曲线和代表性 replay。
2. 对“高完成率但长绕路”以及“低碰撞但无吞吐”进行反例检查。
3. 保留所有预先指定的 seed 和失败案例，不按结果更换 checkpoint 或配置。
4. 由人工与 AI 分别检查后再形成结论；存在分歧时保留为未决问题，而不是套用固定
   指标自动裁决。
