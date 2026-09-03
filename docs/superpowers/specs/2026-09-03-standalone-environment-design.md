# 独立仿真环境仓库迁移设计

## 目标

从当前 `llm-a-mappo` 中提取一个可独立安装、测试和运行的仿真
环境仓库，目标目录为 `D:\codeProject\LLM_MAPPO`。新仓库服务于后续
环境维护和新的强化学习项目，不携带当前项目的训练、教师蒸馏或论文
实验历史。

## 保留的能力

新仓库必须保留目前实际使用的环境合同：

- 仓库地图、AGV、货架、任务队列和动态入库；
- 任务优先级、充放电、电量耗尽与恢复；
- DirectGoal 613D 物理观测、5 动作空间和物理动作掩码；
- Reward-v2；
- 无界面渲染、单回合 GIF 回放和完整 episode 指标；
- Gymnasium `reset()` / `step()` 接口；
- 4 AGV LowLoad 与 5 AGV Formal 的版本化 YAML 配置；
- 一个只使用合法随机动作的最小运行入口。

## 明确不迁移的能力

下列代码不属于独立环境仓库，不能通过复制方式带入：

- MAPPO、QMIX、buffer、PPO 更新、checkpoint、TensorBoard 与训练调度；
- A* Pure Motion Teacher、Reward Calibration、shadow rollout；
- LLM 三维语义、离线标签、OOD、蒸馏损失和教师对照；
- O1/O2/O3/E1/E2/R1 实验协议、产物、模型权重和论文证据；
- legacy waypoint 执行期观测与 `legacy-v1` 奖励。

## 目标结构

```text
LLM_MAPPO/
  pyproject.toml
  README.md
  src/llm_mappo_env/
    core/          # rware 地图、任务、AGV、动态入库与充电
    adapter/       # Gym Phase2 环境、DirectGoal、动作掩码和指标
    rewards/       # 唯一的 Reward-v2
    rendering/     # headless 帧和 GIF
    configs/       # 4-AGV LowLoad、5-AGV Formal
  scripts/
    run_random_policy.py
    render_episode.py
  tests/
  docs/
    handoff.md
    migration-map.md
```

`core` 不依赖任何训练模块；`adapter` 只依赖 `core`；`rewards` 只依赖
环境状态；`rendering` 只读取环境状态。这个单向依赖关系保证替换训练
算法时不必触及环境代码。

## 迁移映射

| 当前来源 | 新职责 | 迁移处理 |
|---|---|---|
| `rware/` | `core/` | 保留必要的 Gym 仓库动力学；移除与旧训练无关的包装 |
| `llm_mappo/phase2.py` | `adapter/environment.py` | 仅保留 DirectGoal、动态任务、充电、动作掩码和 episode 指标 |
| `llm_mappo/optimization_observation.py` | `adapter/observation.py` | 仅保留 DirectGoal 构建器与 613D 合同 |
| `llm_mappo/reward_v2.py` | `rewards/reward_v2.py` | 原样迁移并删除 legacy 分支 |
| `llm_mappo/visualization.py` | `rendering/headless.py` | 原样迁移，保持无 GUI |
| 环境回归测试 | `tests/` | 只保留环境、奖励、观测、充电、任务和渲染测试 |

## 对外接口

环境对象提供以下稳定入口：

```python
observations = environment.reset(seed=...)
action_masks = environment.action_masks()
transition = environment.step(actions)
```

`transition` 至少含有下一步观测、团队奖励、终止状态和完整 episode
指标。随机策略脚本只能从动作掩码中选择动作；它不是学习算法或基线。

## 迁移安全与验收

1. 新目录必须为空或由研究所有者明确允许写入；不得覆盖现有文件。
2. 新仓库使用新的 Git 历史；旧仓库保持不移动、不删除、不改写。
3. 迁移后必须通过 DirectGoal 维度、动作掩码、Reward-v2、动态入库、
   充电和 headless 渲染回归测试。
4. 随机策略必须在两个冻结 YAML 配置上完成一个可重放 episode。
5. `docs/migration-map.md` 必须逐项记录源文件、目标文件、保留与删除
   的责任边界；`docs/handoff.md` 必须说明安装、运行、已知限制和禁止
   的论文主张。

## 非目标与风险

这次迁移不改变环境动力学、Reward-v2 或 DirectGoal 定义。若清理旧
`rware` 代码时发现某个看似底层的函数被训练链路隐式调用，应先把它
保留在环境侧并用测试固定，再决定是否删除；不能为了目录整洁而改变
环境行为。
