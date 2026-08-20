# G2-3 核心充电配置选择记录

## 决策

四个核心组统一采用：

- `battery_cost_scale = 1.10`
- `charge_threshold = 0.30`
- `charge_release_threshold = 0.80`

该配置只定义规则层的固定充电安全机制，不支持“MAPPO 自主学习充电时机”的主张。

## 证据范围

四个候选均采用相同教师、奖励、PPO 参数、训练 seed-group 和 200 episode 预算。最终对
`1.10/0.30/0.80` 与 `1.10/0.25/0.80` 的 final checkpoint 使用确定性动作完成
seed `0–9 × 20 episodes` 的诊断评估。seed `0–9` 已参与配置选择，因此不得用于正式
G6 泛化结论；正式评估改用未见 seed `200–209`。

| 指标 | 1.10/0.30/0.80 | 1.10/0.25/0.80 |
|---|---:|---:|
| task completion rate | 1.000 | 1.000 |
| completed tasks / 1000 steps | 113.4598 | 115.4659 |
| mean steps / episode | 440.905 | 433.115 |
| mean reward / episode | 79.7406 | 81.4812 |
| episodes with charging | 20% | 1% |
| episodes with low battery | 34% | 1% |
| mean energy deaths / episode | 0.000 | 0.005 |

0.25 阈值的吞吐优势在 10 个诊断 seed 上一致，但它的充电暴露几乎消失且发生一次能源
死亡，不满足安全优先的核心配置选择。0.30 阈值保留更多可观测充电行为并保持零能源死亡，
因此被选为核心配置。`1.20/0.20/0.80` 继续只作为所有方法共同的能源压力评估场景。

原始诊断文件位于本地忽略目录：

- `artifacts/g2_charging_retrain_candidate/seed_007/evaluation_heldout_10x20.json`
- `artifacts/g2_charging_retrain_candidate_scale110_threshold025/`
  `g2_charging_retrain_candidate_scale110_threshold025/seed_007/`
  `evaluation_heldout_10x20.json`
