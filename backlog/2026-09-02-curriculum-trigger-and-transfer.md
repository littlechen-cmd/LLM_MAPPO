# Backlog — curriculum trigger and transfer

## Date
2026-09-02

## Context

R1-C采用4-AGV LowLoad诊断，但研究所有者批准暂不把它作为正式curriculum stage或warm start。
需要保留5-AGV正式环境失败后的难度定位与公平迁移问题。

## Findings / notes

- 先运行5-AGV Formal scratch；成功则不引入curriculum。
- 若4-AGV LowLoad成功而5-AGV Formal失败，再运行5-AGV LowLoad scratch。
- 5-AGV LowLoad也失败更支持协调规模/观测/DirectGoal瓶颈；只有LowLoad成功而Formal失败才支持
  高负载探索困难。
- 若curriculum进入正式协议，所有方法必须遵循相同difficulty schedule、保留各自teacher条件，
  并报告所有stage的总环境交互预算。
- 共享预训练若被采用，应来自MAPPO-DG而非RC-AStarKD，但这会把研究问题改为fine-tuning。

## Action items

- [ ] 仅在R1-D 5-AGV Formal scratch失败后，把5-AGV LowLoad加入R1-F诊断任务。
- [ ] 仅在5-AGV LowLoad成功且Formal失败后，另行制定curriculum/transfer规格与公平预算合同。
- [ ] 在任何迁移实现前冻结Actor、Critic、optimizer与Reward Calibration状态的加载边界。
