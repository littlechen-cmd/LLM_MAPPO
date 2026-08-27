# 优化路线资源重规划审查

日期：2026-08-27
范围：只审查 optimization 路线从 O1 到 E3 的剩余任务，不修改 stable 路线合同。

## 1. 审查结论

本次从规格一致性、GPU 资源、实验有效性三个独立角度审查。核心裁决是：不降低正式 65 次训练
证据规模，优先减少正式实验前的重复 A600 调度和非必要长校准，同时把 O3 的高风险跨拓扑性能
主张降为可选探索性压力测试。

### Fix now

1. O1 常规 CUDA Gate 从 baseline/H4/H12 精简为 baseline/H12；原 workers、repeats、windows、
   `3×` runtime 与 persistent-memory 阈值不变。H4 只在 Gate 失败后进入独立诊断。
2. O1 与 O2 由同一个 owner A600 作业 fail-fast 编排，但必须保留独立产物和状态；O1 未通过时
   O2 不得启动。
3. O2 从 9 次降为 6 次，只保留 `MAPPO-DG/RC-AStarKD × 107/117/127 × 150000 steps`。
   Fixed/RC 链路等价性由 MateBook 确定性短受控 smoke 验证。
4. E1/E2 的 65 次正式/诊断训练预算保持原样，总学习运行数从 74 改为 71。
5. canonical core topology 的 `200–209 × 20` 成为唯一正式必需 held-out 随机鲁棒性证据；O3
   不再是性能门，也不支持跨拓扑泛化主张。
6. 删除 O3 内重复 E1/E2 的 O3-G checklist，改为引用 Roadmap 下游合同。

### Improve later

- O3-C～F 全部在 MateBook 上完成，并用参数化测试复用两张图的确定性、接口和防泄漏断言；
- E1 用一个 CPU 配置矩阵覆盖每条独立代码路径，不按 65 个正式 run 重复 smoke；
- 正式 checkpoint 评估保持 owner-run A600-eligible；若 E1 在查看结果前另行冻结为
  MateBook/CPU，只允许改变执行平台，不得改变 seed、episode、checkpoint 或统计合同；
- E1 在查看任何 O3 policy performance 前记录 O3 探索矩阵为 `execute` 或 `defer`。

### Reject / defer

- 拒绝缩减正式 65 次训练、8 个确认性 seed、QMIX-DG、RuleKD-v3 或三个既有诊断组；
- 拒绝删除 O3 双哈希、确定性、DirectGoal/NoGoalHint、Pure Teacher、shadow restore 和防泄漏
  测试；这些均为低成本本地正确性证据；
- 拒绝用同拓扑 held-out seed 冒充未见拓扑泛化；论文必须明确两者的证据边界；
- stable 路线的 S1/S2 不在本次重规划范围内。

## 2. A600 资源清单

| 阶段 | 必需 A600 工作 | 数量 | 目的 |
|---|---|---:|---|
| O1 | baseline/H12 CUDA runtime/memory Gate | 1 个短作业 | 排除正式 H12 开销失控、显存与对象持续泄漏 |
| O2 | MAPPO-DG | 3 次训练 | 提供无 A*KD 校准参照 |
| O2 | RC-AStarKD | 3 次训练 | 检查覆盖率、数值稳定性与 AUC 退化门 |
| E2 | 原冻结正式/诊断矩阵 | 65 次训练 | 形成论文确认性与诊断性证据 |

因此 optimization 路线必需的 A600 学习运行是 `6+65=71` 次，另有一个短 O1 Gate。正式评估
可使用 A600，但不要求 GPU；O3、E1 smoke、标签处理和 E3 统计均不需要 A600。

## 3. O3 下游备选合同

正式必需主张固定为“canonical core topology 下对未见随机实例的鲁棒性”，不是跨拓扑泛化。
O3 两张地图只保留以下探索性矩阵：

```text
MAPPO-DG / RC-AStarKD+LLMKD
× training seeds 7/17/27/37/47/57/67/77
× two O3 topologies
× held-out seeds 200–209
× 20 episodes
```

E1 必须在读取任何 O3 policy performance 前，仅依据资源记录 `execute` 或 `defer`。执行时无最低
性能阈值且完整报告；延期时不得运行或选择性报告。两种选择都不改变 D1、正式训练或核心论文
主张。

## 4. 推荐实施顺序

1. 在 MateBook 继续 O3-C～F；
2. 本地修改并验证 O1 runner 的 baseline/H12 与 failure-only H4 合同，以及 O1→O2 编排器；
3. 建立唯一 O2 feature spec 和六次训练 runner；
4. A600 可用时，由研究所有者用一个作业先运行 O1，只有通过后才进入 O2；
5. O1/O2/O3 通过后进入 D1 和 E1，正式冻结 65 次训练的其余运行参数与 O3 execute/defer 决策。

## 5. 已批准决策与开放问题

研究所有者已于 2026-08-27 批准 O1/O2 合并编排、failure-only H4、O2 六次校准，以及 O3
降级方案；明确要求正式 65 次训练规模保持不变。当前重规划无未决实现选择。
