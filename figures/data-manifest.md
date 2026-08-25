# 论文图数据清单

当前处于 D1 前准备状态，不把历史 Phase 3 图片登记为正式论文证据，也不包含模拟数据。
具体输入根目录、seed 数和图表集合必须在所选路线的 E1 重新冻结。

## 已实现流水线

| Figure | Measures | Required real data | Script | Outputs | Status |
|---|---|---|---|---|---|
| F1 Core learning curves | 四核心组随环境步数的完成率与 team reward，曲线为20 episode滚动均值，阴影为训练 seed 间 SD | 四组、每个训练 seed 的 `episodes.csv`，必须含 `environment_steps` | `figures/core/plot_learning_curves.py` | `figures/core/core_learning_curves.png`、`.svg` | 八 seed 兼容脚本已存在；稳定路线五 seed 适配待 E1 |
| F2 Core held-out comparison | 完成率、主要任务吞吐、reward、碰撞及训练 seed 的95% bootstrap CI | `artifacts/<selected_route>/formal_aggregate/group_summary.csv` | `figures/core/plot_core_comparison.py` | `figures/core/core_comparison.png`、`.svg` | 待 D1/E1 绑定路线 |

聚合入口为 `eval/aggregate_formal_results.py`。现有实现只接受四组 × 八训练 seed，是优化
路线兼容入口；若 D1 选择稳定路线，E1 必须先适配并测试四组 × 五 seed。每个模型最终均使用
固定的 10 个评估 seed、每 seed 20 episodes，输入路径由路线 manifest 决定。

## 后续图

| Figure | Measures | Required real data | Planned script | Status |
|---|---|---|---|---|
| F3 Semantic-label controls | 连续三维 LLM、RuleKD-v3、ShuffleKD-v3、NoOOD-v1 与盲审 | label audit CSV + behavior JSON | `figures/core/plot_semantic_behavior.py` | 待 E1/E3 |
| F4 Optimization topology robustness | 正常环境与 O3 两个真正未见拓扑的性能退化 | optimization robustness JSON | `figures/core/plot_robustness.py` | 仅优化路线，待 O3/E3 |
| F5 Efficiency | 固定10k-step网格吞吐AUC、达阈值步数、A*与控制循环运行效率 | AUC CSV + episodes/updates/resource windows | `figures/core/plot_efficiency.py` | 待 E3 |

稳定路线不生成跨拓扑图，也不得用同图 8-AGV 替代。正式绘图必须使用冻结后的真实数据，
同时输出 450 DPI PNG 与矢量 SVG。图注只说明
测量对象、统计单位和不确定性含义，不预设结果方向。当前脚本采用色觉友好配色。
