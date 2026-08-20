# 论文图数据清单

当前仅冻结数据契约，不把历史 Phase 3 图片登记为正式论文证据，也不包含模拟数据。

## 已实现流水线

| Figure | Measures | Required real data | Script | Outputs | Status |
|---|---|---|---|---|---|
| F1 Core learning curves | 四核心组随环境步数的完成率与 team reward，曲线为20 episode滚动均值，阴影为8个训练 seed间SD | 四组、每个训练 seed 的 `episodes.csv`，必须含 `environment_steps` | `figures/core/plot_learning_curves.py` | `figures/core/core_learning_curves.png`、`.svg` | 脚本就绪，待 G4/G5 真实数据 |
| F2 Core held-out comparison | 完成率、主要任务吞吐、reward、碰撞及8个训练 seed的95% bootstrap CI | `artifacts/formal_aggregate/group_summary.csv` | `figures/core/plot_core_comparison.py` | `figures/core/core_comparison.png`、`.svg` | 脚本就绪，待 G5 真实数据 |

聚合入口为 `eval/aggregate_formal_results.py`。它只接受完整的四组 × 八训练 seed
矩阵；每个模型必须含固定的 10 个评估 seed，且每个评估 seed 为 20 episode。
输入路径由 `configs/g3_experiment_manifest.yaml` 的 `artifact_slug` 决定。

## 后续图

| Figure | Measures | Required real data | Planned script | Status |
|---|---|---|---|---|
| F3 Semantic-label controls | LLM、规则与扰乱标签及盲审结果 | label audit CSV + behavior JSON | `figures/core/plot_semantic_behavior.py` | 待 G3-5/G6 |
| F4 Robustness degradation | 正常、两个未见布局与能源压力的性能退化 | robustness evaluation JSON | `figures/core/plot_robustness.py` | 待 G3-6/G6 |
| F5 Efficiency | 标准化吞吐AUC、达阈值步数、A*与控制循环运行效率 | AUC CSV + episodes/updates/runtime/summary | `figures/core/plot_efficiency.py` | 待 G6 |

正式绘图必须使用冻结后的真实数据，同时输出 450 DPI PNG 与矢量 SVG。图注只说明
测量对象、统计单位和不确定性含义，不预设结果方向。当前脚本采用色觉友好配色。
