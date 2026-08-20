# 论文图数据清单

当前仅冻结数据契约，不把历史 Phase 3 图片登记为正式论文证据，也不包含模拟数据。

## 已实现流水线

| Figure | Measures | Required real data | Script | Outputs | Status |
|---|---|---|---|---|---|
| F1 Core learning curves | 四组随环境步数的完成率与 team reward，曲线为 20 episode 滚动均值，阴影为训练 seed 间 SD | 四组、每个训练 seed 的 `episodes.csv`，必须含 `environment_steps` | `figures/core/plot_learning_curves.py` | `figures/core/core_learning_curves.png`、`.svg` | 脚本就绪，待 G4/G5 真实数据 |
| F2 Core held-out comparison | 完成率、tasks/1000 steps、reward、碰撞及训练 seed 95% bootstrap CI | `artifacts/formal_aggregate/group_summary.csv` | `figures/core/plot_core_comparison.py` | `figures/core/core_comparison.png`、`.svg` | 脚本就绪，待 G5 真实数据 |

聚合入口为 `eval/aggregate_formal_results.py`。它只接受完整的四组 × 五训练 seed
矩阵；每个模型必须含固定的 10 个评估 seed，且每个评估 seed 为 20 episode。
输入路径由 `configs/g3_experiment_manifest.yaml` 的 `artifact_slug` 决定。

## 后续图

| Figure | Measures | Required real data | Planned script | Status |
|---|---|---|---|---|
| F3 Semantic behavior | LLMKD 语义输出与优先级/场景关系 | engagement samples + behavior JSON | `figures/core/plot_semantic_behavior.py` | 待 G6 |
| F4 Robustness degradation | 正常与压力场景性能退化 | robustness evaluation JSON | `figures/core/plot_robustness.py` | 待 G6 |
| F5 Efficiency | AUC、达阈值步数和运行效率 | episodes/updates/runtime/summary | `figures/core/plot_efficiency.py` | 待 G6 |

正式绘图必须使用冻结后的真实数据，同时输出 450 DPI PNG 与矢量 SVG。图注只说明
测量对象、统计单位和不确定性含义，不预设结果方向。当前脚本采用色觉友好配色。
