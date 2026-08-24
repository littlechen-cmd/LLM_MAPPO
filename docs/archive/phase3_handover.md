# Phase 3 可视化链路 — 改动协作说明

> 用途：供团队成员了解 Phase 3 绘图链路的当前改动、数据流与验证方式。
> 关联任务：Phase 3a 可行性训练前的 P0/P1 出图能力建设。

---

## 1. 改动清单

### 1.1 本次会话新增（Phase 3 可视化链路）

| 文件                          | 类型              | 作用                                                                                    |
| ----------------------------- | ----------------- | --------------------------------------------------------------------------------------- |
| `llm_mappo/plotting.py`       | 新增（约 517 行） | Phase 3 绘图核心模块：CSV 加载、平滑、P0/P1 全部图形绘制、`render_all_figures` 顶层驱动 |
| `scripts/plot_phase3.py`      | 新增              | 静态绘图 CLI：从已完成/进行中的训练目录出图到 `figures/`                                |
| `scripts/plot_phase3_live.py` | 新增              | 实时绘图 CLI：训练期轮询 CSV mtime，持续刷新 `figures/`                                 |



### 1.2 本次会话修改

| 文件                           | 改动点                                                                                                                                                                                                                                                   |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llm_mappo/phase3_training.py` | ① `evaluate_phase3` 新增 `collect_engagement` / `engagement_sample_rate` 参数，返回 `engagement_samples` 键；② 新增 `_engagement_label()` 辅助函数（按优先级标签 A/B/C/none 采样投入度）；③ `train_phase3` 训练开始时向 TensorBoard 写入 `config/*` 标量 |
| `eval/evaluate_phase3.py`      | CLI 新增 `--collect-engagement`、`--engagement-sample-rate`、`--engagement-csv` 参数，支持落盘 `(label, engagement)` 采样数据                                                                                                                            |

### 1.3 历史遗留改动（非本次会话，工作区未提交）

以下文件在会话开始前已有改动，`git status` 中一并列出，供成员知悉：

- 修改：`.gitignore`、`configs/phase2_medium_3ag_astar_bc.yaml`、`docs/phase2_medium_3ag_expert_baseline.md`、`docs/phase2_small_astar_bc_800.md`、`llm_mappo/phase2_training.py`、`llm_mappo/planner.py`、`rware/rendering.py`、`大规模动态仓储LLM-MAPPO总体方案.md`
- 删除：`LICENSE`、`visualize_phase2.py`
- 未跟踪：`docs/MARL-LLM_学习文档_压缩版.md`、`order.md`、`repomix-output.xml`、`tests/test_astar_heuristic.py`、`visualize.py`

> 建议：提交前请区分上述历史改动与本会话改动，避免混入单个 commit。

- 同时，对于A*算法也做了修改，将旋转即(LEFT,RIGHT动作)添加到启发函数的代价计算中，减少AGV的无效旋转动作
- 对应文件含有`llm_mappo/phase2_training.py`、`llm_mappo/planner.py`、`rware/rendering.py`等

---

## 2. 数据流总览

```
train_phase3()
  ├─ episodes.csv   (episode 级：completion / collisions / deadlock / reward /
  │                    agent_deaths / blocked_forwards / priority_*_completion_steps)
  ├─ updates.csv    (update 级：policy_loss / value_loss / entropy / engagement_loss)
  ├─ summary.json
  └─ tensorboard/   (episode/* 与 training/* 标量，训练期实时可看)

evaluate_phase3(collect_engagement=True)
  └─ engagement_samples  →  eval/evaluate_phase3.py --engagement-csv
                            →  engagement_samples.csv (label, engagement)

渲染入口：
  scripts/plot_phase3.py --run-dir <dir>            → figures/*.png（离线）
  scripts/plot_phase3_live.py --run-dir <dir>       → figures/*.png（实时轮询）
```

---

## 3. 关键接口说明

### 3.1 `llm_mappo/plotting.py`

- `load_training_data(run_dir)` → `TrainingData`：加载 `episodes.csv` / `updates.csv` / `summary.json`，自动做类型强转
- `render_all_figures(run_dir, output_dir, *, phase2_comparison=None, engagement_samples=None)` → `List[Path]`：顶层驱动，缺数据自动跳过
- 产出图形（9 + 2 张可选）：
  - P0：`episode_completion.png`、`episode_collisions.png`、`episode_deadlock.png`、`episode_reward.png`、`episode_deaths.png`、`episode_blocked.png`、`priority_completion_steps.png`、`training_losses.png`、`engagement_loss.png`
  - 可选：`reservation_kl.png`（3b 才有数据）、`phase2_vs_phase3.png`、`engagement_by_priority.png`
- 常量：`GATE_COMPLETION=0.95`、`GATE_COLLISIONS=2.0`、`GATE_DEADLOCK=0.05`、`PHASE2_BASELINE_COLLISIONS=9.585`（Phase 2 No-Go 基线）
- 强制 `matplotlib.use("Agg")`，可无头运行（headless Windows 服务器）

### 3.2 `eval/evaluate_phase3.py` 扩展

```powershell
python eval/evaluate_phase3.py artifacts/phase3a_dual_head/seed_007/checkpoint_final.pt `
    --output artifacts/phase3a_eval.json `
    --collect-engagement --engagement-csv artifacts/engagement_samples.csv
```

`engagement_samples.csv` 格式：表头 `label,engagement`，label 取 `A/B/C/none`。

### 3.3 `_engagement_label()` 的依赖链

`Phase2Warehouse.env.agents` → `task_queue.task_for_agent(id)` → `task.label[0]`（首字母）。与训练侧 `_engagement_targets`（A=0.8 / B=0.5 / C=0.3）使用同一标签语义。

---

## 4. 使用示例

```powershell
# 1) 训练结束后离线出图
python scripts/plot_phase3.py --run-dir artifacts/phase3a_dual_head/seed_007

# 2) 训练期间实时刷新（30 秒轮询）
python scripts/plot_phase3_live.py --run-dir artifacts/phase3a_dual_head/seed_007 `
    --interval 30 --output-dir figures/phase3a

# 3) 评估时采样投入度 + 出散点图
python eval/evaluate_phase3.py <ckpt> --collect-engagement --engagement-csv eval_out.csv
python scripts/plot_phase3.py --run-dir <run> --engagement-csv eval_out.csv
```

---

## 5. 验证结果（已执行）

| 检查项                                    | 结果                 |
| ----------------------------------------- | -------------------- |
| flake8（5 个文件，py310）                 | 0 警告               |
| IDE lint（read_lints）                    | 0 errors             |
| 冒烟测试：合成 CSV → `render_all_figures` | 生成 9 张 PNG 且非空 |
| 冒烟测试：`plot_phase3_live.py --once`    | 生成 9 张 PNG        |
| 临时文件清理                              | 已全部删除，无残留   |

**未验证项**：真实训练产物出图（因 `artifacts/phase3a_dual_head/` 尚无训练数据），待 800-episode 训练完成后用 `--run-dir` 指定真实目录即可。

---

## 6. 下一步建议

1. 跑 Phase 3a 的 800-episode 本地 CPU 可行性训练，产出真实 `episodes.csv` / `updates.csv`
2. 训练结束后运行 `plot_phase3.py` 出图，对照 Phase 2 基线（completion 0.973 / collisions 9.585）
3. 评估时开启 `--collect-engagement`，确认投入度头是否学到 A>B>C>none 的语义排序
4. 通过 Go/No-Go 后进入 3b，打开 `reservation_kl_coefficient > 0`，用 `plot_reservation_kl` 画蒸馏曲线
