## Task Packet

- Scope: 完成 G3-4 与 G3-7 的预冻结实现：八训练 seed 数据契约、学习曲线 AUC、
  主要指标/确认性比较、统计聚合、表格图数据清单和回归测试。
- Files to read: `CONSTITUTION.md`、`TASKS.md`、`configs/g3_experiment_manifest.yaml`、
  `plan/experiment-protocol.md`、`plan/review/method-experiment-traceability.md`、
  `tables/table-schema.md`、`figures/data-manifest.md`、`llm_mappo/formal_results.py`、
  `eval/aggregate_formal_results.py`、相关测试。
- Files allowed to edit: 上述文件、`tests/test_formal_results.py`、
  `tests/test_formal_figures.py`、`plan/progress.md`、`plan/review/` 与本任务包。
- Required skills: using-research-writing、paper-orchestration、
  experiment-results-planning、statistical-analysis、verification。
- Evidence/data inputs: G2-3 已选择的能源配置、冻结候选的四核心组、未见评估 seed
  `200–209`、八个正式训练 seed；不使用或生成正式结果。
- Required artifacts: manifest v2、八 seed 核心聚合、AUC 原始/汇总 CSV、主要指标和
  确认性比较契约、更新的协议/追溯矩阵/表格/图清单、回归测试与审查记录。
- Rejection checks: 不得将 manifest 标为 frozen；不得填入正式步数、结果数值或训练结论；
  不得把同一 checkpoint 的评估回合当作独立统计样本；不得启动训练或长评估。
- Validation commands: `python -m pytest tests/test_formal_results.py tests/test_formal_figures.py tests/test_phase4.py -q`、
  `python -m flake8 llm_mappo eval figures/core tests/test_formal_results.py tests/test_formal_figures.py`、
  manifest/任务 ID 一致性检查与 `git diff --check`。
