# G3-4 / G3-7 预冻结实现审查

## 规格符合性审查

- 通过：manifest v2 固定8个正式学习seed、3个诊断seed、10个未见评估seed及每seed 20回合；
- 通过：训练日志AUC以训练seed为统计单位，缺失任一核心组/seed日志会拒绝聚合；
- 通过：完成任务数/1000 environment steps、标准化吞吐AUC、安全约束和五项确认性比较已在
  manifest、协议与表格schema中一致记录；
- 通过：标签盲审样本量、类别配额、抽样seed、双评价者、评分增量和汇总边界均已固定；
- 通过：manifest仍为`provisional_g3`，未填入正式步数、冻结commit或任何实验结果。

## 质量审查

- 通过：AUC在相同environment-step的并行episode完成记录上先汇总，避免将并发完成误作顺序
  独立观察；最终统计单位仍是训练seed；
- 通过：确认性比较和2×2交互/诊断分析已区分，避免把ShuffleKD或NoWP写成显著性证据；
- 通过：外部MARL、RuleKD、未见布局仍被标为G3-5/G3-6待实现，不以文档替代实现；
- 通过：更新后的协议、追溯矩阵、表格schema和图数据清单没有填充任何模拟或真实结果数值。

## 验证证据

- `python -m pytest tests/test_formal_results.py tests/test_formal_figures.py tests/test_phase4.py -q`
  → `32 passed`；
- 其余测试文件分组执行 → `129 passed`；合计完整套件 `161 passed`；
- `python -m flake8 llm_mappo/formal_results.py eval/aggregate_formal_results.py`
  `tests/test_formal_results.py tests/test_formal_figures.py` → 通过；
- `python -m flake8 rware llm_mappo eval train scripts figures/core` → 通过；
- `git diff --check` → 通过。

## 剩余风险

- G2-1d/e仍阻塞最终冻结；
- G3-5/G3-6尚未实现，当前核心聚合器只读取四个核心组；
- 正式G5训练命令必须把日志写入manifest约定的`artifacts/formal_training`路径。
