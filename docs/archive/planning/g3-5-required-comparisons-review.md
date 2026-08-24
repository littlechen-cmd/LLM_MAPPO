# G3-5 必需比较实现审查

## 规格符合性审查

- 通过：manifest 为五个必需组记录了配置、artifact slug、角色、seed、waypoint/KD 因子和
  confirmatory 边界；八 seed 正式组与三 seed 诊断组未混淆。
- 通过：RuleKD 与 ShuffleKD 仅转换缓存 JSONL；不含 provider 或在线模型调用。ShuffleKD 固定
  seed `20260820` 且实施无固定点置换。
- 通过：NoWP 不改变 actor 观测长度，不生成规划 waypoint，并关闭 A* KL 和 waypoint reward。
- 通过：QMIX-WP 和启发式 Dispatcher+A* 均复用 Phase2Warehouse 的动作 mask、动态任务和
  规则安全边界；没有为对照组创建单独奖励或环境设置。

## 质量审查

- 通过：确认性 RuleKD/QMIX 比较与三 seed ShuffleKD/NoWP 诊断在协议和 manifest 中分开，
  避免以诊断组形成显著性结论。
- 通过：QMIX 采用中心化单调 mixer 与共享去中心化 agent Q 网络，避免把 A* 专家动作或 LLM
  标签泄漏进外部 MARL 基线。
- 通过：启发式基线被明确为端到端非学习参照，而非只报告静态 A* 路径规划。

## 验证证据

- `python -m pytest tests/test_g3_comparisons.py tests/test_phase2.py tests/test_phase4.py -q`
  → `39 passed`；
- `python -m flake8 rware llm_mappo eval train scripts figures/core` → 通过；
- manifest 与全部比较配置 YAML 解析 → 通过；
- `evaluate_heuristic_dispatcher_astar.py --help` 与 `train_qmix.py --help` → 通过；
- `git diff --check` → 通过。

## 剩余风险

- G4-5 才能证明五组的短预算端到端训练、日志、checkpoint 和评估链路兼容；
- QMIX 的样本效率和资源占用尚无结果，不能作任何算法性能主张；
- 生成实际 RuleKD/ShuffleKD 缓存前，应核对源文件 SHA-256 与 manifest 一致。
