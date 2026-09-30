# 正式环境训练配置任务暂停记录（2026-09-30）

应研究所有者要求，当前停止任务分析、短测和后续实施。继续时从本记录恢复。

## 当前结果

- 工作树：`D:\codeProject\llm-a-mappo\.worktrees\formal-environment-training-spec`；
  分支：`codex/formal-environment-training-spec`。主 `core-integration` 工作树仍用于
  baseline 优化，未改动其文件。
- 三场景合同及 12 份临时 YAML 已在此前提交 `26cb758`；三种场景为
  tiny-3ag（3/3/2）、medium-5ag（5/5/2）、formal-8ag（8/8/10），
  每场景 baseline、A* KD、LLM KD、联合组四组。
- 最新初版数值见 `docs/handoff/formal_training_initial_configuration.md`：
  候选累计训练预算为 tiny/medium/formal 各 `600k/600k/1.2M`；16 worker、
  rollout 64、5 PPO epoch、1 minibatch、学习率 `3e-4` 线性降至 0，
  同场景训练与主评估 `max_steps=800/800/1200`。该预算仅为审阅草案，
  当前 YAML 的预算字段仍为 `null`、批准状态仍为 `provisional`。
- 已核对 `core-integration` 工作树旧 medium 产物：reward-v2 的三个 300k
  训练 seed 在固定诊断评估分别交付 2/0/0 单；C1 300k seed 7 交付 2 单；
  C1 600k seed 7 交付 43 单，但有 42 次能量死亡、4/10 回合死锁、0 次
  充电站抵达及 0 次完成充电。C1-v3 执行交接尚无新合同训练结果。
- 本次核对了 12 份 YAML 均仍为临时状态，文档矩阵恰有 12 行；`git diff --check`
  通过。没有启动正式长训练。

## 运行与 Git 状态

- tiny 的 50k 单 seed 诊断已按最新优先级要求中止；仅跑到 11,264 个累计环境步，
  没有 final checkpoint 或固定评估，不能当作已完成短测。其产物在
  `artifacts/formal_environment_training/diagnostics/tiny-3ag/seed_0007/`
  下时间戳为 `20260930T012433Z` 的目录，`state.json` 留有旧的 `running` 状态。
  停止后已检查没有 Python 训练进程。
- 本次新增的初版配置文档与实验合同中的引用已加入 Git 暂存区。若暂停记录尚未
  提交，恢复时先核对 `git status`，保留全部工作树改动。当前阶段不改动
  `specs/roadmap.md`，不改正式 YAML 的预算与批准状态。

## 恢复后的下一步

1. 先确认本次文档提交和工作树状态，阅读研究所有者对初版预算的反馈；
   若需要调整，应在三场景四组内一致修改，不根据某个方法的早期评估挑选预算。
2. 按既定 50k 上限完成三个 baseline 单 seed 短测及四组短生命周期验证，
   记录实际吞吐、交付、充电和安全事件。已中止 tiny 运行需新建独立运行目录。
3. 取得 C1-v3 多 seed 新结果（若已产生）、真实 A* 教师版本及 LLM 表示产物，
   向研究所有者提交最终训练预算和配置身份供批准。批准前不运行 96 次长训练。
