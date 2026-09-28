# 正式环境训练配置实施暂停记录

用户要求重启电脑，实施已暂停。恢复工作树：
`D:\codeProject\llm-a-mappo\.worktrees\formal-environment-training-spec`，
分支：`codex/formal-environment-training-spec`，当前 WIP 提交：`1dde110`。

已保存：获批的中文规格；按用户确认修订的 `mission.md`、`tech-stack.md`；
三场景、四方法的 12 份 provisional YAML；正式图 JSON/图片及中文说明；tiny/formal 注册；
16 worker 和按场景 FP Critic 维度的实现；受限 50k baseline 诊断入口；聚焦测试。
当前所有新配置仍不允许正式长训练，`environment_steps` 等待短测及用户批准。

已验证：新聚焦测试 6 项通过；与旧 runner 合跑 15 项通过。工程实现者报告当前 flake8
还有 6 项问题（C901 5 项、E501 1 项），尚未修复。未运行三场景 50k 诊断或正式长训练。

恢复后按顺序处理：修复 flake8；补四组共同字段锁定；验证完整配送补货、充电和多 AGV
占位；试跑 16 worker 短 rollout；运行适用回归和三场景受限 baseline 短测；审查证据、
填写逐场景长训练建议，并交用户批准最终 `environment_steps`、reward 与日程。
LLM 表示产物和 A* 教师版本尚未获批，必须保持相应组的长训练准入阻断。

任务组 1、2、3 均未标记完成。`specs/roadmap.md` 未读取或修改。
