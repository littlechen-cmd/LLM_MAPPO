# Plan — O3 真正未见拓扑与评估隔离

本计划在当前优化路线分支执行，不创建额外 worktree。严格测试先行；每个 task group 完成后更新
`CHANGELOG.md` 并形成聚焦 commit。长训练、长评估和学习策略性能检查全部禁止。

## Task group O3-A：依赖重排与唯一规格

- [x] 从正确性、架构质量和规格一致性三个角度完成只读重规划审查；
- [x] 将依赖图改为 O0 后并行 `O1→O2` 与 O3，并在 D1 汇合；
- [x] 保持 O1=A600 gate pending、O2=blocked，修正 O3/E2 性能门边界；
- [x] 同步 Roadmap、TASKS、canonical architecture、实验协议、progress、治理 manifest 与
  `CHANGELOG.md`；
- [x] 建立根 `terminology.md`，以通俗/专业双层解释统一 O3 与实验治理概念；
- [x] 创建 O3 唯一 feature spec，并记录 2026-08-26 owner 决策。

## Task group O3-B：从零设计地图与 owner preview gate

- [x] 先写失败测试，冻结文件格式、20×24、144 `X`、334 `.`、2 `G`、8 stations 和两个图论
  结构证书；
- [x] 仅根据 requirements 第 3–4 节生成两张新 ASCII 地图，不读取、恢复或仿照 P0 rejected
  drafts；
- [x] 生成静态 lint/结构审计摘要与两张 PNG 预览，不创建环境、不加载 checkpoint；
- [x] 暂停并交由研究所有者人工审核地图几何；批准前不得进入 O3-C；
- [x] owner 于 2026-08-27 批准首版预览，并在 core 尺寸核验后批准将两图确定性旋转为
  `20×24 v2`；若拒绝，只按结构意见重做，不得
  使用策略性能选择地图。
- [x] owner 于 2026-08-27 人工确认 `v2` PNG、旋转后的精确坐标与不变的图论证书。

## Task group O3-C：TopologySpec、package data 与双哈希

- [x] 先写失败测试，冻结两个环境 ID、evaluation-only usage、资源加载和 source hash fail-closed；
- [x] 在 `llm_mappo/o3_topologies.py` 实现只读 `TopologySpec`、显式 evaluation factory 与按需注册，
  禁止普通包导入时全局注册 O3；
- [x] 将 `rware/layouts/o3/*.txt` 纳入 package data，验证 editable/installed resource bytes 一致；
- [x] 计算并写入 owner-approved source SHA-256 与 effective layout hash，任一漂移即拒绝构造；
- [x] 生成版本化 O3 evidence manifest，记录 commit、IDs、两种 hashes、站点与结构证书。

## Task group O3-D：确定性、结构、安全与接口验证

- [x] 使用 test-only seeds `9301/9302` 验证同 seed reset、任务流、固定动作轨迹、observation 与
  mask 的逐字节确定性；
- [x] 验证 highway 连通、货架邻接/往返可达、目标/充电站合法、窄通道与中央 cut 证书；
- [x] 验证 5 动作、`[5,5]` mask、reward/termination/info、安全约束与 canonical 环境一致；
- [x] 验证 DirectGoal/NoGoalHint `[5,613]`、semantic-view-v3 `[5,61]`、goal block 与字段顺序；
- [x] 在 throwing planner 替身下完成 reset/短 step，断言 planner queries=0；
- [x] 验证 Pure Motion Teacher query 与 shadow capture/restore 使用当前 effective hash 且保持 O0
  合同；不读取性能指标。

## Task group O3-E：训练与数据防泄漏

- [ ] 先写失败测试，证明 optimization、Phase 3/4 training、O2 和 label collection 入口拒绝 O3 IDs；
- [ ] 增加不可由普通布尔 flag 绕过的 evaluation-only guard，并保留训练 core environment 白名单；
- [ ] 扫描所有训练/O2/60/800 label/prompt/scenario/OOD 配置，证明不含 O3 IDs、hashes 或路径；
- [ ] 证明 O3 代码路径不创建 optimizer、不加载 checkpoint、不调用在线 LLM、不使用 `200–209`；
- [ ] 固化 O1 No-Go 后地图证书可保留、接口证据必须重验的 manifest 状态规则。

## Task group O3-F：本地收口与证据交接

- [ ] 运行 O3 focused tests、完整 pytest、Flake8、CLI/static scan 与 `git diff --check`；
- [ ] 审计 O3 改动未改变 reward、energy、O0 Teacher/Calibration、正式 seeds、预算或稳定路线；
- [ ] 更新 `TASKS.md`、`CHANGELOG.md`、Roadmap 与 evidence manifest，记录 literal commands/results/
  commit/hash；
- [ ] 提交 owner handoff：已完成范围、文件、测试、风险、禁止主张与工作树状态；
- [ ] 只将 O3 标记为“拓扑/接口就绪”。不得运行或声称 learned-policy performance。

## Task group O3-G：E2 延后事项（本阶段不执行）

- [ ] D1 选择优化路线后，由 E1 冻结 topology×group×seed×episode 正式矩阵；
- [ ] 研究所有者在 E2 使用 final checkpoints 与 held-out evaluation seeds `200–209 × 20` 运行；
- [ ] E3 完成统计后，才允许依据证据形成跨拓扑可靠性主张。
