# 研究进度

## 2026-08-30 — P1/O1 通过，O2 实验层实现中

- 研究所有者已完成 Linux P1 smoke；服务器的 Python、GPU 0 绑定、资源预检与 provenance
  均通过。
- O1 CUDA Gate 通过：artifact 为
  `artifacts/optimization/o1_cuda_gate/20260829T075212Z_7c305ea2`，commit 为
  `7c305ea24cdca34467c2e7e8a5a9d66ba1133d1e`，H12/baseline runtime ratio 为 `2.880`，
  memory Gate 通过。因此 O2 不再受 O1 阻塞。
- O2 处于实现阶段，尚未开始任何 150000-step 正式训练。其唯一矩阵保持
  `MAPPO-DG/RC-AStarKD × 107/117/127`，每项 150000 个 joint environment transitions；
  六项正式作业将顺序使用 GPU 0，且只在最终汇总达到覆盖率和吞吐 AUC 门槛后才判为 Go。

## 2026-08-26 — O1 门禁等待与 O3 并行启动

- P0、O0 已完成；O1 本地实现、223 项回归、短 smoke 与静态验证已完成。
- O1 正按 2026-08-27 资源修订调整 runner：常规 Gate 只含 baseline/H12，H4 仅失败后诊断；
  修订实现和 owner A600 结果均完成前不得标记为通过。
- A600 被占用期间，O3 只并行推进规格、地图冻结、确定性、接口和防泄漏验证；O2 继续被
  O1 Gate 硬阻塞。
- O3 唯一规格为 `specs/2026-08-26-o3-unseen-topologies/`；两个布局从零设计，不恢复 P0
  删除的 rejected drafts。
- O3 是拓扑与评估协议就绪门，不加载学习策略、不观察性能，也不是 D1 性能门。正式必需证据
  是 canonical core topology 的 held-out-seed 鲁棒性；O3 只可按 E1 预先决定的探索矩阵在 E2
  执行或整体延期。

## 当前冻结决策

- 优化路线保持 Pure Motion A*、H=12 Reward Calibration、三维离线 LLM 语义与 MAPPO 主线；
- O3 使用 5 AGV、20×24、144 个货架格、2 个目标和 8 个显式充电站，仅改变静态拓扑；
- O3 描述限定为“未见的载货运输 highway 拓扑与载货流瓶颈”，不新增静态墙语义；
- 放弃同图 8-AGV 压力实验；O3 布局禁止进入训练、O2、标签生成、OOD reference、路线选择或
  超参数选择；
- 正式 held-out evaluation seeds `200–209` 只能在 D1 选定唯一方向后使用；
- O2 固定为 6 次校准，E1/E2 正式训练保持 65 次，优化路线总学习运行数为 71；
- 所有长训练和长评估由研究所有者在 A600 执行。

## 历史证据入口

Phase 2–4、充电配置、A* 吞吐诊断和旧规划审查已迁入 `docs/archive/`；路径映射见
`docs/archive/README.md`。历史结论不自动升级为新路线的正式证据。
