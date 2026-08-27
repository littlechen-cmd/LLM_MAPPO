# Validation — O3 真正未见拓扑与评估隔离

## 1. Definition of done

- [x] 研究所有者批准两张 PNG 预览，ASCII 内容、坐标、version 与双 hash 随后冻结；
- [x] 两张地图满足共同静态合同及各自图论证书，环境构造对 hash 漂移 fail closed；
- [x] 两个 O3 ID 只能经 evaluation-only 工厂使用，训练/O2/label/OOD 路径明确拒绝；
- [x] `9301/9302` 确定性、结构、安全、DirectGoal/NoGoalHint、61D semantic、Pure Teacher、shadow
  restore 与 zero-planner-query 验证全部通过；
- [x] 没有 checkpoint、optimizer、策略性能、正式 seed、在线 LLM、8-AGV 或长任务进入 O3；
- [x] evidence manifest 记录 commit、source/effective hashes、IDs、shared contract、证书和验证结果；
- [x] 完整回归与静态检查通过，治理文档同步，工作树清洁；
- [x] O3 只声明拓扑/接口就绪；canonical core 正式评估与 O3 探索矩阵的 E1 execute/defer 决策
  均保持 pending。

## 2. 人工地图门

O3-B 必须输出：

1. 两张 20×24 ASCII 地图；
2. 每图 `X`、`.`、`G` 计数、显式充电站、连通分量与图论证书摘要；
3. 两张确定性 PNG 预览；
4. 明确说明预览没有创建环境、加载 checkpoint 或执行策略。

研究所有者批准前停止。批准不得依据任何学习策略性能；批准后地图修改必须提高 version 并重新
审核，不能静默更新 hash。

## 3. 结构与哈希验证

- [x] source bytes 为 UTF-8 无 BOM、LF、末尾单换行、20×24、合法 glyph；
- [x] 两图各有 144 shelves、334 ordinary highways、2 goals，highway graph 单连通；
- [x] 窄通道证书满足 width=1、length>=6、左右分区与 articulation 合同；
- [x] 中央交叉满足中心 degree=4、四臂 length>=3、移除中心得到 4 个作业分量；
- [x] 所有货架邻接并可往返 goal，8 stations 唯一/合法/可达且不占关键 cut；
- [x] 两个 source hashes、两个 effective hashes 彼此不同且不同于 canonical core layout；
- [x] 修改任一 map byte、station 或 expected hash 均使 factory 明确失败。

## 4. 接口与确定性验证

- [x] 同 topology+seed 两次 reset 的实体、任务、observation、mask 与固定动作短轨迹逐字节一致；
- [x] action space=5、mask=`[5,5]`、DirectGoal/NoGoalHint=`[5,613]`、semantic=`[5,61]`；
- [x] DirectGoal 与 NoGoalHint 仅在冻结九位 geometry block 存在预期差异；
- [x] reward、termination、truncation、info 和规则安全与 core environment schema 相同；
- [x] throwing planner 替身下 reset/step 通过，planner query count=0；
- [x] Pure Teacher query 和 shadow snapshot 使用 O3 effective hash，restore 后 state/RNG 等价。

## 5. 防泄漏验证

- [x] 所有 training config parser 和训练入口拒绝两个 O3 IDs；
- [x] label generation、pilot/formal manifests、prompt/scenario 与 OOD reference 不含 O3 ID/hash/path；
- [x] O3 测试只出现 `9301/9302`，不出现 `200–209`；
- [x] O3 执行路径没有 `torch.load`、optimizer、training loop、online LLM 或 policy performance 输出；
- [x] 静态扫描证明旧 rejected drafts 与同图 8-AGV 未重新进入活动实现；
- [x] O1 No-Go 失效规则写入 manifest 并有回归覆盖。

## 6. Tests to run

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pytest `
  tests/test_o3_topology_registry.py `
  tests/test_o3_topology_structure.py `
  tests/test_o3_topology_interfaces.py `
  tests/test_o3_topology_leakage.py -q

& "D:\Anaconda3\envs\py310\python.exe" -m pytest -q
& "D:\Anaconda3\envs\py310\python.exe" -m flake8 rware llm_mappo eval train scripts figures/core
& "D:\Anaconda3\envs\py310\python.exe" visualize.py --help
& "C:\Users\28016\bin\rg.exe" -n `
  "llm-mappo-o3-unseen|unseen_narrow_passage|unseen_central_cross" `
  configs train llm_mappo tests specs plan
git diff --check
git status --short
```

搜索结果必须逐项归类：只允许 O3 registry/factory/tests/spec/evidence 和明确的拒绝 guard；任何训练、
O2、label、prompt、scenario 或 OOD 数据引用均为 No-Go。

## 7. Merge criteria

- [x] O3-A 至 O3-F 全部完成；下游性能事项只引用 Roadmap E1/E2，不在 O3 重复建任务组；
- [x] owner preview approval、双 hash 与 evidence manifest 均为 literal values；
- [x] focused/full/static checks 全通过，`CHANGELOG.md` 与 `TASKS.md` 同 commit 更新；
- [x] 没有未提交文件、凭据、artifact、checkpoint 或策略性能结果；
- [x] 架构师确认代码、实验合同和允许论文主张一致；
- [x] 研究所有者于 2026-08-27 批准 O3“拓扑/接口就绪”，且未将其误写为性能通过。
