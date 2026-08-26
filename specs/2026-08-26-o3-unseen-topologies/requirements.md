# Requirements — O3 真正未见拓扑与评估隔离

## 1. 目标与阶段含义

O3 建立两个从零设计、版本化、仅用于优化路线独立评估的仓储拓扑，并证明它们与既有
DirectGoal/NoGoalHint、semantic-view-v3、Pure Motion Teacher 和 shadow snapshot 接口兼容。
O3 只构成“拓扑与评估协议就绪门”，不加载学习策略、不衡量难度或完成率，也不形成跨拓扑
性能主张。

本规格涉及的 TopologySpec、双哈希、割点、图论证书、evaluation-only 和防泄漏等概念统一引用
根 `terminology.md`。实现或评审中新引入的重要概念必须先补充该术语表。

O3 可在 O1 等待研究所有者运行 A600 runtime/memory gate 时并行。O1 未通过前 O2 仍被硬阻塞；
O1、O2、O3 全部通过仍只是进入 D1 的前置条件。真正的未见拓扑性能评估仅在 D1 选择优化路线
后于 E2 执行。

## 2. 范围

### 2.1 包含

- 两个显式、不可随机再生成的 ASCII 地图：
  - `rware/layouts/o3/unseen_narrow_passage_v1.txt`；
  - `rware/layouts/o3/unseen_central_cross_v1.txt`。
- 两个唯一环境 ID：
  - `llm-mappo-o3-unseen-narrow-passage-5ag-v1`；
  - `llm-mappo-o3-unseen-central-cross-5ag-v1`。
- 只读 `TopologySpec` 注册表、evaluation-only 环境工厂、地图加载、静态 lint、结构证书、
  source/effective hash 验证和证据 manifest。
- 确定性 reset/step、动作与观察接口、安全规则、DirectGoal/NoGoalHint、61D semantic view、
  Pure Motion Teacher、shadow restore 与 planner-query=0 的本地测试。
- 对训练、O2、标签生成、OOD reference、路线选择、checkpoint 选择和调参入口的代码门与静态
  防泄漏审计。
- 地图设计完成后的 PNG 预览人工批准门；批准前不冻结哈希或继续接口实现。

### 2.2 不包含

- A600 runtime/memory gate、O2 训练、长评估或长回放；
- 任何已训练或训练中策略、checkpoint、吞吐、完成率、碰撞率、死锁率或难度比较；
- held-out evaluation seeds `200–209`；O3 测试只使用 `9301/9302`；
- LLM API 调用、60/800 标签生成、prompt/OOD/Teacher/reward/KL/network 修改；
- 同图 8-AGV 压力场景、旧 rejected layout、随机布局生成器或结果驱动的地图替换；
- 新增静态墙、改变 RWARE `X`、`.`、`G` 语义，或声称所有空载 AGV 都被墙式瓶颈限制；
- 正式 topology×group×seed×episode 评估矩阵；该矩阵在 E1 冻结并于 E2 执行。

## 3. 冻结环境合同

两个拓扑除静态地图与显式充电站坐标外，必须共享 canonical optimization environment 合同：

| 字段 | 冻结值 |
|---|---:|
| grid size | `24 × 20`（width × height） |
| shelf cells | `144` |
| goals | `2` |
| agents | `5` |
| charging stations | `8` 个显式坐标 |
| sensor range | `4` |
| actions | `5`（NOOP/FORWARD/LEFT/RIGHT/TOGGLE_LOAD） |
| dynamic ingress interval | `40` |
| batch size | `[4,8]` |
| request queue | `8` |
| task target | `50` |
| max steps | `1000` |
| deadlock steps | `180` |
| battery cost scale | `1.10` |
| charge/release threshold | `0.30/0.80` |
| physical observation | DirectGoal 或 NoGoalHint，`[5,613]` |
| semantic view | semantic-view-v3，`[5,61]` |

reward、task/rule layer、hard action mask、termination、truncation、info schema 与 canonical core
environment 保持一致。禁止 padding、截断、改变字段顺序或为 O3 创建新网络。

## 4. 地图语义与结构证书

地图文件必须是 UTF-8（无 BOM）、LF、末尾一个换行、20 行、每行 24 个字符，只允许 `X`、`.`、
`G`。`X` 是货架格而非静态墙，`.`/`G` 是 highway；因此 O3 论文术语固定为“未见的载货运输
highway 拓扑与载货流瓶颈”。结构证书在 loaded-transport highway graph 上计算：节点为 `.`/`G`
格，边为四邻接。

### 4.1 共同静态有效性

- highway graph 连通分量数为 1；
- 恰有 144 个 `X`、334 个 `.` 和 2 个 `G`；
- 每个货架格至少邻接一个 highway；所有货架均可经 highway 到达目标并返回原位；
- 8 个显式充电站唯一、不与 `G` 重合、属于 highway、有 highway 邻居且相互可达；
- 5 AGV reset 可生成唯一合法位置，queue/batch 始终有足够货架；
- 不存在孤立 highway、孤立目标或不可达作业分区。

### 4.2 长窄单通道

- 两个非平凡作业区之间只有一条宽 1 格、连续长度至少 6 格的 highway 通道；
- 移除该通道内部任一预注册 articulation 节点会把左右作业区分开；
- 通道内部不得放置 `G` 或充电站；
- 两个 `G` 位于同一侧，且两侧均有货架，使跨区载货运输在任务支持上存在；
- 证书记录两侧节点数、通道坐标序列、长度及 articulation 集合。

### 4.3 中央四向交叉瓶颈

- 四个非平凡作业区通过一个预注册中央 articulation 节点连接；
- 中心节点在 highway graph 中度数为 4，四条宽 1 格的臂在进入各作业区前长度至少 3；
- 移除中心节点后恰好得到 4 个含作业区的连通分量；
- 中心节点与四条臂不得放置 `G` 或充电站；两个 `G` 分置相对象限；
- 证书记录中心坐标、四臂坐标、分量大小和 articulation 结果。

## 5. 地图设计与人工冻结门

架构师根据第 3–4 节生成两张候选 ASCII 地图、结构审计摘要和 PNG 预览。研究所有者只审核地图
几何与合同符合性，不查看学习策略性能。owner 批准后才允许：

1. 将地图内容视为冻结输入；
2. 计算并写入精确 source/effective hashes；
3. 实现注册、环境工厂与后续接口验证。

批准后不得因为任何性能结果修改、替换或选择性删除地图；结构或文件错误只能修复后提高地图
版本并重新经过 owner 批准。

## 6. 注册、加载与哈希

`llm_mappo/o3_topologies.py` 定义冻结 `TopologySpec`：environment ID、version、资源路径、
`usage=evaluation_only`、5 AGV、显式充电站、source SHA-256、effective layout hash 和结构证书。

- O3 地图作为 package data 安装，editable 与非 editable 安装读取相同字节；
- source hash 对冻结文件原始字节计算 SHA-256；不允许自动重写换行或接受新 hash；
- effective hash 复用 `DynamicWarehouse.shadow_layout_hash()`，覆盖 grid、goals、highways、charging
  与 picking stations；
- evaluation factory 构造环境后必须同时核对两种 hash，任一不匹配即 fail closed；
- O3 环境只由显式 evaluation factory 注册/构造，不在普通包导入时全局注册；
- PNG 预览与结构报告是证据，不是环境输入。

## 7. 防泄漏合同

以下约束必须由测试和运行时代码共同保证，不能只写在文档中：

- optimization training config、legacy Phase 3/4 training、O2 runner 与标签收集入口遇任一 O3 ID
  必须明确拒绝；不得提供一个普通布尔开关绕过；
- O3 ID、source/effective hash 和地图路径不得出现在训练/O2 配置、60/800 label manifest、prompt、
  scenario generator 或 OOD reference corpus；
- O3 阶段不得实例化 optimizer、加载 checkpoint 或调用 policy evaluation；
- test-only seeds 固定为 `9301/9302`，不得使用 `200–209`；
- O3 结果不得参与 Teacher、reward、prompt、OOD、网络、checkpoint、路线或超参数选择；
- E2 启动前必须重新验证冻结 commit、两种 hash 和 O3 evidence manifest。

semantic query 中的 layout hash 等于当前 O3 effective hash；offline corpus 保留其来源 hash。跨布局
可靠性仅由已冻结的 61D OOD 公式决定，可靠性为 0 时正常 fail closed，禁止回退旧标签、规则标签
或重生成标签。

## 8. 接口与确定性

- 相同 topology ID + seed 的 reset state、初始任务、允许的动态入库随机事件、observation、mask 和
  固定动作短轨迹必须逐字节一致；不同 seed 只改变既有合同允许的随机量；
- action space、`[5,5]` hard mask、reward/termination/info schema 与 canonical 环境一致；
- DirectGoal 与 NoGoalHint 均为 `[5,613]`，除九位 goal geometry block 外其余槽一致；
- semantic-view-v3 为 `[5,61]`，当前环境 effective hash 进入 semantic JSON provenance；
- throwing planner 替身下 reset + 固定动作短 step 成功，planner query 始终为 0；
- Pure Motion Teacher 能按当前 effective hash 查询，且不改变其 O0 输入、输出或 cache 合同；
- shadow snapshot capture/restore 必须校验当前 effective hash 并保持 state/RNG 等价。

## 9. 失效与主张边界

若 O1 A600 gate No-Go 并返回 O0，地图文件、source hash 和纯图论结构证书可保留；所有依赖
Student、Teacher、observation、mask 或 shadow snapshot 接口的 O3 证据自动失效，必须在新批准
commit 上重跑。

O3 通过只允许声明“两个真正未见的 evaluation-only 载货运输拓扑已经冻结且接口就绪”。不得声明
策略已经泛化、方法更稳健、场景更困难或达到任何性能水平。

## 10. 已批准决策

上述范围、依赖、两个拓扑定义、5 AGV/24×20/144 shelves/2 goals/8 stations、显式文件方案、双
哈希、防泄漏、test-only seeds、owner map-preview gate、E2 性能归属与 O1 No-Go 失效规则已由
研究所有者于 2026-08-26 批准；其中 shelf/station 数量经 canonical runtime 核验后由 owner
明确修正为 `144/8`。

## 11. Open questions

无。地图的精确坐标由 O3-B 按冻结结构约束生成，并由预注册 owner preview gate 作唯一人工选择；
该选择不允许读取任何学习策略性能。
