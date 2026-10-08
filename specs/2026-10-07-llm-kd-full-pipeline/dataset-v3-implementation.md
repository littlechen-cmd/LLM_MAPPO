# v3 数据流水线实施包

日期：2026-10-08。依据已批准评分与 medium 采集合同执行。
本文件具体化工程接口，不增加 API 预算、外发范围或长任务权限。

## 产物合同

- dataset manifest：`llm-dataset-build-run-v3`。
- label manifest：`llm-dataset-label-run-v3`。
- 主分布：`native-209d-goal-full-main-v3`。
- semantic contract：`observable-priority-coordination-four-score-v3`。
- manifest `semantic_dimensions` 使用四个 canonical 名的有序列表，
  `semantic_sources` 使用 `['llm','llm','llm','observable-rule-v3']`。
- 每条 label 的 scores、known_masks、sources、evidence、unknown_reasons
  均为四个 canonical 键的对象；未知分数原始为 null，训练占位为 0/mask=false。
- receipt 同时绑定 view、prompt、模型、请求及规则版本；前三维保留连续数值。

## 采集与选样

1. 来源 Actor 先按原始配置、checkpoint、导出哈希及来源身份严格加载；独立按
   获批新配置构造 rollout 环境，禁止改写 source 身份或旧 collector 的精确约束。
2. 新采集入口按真实 n_agents 分配 hidden/observation/mask 数组；支持结构上的
   3/5/12 规模，实际启用环境仍必须经过各自配置冻结。
3. 原始完整 episode 逐个压缩保存，绑定来源和 rollout 两类哈希；未到边界不能
   把截断诊断当完整 episode。恢复校验已完成文件，不覆盖失败和原始轨迹。
4. 先以确定性盐值将完整 episode 切成 70/15/15，再在分区内选择 8 帧窗口。
   同一 episode 不跨分区，同一 agent 的窗口至少间隔 8 步，跨分区语义视图去重。
5. 候选索引存储在磁盘，内存最多保留有界 episode 缓存及最终最多 8,000 个选择
   索引；不得把所有 episode 的所有 209D 窗口同时载入内存。
6. 选样只用原始可观察状态及已冻结规则参考，不读取 LLM 分数；明确记录结构、
   竞争、普通及充电服务覆盖。正式 labels 之前先报告候选覆盖不足。
7. 试标固定来自 train，120 条主试标计入正式配额；40 条复标/Pro 与 60 条审阅
   集合在获取标签前固定。失败结果不删除，不因结果不好重选 held-out。
8. 首段实际采集兼作连通与覆盖检查，继续同一采集身份；无需单独重复 smoke。

tiny/medium/formal 数据窗口配额依次为 700/150/150、2,100/450/450、
5,600/1,200/1,200。具体 episode 种子集合、选样清单、哈希及覆盖报告在数据冻结
时写入身份；本文件不将尚未生成的清单假写为完成。

## 标注与训练交接

- 新 prompt 仅要求前三维连续标签；第四维从同一 view 独立派生，来源明确。
- 请求、响应、失败、重试及费用预留继续留档；标签缓存键包含新 prompt 与合同。
- 不向旧 medium labels 目录追加新语义，不将旧四维重新命名为新标签。
- parser 通过后仍执行语义证据、服务排除、unknown、覆盖及一致性质量门。
- 只在数据与 label manifest 完整、合同一致且质量通过时设置 pretraining_eligible。
- 首批试标质量通过后继续同一次主标注；正式 pretrain 使用批准参数和真实哈希。
- long training 及多 seed 正式评价由所有者运行，命令必须实际可加载并无占位符。

## 最小回归范围

来源/rollout 哈希不混用；3/5/12 动态数组；中断恢复和 episode 对齐；窗口分区、
间隔与去重；连续分数不量化；unknown 梯度不参与；规则服务排除；严格新旧身份。
合成数据仅证明工程正确性，不作真实数据质量或训练效果证据。
