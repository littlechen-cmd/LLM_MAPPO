# 三场景核心实验训练初版配置

更新日期：2026-10-05。负责人已授权本轮自主敲定初版数值，不再等待参数逐项确认。
完整交付为 [中文 XLSX](三场景核心实验训练配置.xlsx)，包含 16 张工作表、12 组配置、
96 次训练清单、数据集、预训练、seed、评估协议、来源身份与启动步骤。
这是可执行的初版合同；baseline 优化产生新证据时允许版本化修订，不覆盖历史运行。

## 核心矩阵

| 场景 | 地图 | AGV/充电/拣选 | 活跃任务 | 训练及主评估 max_steps | 每 seed 累计训练步数 |
| --- | --- | --- | ---: | ---: | ---: |
| tiny-3ag | 官方 RWARE 10×11 | 3/3/2 | 3 | 800 | 450,000 |
| medium-5ag | 现有 20×24 | 5/5/2 | 5 | 800 | 900,000 |
| formal-8ag | 正式 50×28 | 8/8/10 | 8 | 1200 | 1,800,000 |

每场景均为 baseline、A* KD、LLM KD、联合组，配对训练 seed 为
`7/17/27/37/47/57/67/77`。总计 96 次 RL 训练、100,800,000 累计环境 transition。
16 worker × rollout 64，每次正常更新采集 1,024 个累计环境 transition。
AGV 的动作样本数不另外累加到环境步数。

## 本版数值

- PPO：5 epochs、1 minibatch、γ=0.99、GAE=0.95、clip=0.2、hidden=64。
- Actor/Critic LR 均 3e-4，按各场景累计环境步线性降至 0；entropy=0.01、
  value loss=0.5、gradient norm=0.5、Huber delta=10。
- 保留 ValueNorm、优势归一化、clipped value loss、active mask 和动作硬掩码。
- Reward 沿用 C1 energy-death local-mean：交付5、取货0.5、进度0.1、受阻/碰撞
  惩罚幅度各0.1、每步0.005、死锁5、能量死亡20、充电事件0。
  局部系数须按现有 team 均值实现理解，不能误写成每次事件固定 team 收益。
- 电量倍率1.1，充电触发/释放阈值0.3/0.8，死锁阈值180。
- checkpoint 每25k保存，另保留预算50%/100%里程碑；主比较固定 final。
- A* 教师采用已实现 `astar-geometric-cost-v2 / no_validity`；禁用语义有效性检查，
  技术无解标签仍按现有 validity/scope 掩码排除。λ=0.05→0.01（前70%预算），
  后30%保持0.01，预算终点0；temperature=1、动态占位衰减0.25。
- LLM 使用已有 medium 原生209D主数据：120回合、600条AGV轨迹、6000个8帧窗口，
  train/validation/test=4200/900/900，按完整回合84/18/18隔离。
  三场景共用一份数据；这是控制标注成本的迁移设定，尚无各场景收益证据。
- LLM 主标注 DeepSeek Flash，试标120计入主集；盲审60、复标40、Pro对照40，
  并发4，预算上限USD50；质量门和API参数全部列入工作簿。
- 离线预训练8个配对seed：batch256、最多100epoch、Adam LR0.001、四维权重各1，
  早停patience10/min_delta0.0001。现有实现导出早停末轮表示，未恢复最佳验证epoch。
  仅导入Actor Encoder+GRU，后续全部参与PPO微调；同seed的LLM/联合组共享表示。

## 实际短测与继承证据

按负责人最新指令，tiny完成后直接检查formal，不要求medium再做50k。

| 运行 | 步数/更新 | 已完成训练回合 | 诊断评估 | 实际含义 |
| --- | --- | --- | --- | --- |
| tiny seed7 | 50,000/49 | 48回合、38,400回合步、21交付、29能量死亡 | 9300–9302共0交付、3/3死锁 | 训练和评估完成；CSV布尔汇总故障已修复并恢复汇总，原failed状态保留 |
| formal seed7 | 4,096/4 | 256步/worker，尚无完成回合 | 9300–9302共0交付、3/3死锁 | final与summary已保存，接口可训练；不据此判断学习效果 |
| 既有medium C1-v3 seed7 | 600,000/586 | 见原日志 | 9400–9409共43交付、7,456实际步、42能量死亡、4/10死锁、0到站/充电恢复 | 支持本版沿用PPO及reward；不证明安全或持续运行 |

原始目录、代码身份、checkpoint哈希和有限loss/KL/entropy记录见
[证据快照](formal_training_short_test_evidence.json)。tiny的21交付只统计完成回合，
不能当作全部50k步的总交付。medium本次中断的短测不作为完成证据。

## 评估与真实依赖

主评估每个final模型使用200实例：base200–209各20个，
`episode_seed=1000*base+index`，index0–19；总计19,200回合。
formal另用400–409的10实例、5000步上限耐久评估：32个模型共320回合，
每1000步记录计数差分，提前终止保留实际时长，未运行窗口为null，不重启拼接。
按8个训练seed配对统计；12/16 AGV与未见拓扑仍是后续探索，不阻碍8 AGV主比较准备。

参数不再有待确认字段。**技术依赖仍据实保留**：2026-10-05 LLM标签处于
`halted / pilot / response_validation_failed_after_retry`，尚无完整标签或获批表示。
修复响应验证后，按工作簿完成标注和8次预训练；生成器读取真实SHA/identity即可生成
另外48份LLM配置，不能用占位文件启动。

## 启动入口与验证

在正式工作树执行 `python -m scripts.prepare_formal_training_configs`，生成48份
baseline/A*的approved配置及48项明确LLM依赖；加dataset/label参数可生成8份
预训练YAML。真实表示完成后重新生成完整96份配置。工作簿命令使用项目固定服务器
解释器，Windows短验证使用 `D:/Anaconda3/envs/py310/python.exe`。
耐久入口为 `scripts.evaluate_mappo_fp_baseline --protocol durability`，严格保留
原配置和checkpoint身份。长训练、真实预训练、付费API和长评估均由负责人运行。

工程师已有完整回归529通过/2项CUDA跳过；耐久分段追加40项聚焦通过。
本次交接再次执行聚焦核验，准确结果见最终交接。不宣称本地CUDA或四组长训练已通过。

2026-10-05最终交接复核：四个聚焦测试文件49项通过，110.33秒；变更代码静态检查
与差异格式检查通过。工作簿ZIP完整性、12组YAML对齐、96运行与200唯一实例核验通过。

最后协议分离核查发现formal主评估误附耐久字段，新增真实环境回归先失败后修复。
修复后的耐久/准备器11项通过（44.98秒）；正式配置统一UTF-8/LF。
