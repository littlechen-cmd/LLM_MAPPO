# 三场景核心实验合同

2026-10-05更新。本轮初版数值已按负责人预算及自主配置授权确定；
[中文 XLSX](三场景核心实验训练配置.xlsx) 为集中审阅入口，
[初版配置说明](formal_training_initial_configuration.md) 提供证据和边界。

## 共同合同

- 三场景tiny3/3/2、medium5/5/2、formal8/8/10，活跃任务分别3/5/8。
- max_steps=800/800/1200；environment_steps=450k/900k/1.8M，四方法同场景相同。
- balanced-random-v1、rware_delivery，每完成一单补一单；规则层随机分配给空闲AGV。
- 16worker、rollout64及C1 local-mean PPO起点；不修改环境机制。
- 四方法baseline、A* KD、LLM KD、联合组，各用8个配对训练seed，主矩阵96次运行。
- A*明确采用当前几何教师v2的no_validity版本；LLM四维语义以Actor可见8帧历史监督。
- LLM共用medium6000窗口，8份配对seed表示；仅Encoder+GRU初始化，PPO继续微调。
- 执行期MAPPO输出逐步动作；A* planner query与在线LLM调用均0。

## 评估、统计和失败处理

每个final模型复用200个预留实例，按8个配对训练seed汇总。主指标是原始交付数与
每千实际步交付；辅报取货、充电恢复、安全事件、实际时长及失败回合占比。
formal另做5000步/10实例耐久评估，1000步分段且不把早终止重启拼接。
统计报告每seed结果、均值/样本SD、配对差值和95%区间；bootstrap20000次，
统计seed20261001。主对照为三个教师方法各对baseline、三个场景共9项；
双侧精确sign-flip配对检验并Holm校正，n=8的局限随报告说明。

基础设施故障保留记录后同身份恢复；算法零交付、NaN、能量死亡和死锁不换seed。
只有medium联合组全部8个训练seed各自9400–9409诊断均零交付，才提交负责人决定
是否暂停formal长训练；baseline零交付不自动触发。旧诊断seed不进入正式比较表。
12/16 AGV压力评估须另定充电容量和Actor-only合同，未见拓扑后议。

## 当前可执行状态

短测tiny50k与formal4096已完成，不再要求medium50k。正式配置生成器支持approved，
同时保留provisional诊断拒绝长训练的门。baseline/A*48配置可启动；LLM48配置需先
完成现有pilot响应验证问题、全量标签及8份真实表示。这是产物依赖，不是待定数值。
本阶段不替负责人运行长任务、发布、推送或合并；保留后续优化的版本化修订空间。

## 2026-10-06 medium A* KD 执行修订

负责人批准 medium A* KD 在总计 900k 环境步内采用前 600k 蒸馏、后 300k 纯 PPO 尾段；
`astar_schedule_steps=600000` 独立于 `optimization_schedule_steps=900000`，
学习率与本次 baseline 全程相同，熵系数仍为 0.01。尾段蒸馏损失及共享梯度为零；
训练期间仍生成教师诊断，不将其表述为停止所有教师查询。评估期教师及 planner 查询均为零。
启动入口与按 seed 配置以 `medium5_formal_astar_kd_execution.md` 和
`configs/training_v1/medium-5ag-formal-astar-kd/` 为准。先运行 seed 7，经负责人/架构师
审查后才开放其余 7 个 seed；不以胜过 baseline 为准入条件，不根据预留评估调参或替换 seed。
其他场景及尚未启动的联合组在启动前须单独核对蒸馏日程，旧 XLSX 不作为本次新日程的执行入口。
