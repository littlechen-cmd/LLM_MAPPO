# 需求 — 正式仿真环境训练初版配置

## 范围与授权

确认环境机制、接入三场景、敲定四方法配置及验证协议，交付中文XLSX。
负责人已指定tiny450k/medium900k/formal1.8M预算，并授权基于现有证据自主选择其余
初版数值，不再逐项确认。baseline优化结果可导致后续版本化修改；本版不是永久冻结。
不修改环境机制，不运行长训练/长评估/付费API，不读取或修改roadmap.md，
不修改正在优化的core-integration工作树。

## 环境合同

| 场景 | 大小 | AGV/充电/拣选 | 活跃任务 | 训练/主评估max_steps | environment_steps |
| --- | --- | --- | ---: | ---: | ---: |
| tiny-3ag | 官方RWARE10×11 | 3/3/2 | 3 | 800 | 450000 |
| medium-5ag | 现有20×24 | 5/5/2 | 5 | 800 | 900000 |
| formal-8ag | 正式50×28 | 8/8/10 | 8 | 1200 | 1800000 |

tiny充电(0,0)/(9,0)/(9,10)，拣选(4,10)/(5,10)；medium沿用现有站点；formal
启用JSON全部8充电/10拣选。正式图SHA为
`2a03a4abfe281de6865f39399d870f5ba9931d2520a6829bdb01788d50ca22fc`，
504货架、896非货架可通行格。Actor均209D；FP Critic随场景为201/689/2197D。

seed决定初态与请求；规则层给空闲AGV随机分配任务。balanced-random-v1均衡随机
A/B/C任务创建后保持优先级，权重2/1.5/1。rware_delivery完成一单即补一单；
无task_target终止或目标完成率分母。拣选暂停3步，低电量规则层指定充电目标。
MAPPO选择五种逐步动作，A*仅训练监督，LLM只离线表示初始化。
max_steps是上限；死锁等可提前终止，不修改终止机制制造满时长。

## 训练与教师

四方法baseline、A* KD、LLM KD、联合组，各场景各8配对seed：7/17/27/37/47/57/67/77。
16worker×rollout64=1024累计transition/正常更新；5epochs、1minibatch、γ0.99、
GAE0.95、clip0.2、hidden64、Actor/Critic LR3e-4线性到预算终点0、entropy0.01、
value loss0.5、gradnorm0.5、Huber10。checkpoint25k及50%/100%预算里程碑。
采用现有C1 local-mean reward、电量倍率1.1、充电0.3/0.8、deadlock180。
细项包括Adam、ValueNorm、优势归一化和采样末批，均在XLSX给出确定数值。

A*采用已实现astar-geometric-cost-v2/no_validity；仅禁语义validity检查，技术无解
仍按validity/scope过滤。λ0.05在前70%预算到0.01，后保持，终点0；τ1、动态衰减0.25。
LLM共享已有medium6000原生209D/8帧窗口，完整回合隔离，4200/900/900标签；
主模型Flash，试标120、盲审60、复标40、Pro对照40、并发4、硬预算USD50。
预训练8个配对seed、batch256、最多100epoch、Adam LR0.001、四维权重1、
patience10/min_delta0.0001，导出早停末轮Encoder+GRU；下游继续训练全部参数。
三场景共享medium数据属于预先声明的迁移方案，不声称已经验证收益。

provisional模板仅诊断；生成器输出approved正式配置。LLM必须有真实representation、
SHA、identity和张量，不允许占位文件；缺失产物以依赖记录表示，不再设参数批准问答。

## 评估

每个final checkpoint有200主实例：base200–209、index0–19，实际seed=1000*base+index。
同场景所有方法与模型复用同实例。formal另做seed400–409的5000步耐久评估，
每1000步报告差分；早终止尾窗记录actual_steps，未运行窗口null，不重置拼接。
正式主指标为原始交付、每千实际步吞吐、取货转交付，另报安全与充电恢复。
按8个训练seed而非200episode做配对统计；区间和检验规则以XLSX为准。

诊断9300–9302、历史9400–9409不混入主评估。只有联合组medium全部8训练seed
各自9400–9409交付总和均0，才提交负责人决定是否暂停formal，不自动停止。
12/16AGV压力评估需另定充电容量与Actor-only接口，未见拓扑后议。
有限5000步证据不等于无限lifelong保证。

## 短测和交付标准

tiny实际50k seed7完成后直接formal短测；formal只须证明训练可进行，不强求50k。
medium50k要求取消。报告零交付、实际步数、更新数、有限loss/KL/entropy、安全和
checkpoint身份；不得从单seed短测声称收敛或方法优劣。
XLSX须包括各规模每组所有初版数值、dataset、预训练与评估seed数量、命令和真实依赖。
已知LLM标签pilot失败及表示缺失不得隐瞒；修复后按给定配置启动，无需再确定参数。
