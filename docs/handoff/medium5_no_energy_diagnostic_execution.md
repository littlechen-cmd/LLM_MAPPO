# medium-5ag 无能源约束诊断执行说明

2026-10-06。负责人批准方案一：暂停新增正式多 seed 训练，先在无能源约束环境中重新训练 baseline 与 A* KD。
本轮为诊断，不改变既有能源环境的正式数据合同，不覆盖历史结果。

## 无能源约束的精确定义

通俗说明：AGV 不会因为电量耗尽退出，也不再被低电量规则引导去充电；只需完成任务和避让。
专业定义：`energy_enabled=false`，电量固定1，耗电/补电/能量死亡事件禁用；低电量惩罚、
任务领取的电量门、充电目标切换、充电预约和充电奖励关闭。能源相关计数为0。
任务队列、固定优先级、随机分配、取货—交付—回库、完成后补单、硬碰撞约束和死锁终止继续生效。

地图与站点坐标保留。原充电站格子作为普通可通行位置；Actor 的充电地图通道和充电 phase 为0，
自身/邻居电量为1，Critic 的充电占用特征为0。维度保持 Actor209 / FP Critic689。
默认 `energy_enabled=true` 保持能源环境行为；新模式用独立 schema/checkpoint 身份防止混用。
它不等同于仅将充电奖励设0（原配置该奖励已是0），也不以无限初始电量掩盖仍然活跃的目标机制。

## 两组训练合同

| 项目 | baseline | A* KD |
|---|---|---|
| method | MAPPO-FP-DG | MAPPO-FP-ASTARKD-NO-VALIDITY |
| 训练 seed | 7 | 7 |
| 总累计环境步 | 900000 | 900000 |
| workers / rollout | 16 / 64 | 16 / 64 |
| PPO epochs / minibatches | 5 / 1 | 5 / 1 |
| Actor/Critic LR | 3e-4，900k线性降至0 | 相同 |
| γ / GAE λ / clip | 0.99 / 0.95 / 0.2 | 相同 |
| 熵 / value / grad norm / Huber | 0.01 / 0.5 / 0.5 / 10 | 相同 |
| KD 日程 / 截断 | 无KD | 600k日程，450k硬截断 |
| KD λ | 0 | 0.05至420k降为0.01，450k起为0 |
| 纯 PPO 尾段 | 全程 | 后450k |
| 评估 | final，9400–9409，共10实例 | 相同 |

两组均从头初始化，不加载既有能源模型，不调用在线LLM。16worker×2learner并行，共32worker，GPU0。
共同环境为medium-5ag，20×24地图、5AGV、5个保留但不启用的充电站格子、2拣选站、
训练/评估800步、5个初始活跃任务、balanced-random-v1 + rware_delivery。
reward保留任务交付、取货、目标进展、阻挡、冲突、步耗时和死锁项，能源事件及惩罚贡献为0。

截断按更新结束时的累计步数判断；16×64为1024 transition/update，首次λ为0的更新结束于450560步，
而非在rollout中途重置参数。末次rollout缩短至精确900000，总879次更新。
KD置零后仍可能生成训练诊断标签，不把纯PPO尾段称为无教师查询；评估不查询教师。

A*教师几何v2 no-validity不变；当前450k截断仅作为起点，不假定它在新环境仍是最优。
未改变其他PPO参数来补偿模式切换；若需要后续修订，应保留本轮结果并生成新身份。
独立 schema `no-energy-diagnostic-v1`；普通正式 loader 拒绝加载本轮诊断配置。

## 验证与结果判读

启动前验证：电量保持1、耗电/充电/死亡/低电量/预约禁用，Actor/Critic能量特征中性，
正常能源模式仍工作；碰撞/死锁仍生效；确定性控制能真实重复完成取货、交付、回库和补单。
确定性控制是机制可解性检查，不是MAPPO已学习该行为的证据。

长诊断主要检查原始交付数、超过初始5单的回合比例、完整800步比例、死锁与任务循环。
实际步吞吐与时长并列报告，防止死锁早停造成分母偏差。完整800步不单独证明持续交付。
能源死亡应恒为0，不能把这个由环境保证的结果写成策略学会能源管理。
末100k训练交付、熵及KD尾段归零用于辅助判读，不以高熵单独宣称学习成功。

- 两组都改善：能源约束可能是重要限制，后续定位能源环境中的具体失败阶段。
- baseline能重复交付而KD受限：检查A*监督设计和优化影响。
- 两组仍低交付/严重死锁：先检查共同任务机制、观测或学习信号，不直接增加seed。

单训练seed、10诊断实例仅用于方向判断，不提供显著性/最优性结论。
本轮无能源环境结果不并入此前能源环境baseline表；不使用正式200实例。
后续是否进入多seed，以及无能源环境是否成为论文正式环境，由结果审查决定。

## 配置和产物

- 配置：`configs/no_energy_diagnostics/medium-5ag/baseline.yaml` 与 `astar-kd.yaml`。
- 启动：`scripts/run_medium5_no_energy_diagnostic.py`，默认预检，`--run`两组并行。
- 结果：`artifacts/no_energy_diagnostics/medium-5ag/baseline/` 与 `astar-kd/`。
- 长任务由负责人在冻结服务器环境、资源预检和项目GPU lease满足后启动；本地仅短验证。
- 失败保留，不换seed，不自动重试，不覆盖原正式结果。

## 服务器同步与启动

### 本地 PowerShell

```powershell
$TrainingSyncHost = "lzx@10.181.115.40"
$TrainingSyncDirectory = "D:\codeProject\llm-a-mappo\.worktrees\formal-environment-training-spec\artifacts\server_sync"
scp "$TrainingSyncDirectory/medium5-no-energy.bundle" "$TrainingSyncDirectory/medium5-no-energy.commit" "$TrainingSyncDirectory/medium5-no-energy.sha256" "${TrainingSyncHost}:/home/lzx/"
ssh $TrainingSyncHost
```

### 服务器 Bash：新 checkout 与预检

此目录独立于此前训练目录；只 clone 一次，不覆盖已存在的 checkout。

```bash
(
set -euo pipefail
cd /home/lzx
sha256sum --check medium5-no-energy.sha256
SYNC_CHECKOUT=/home/lzx/llm-a-mappo-workspace/medium5-no-energy-20261006
git clone --single-branch --branch codex/formal-environment-training-spec \
  /home/lzx/medium5-no-energy.bundle "$SYNC_CHECKOUT"
cd "$SYNC_CHECKOUT"
test "$(git rev-parse HEAD)" = "$(cat /home/lzx/medium5-no-energy.commit)"
nvidia-smi
env -u PYTHONHOME PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python \
  "$PWD/scripts/run_medium5_no_energy_diagnostic.py" --device cuda
)
```

预检应列出 baseline 和 astar-kd 两组，`energy_enabled=false`、900000步、seed7，
且 `launch_blockers=[]`。完成既有资源预检和项目 GPU lease 后启动；入口自身的批次锁
只限制本 checkout 重复启动，不代替共享服务器 GPU lease。

### 服务器 Bash：两组并行长诊断

```bash
(
set -euo pipefail
cd /home/lzx/llm-a-mappo-workspace/medium5-no-energy-20261006
test "$(git rev-parse HEAD)" = "$(cat /home/lzx/medium5-no-energy.commit)"
TRAIN_LOG="/home/lzx/medium5-no-energy-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup env -u PYTHONHOME PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python -u \
  "$PWD/scripts/run_medium5_no_energy_diagnostic.py" --device cuda --run \
  > "$TRAIN_LOG" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "$TRAIN_LOG"
)
```

两组在等待任何一组结束前均启动，32个CPU-only worker；批次manifest与独立日志在
`artifacts/no_energy_diagnostics/medium-5ag/launches/`。一组学习失败时保留结果，
另一组继续；不自动重试。driver退出/中断时只清理它自己启动的子进程。

### 完成后汇总

```bash
(
set -euo pipefail
cd /home/lzx/llm-a-mappo-workspace/medium5-no-energy-20261006
env -u PYTHONHOME PYTHONPATH="$PWD" \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python \
  "$PWD/scripts/analyze_medium5_no_energy_diagnostic.py" \
  --source artifacts/no_energy_diagnostics/medium-5ag \
  > /home/lzx/medium5-no-energy-review.json
)
```

分析器只读原始结果，检查两组完整900k/879次更新、配置/代码身份、诊断评估实例、
能源计数与蒸馏尾段；输出交付、完整800步比例、死锁、超过5单回合和最后100k统计。
回传整个 `artifacts/no_energy_diagnostics/medium-5ag/` 和汇总JSON，保留final checkpoint。
本地验证记录在 `docs/handoff/medium5_no_energy_preflight.json`；服务器CUDA及两组900k
尚未执行，不将本地接口短测称为学习性能验证。

## 本地验证和交接状态

- 任务ID：`NE-DIAG`；实施范围仅无能源约束诊断，不修改正式实验宪章及已完成能源结果。
- 全量：canonical Python执行 `-m pytest -q`，672 passed、2 skipped（无CUDA）。
- 最终专项：`tests/test_no_energy_{environment,launch,analysis}.py`，46 passed。
- 8个修改/新增生产Python文件flake8通过；`git diff --check`通过。
- baseline与A*各16worker/2048步CPU短测、seed9300评估完成；损失有限、能源计数0，
  A*缩放截断后蒸馏λ/损失/共享梯度0、PPO尾段学习率为正，维度209/689。
- 短测缩放预算、rollout与KD日程只验证接口，原900k配置及9400–9409评估未在本地执行。
- 修改文件：环境和工厂、Actor/Critic观测、runner、两个诊断脚本、两份配置、
  三份测试、术语及交接记录、TASKS/CHANGELOG；不含roadmap或共享环境安装修改。
- 已知限制：无能源模式不能验证充电恢复；450k在本模式仍是诊断起点；单seed不证明显著性。
- 本地提交ID在同步包同名`.commit`中；完成后工作树应干净，仅ignored的本地产物保留。
  不推送、不合并；服务器owner-run命令见上文。
