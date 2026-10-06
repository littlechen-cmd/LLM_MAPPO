# medium-5ag A* KD 截断时点诊断

2026-10-06。负责人批准新增四组并行、单训练 seed 的诊断。先比较不同纯 PPO 尾段，再决定正式 A* KD 截断配置。

## 四组合同

| 组名 | A* KD 截断累计步数 | 名义纯 PPO 尾段 | 首个 λ=0 的 update 步数 |
|---|---:|---:|---:|
| cutoff_450k | 450000 | 450000 | 450560 |
| cutoff_600k | 600000 | 300000 | 600064 |
| cutoff_750k | 750000 | 150000 | 750592 |
| cutoff_900k | 900000 | 0；无尾段对照 | 900000；此时学习率也为 0 |

四组均从头初始化，训练 seed=7，总预算 900000 累计 joint-environment transitions。
不从正式 baseline 或已有 A* KD checkpoint 接续。
所有环境、reward、Actor/Critic、PPO 参数及 900k 学习率日程与本次正式 baseline 一致。
每组 16 个 CPU-only worker、rollout 64、5 PPO epochs、1 minibatch；四个独立 learner 并行，
共 64 个 worker。GPU 使用 physical GPU 0，GPU 1 不参与；资源预检和项目 lease 仍须满足。

教师使用几何 v2 no-validity。四组共同使用 `astar_schedule_steps=600000`，
λ 在前 420k 从 0.05 线性降到 0.01，之后保持 0.01；
`post_schedule_astar_lambda=0.01` 保持共同曲线，独立 `astar_stop_steps` 在组内截断时将 λ 强制置零。
这样各组共有训练前缀的监督强度一致，不把不同衰减速度与截断时点混在一起。
相同 seed 不保证并行 CUDA 运行逐位相同；这是一轮配对诊断，不是多个独立训练 seed。

截断按 update 时的累计步数生效；末尾不足 1024 个 transition 的 rollout 缩短到精确 900k。
截断后不重置网络/优化器，熵系数保持 0.01。纯 PPO 是优化目标；教师诊断标签仍会生成，
不能把尾段描述为零训练教师查询。执行/评估时教师与 planner 查询须为零。
900k 对照最后一次零学习率 update 不算有可学习的纯 PPO 尾段。

## 评估和选择依据

四组 final checkpoint 各用 9400–9409，共 10 个诊断实例；masked argmax、800 步、
5 个初始活跃任务、balanced-random-v1 + rware_delivery，与训练工作负载一致。
不消费正式 200 个评估实例，不使用正式 baseline 200 实例均值与本诊断 10 实例做配对比较。

主要检查：

- 固定时长内原始交付数；实际运行时长、完整 800 步比例、死锁及全体 AGV 能量死亡比例；
- 实际步吞吐并列展示，防止早死亡缩短分母造成虚高；
- 多于初始 5 单的回合、充电发生回合及补单循环；不把充电 AGV-step 数当作恢复交付成功；
- 最后 100k 的交付、死锁、能量死亡及熵；不要把熵较高单独当作任务学习成功；
- 截断前蒸馏梯度存在，截断后 λ、蒸馏损失及共享梯度为零，PPO 学习率在 900k 前仍为正；
- 配置/代码身份一致、879 次 update、最终评估完整、无数值异常。

共同评估实例上做描述性比较，并检查训练曲线是否支持该比较。若交付领先但死亡/死锁明显恶化，
不据单一吞吐指标定胜负；接近时保留不确定性。此轮只提出正式截断候选，不能证明全局最优截断，
不能把 10 个评估回合当作 10 个独立训练 seed。失败也保留，不替换 seed 或静默重跑。

## 产物和运行入口

- 四组配置：`configs/astar_tail_diagnostics/medium-5ag/cutoff_*k.yaml`。
- 专用 schema：`astar-tail-diagnostic-v1`；普通正式 loader 拒绝这些配置。
- 专用入口：`scripts/run_medium5_astar_tail_diagnostic.py`；默认只预检，`--run` 启动四个独立 learner。
- 结果：`artifacts/astar_tail_diagnostics/medium-5ag/cutoff_*k/seed_0007*/`。
- 一组失败不会销毁其他组；其他已启动组继续并记录结果，不自动重试失败组。
- 批次锁只保护本 checkout 的诊断入口，不代替 GPU lease，也不限制其他程序。

诊断完成后回传四组完整目录；使用分析入口汇总交付、安全、熵与日程验证，再确定正式配置。
此前 600k/300k 正式配置保留为候选；已生成命令不视为本次诊断已经通过。
服务器长训练由负责人运行，本地只做确定性验证和短 smoke。

## 服务器同步与启动命令

### 本地 PowerShell

```powershell
$TrainingSyncHost = "lzx@10.181.115.40"
$TrainingSyncDirectory = "D:\codeProject\llm-a-mappo\.worktrees\formal-environment-training-spec\artifacts\server_sync"
scp "$TrainingSyncDirectory/medium5-astar-tail.bundle" "$TrainingSyncDirectory/medium5-astar-tail.commit" "$TrainingSyncDirectory/medium5-astar-tail.sha256" "${TrainingSyncHost}:/home/lzx/"
ssh $TrainingSyncHost
```

### 服务器 Bash：新 checkout 与预检

仅执行一次 clone；不覆盖已存在的目录。此 checkout 与上一轮正式训练准备目录分开。

```bash
(
set -euo pipefail
cd /home/lzx
sha256sum --check medium5-astar-tail.sha256
SYNC_CHECKOUT=/home/lzx/llm-a-mappo-workspace/medium5-astar-tail-20261006
git clone --single-branch --branch codex/formal-environment-training-spec \
  /home/lzx/medium5-astar-tail.bundle "$SYNC_CHECKOUT"
cd "$SYNC_CHECKOUT"
test "$(git rev-parse HEAD)" = "$(cat /home/lzx/medium5-astar-tail.commit)"
nvidia-smi
env -u PYTHONHOME PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python \
  "$PWD/scripts/run_medium5_astar_tail_diagnostic.py" --device cuda
)
```

确认预检四组均已列出、`launch_blockers=[]`，并完成既有资源预检及 GPU lease 后启动。
四组并发不表示一定比单组快四倍；CPU worker、线程和 GPU 会共享资源。

### 服务器 Bash：四组并行长诊断

```bash
(
set -euo pipefail
cd /home/lzx/llm-a-mappo-workspace/medium5-astar-tail-20261006
test "$(git rev-parse HEAD)" = "$(cat /home/lzx/medium5-astar-tail.commit)"
TRAIN_LOG="/home/lzx/medium5-astar-tail-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup env -u PYTHONHOME PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python -u \
  "$PWD/scripts/run_medium5_astar_tail_diagnostic.py" --device cuda --run \
  > "$TRAIN_LOG" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "$TRAIN_LOG"
)
```

`--run` 一次启动四组，而不是轮流等待。批次 manifest 记录每组 PID、命令、配置 SHA、
退出状态和独立日志；路径在 `artifacts/astar_tail_diagnostics/medium-5ag/launches/`。
driver 的 PID 不等于已成功完成训练；启动后检查 driver 日志和四组日志。

### 完成后汇总（服务器 Bash）

```bash
(
set -euo pipefail
cd /home/lzx/llm-a-mappo-workspace/medium5-astar-tail-20261006
env -u PYTHONHOME PYTHONPATH="$PWD" \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python \
  "$PWD/scripts/analyze_medium5_astar_tail_diagnostic.py" \
  --source artifacts/astar_tail_diagnostics/medium-5ag
)
```

只有四组完整结果均通过身份及日程检查才生成比较报告。
输出 `review/review.md`、`comparison.csv`、`comparison.json`；原始训练文件不被修改。
回传整个 `artifacts/astar_tail_diagnostics/medium-5ag/` 目录，保留所有失败结果、批次 manifest
及 final checkpoint。分析器输出待负责人确认的截断候选，不自动修改正式配置。
