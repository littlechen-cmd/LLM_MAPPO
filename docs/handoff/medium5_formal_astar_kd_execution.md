# medium-5ag 正式 A* KD 训练执行说明

2026-10-06。适用于本次 900k × 8 seed 正式比较；先运行 seed 7，审查后再运行其余 7 个 seed。
本文件与 `configs/training_v1/medium-5ag-formal-astar-kd/` 是本次执行入口。

## 配置审查结论

采用 `MAPPO-FP-ASTARKD-NO-VALIDITY`、`astar-geometric-cost-v2`。
教师仅提供静态几何三动作软标签，监督独立 MotionPriorHead 并影响共享 Encoder/GRU；
不直接替换 policy action head，不分配任务，不在评估期间参与逐步动作。
no-validity 关闭语义有效性门，但保留技术有效性、任务适用范围及动态阻挡衰减；
临时 picking lock / 拾取边界仍保留适用范围，避免截断空载至载货的监督衔接。

| 参数 | 本次值 |
|---|---|
| 训练 seed | 7、17、27、37、47、57、67、77；8 个配对 seed |
| 累计环境步 | 每个 seed 900000；所有 worker 累计，不是每个 AGV 或 worker 的步数 |
| episode max_steps | 训练与评估均 800 |
| AGV / 充电站 / 拣选站 | 5 / 5 / 2 |
| 初始活跃任务 | 5；每交付一单补一单 |
| 任务机制 | balanced-random-v1 + rware_delivery |
| 并行与 rollout | 16 个 CPU worker；64；完整更新 1024 个环境 transition |
| PPO epochs / minibatches | 5 / 1 |
| γ / GAE λ | 0.99 / 0.95 |
| PPO clip | 0.2 |
| Actor / Critic 初始 LR | 3e-4 / 3e-4；900k 全程线性衰减至 0，与 baseline 相同 |
| 熵 / value loss 系数 | 0.01 / 0.5 |
| 梯度裁剪 / Huber δ | 0.5 / 10 |
| hidden / Actor / Critic 维度 | 64 / 209 / 689 |
| A* 独立日程 | astar_schedule_steps=600000 |
| A* λ | 起始 0.05；至 420k 线性衰减为 0.01；保持至 600k；此后为 0 |
| 尾段 | 600k–900k 仅 PPO + 熵损失更新；不重置网络或优化器 |
| 600k 处 LR | 约 1e-4；仍有可学习尾段，与 baseline 日程相同 |
| 检查点 | final + latest；每 25k 更新 latest；450k / 900k 命名里程碑 |
| 主评估 | 每个 final checkpoint 200 个实例；masked argmax |
| 评估 seed 派生 | base 200–209，各 20 个；实际 seed=1000×base+index，index=0–19 |
| 评估教师 / planner / 在线 LLM | 均 0；训练期 A* 教师查询另记 |
| LLM dataset / 初始化 | 本 A* 单独组不使用 LLM dataset 或表示初始化 |
| 产物目录 | artifacts/formal_environment_training/medium-5ag/astar-kd/ |

reward 与本次 baseline 配置逐字段一致；不沿用旧诊断的训练 500 步/评估 800 步、
旧 reward、旧 6 worker 或旧 450k 预算。λ 日程使用 update 时的累计步数，因此跨越
600k 的首个 update 起置零；不把向量化更新边界说成恰好逐 transition 切换。
尾段仍生成教师标签和诊断，不意味着训练期所有教师查询停止；蒸馏损失和共享梯度必须为零。
熵奖励保留探索动力，但不保证持续探索或充电策略学习成功。

## seed 7 审查及论文边界

1. run manifest、配置 SHA、代码 commit、final checkpoint 身份一致；900k、879 updates、200 个评估实例完整。
2. 所有关键训练数值有限；前段教师有效监督确实发生，不能只有标记而无蒸馏梯度。
3. 600k 后 `astar_lambda`、`astar_loss`、`astar_shared_grad_norm` 为 0；900k 前 PPO LR 为正。
4. 评估采用 final checkpoint，200 个既定实例，teacher/planner query 为 0。
5. 同时检查交付数、实际时长、死锁、能量死亡、充电与补单；不单独按每千实际步吞吐判断。
6. 完成数全零、严重异常或日程未生效时暂停追加 seed 并提交负责人判断；不自动替换 seed。

seed 7 属于正式样本，失败也保留。审查仅决定是否继续同一配置；不根据这次预留评估
调整超参数再将新结果当作原合同的确认性比较。若需改方案，应版本化并明确探索/确认界限。
无需胜过 baseline 才能放行；单 seed 不支持优劣结论，后续报告必须包含全部 8 个 seed。
纯 PPO 尾段指优化目标，已有 A* 训练历史和附加 head 不被删除，不等同于重新训练 baseline。

## 服务器执行

长训练由负责人在冻结环境和 GPU lease 满足后手动启动。服务器使用独立新 checkout，
不覆盖已完成 baseline 的结果。同步包和命令见本次交付的 `server_sync` 清单及聊天命令。
先做 CUDA dry-run，确认 blockers 为空，再启动 `--seeds 7 --run`。
完成后交回 seed 7 的完整目录；审查通过再执行其余 seed，不自动追加。

### 同步（本地 PowerShell）

本轮同步清单会包含 bundle、commit 和 sha256 三个文件。复制后执行：

```powershell
$TrainingSyncHost = "lzx@10.181.115.40"
$TrainingSyncDirectory = "D:\codeProject\llm-a-mappo\.worktrees\formal-environment-training-spec\artifacts\server_sync"
scp "$TrainingSyncDirectory/medium5-astar-kd.bundle" "$TrainingSyncDirectory/medium5-astar-kd.commit" "$TrainingSyncDirectory/medium5-astar-kd.sha256" "${TrainingSyncHost}:/home/lzx/"
ssh $TrainingSyncHost
```

### 新 checkout 与 CUDA 预检（服务器 Bash）

```bash
(
set -euo pipefail
cd /home/lzx
sha256sum --check medium5-astar-kd.sha256
SYNC_CHECKOUT=/home/lzx/llm-a-mappo-workspace/medium5-astar-kd-20261006
SYNC_COMMIT=$(cat /home/lzx/medium5-astar-kd.commit)
git clone --single-branch --branch codex/formal-environment-training-spec \
  /home/lzx/medium5-astar-kd.bundle "$SYNC_CHECKOUT"
cd "$SYNC_CHECKOUT"
test "$(git rev-parse HEAD)" = "$SYNC_COMMIT"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
env -u PYTHONHOME PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python \
  "$PWD/scripts/run_medium5_formal_astar_kd.py" --device cuda --seeds 7
)
```

clone 仅执行一次；已存在目录时进入目录核对 commit，不能覆盖旧目录。
预检仅检查配置和依赖，不进行训练或策略评估。共享 GPU/lease 按项目既有流程确认。

### 启动 seed 7（服务器 Bash）

预检 `launch_blockers=[]` 且资源条件满足后执行：

```bash
(
set -euo pipefail
cd /home/lzx/llm-a-mappo-workspace/medium5-astar-kd-20261006
test "$(git rev-parse HEAD)" = "$(cat /home/lzx/medium5-astar-kd.commit)"
TRAIN_LOG="/home/lzx/medium5-astar-kd-seed7-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup env -u PYTHONHOME PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python -u \
  "$PWD/scripts/run_medium5_formal_astar_kd.py" --device cuda --seeds 7 --run \
  > "$TRAIN_LOG" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "$TRAIN_LOG"
)
```

检查启动后的日志，而不将得到 PID 等同于训练成功。完整训练和固定评估完成后，
保存并回传 `artifacts/formal_environment_training/medium-5ag/astar-kd/seed_0007_*`，
其中应有 manifest、state、summary、updates、episodes、evaluation 和 final checkpoint。

### 审查通过后的其余 7 seed（暂不执行）

仅在负责人/架构师完成上述审查后执行；`--confirm-seed7-review` 是人工审查声明，
不是由 seed 7 训练自动产生的通过标记。启动器会复核 seed 7 完整产物和同一代码/配置身份。

```bash
(
set -euo pipefail
cd /home/lzx/llm-a-mappo-workspace/medium5-astar-kd-20261006
test "$(git rev-parse HEAD)" = "$(cat /home/lzx/medium5-astar-kd.commit)"
SEED7_RUNS=("$PWD"/artifacts/formal_environment_training/medium-5ag/astar-kd/seed_0007_*)
test "${#SEED7_RUNS[@]}" -eq 1
test -d "${SEED7_RUNS[0]}"
env -u PYTHONHOME PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python \
  "$PWD/scripts/run_medium5_formal_astar_kd.py" --device cuda \
  --seeds 17 27 37 47 57 67 77 \
  --seed7-reviewed-run "${SEED7_RUNS[0]}" --confirm-seed7-review
TRAIN_LOG="/home/lzx/medium5-astar-kd-remaining-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup env -u PYTHONHOME PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python -u \
  "$PWD/scripts/run_medium5_formal_astar_kd.py" --device cuda \
  --seeds 17 27 37 47 57 67 77 \
  --seed7-reviewed-run "${SEED7_RUNS[0]}" --confirm-seed7-review --run \
  > "$TRAIN_LOG" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "$TRAIN_LOG"
)
```

其余 seed 串行执行；有失败立即停止，保留结果，不覆盖或静默重跑。
启动锁只保护同 checkout 的本启动器，不代替共享服务器 GPU lease。
