# S1 Owner-run 命令、manifest 与产物 schema（Linux 服务器）

- 任务：S1 稳定路线旧 Phase 3 A* 行为兼容恢复验收。
- 状态：命令与产物 schema 已定稿；正式验收由研究所有者在 Linux 服务器运行。
- 运行目标：`lzx@10.181.115.40`，仓库 `/home/lzx/llm-a-mappo`。

## 1. 运行环境声明

- Python：`/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python`（py310）。
- 代码 commit：`codex/stable`，正式运行前用 `git rev-parse HEAD` 记录实际 SHA。
- 配置文件：`configs/s1_phase3_dynamic_ingress.yaml`，运行前记录其 SHA-256：
  `sha256sum configs/s1_phase3_dynamic_ingress.yaml`。
- 设备：S1 验收为 CPU 单进程评估，无需 GPU；RTX 4090（GPU 0）仅供训练使用。

## 2. 冻结配置

`configs/s1_phase3_dynamic_ingress.yaml` 已冻结：3 AGV、目标 9、动态入库
（`batch_interval: 40`、`batch_size_range: [1, 3]`）、`battery_cost_scale: 1.1`、
`charge_threshold: 0.3`、`charge_release_threshold: 0.8`。冻结断言由
`tests/test_s1_stable_route.py::test_s1_config_freezes_stable_environment_contract` 覆盖。

## 3. 命令

### 3.1 短 smoke（工程师本地已跑，非验收）

```powershell
python eval/evaluate_dynamic_ingress_astar.py `
  --config configs/s1_phase3_dynamic_ingress.yaml `
  --seeds 300 --episodes-per-seed 2 `
  --reservation-mode legacy --coordinator-yield-action right `
  --output artifacts/stable/smoke/s1_smoke_frozen_legacy_right.json
```

### 3.2 S1 验收（研究所有者在 Linux 服务器运行）

运行前先检查 GPU 占用与内存（本验收虽为 CPU，仍确认共享服务器状态；不得终止他人进程）：

```bash
nvidia-smi; free -g; uptime
```

长任务用 `nohup`，日志写入 `/home/lzx/`（不写 Git worktree），不用 tmux：

```bash
cd /home/lzx/llm-a-mappo
RUN=run_$(date +%Y%m%d_%H%M%S)
mkdir -p artifacts/stable/s1_acceptance/$RUN
nohup /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python \
  eval/evaluate_dynamic_ingress_astar.py \
  --config configs/s1_phase3_dynamic_ingress.yaml \
  --seeds 300 301 302 303 304 305 306 307 308 309 \
  --episodes-per-seed 20 \
  --reservation-mode legacy --coordinator-yield-action right \
  --output artifacts/stable/s1_acceptance/$RUN/aggregate.json \
  > /home/lzx/s1_acceptance_$RUN.log 2>&1 &
echo $! > /home/lzx/s1_acceptance_$RUN.pid
```

- seed 集合：`300–309`，每 seed 20 episodes，共 200 episodes。
- 设备：CPU（评估脚本无 CUDA）；并行度：单进程（脚本内串行）。
- 覆盖/续跑：`RUN` 目录唯一，不静默覆盖；基础设施故障才允许同 seed、同配置重跑并保留故障记录。
- 阈值：完成率 ≥95%、碰撞 =0、能量死亡 =0、终止死锁率 ≤1%（四项同时满足才算通过）。

## 4. 产物 schema 与 run 目录

```
artifacts/stable/s1_acceptance/run_<ts>/
└─ aggregate.json          # 聚合结果（eval CLI 原生 schema）
/home/lzx/s1_acceptance_run_<ts>.log   # nohup stdout/stderr 全量
/home/lzx/s1_acceptance_run_<ts>.pid   # 进程号
```

`aggregate.json`（eval CLI 原生）关键字段：

- 顶层：`reservation_mode`、`coordinator_yield_action`、`seeds[]`、`episodes`、
  `task_completion_rate`、`mean_collisions_per_episode`、`deadlock_rate`、
  `mean_energy_deaths_per_episode`、`reservation_teacher{}`、`gate{passed}`。
- 逐 seed：`seed`、`episodes`、`task_completion_rate`、`mean_collisions`、`deadlock_rate`、
  `success_rate`、`mean_energy_deaths`、`path_livelocks`、`state_deadlocks`、
  `reservation_teacher{}`、`stall_diagnostics{}`。

schema 说明：eval CLI 当前只输出「逐 seed 聚合」，不输出「逐 episode 记录」。S1 验收以逐 seed
为统计单位（交接 §8 要求逐 seed 报告），接受现状；如需逐 episode 落盘属后续实现项。

## 5. 进度查看 / 正常终止 / 完整性检查 / 聚合分析

```bash
# 进度查看（tail nohup 日志）
tail -n 20 /home/lzx/s1_acceptance_run_<ts>.log

# 正常终止判定：aggregate.json 存在且含 gate 段
python -c "import json;d=json.load(open('artifacts/stable/s1_acceptance/run_<ts>/aggregate.json'));print(d['gate'])"

# 完整性检查：seeds 覆盖 300-309 各 20 episodes、无缺失
python -c "import json;d=json.load(open('artifacts/stable/s1_acceptance/run_<ts>/aggregate.json'));print(sorted(s['seed'] for s in d['seeds']), [s['episodes'] for s in d['seeds']])"

# 聚合分析（eval CLI 已输出逐 seed + 总体聚合，直接读 JSON）
python -c "import json;d=json.load(open('artifacts/stable/s1_acceptance/run_<ts>/aggregate.json'));print('completion',d['task_completion_rate'],'collisions',d['mean_collisions_per_episode'],'deadlock',d['deadlock_rate'],'energy_deaths',d['mean_energy_deaths_per_episode'])"
```

注：`eval/aggregate_formal_results.py` 的接口是 `--manifest/--evaluation-root/--output-dir/...`，
面向 E2 正式多组结果聚合，不适用于 S1 单 JSON 验收；S1 聚合直接读 `aggregate.json`。

## 6. 失败分类与处理

| 原因 | 判定 | 处理 |
|---|---|---|
| `target_reached` | 正常达成 9 件 | 计入完成率 |
| `deadlock` | 终止死锁 | 计入死锁率；>1% 即失败 |
| `time_limit` | 1000 步未达成 | 计入未完成 |
| `energy_failure` | 能量死亡 | 计入能量死亡率；>0 即失败 |
| 基础设施故障 | 进程/IO/环境异常 | 允许同 seed、同配置重跑，保留故障记录 |
| 数值/安全失败 | NaN/碰撞/能量死亡 | 保留为结果，不得静默重试 |

## 7. 任务标注（禁止混用）

- `artifacts/stable/smoke/`：短 smoke（工程师验证，非验收）。
- `artifacts/stable/s1_acceptance/`：S1 验收（研究所有者 300–309 × 20）。
- `artifacts/stable/predecision/`：S2 预备训练（仅 S1 通过后，用预声明非正式 seed）。
- E2 正式评估（`200–209`）另行冻结协议，与本命令隔离。

## 8. 禁止主张

不依据 smoke 或单次验收声称 S1 通过；不主张复现 99.72%；不主张未见拓扑或 8-AGV 泛化；不把
S2 预备结果计入最终统计。

## 9. S2 决策前预备训练（仅 S1 通过后启动）

- 启动条件：S1 验收通过、稳定路线配置已冻结、Linux 服务器 GPU 0 空闲。
- 隔离：产物写入 `artifacts/stable/predecision/`；结果不参与路线选择、不进入最终统计、不作为论文
  正式实验；若最终正式实验，必须按冻结协议从头重跑。
- seed：预声明且与正式 seed（`7/17/27/37/47`）不同。提议 `1/11/21` 三个诊断 seed，请研究所有者
  确认后再跑（本命令仅为链路验证与资源估算）。

运行前检查 GPU（仅用 GPU 0，不得占用他人进程）：

```bash
nvidia-smi; free -g; uptime
```

```bash
cd /home/lzx/llm-a-mappo
BASE=artifacts/stable/predecision/mappo_wp_astar_kd_eng
for S in 1 11 21; do
  CUDA_VISIBLE_DEVICES=0 nohup /home/lzx/.conda/envs/llm-a-mappo-py310/bin/python \
    train/train_phase3.py \
    --config configs/s2_phase3b_dynamic_ingress_astar_kl.yaml \
    --seed $S --episodes 200 --device cuda --parallel-envs 12 \
    --output-dir $BASE \
    > /home/lzx/s2_predecision_seed_$S.log 2>&1 &
  echo $! > /home/lzx/s2_predecision_seed_$S.pid
done
```

- GPU：训练只用 RTX 4090（GPU 0）。`device: cuda` 在代码里解析为 `cuda:0`（`_resolve_device`），
  再加 `CUDA_VISIBLE_DEVICES=0` 显式隐藏 4080 SUPER（GPU 1），双重保证不占用 GPU 1。
- 变体：`configs/s2_phase3b_dynamic_ingress_astar_kl.yaml` 为 Phase 3b（MAPPO-WP + A*KD +
  规则 engagement，`reservation_kl_coefficient: 0.05`、`engagement_coefficient: 0.10`）。
  离线 LLM 二维教师（`task_commitment`/`local_assertiveness`）属 E1 正式协议冻结项，本预备
  训练暂用规则 engagement 验证训练链路与资源。
- 预算：每 seed 200 episodes（资源估算用）；正式预算由 E1 冻结，不由 S2 决定。
- 检查：训练结束后核对 `episodes.csv`/`updates.csv`/`summary.json` 与 `checkpoint_final.pt` 存在、
  无 NaN/Inf；GPU 显存峰值与 wall-clock 记录进资源估算。
