# Requirements — P1 Linux 优化路线执行基础设施

## 1. 目标

把优化路线从已退役的 Windows/A600 假设迁移到研究所有者通过 SSH 使用的双 GPU Linux
服务器，并在不改变任何算法或实验合同的前提下，建立可安装、可审计、可等待共享资源、可恢复
基础设施故障的执行链路。P1 通过后，下一项工作必须是 owner-run O1 CUDA Gate；O1 Go 后立即
进入 O2 实现与六次校准运行，不得因服务器长期共享占用而跳过或降低 Gate。

## 2. 已确认服务器事实

| 项目 | 冻结值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS，Linux 6.8，x86_64 |
| CPU | 双路 AMD EPYC 7542；运行时仍须由 manifest 记录实际 socket/core/thread |
| RAM | 128 GB |
| GPU 0 | NVIDIA GeForce RTX 4090，49140 MiB，PCI `00000000:01:00.0` |
| GPU 1 | NVIDIA GeForce RTX 4080 SUPER，16376 MiB，PCI `00000000:21:00.0` |
| 驱动 | 580.173.02 |
| 系统盘 | 984 GB；审计时可用 670 GB |
| Conda | `/opt/miniconda3`；共享 base 为 Python 3.14.6，不得修改 |
| 项目目录 | `/home/lzx/llm-a-mappo` |
| 资源状态 | 服务器通常存在其他计算进程；审计时同一 Python PID 同时占用两张 GPU |

上述占用状态只说明当前不可运行 Gate，不是永久机器合同。任何实现不得终止、挂起、降优先级或
抢占其他用户的进程。

## 3. 范围

### 3.1 包含

- 只服务优化路线；稳定路线的 Linux 迁移不属于 P1；
- 更新 Mission、Tech Stack、Roadmap、TASKS、实验协议、机器 manifest 与术语；
- 在用户目录创建独立 Python 3.10.19 Conda prefix 的可复制安装指引；
- 冻结 Linux 依赖约束和 PyTorch 2.10.0 CUDA 12.8 wheel；
- 提供无副作用的机器采集、一次性预检和限时等待预检；
- 通过物理 GPU index、UUID、PCI bus ID 与 `CUDA_VISIBLE_DEVICES` 固定设备；
- 提供项目级 GPU 文件锁，防止本项目两个作业误用同一卡；
- 修正 O1 正常 Gate 为 `baseline/H12`，H4 仅为失败后诊断；
- 为 O1 Gate 提供原子分片、显式恢复、状态文件和完整 provenance；
- 提供 owner 在 `tmux` 中手动启动的 wait→preflight→O1 编排入口；
- 为 O2 预留只读消费 O1 Go 证据的接口，但不在 P1 运行或伪造 O2；
- Windows 本地 mock 回归与 owner-run Linux 安装/短 CUDA smoke。

### 3.2 不包含

- 不运行 O1 Gate、O2、正式训练、正式评估或长回放；
- 不修改 MAPPO、Pure Motion Teacher、Reward Calibration、LLM 标签、环境、奖励、能源、seed、
  12 workers、runtime/memory 阈值或训练预算；
- 不引入 DDP、DataParallel、单次训练跨双 GPU、Slurm、Docker 或 Apptainer；
- 不让 P1 自动安装环境、登录 SSH、传输凭据、push、kill 进程或删除旧 artifact；
- 不宣称 4080 与 4090 的训练结果天然可互换；正式 GPU 分配到 E1 预注册。

## 4. Python 与依赖合同

- Windows canonical interpreter 保持
  `D:\Anaconda3\envs\py310\python.exe`；
- Linux canonical interpreter 固定为
  `/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python`；
- Linux 环境是用户级 Conda prefix，不写 `/opt/miniconda3/envs`，不修改 base，不依赖
  `conda activate`；
- Python 固定 `3.10.19`；PyTorch 固定 `2.10.0+cu128`；
- `constraints/linux-py310-cu128.txt` 至少固定：NumPy 2.2.6、SciPy 1.15.3、Gymnasium 1.2.3、
  PyYAML 6.0.3、psutil 7.2.2、Matplotlib 3.10.9、NetworkX 3.4.2、Pyglet 1.5.31、pytest
  9.0.2、Flake8 7.3.0、TensorBoard 2.21.0、build 1.5.0、Pillow 12.1.1；
- 安装顺序固定为创建 prefix、安装官方 cu128 Torch、使用 constraints 安装 editable extras；
- 验证后导出 `environment.freeze.txt` 与 SHA-256 到 P1 artifact，不把服务器绝对环境副本提交仓库。

## 5. GPU 与共享服务器合同

### 5.1 设备身份

- O1 Gate 和 O2 校准的 canonical physical device 为 GPU index 0；名称必须精确等于
  `NVIDIA GeForce RTX 4090`，总显存必须不少于 48000 MiB；
- launcher 在任何 Torch 子进程启动前设置 `CUDA_VISIBLE_DEVICES=0`；子进程只允许使用
  logical `cuda:0`；
- manifest 同时记录 physical index、运行时 UUID、PCI bus ID、name、total/free memory、driver、
  Torch 与 CUDA runtime；首次 P1 smoke 记录 UUID，但不把实测前未知的 UUID 猜写进源码；
- immutable machine identity hash 只覆盖 OS/architecture、CPU 型号与拓扑、GPU
  index/UUID/PCI/name/total memory 和 driver；CPU load、free RAM/GPU/disk 等动态量只进入 preflight
  report，不进入 resume identity；
- GPU 1 在 P1 只允许短诊断或未来扩展验证，不用于 O1/O2，也不改变正式 GPU 分配合同。

### 5.2 Fail-closed 预检

`configs/optimization/p1_linux_server.yaml` 固定：

- `physical_gpu_index: 0`；
- `expected_gpu_name: NVIDIA GeForce RTX 4090`；
- `minimum_total_gpu_memory_mib: 48000`；
- `minimum_available_ram_gib: 64`；
- `minimum_free_disk_gib: 200`；
- `maximum_cpu_percent: 50`；
- `poll_seconds: 60`、`required_consecutive_free_samples: 5`、
  `wait_timeout_hours: 48`；
- `require_clean_git: true`、`required_python: 3.10.19`、
  `required_torch: 2.10.0+cu128`。

每次 free sample 必须同时满足：目标 GPU 无外部 compute PID、目标 GPU free memory 不少于总量
95%、available RAM/磁盘达标、最近采样 CPU utilization 不超过 50%。五次连续样本中任一次失败
即清零计数。48 小时超时后以非零退出并保留 wait log；owner 可重新启动，不得自动放宽阈值。

### 5.3 Lease 与并行

- launcher 通过 Linux `fcntl.flock` 持有 `/tmp/llm-a-mappo-optimization-gpu-0.lock`；
- 锁只能防止本项目内部冲突，不能替代 `nvidia-smi` 外部进程检查；
- O1 Gate 独占 GPU 0，Gate 期间不得启动本项目第二个 GPU 作业；
- P1 可以记录 GPU 1，但不提供绕过 canonical GPU 0 的 O1 参数；
- 未来 E1 若启用双 GPU 并行，必须按比较组/seed 配对预注册设备分配并另行验证。

## 6. O1 Gate 与 H4 诊断

- 正常 Gate CLI 只接受 `--modes baseline h12`；任何包含 H4、缺少 baseline/H12 或顺序不同的
  调用均失败；
- H4 由独立 `diagnose-h4` 子命令执行，只在正常 Gate No-Go 后允许；
- H4 artifact 必须写 `diagnostic_only=true`、`can_change_gate_result=false`，且不含 `gate_pass`；
- 正常 Gate 保留 12 workers、5 repeats、16 warm-up vector steps、128 measure vector steps、
  2 warm-up memory windows 与 10 measure memory windows；
- runtime Go 仍为 `median(H12)/median(baseline) <= 3.0`；memory Go 仍使用既有 64 MiB/5%、
  Spearman `rho>=0.80` 与对象增长合同；
- O1 summary 只有 runtime 与 memory 同时 Go 才写 `gate_pass=true`；
- runner 必须确认训练 config 使用 `device: cuda`，且实际 Torch tensor 位于 logical `cuda:0`。

## 7. 产物、恢复与 O2 衔接

- P1 根目录：`artifacts/optimization/p1_linux_server/`；
- O1 根目录：`artifacts/optimization/o1_cuda_gate/<run_id>/`；
- `run_id` 为 UTC 时间戳加 8 位 commit 前缀，不由用户自由覆盖；
- 所有 JSON/CSV 先写同目录 `.tmp`，flush+fsync 后以 `os.replace` 原子落位；不可变 artifact
  若已存在即拒绝覆盖，只有 `state.json` 允许以相同 run identity 原子替换；
- 每个 runtime repeat 与 memory window 形成独立、带 config/code/machine hash 的 shard；
- `--resume <run_dir>` 只复用 hash 全部一致且 schema 校验通过的 shard；不一致、损坏或缺 provenance
  时 fail closed；
- 可恢复基础设施失败只包括：owner 发送 SIGINT/SIGTERM、子进程因此以 130/143 退出、目标 GPU
  在运行中出现外部 PID 并被监控器检出，以及 `errno` 为 `EIO/ENOSPC/EDQUOT/ESTALE` 的
  `OSError`。它们写 `status=interrupted_infrastructure`；受污染或未完成 shard 不得复用；
- CUDA OOM、NaN/Inf、算法异常、安全失败、断言失败与所有未知异常写
  `status=algorithm_failure`，禁止 resume 覆盖；
- O1 Go 后输出 `o1_gate_receipt.json`。O2 后续入口必须验证 receipt、commit、config hash、GPU
  identity 与 P1 environment hash；P1 只实现和测试 receipt API，不启动 O2；
- owner 在 P1 通过后启动 O1；Codex 审核 O1 artifact。Go 后必须立即进入 O2 工程/六次运行，
  No-Go 则按 O1 合同返回 O0，不允许因资源占用跳过。

## 8. 允许主张

P1 通过只允许声称：优化路线具有经验证的 Linux Python/CUDA 环境、共享服务器资源门禁、可追溯
GPU 绑定和可恢复 O1 基准链路。不得声称 O1、O2、算法性能、收敛、泛化或正式实验已经完成。

## 9. Open questions

无。GPU UUID 不是开放决策；它由 P1 owner smoke 实测并进入证据 manifest。
