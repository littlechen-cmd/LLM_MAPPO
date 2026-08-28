# P1 后的 O1 Linux CUDA Gate

只有 P1-H 的短 CUDA smoke 通过后，研究所有者才能在服务器运行本命令。它会等待共享 GPU 0
连续五次满足预检条件，最多等待 48 小时；不会结束、暂停或改变其他用户的进程。

先确认已按 [P1 Linux 环境安装](/home/lzx/llm-a-mappo/docs/runbooks/p1-linux-environment-setup.md)
生成 `artifacts/optimization/p1_linux_server/environment_report.json`，然后执行：

```bash
cd /home/lzx/llm-a-mappo
tmux new -s p1-o1
/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python scripts/run_o1_when_available.py \
  --server-config configs/optimization/p1_linux_server.yaml \
  --gate-config configs/optimization/o1_reward_calibration_smoke.yaml \
  --output-root artifacts/optimization
```

常用 tmux 操作：

```bash
# 不停止任务而离开会话：按 Ctrl-b，随后按 d
tmux attach -t p1-o1
tmux ls
```

若基础设施中断，先保留 `artifacts/optimization/p1_linux_server/` 与
`artifacts/optimization/o1_cuda_gate/` 中的文件。只在同一 run directory、同一 commit、配置、
机器身份和环境冻结哈希都一致时，才可在原命令末尾加：

```bash
  --resume artifacts/optimization/o1_cuda_gate/<run_id>
```

Launcher 的输出只会声明下一阶段：`O2` 表示 O1 Gate 已通过；`O0` 表示 No-Go。它绝不会自动
启动 O2。不要把短 smoke 或等待日志当成训练、收敛或论文性能证据。
