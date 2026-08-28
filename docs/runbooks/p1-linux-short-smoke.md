# P1 Linux 短 smoke（仅研究所有者执行）

本步骤必须在 P1 环境安装验证通过后执行。它只确认 GPU 0 的 CUDA 绑定、
预检记录和一次 128 环境步的功能 smoke；它不是 O1 Gate，不得用于论文性能、
收敛或泛化结论。

在 `/home/lzx/llm-a-mappo` 中执行以下命令。所有结果均写入独立、带时间戳的
P1 证据目录；命令不会终止、暂停或调整其他用户的进程。

```bash
cd /home/lzx/llm-a-mappo
PYTHON_BIN=/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python
RUN_ID=p1_smoke_$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=artifacts/optimization/p1_linux_server/$RUN_ID
mkdir -p "$RUN_DIR"

# GPU 0 忙碌时，该命令以 No-Go（退出码 2）结束并留下只读预检报告；不要绕过它。
"$PYTHON_BIN" scripts/check_optimization_server.py \
  --config configs/optimization/p1_linux_server.yaml \
  --once --output "$RUN_DIR"

# 仅在上一条命令退出码为 0 后继续：连续五个合格样本，最多等待 48 小时。
"$PYTHON_BIN" scripts/check_optimization_server.py \
  --config configs/optimization/p1_linux_server.yaml \
  --wait --output "$RUN_DIR"

export CUDA_VISIBLE_DEVICES=0
"$PYTHON_BIN" -c 'import json, torch; x=torch.zeros((1,), device="cuda:0"); print(json.dumps({"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"logical_device":str(x.device),"device_name":torch.cuda.get_device_name(0),"finite":bool(torch.isfinite(x).all().item())}, sort_keys=True))' \
  | tee "$RUN_DIR/cuda_tensor_smoke.json"

"$PYTHON_BIN" train/train_optimization.py \
  --config configs/optimization/o1_reward_calibration_smoke.yaml \
  --output "$RUN_DIR/optimization_128_step_smoke" \
  | tee "$RUN_DIR/optimization_128_step_smoke.stdout.txt"

git rev-parse HEAD | tee "$RUN_DIR/git_commit.txt"
git status --short | tee "$RUN_DIR/git_status.txt"
sha256sum "$RUN_DIR"/preflight_*.json \
  "$RUN_DIR"/wait_*.jsonl \
  "$RUN_DIR"/cuda_tensor_smoke.json \
  "$RUN_DIR"/optimization_128_step_smoke.stdout.txt \
  "$RUN_DIR"/git_commit.txt "$RUN_DIR"/git_status.txt \
  | tee "$RUN_DIR/SHA256SUMS.txt"
```

保留整个 `RUN_DIR`，并把其路径及命令输出交给架构审查。若预检、CUDA 或 128 步
smoke 有任一失败，请停止；不要更换 GPU、降低阈值、缩短正式 H=12 或启动 O1/O2。

**术语说明：**“smoke”是很短的通路检查，类似先确认发动机能点火；它并不测试
车辆是否能完成长途赛程。
