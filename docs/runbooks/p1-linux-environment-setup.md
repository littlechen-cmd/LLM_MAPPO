# P1 Linux 优化路线环境安装

本说明只由研究所有者在 Linux 服务器执行。它创建一个用户级 Conda prefix，不修改共享
`/opt/miniconda3` 的 base，也不需要激活 Conda 环境。

在项目目录 `/home/lzx/llm-a-mappo` 中执行：

```bash
CONDA_BIN=/opt/miniconda3/bin/conda
PYTHON_BIN=/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python

"$CONDA_BIN" create --yes --prefix /home/lzx/.conda/envs/llm-a-mappo-py310 python=3.10.19
"$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128
"$PYTHON_BIN" -m pip install --constraint constraints/linux-py310-cu128.txt \
  numpy scipy gymnasium PyYAML psutil matplotlib networkx pyglet pytest flake8 \
  tensorboard build Pillow
"$PYTHON_BIN" -m pip install --constraint constraints/linux-py310-cu128.txt -e .

mkdir -p artifacts/optimization/p1_linux_server
"$PYTHON_BIN" scripts/verify_linux_environment.py \
  --constraints constraints/linux-py310-cu128.txt \
  --report artifacts/optimization/p1_linux_server/environment_report.json \
  --write-freeze artifacts/optimization/p1_linux_server/environment.freeze.txt
```

成功时最后一条命令退出码为 `0`，报告中的 `pass` 为 `true`。该步骤只验证软件环境；它不表示
P1、O1 或 O2 已通过，也不会占用 GPU 运行训练。

若核验失败，不要手工放宽版本、替换 GPU 或混用 base 环境。保留生成的报告并交给 Codex 分析。
