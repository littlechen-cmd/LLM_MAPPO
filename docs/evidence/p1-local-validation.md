# P1 本地交叉平台验证记录

日期：2026-08-28  
验证提交：`99d8610`（P1-G 记录提交前）  
范围：`codex/optimization` 的 Linux 服务器执行边界；不含 Linux 实机、O1 Gate、O2 或任何长训练。

## 本地结果

所有命令从仓库根目录使用既有 Windows `py310` 解释器执行。

| 检查 | 结果 |
| --- | --- |
| P1 聚焦回归 | 通过（见 P1-G 提交前复跑命令） |
| 完整 pytest | 通过：283 tests，56.06 s |
| Flake8 | 通过：`rware llm_mappo eval train scripts figures/core` |
| 受影响 CLI `--help` | 通过：环境验证、服务器预检、O1 Gate、wait-to-O1 launcher、`visualize.py` |
| YAML 解析 | 通过：全部 `configs/**/*.yaml` |
| 构建 | 通过：`python -m build` 生成 sdist 与 wheel |
| 静态安全审计 | 通过：活动运行入口无凭据、进程控制、A600 假设或普通 GPU 覆盖参数 |
| Git 审计 | 通过：`git diff --check`；P1-G 记录前工作树干净 |

历史 O0/O3 规格中保留了其形成时的 A600 叙述，故不应将“全仓库不含 A600”误读为
验收条件；本次扫描限定于当前活动的运行脚本、配置、协议、任务入口、runbook 和术语表。

## 冻结合同差异审计

P1 仅增加或修订以下类别：

- **治理与文档：**Linux 服务器依赖顺序、术语、协议、manifest、任务和 P1 规格；
- **环境与运行基础设施：**版本约束、只读环境/资源预检、GPU 0 租约、原子证据与显式恢复；
- **O1 启动接口：**将正常 Gate 固定为 baseline/H12，H4 仅允许在失败后诊断；
- **验证：**P1 单元测试、模拟 `nvidia-smi` 夹具、runbook 和本记录。

以下冻结项未被改变：MAPPO/Student 架构、Pure Motion A*、LLM 语义、reward、充放电
`1.10/0.30/0.80`、训练/正式 seed、12 workers、5 repeats、16/128 窗口、H=12、
1/16 sampler、512 expansion budget、O1 runtime/memory 阈值及 O2 预算。唯一的流程修正是
禁止 H4 作为正常 Gate 组成部分；这落实既有冻结合同，而非改变实验参数。

## 未完成项与禁止主张

- Linux Conda prefix、真实 RTX 4090/CUDA 绑定、五次连续空闲样本和 128-step smoke 仍由研究所有者执行；
- 共享服务器忙碌是预期状态。预检不通过时应保留证据并等待，不得干预其他人的进程；
- P1 尚未完成，O1 Gate 和 O2 均尚未开始；
- 本地模拟验证不等同于服务器可用性，也不支持性能、收敛、泛化或论文方法优越性的主张。

## 所有者下一步

先按 [环境安装 runbook](../runbooks/p1-linux-environment-setup.md) 创建并验证用户级 prefix，
再按 [短 smoke runbook](../runbooks/p1-linux-short-smoke.md) 产生 P1-H 证据目录。仅当架构审查
这些证据并明确验收 P1 后，才可运行 `docs/runbooks/p1-o1-tmux.md` 中的 O1 Gate 命令。
