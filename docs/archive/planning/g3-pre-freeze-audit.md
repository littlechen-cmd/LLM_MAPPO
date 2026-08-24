# G3-1 预冻结核验记录

**日期**：2026-08-20  
**状态**：`PRE_FREEZE_AUDITED`，不是 `FROZEN`。本记录只核验可静态确定的
G3-1 输入，不改变正在运行训练的代码、配置、标签或产物。

## 已核验的候选输入

| 项目 | 核验结果 | 证据 |
|---|---|---|
| 候选代码基线 | `f6f06507f9d7c0e9d89873e0a7990934bde6fe65` | 分支 `feature/phase4-parallel-cuda` 与 `origin` 同步 |
| 四组定义 | 完整 `2×2`：`MAPPO-WP`、`MAPPO-WP+A*KD`、`MAPPO-WP+LLMKD`、`MAPPO-WP+A*KD+LLMKD` | `configs/g3_experiment_manifest.yaml` |
| 共同环境 | 四组 YAML 的环境区段完全一致；5 AGV、1,000 step 上限、动态入库、`1.10/0.30/0.80` 能源参数一致 | YAML 解析核验 |
| 共同训练字段 | phase、seed、环境 seed-group、并行环境数、环境步数预算、PPO 共同字段、标签路径及近邻数完全一致 | YAML 解析核验 |
| 唯一允许的组间差异 | 输出目录、A* KL 开关与系数、离线 LLMKD 开关与语义蒸馏系数 | `tests/test_phase4.py::test_phase4_formal_configs_encode_the_independent_teacher_matrix` |
| 离线标签 | 文件存在、400 条 JSONL 记录；SHA-256 与 manifest 一致 | `9928f5756c1261589946eb3aedf8dc2c1fa6f73f037cd05c31afca0683161797` |
| 评估协议 | 未见 seed `200–209`、每 seed 20 episode、确定性动作、统计单位为训练 seed | manifest `evaluation` 区段 |
| 结果与图表链路 | protocol、traceability、table schema、data manifest、聚合器和 F1/F2 绘图脚本均存在 | 29 项相关回归测试通过 |

## 文件指纹

| 文件 | SHA-256 |
|---|---|
| `configs/g3_experiment_manifest.yaml` | `32ca3aee98c59d2141f932a308c288df133994831f7b1813651bf2f3d31b0b25` |
| `configs/g3_core_mappo_wp.yaml` | `3a80b3ad7bb374e2b35b3684458dce4b78a15bf0da6b4f1f1d767d2ec4f94c5e` |
| `configs/g3_core_mappo_wp_astar_kd.yaml` | `81c9d177c7e9600170ea842ea2cfbf3048b844c4850e08da2687a0ed907c0d9b` |
| `configs/g3_core_mappo_wp_llm_kd.yaml` | `ac6ec88c561e53090bdced07440e903d366be8059555dd6bf15f0b8d0b6d770d` |
| `configs/g3_core_mappo_wp_astar_llm_kd.yaml` | `b1967b6369093940966797860db893e76cad21d3a6dbe48073f27a9b53996d49` |

## 不得提前解除的阻塞项

1. **G2-1d/e**：必须由项目负责人完成 legacy/fixed 的 `300–309 × 20 episodes`
   配对诊断并按预注册阈值审计。若修复不被采用，G3 候选代码基线必须相应调整。
2. **正在运行训练的可追溯性**：训练结束后，必须记录其启动 commit、配置 SHA-256、
   标签 SHA-256、命令、设备与产物路径；若任一项不匹配最终冻结协议，产物只能作诊断证据。
3. **G4 共同预算选择**：manifest 的 `formal_environment_steps` 仍为 `null`。只能在四组
   matched G4 曲线完成后按同一规则填入一个共同值，不能逐组设定。
4. **manifest 状态与代码引用**：manifest 当前为 `provisional_g3`，并记录旧的
   `preparation_base_commit=18d52f0...`。G2-1e 通过后，须在单独的冻结 commit 中将其更新为
   最终代码基线，并填入 `frozen_commit`；在此之前不得改为 `frozen_g3`。

## 正式冻结前的最小复核顺序

1. 审计 G2-1 配对结果并决定采用 fixed 或保留 legacy；
2. 记录正在运行训练的完整 provenance，确认其输入与候选协议一致；
3. 以同一代码基线重算本表全部 SHA-256，并运行配置矩阵、正式汇总和绘图链路测试；
4. G4 结束后按预注册规则写入一个共同正式步数；
5. 在独立 Git commit 内更新 manifest 的 `status`、`frozen_commit`、预算及最终指纹，再将
   G3-1 标为完成。

## 本次验证

- `python -m pytest tests/test_phase4.py tests/test_formal_results.py tests/test_formal_figures.py -q`
  → `29 passed`；
- 已确认 protocol、traceability、table schema、figure data manifest、正式聚合器和核心绘图
  脚本存在；
- 本次没有启动训练、多 seed 评估或长回放。
