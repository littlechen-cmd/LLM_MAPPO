# P0 集成候选验证报告

## 1. 候选与环境

- 验证日期：2026-08-24
- 候选分支：`feature/phase4-parallel-cuda`
- 候选 commit：`218007aab7b5596b08730c27606b92d0cbd775af`
- 干净检出：`artifacts/p0_clean_checkout/repo/`（由 Git 忽略）
- Python：`3.10.19`，解释器 `D:\anaconda3\envs\py310\python.exe`
- 关键版本：PyTorch `2.10.0+cpu`、Gymnasium `1.2.3`、NumPy `2.2.6`
- 平台：Windows，本地 CPU 验证；未使用 A600，未运行长训练或长评估。

干净检出由候选 commit 直接克隆，初始 `git status --short` 无输出。验证结束后移除人工
写入探针，检出目录仍无 tracked 或 untracked 修改。主工作区仅包含本报告及对应任务状态更新。

## 2. 安装合同

为避免修改现有 `py310` 环境，使用同一解释器、无网络、无依赖重装，并把 editable 安装写入
Git 忽略的独立 prefix：

```powershell
D:\anaconda3\envs\py310\python.exe -m pip install `
  --ignore-installed --no-index --no-build-isolation --no-deps `
  --prefix D:\codeProject\llm-a-mappo\artifacts\p0_clean_install `
  -e D:\codeProject\llm-a-mappo\artifacts\p0_clean_checkout\repo[dev,train]
```

退出码为 `0`。生成了 `llm_mappo-0.1.0.dist-info` 和
`__editable__.llm_mappo-0.1.0.pth`，证明当前源码可按既有 editable 安装合同构建并安装。
直接使用 pip build isolation 的首次探测因沙箱禁止联网获取构建依赖而失败；一次虚拟环境
探测又因 Windows 将该环境判为不可写并尝试写用户目录而失败。两次均未修改项目或现有
`py310` 安装，随后采用上述隔离 prefix 合同成功完成验证。

## 3. 自动验证结果

以下命令均在干净检出的项目根目录运行。嵌套、被忽略的检出目录中，pytest 的默认
`cacheprovider` 在测试完成后的缓存写入阶段无法退出；单测试对照确认禁用该插件后正常退出。
该插件只管理 `.pytest_cache`，不参与测试收集或断言，因此干净检出测试统一显式使用
`-p no:cacheprovider`。

| 验证项 | 原始命令 | 结果 |
| --- | --- | --- |
| 全量测试 | `python -m pytest -p no:cacheprovider -q` | 退出码 0；184 passed，53.87s |
| A* 诊断回归 | `python -m pytest -p no:cacheprovider tests/test_astar_diagnostics.py -q` | 退出码 0；11 passed，5.88s |
| 布局预览回归 | `python -m pytest -p no:cacheprovider tests/test_layout_preview.py -q` | 退出码 0；8 passed，7.35s |
| 配置合同回归 | `python -m pytest -p no:cacheprovider tests/test_formal_results.py::test_g3_manifest_freezes_eight_formal_seeds tests/test_g3_comparisons.py::test_g3_comparison_configs_share_the_frozen_environment_contract -q` | 退出码 0；2 passed，7.28s |
| Flake8 | `python -m flake8 rware llm_mappo eval train scripts figures/core` | 退出码 0 |
| 可视化 CLI | `python visualize.py --help` | 退出码 0 |
| A* 评估 CLI | `python eval/evaluate_dynamic_ingress_astar.py --help` | 退出码 0 |
| YAML 解析 | 对 `configs/**/*.yaml` 逐一执行 `yaml.safe_load` | 退出码 0；30 份配置 |
| diff 检查 | `git diff --check` | 退出码 0 |

配置合同回归保留历史 G3 manifest 的复现检查，但这些配置不再是活动路线合同；当前阶段的唯一
入口仍是 `TASKS.md` 与本 P0 spec。

## 4. Smoke 与静态核验

- 从无 BOM 的四行 ASCII 布局执行
  `python visualize.py --layout-preview <layout> --layout-preview-output <png> --cell-size 8`，
  退出码 `0`，生成 171 字节、`40x32` 的 PNG。对应回归证明 CLI 在预览路径中不会创建环境、
  加载 checkpoint 或运行策略。首次使用 PowerShell `Set-Content -Encoding utf8` 的输入带 BOM，
  因首行宽度不一致被正确拒绝；改用无 BOM 输入后通过。
- `TASKS.md` 检出 11 个且仅 11 个阶段入口：P0、O0–O3、S1–S2、D1、E1–E3。
- 活动文档中的 `8-AGV` 均为明确的禁止/不实施边界，不存在活动压力实验合同。
- 根 `CONSTITUTION.md` 已删除；仍出现该文件名的位置仅用于记录 P0 迁移和删除事实。
- 两个 rejected layout 仅在迁移审计和 P0 删除要求中留下文字证据，不存在文件、复制或归档。
- `artifacts/p0_safety_backup/`、干净检出、隔离安装与 smoke 产物均被 Git 忽略且未进入暂存区。

## 5. 人工边界与结论

- P0-A 迁移矩阵已唯一归类每项初始 dirty 内容，安全备份清单和哈希可读；两个 rejected
  layout 按研究所有者要求永久删除且未备份。
- 归档索引保留旧 Phase 3/4 artifact、配置和原路径的追溯关系。
- A* 诊断回归覆盖诊断关闭时的行为等价、`NOOP/RIGHT` yield 边界、动作流水线计数守恒、
  stall reason 聚合以及未知项显式分类。
- P0 没有改变训练算法、教师语义、奖励、观测、能源参数或正式 seed。
- 未执行训练、多 seed 评估、长回放、O0/O1 重设计、O3 未见拓扑或 S1 历史行为恢复。

P0-E 集成候选通过。P0-F 仍需完成稳定路线工程师交接、最终文档状态提交以及三个本地分支
引用落位；这些事项完成前，不宣称整个 P0 已结束。
