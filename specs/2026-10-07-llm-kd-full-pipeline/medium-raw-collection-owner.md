# medium 硬充电原始数据：所有者运行说明

日期：2026-10-08。使用 `core-integration` 工作树，在当前电脑运行。
本命令完成已经启动的正式采集，不运行训练，也不调用 API。

## 当前产物

运行目录：
`D:\codeProject\llm-a-mappo\.worktrees\core-integration\artifacts\llm_kd\phase_20261007\dataset\goal-full-v3-bf8fad003f9033f8`

首个完整 episode 已成功保存，seed=10000，800 步，数组形状 `[5,800,209]`，
500 个非重叠候选窗口。它计入下述 120 个 episode，恢复会跳过，不能重新创建目录。
当前仍未冻结 3,000 条选样，未产生新标签或 API 费用。

首条轨迹的规则参考诊断：普通窗口 462、结构受限 33、停滞正分 5；没有检出
竞争或目标站充电服务。这些是原始候选诊断，不是 LLM 标签或质量结论。
完整采集后统一检查覆盖；不能因首条缺失就断言整套来源不可用。

## 一次运行：补齐采集，再检查选样资格

在 PowerShell 粘贴以下整段。失败会停止；保留所有输出，再将结果交回本对话。
长采集按已批准分工由所有者运行。

```powershell
Set-Location 'D:\codeProject\llm-a-mappo\.worktrees\core-integration'
& D:/Anaconda3/envs/py310/python.exe scripts/run_goal_full_llm_dataset.py collect `
  --source-config artifacts/llm_kd/dataset/medium-5ag/source/medium-5ag-baseline-seed_0007.yaml `
  --source-checkpoint artifacts/reward_diagnostics/group2/medium5_c1_v3/baseline/seed_0007_20260928T133735Z/checkpoint_final.pt `
  --actor-artifact artifacts/llm_kd/dataset/medium-5ag/source/collection_actor_final.pt `
  --actor-sha256 c6d2df435f63094acd579d344dfb65488d20636102ef84fb9736e5c606ac0ffb `
  --rollout-config configs/llm_kd/phase_20261007/medium-5ag-baseline-goal-full.yaml `
  --output-root artifacts/llm_kd/phase_20261007/dataset `
  --episode-count 120 --resume
if ($LASTEXITCODE -ne 0) { throw 'Raw collection failed; preserve evidence and report.' }
& D:/Anaconda3/envs/py310/python.exe scripts/run_goal_full_llm_dataset.py select `
  --run-directory artifacts/llm_kd/phase_20261007/dataset/goal-full-v3-bf8fad003f9033f8
if ($LASTEXITCODE -ne 0) { throw 'Selection failed; preserve evidence and report.' }
```

代码、配置、来源文件必须保持与首条采集一致。恢复若报告身份变化，不改写旧身份，
不删除产物重试；先回到本对话检查。20 份相关实现的原始字节快照已保存在运行目录
`implementation_snapshot`；其中包含当前实际 runner 的本地改动，不能仅凭 Git 提交
重建这些字节。

## 返回结果与后续处理

请提供上述命令输出及运行目录。重点核验 `collection_state.json`、`manifest.json`、
`state.json`、`selection_manifest.json`、`selection_frozen.json` 和
`label_selection_lists.json`。覆盖统计和缺口位于 manifest 与 selection_manifest。
选样命令退出码 0 只表示报告成功写出；`insufficient_coverage` 仍为不合格，
只有 eligible=true 且已生成 `selection_frozen.json` 才能准备标注。

只有完整采集、2,100/450/450 选样及原始覆盖门通过后，才准备真实 120 条试标和
同 40 条复标/Pro 对照；通过语义质量与 60 条匿名审阅后才继续主集。
原始覆盖不足会留下缺口报告；后续由所有者选择新的硬充电 baseline Actor，或批准
独立标记的场景补充。不能自动混入 A* 行为、人工摆放场景或降低语义质量门。

本说明不启动 tiny/formal，不决定 formal AGV 数，也不代表数据 Gate 已通过。
