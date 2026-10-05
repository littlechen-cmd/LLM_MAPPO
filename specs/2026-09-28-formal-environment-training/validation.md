# 验证 — 正式环境训练配置

## 已完成的本阶段验收

- [x] 三场景站点、任务池、布局身份、Actor/FP Critic维度与确定性任务循环回归。
- [x] 12模板预算与日程一致；正常rollout1024步，末批按实际累计transition计数。
- [x] provisional诊断与approved正式入口隔离；跨配置checkpoint身份严格校验。
- [x] 正式配置准备器输出48份baseline/A*与48项LLM依赖；真实表示导入需SHA/identity。
- [x] 8份配对离线预训练配置生成接口，参数与工作簿一致。
- [x] tiny实际50000步/49更新与formal4096步/4更新证据；medium按指令不继续短测。
- [x] CSV布尔汇总修复；tiny训练/评估产物未重跑，原failed状态和恢复哈希均保留。
- [x] 5000步耐久入口保持原checkpoint身份，并记录1000步窗口和真实早终止。
- [x] 96运行、8训练seed、200唯一主实例、10耐久实例、总100800000预算的工作簿校验。
- [x] 中文XLSX、初版说明、实验合同与已知LLM依赖记录已交付；没有伪称四组实测收敛。

## 已有工程核验

工程师完整回归529 passed / 2 skipped，206.49秒；跳过原因为本地CUDA不可用。
耐久分段追加40项聚焦通过，含正式环境真实NOOP策略180步死锁、不重启、边界差分。
本次交接重新运行下列检查，最终实际计数与输出写入交接记录：

```powershell
& D:/Anaconda3/envs/py310/python.exe -m pytest tests/test_formal_training_contract.py tests/test_formal_training_preparation.py tests/test_formal_durability_evaluation.py tests/test_mappo_fp_runner.py -q
& D:/Anaconda3/envs/py310/python.exe -m flake8 llm_mappo/formal_training.py llm_mappo/mappo_baseline/runner.py scripts/evaluate_mappo_fp_baseline.py scripts/prepare_formal_training_configs.py scripts/build_formal_training_workbook.py
& D:/Anaconda3/envs/py310/python.exe -m scripts.build_formal_training_workbook
```

## 后续运行门（不是本阶段已完成结果）

- LLM标签须完成真实pilot/盲审/6000主标签及质量门，当前响应验证失败需修复。
- 8个真实表示权重须按实际SHA/identity导入；测试用数值产物不能充当论文训练产物。
- 正式96次RL、19200主评估回合与320耐久回合由负责人运行；本阶段没有执行。
- 本地CPU短测与回归不代表服务器CUDA验证、预算充分性、安全或lifelong性能通过。
- 12/16AGV容量设计另立合同；不读取或更新roadmap.md，不推送或合并。

2026-10-05最终交接复核：四个聚焦测试文件49项通过，110.33秒；变更代码静态检查
与差异格式检查通过。工作簿ZIP完整性、12组YAML对齐、96运行与200唯一实例核验通过。

最后协议分离核查发现formal主评估误附耐久字段，新增真实环境回归先失败后修复。
修复后的耐久/准备器11项通过（44.98秒）；正式配置统一UTF-8/LF。
