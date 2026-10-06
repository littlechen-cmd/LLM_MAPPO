# 当前工作树任务记录

本文件仅记录 `codex/formal-environment-training-spec` 的本轮配置交付；
不替代core-integration中负责人维护的其他阶段任务与Gate记录。

- [x] FT-1：核验并接入tiny/medium/formal场景与维度、站点和任务合同。
- [x] FT-2：确定12组初版配置、预算、seed、LLM数据/预训练与正式配置准备入口。
- [x] FT-3：完成tiny50k和formal4096短测，交付中文XLSX、启动命令与验证材料。

上述完成指本阶段配置交付，不表示LLM标签/表示、四组长训练或论文性能Gate已经通过。
LLM标签当前pilot失败待数据工程修复；long runs由负责人按工作簿执行。

- [x] MB-PREFLIGHT：完成medium baseline正式训练最终审查，准备900k×8seed与200预留评估实例的独立配置和串行启动器；88项回归及4096步CPU短测通过。GPU长训练与性能结论由负责人后续运行。

- [x] MB-IMPORT：修复服务器ROS同名scripts包遮蔽，三个真实模块入口先失败后通过；全量556项通过、2项因无CUDA跳过。提供从c988157更新的增量Git包，服务器CPU验收由负责人执行。

- [x] AK-PREFLIGHT：审查 medium A* KD 正式配置并实现独立 600k 蒸馏/300k 纯 PPO 尾段，保持 baseline 900k 学习率合同；132 项回归、26 项最终专项复测及 16 worker/2048 步 CPU 短测通过。已准备 seed 7 优先、审查后才开放其余 7 seed 的启动器；服务器 CUDA 验收及长训练仍待负责人运行。

- [x] AK-TAIL：准备四组 seed 7/900k 截断诊断（450k/600k/750k/900k），共用 λ 曲线、四 learner 并行、独立诊断 schema 与 9400–9409 评估；200 项回归、9 项最终分析器复测及 16 worker/2048 步短测通过，交付同步/启动/分析命令。长诊断、结果审查与正式截断选择尚未完成。

- [x] NE-DIAG：准备无能源约束medium-5ag的baseline/A* KD两组seed7、900k并行诊断，A*采用450k硬截断与450k纯PPO尾段；完成672项全量回归、46项最终专项复测和两组各16worker/2048步CPU短测，交付独立配置、启动/分析入口与同步包。服务器CUDA及900k长诊断待负责人运行；正式多seed保持暂停，不能以此宣称性能改善。
