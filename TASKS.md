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
