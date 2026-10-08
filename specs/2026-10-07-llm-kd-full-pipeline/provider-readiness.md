# v3 标注服务核对

核对日期：2026-10-08。当前仅完成公开资料核对，未发送新仿真样本或付费请求。

## 模型与费率

依据 [DeepSeek 官方价格页](https://api-docs.deepseek.com/quick_start/pricing/)，
当前请求名仍为 `deepseek-flash` 和 `deepseek-v4-pro`，官方列出的版本分别为
DeepSeek-V4.1-Flash、DeepSeek-V4-Pro-0813。每条实际响应仍须记录 returned model，
不能仅凭请求别名宣称服务版本一致。

预算按峰时、输入缓存未命中预留，不依赖缓存折扣或请求时段：

| 模型 | 输入 USD/百万 tokens | 输出 USD/百万 tokens |
| --- | ---: | ---: |
| Flash | 0.30 | 1.20 |
| Pro | 1.32 | 3.96 |

按 [官方 thinking mode 文档](https://api-docs.deepseek.com/guides/thinking_mode/)，
Chat Completions 可用 `thinking.type=disabled` 显式关闭 thinking；本合同延续该设置，
避免默认开启的额外输出影响费用预留和结果可比性。JSON 输出能力已在官方价格页确认。

## 真实请求前的依赖

- medium 完整原始轨迹与候选覆盖检查通过，冻结 3,000 条窗口和 120/40/60 名单。
- 保存最终 prompt、原始 view 哈希、服务参数及上述费率证据；首次真实请求即试标。
- 依据实际各条载荷的输入上限和输出 cap 做逐请求费用预留，失败、重试和未知计费结局
  均占用账本；新 phase 累计上限 100 USD，历史费用单列。
- 现有外发授权不自动扩展至 tiny/formal；在各自实际载荷与模型冻结后处理适用授权。
- 本次公开资料核对不能证明实际模型可调用、标签质量合格或 API 已完成真实验收。

离线交接核对中的一个真实 medium train view 生成 22,339 字符的 user prompt，
模拟响应的 0.37/0.63 经解析、合并和训练加载保持原值。该单条字符数不作为完整数据集
的 token 估计或预算保证；第四辅助维没有新增 API 请求。
