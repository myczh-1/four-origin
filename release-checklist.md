# 发布检查清单：首发已公开

2026-10-20。Copyright 2026 myczh（个人公开笔名）。

- [x] 用户确认代码和Adapter采用Apache-2.0，数据和报告采用CC BY 4.0，允许商用。
- [x] 完整许可证文本及LICENSING.md范围说明已补齐。代码示例、测试和运行配置归Apache；数据/结果集合和报告正文归CC BY；第三方内容保留原许可。CC BY仅覆盖实际拥有的相关权利。
- [x] 保留Qwen3-0.6B固定revision的原始LICENSE及Copyright 2024 Alibaba Cloud；上游该revision无独立NOTICE。新增NOTICE明确为项目自写归属，不冒称上游文件。
- [x] 四份Adapter均有独立LICENSE、NOTICE、MODIFICATIONS.md、licenses/、模型卡、数据说明、加载器及权重许可侧文件；HF模型卡license为apache-2.0，数据卡为cc-by-4.0。
- [x] 文档注明LoRA训练改动及Origin自定义加载，权重本身保持不变。未打包基座或tokenizer。
- [x] 当前文档已清除“许可证未选择”的过期状态。evidence/内历史审计快照保持原样，其当时状态不代表当前许可状态。
- [x] 原始研究与V4.1文件不修改；558个受保护文件哈希未变。此次发布没有训练、微调或新推理。
- [x] 初次打包时干净Python3.14环境安装、pip check、九个数据分割逐字节重建、960条V3已有输出评分核对、9项护栏通过。本次许可更新重新执行标准库检查和许可/归属完整性检查；不声称重跑模型成绩。
- [x] V4评分缺陷与未获人工验证状态持续明确披露，不将旧评分当作正式长期结论。公开指标有130项JSON引用可查。
- [x] 已核验GitHub账号myczh-1和HF账号myczh；GitHub仓库four-origin。
- [x] 首发仅V3 Text和Origin Adapter；数据随GitHub发布，不建立独立数据仓库。
- [x] 首发模型卡和README使用实际仓库URL；以myczh个人署名按所选许可证发布。
- [x] 用户已明确授权按上述首发范围创建仓库、上传及公开。

后续维护见docs/publication-guide.zh-CN.md。隐私扫描只是模式检查，不能证明所有敏感信息均已排除；最终发布清单仍需人工通读。

- [x] GitHub公开main已推送；HF V3 Text和Origin完整文件提交已确认。
