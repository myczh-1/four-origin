# 4-Origin 发布与后续维护

2026-10-20 首次公开：

- GitHub：https://github.com/myczh-1/four-origin
- Text Adapter：https://huggingface.co/myczh/four-origin-qwen3-0.6b-v3-text
- Origin Adapter：https://huggingface.co/myczh/four-origin-qwen3-0.6b-v3-origin

个人署名myczh。代码和Adapter采用Apache-2.0，数据和报告采用CC BY 4.0，允许商用。首发包括V2/V3实验代码、数据、原始结果与V3两份Adapter；未发布V2 Adapter或基座权重，没有独立HF数据仓库。

## 后续维护

保留每次发布的提交版本与文件哈希。后续修改应先在独立发布目录检查，再推送到对应仓库。不要覆盖原始实验记录。版本标签和GitHub Release尚未创建；可以在选定版本号后补充。模型加载与生成复验没有在发布期间运行，下载哈希核对不代表重新验证实验成绩。

## 公开时的表述

说明“显式来源身份在受控小模型任务中可影响决策与短回答”，保留Text-SFT对照和跨模板失败。明确V4评分缺陷、V4.1没有人工验证、长期净收益未证实。不要称为已解决幻觉或生产抗幻觉模型。

许可证允许商业使用不等于模型可靠，也不强制你以后提供维护支持。保留每个已发布版本的哈希与许可快照，便于引用和复现。

## 官方参考

- GitHub创建仓库：https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- Hugging Face模型卡：https://huggingface.co/docs/hub/model-cards
- CC BY 4.0署名允许使用指定笔名（3(a)(1)(A)(i)）：https://creativecommons.org/licenses/by/4.0/legalcode
