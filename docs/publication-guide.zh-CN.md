# 4-Origin 后续发布流程

当前：个人署名myczh，代码与Adapter Apache-2.0，数据与报告CC BY 4.0，允许商用。已完成本地文件准备，未创建远程仓库、未上传。网名作为公开笔名使用；它不等同于GitHub/Hugging Face账号，不能据此猜测账号或URL。

## 你需要决定和处理的事项

1. 提供GitHub和Hugging Face实际账号/组织名。登录和双因素验证由你完成；不要把Token、密码或恢复码贴进聊天。
2. 确认仓库名称。建议GitHub为four-origin；HF主发布两份：four-origin-qwen3-0.6b-v3-text、four-origin-qwen3-0.6b-v3-origin。这里只是建议，不代表名称可用或已经创建。
3. 确认首发范围：建议先发布V2/V3代码与结果、V3两份Adapter；V2两份Adapter可随后补充。合成数据已在GitHub，不必首发就建独立HF数据仓库。
4. 确认你有权以个人名义许可这些材料；若有其他贡献者或雇主权利，应先处理相关授权。
5. 检查最终文件清单和项目介绍，再明确下达发布指令。此前所有工作停留本地。

## 实际执行顺序

1. 根据你确认的真实账号/名称，在本地替换所有占位链接，更新GitHub和HF模型卡双向引用。重新检查许可证范围、权重哈希、隐私和指标出处。
2. 在你允许远程操作后，创建GitHub仓库。上传release/four-origin/内的内容作为仓库根目录，不上传原始工作目录、release顶层构建脚本、虚拟环境、缓存或私人日志。不要让网站模板覆盖已有LICENSE；混合许可范围由LICENSING.md说明。
3. 分别创建两个HF model仓库。v3-text/和v3-origin/各自目录内的内容放到对应仓库根目录，保留lora/子目录以及Origin的origin_embedding.pt。LICENSE、NOTICE、MODIFICATIONS.md、licenses/、加载器和权重.license文件一起上传。
4. 推荐先在私有仓库核对文件及页面显示（私有上传同样需要你的明确授权），再按批准范围切换公开。核对license元数据、base_model、链接、文件哈希和下载目录结构。
5. 确认公开后再为GitHub建立v0.1.0等版本标签和Release，记录对应HF提交版本与清单。版本号待确认，尚未创建。
6. 下载检查可只校验文件及哈希。端到端模型加载或生成验证属于新增模型执行，当前没有执行；如果希望补做，需要另行授权，并与原实验成绩区分。

## 公开时的表述

说明“显式来源身份在受控小模型任务中可影响决策与短回答”，保留Text-SFT对照和跨模板失败。明确V4评分缺陷、V4.1没有人工验证、长期净收益未证实。不要称为已解决幻觉或生产抗幻觉模型。

许可证允许商业使用不等于模型可靠，也不强制你以后提供维护支持。保留每个已发布版本的哈希与许可快照，便于引用和复现。

## 官方参考

- GitHub创建仓库：https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- Hugging Face模型卡：https://huggingface.co/docs/hub/model-cards
- CC BY 4.0署名允许使用指定笔名（3(a)(1)(A)(i)）：https://creativecommons.org/licenses/by/4.0/legalcode
