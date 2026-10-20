# 4-Origin：显式运行时来源身份

[English](README.md) · [技术报告](technical-report.md) · [指标溯源](metric-ledger.json)

本项目分享 Qwen3-0.6B 的受控实验、原始失败和阶段性发现，不是生产级抗幻觉模型。来源身份能影响短任务决策与回答；普通文字标签同样有效，尚无额外架构优势证据。代码与Adapter采用Apache-2.0，数据与报告采用CC BY 4.0，允许商用。个人公开署名为myczh，仓库链接见下方。

## 方法

SYSTEM=0（系统任务）、USER=1（用户请求）、EXTERNAL=2（外部参考或工具内容）、SELF=3（模型输出或构造的内部候选）。身份由运行时分配，不由模型判断；EXTERNAL不保证真实性，SELF复述外部事实也可有依据。主要实验只操纵 EXTERNAL/SELF，不能称为完整验证四种身份。

Origin 将四行可学习向量加到 token embedding；基座冻结，同时训练 q_proj/v_proj 的 rank=8、alpha=16 LoRA。Text-SFT 使用文字来源标签；没有 Attention Bias。基础版本固定为 `c1899de289a04d12100db370d81485cdf75e47ca`，使用实验自定义纯文本输入，不使用聊天模板。

## 第二轮：相同文本的来源反事实

每对仅改变事实片段的身份，目标分别 YES/NO。训练池2000对，实际训练100对；测试500对，跨模板200对，单训练种子。

| Condition | Split | Sample accuracy | Pair accuracy | Source flip |
|---|---|---:|---:|---:|
| text/tagged | test | 100.00% | 100.00% | 100.00% |
| text/tagged | cross_template | 100.00% | 100.00% | 100.00% |
| origin/normal | test | 99.90% | 99.80% | 99.80% |
| origin/zero | test | 50.00% | 0.00% | 0.00% |
| origin/swap | test | 0.10% | 0.00% | 99.80% |
| origin/shuffle | test | 49.50% | 23.20% | 47.40% |
| origin/normal | cross_template | 94.25% | 88.50% | 88.50% |
| origin/zero | cross_template | 50.00% | 0.00% | 0.00% |
| origin/swap | cross_template | 5.75% | 0.00% | 88.50% |
| origin/shuffle | cross_template | 53.75% | 24.50% | 41.50% |

归零跳过来源贡献；互换只变事实片段；置乱在样本间平衡置乱身份，不逐token打乱。第二轮互换/置乱按原标签评分，因此忠实跟随互换身份会得到接近零的原标签正确率。Text/tagged才是保留来源信息的对照。

跨模板88.50%不是稳定泛化。事后四格诊断 F0/Q0、F1/Q0、F0/Q1、F1/Q1 分别100%、100%、93%、88.50%；配置类33/33、33/33、21/33、14/33。主要触发因素是问法变化，与事实句式存在交互，不能单独归因于被动语态或词长。[原诊断](evidence/origin_counterfactual_v2/template_diagnosis/diagnosis-report.txt)。

## 第三轮：单轮混合生成

两组从相同基座重新训练，各200步/400条；每题两个给定陈述，测试和跨模板各96条。SELF候选分母96，确定外部事实分母72，要求保留推断的分母48。这里衡量给定候选的保留与限定，不是自主推理能力。

| Condition | Split | SELF candidate asserted | External retained | Requested candidate retained | Complete content + attribution |
|---|---|---:|---:|---:|---:|
| text/normal | test | 0.00% | 100.00% | 100.00% | 100.00% |
| text/normal | cross_template | 0.00% | 98.61% | 97.92% | 98.96% |
| origin/normal | test | 0.00% | 100.00% | 100.00% | 100.00% |
| origin/zero | test | 41.67% | 55.56% | 50.00% | 50.00% |
| origin/swap | test | 0.00% | 100.00% | 100.00% | 100.00% |
| origin/normal | cross_template | 0.00% | 97.22% | 97.92% | 93.75% |
| origin/zero | cross_template | 48.96% | 65.28% | 50.00% | 48.96% |
| origin/swap | cross_template | 0.00% | 97.22% | 97.92% | 93.75% |

第三轮互换按交换后有效来源评分，与第二轮口径不同；第三轮没有随机置乱条件。正常→互换改变全部Origin回答，并等于反事实伙伴；归零使配对输出相同。

普通测试两组均96/96完整正确；跨模板Text95/96、Origin90/96。Origin仍有路径/标识符内容错误，“未检测到原SELF候选冒充事实”不等于零幻觉。Base所有回答达到长度上限，不能据其低解析覆盖给出可靠的总体幻觉改善幅度。[第二轮原报告](evidence/origin_counterfactual_v2/experiment-report.md)和[第三轮原报告](evidence/origin_generation_v3/experiment-report.txt)保留所有结果；机器可读指标路径见ledger。

## 第四轮与公开边界

16600条输出的旧评分存在hypothesis漏识别、and that漏拆分和不确定性作用域错误。V4.1的250条标注仍未人工核验，正式重评分0条。长期错误趋势、来源恢复净收益、通用抗幻觉能力都尚未证实。[详细披露](docs/long-horizon.md)。

## 运行

在新副本使用 Python 3.10+（实际验证环境见release-checklist.md）；原实现支持CPU/MPS，不自动使用CUDA。

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/reproduce.sh check
bash scripts/reproduce.sh smoke
```

check不需要模型依赖；smoke只加载tokenizer、生成合成数据并核对已有评分，没有模型推理。首次可能下载固定版本tokenizer。训练/推理命令仅供以后手动执行，本次未运行：依次data、train-v2、eval-v2、transfer、train-v3、eval-v3，均通过 `bash scripts/reproduce.sh 命令` 调用。V2 Text跟随Origin实际步数；V3固定200步。新运行输出与evidence原始快照分离。

Origin需要自定义来源向量加载与逐token身份，不能仅用普通PEFT加载即得到完整模型。加载方式见Hugging Face材料及英文README。本次首发包含GitHub实验材料与V3两份Adapter。数据均为生成器构造；/srv路径是合成事实内容，并非个人路径。

GitHub: `https://github.com/myczh-1/four-origin`  
HF Text: `https://huggingface.co/myczh/four-origin-qwen3-0.6b-v3-text`  
HF Origin: `https://huggingface.co/myczh/four-origin-qwen3-0.6b-v3-origin`
