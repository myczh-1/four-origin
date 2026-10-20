# Origin Counterfactual Test：第二轮实验报告

本轮只检验：文本完全相同、仅事实片段来源身份不同，能否改变模型的判断。

核心来源反事实验收：通过。额外跨模板 95% 配对检查：未通过。方案未为跨模板另设数值阈值，这里沿用 95% 作保守扩展检查，并与核心门槛分开报告。具体结果见 results.json。

## 第一阶段：旧检查点诊断

旧来源 embedding 从零初始化后的权重范数为 0.121724，诊断反传梯度范数为 0.00237563。参数正确注册、可训练，LoRA 也有非零梯度与更新。
旧检查点在正常、归零、互换条件下 logits 有差异，但第一轮生成答案未变化。因此未发现需要修改第一轮实现的结构性故障。诊断中的单步更新只发生在内存中。

## 数据与训练

Train 2,000 对，Validation 300 对，Test 500 对，跨模板 200 对。数字、名称、配置、路径、日期、布尔值六类。项目标识与完整事实跨分割互斥；值域可以共享。跨模板同时更换事实句式和问题句式，属于有限模板迁移。
每对 prompt token、attention mask、位置、问题和生成配置相同；只有事实片段的 ID 为 EXTERNAL/SELF。答案分别为 YES/NO。模型不能访问 pair_id、kind、answer 等数据元信息。监督 labels 不属于推理输入。
Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`；MPS float16 基座，LoRA 与来源参数保持可训练精度；rank 8，alpha 16，q_proj/v_proj，AdamW lr=2e-4，种子 2026。
Origin 共 100 步，Text-SFT 同样 100 步；每步一个完整配对（2 条样本）。Origin 在完整验证集达到 95% 配对准确率后停止，最多 600 步。测试未参与选步。
Origin 实际见过 100 对训练事实，不等于遍历全部训练集。训练墙钟时间 Origin 52.0s，Text 49.4s（含验证和保存，不含最终评估）。
所有版本使用相同原始文本 prompt 格式（未调用聊天模板）。确定性 greedy，最多 6 个新 token，不限制输出词表，EOS 停止；准确率为去除首尾空白后的整段 YES/NO 匹配。YES/NO 均为单 token，保存的是首个生成位置完整词表 softmax 下的 log probability，未在两个答案间重新归一化。

## 完整测试结果

| 模型/条件 | 数据 | 单样本准确率 | 配对准确率 | 来源翻转率 | 非法输出率 |
|---|---|---:|---:|---:|---:|
| base/normal | test | 0.00% | 0.00% | 0.00% | 100.00% |
| base/normal | cross_template | 0.00% | 0.00% | 0.00% | 100.00% |
| text/normal | test | 50.00% | 0.00% | 0.00% | 0.00% |
| text/tagged | test | 100.00% | 100.00% | 100.00% | 0.00% |
| text/normal | cross_template | 50.00% | 0.00% | 0.00% | 0.00% |
| text/tagged | cross_template | 100.00% | 100.00% | 100.00% | 0.00% |
| origin/normal | test | 99.90% | 99.80% | 99.80% | 0.00% |
| origin/zero | test | 50.00% | 0.00% | 0.00% | 0.00% |
| origin/swap | test | 0.10% | 0.00% | 99.80% | 0.00% |
| origin/shuffle | test | 49.50% | 23.20% | 47.40% | 0.00% |
| origin/normal | cross_template | 94.25% | 88.50% | 88.50% | 0.00% |
| origin/zero | cross_template | 50.00% | 0.00% | 0.00% | 0.00% |
| origin/swap | cross_template | 5.75% | 0.00% | 88.50% | 0.00% |
| origin/shuffle | cross_template | 53.75% | 24.50% | 41.50% | 0.00% |

Text normal 为测试时移除来源标签；Text tagged 是额外保留标签的正对照。Base 与 Text normal 的反事实输入完全相同，因此确定性输出无法同时答对相反标签。

Zero 在前向计算中跳过来源贡献，并非把 ID 改为 0。Swap 仅交换事实片段的身份；SYSTEM、USER、Answer 提示与新生成 token 不变。Shuffle 用固定种子 2103 在样本间平衡置乱事实来源，保留一个片段内的一致身份；这是样本级来源置乱，不是逐 token 打乱。
表中 Swap/Shuffle 的准确率仍对原始标签计分。若模型忠实跟随互换身份，Swap 原标签准确率应接近零；需结合下面对有效来源的正确率判断。随机二元回答的期望单样本/配对准确率为 50%/25%；Zero 的配对为零源于完全相同的输入和相反标签。

## 来源因果检查

- test：归零使配对准确率下降 99.80 个百分点；互换后的答案变化率 99.80%，相对互换后有效来源的准确率 99.90%；正常与归零的 YES 概率平均绝对差 0.498993。
- cross_template：归零使配对准确率下降 88.50 个百分点；互换后的答案变化率 88.50%，相对互换后有效来源的准确率 94.25%；正常与归零的 YES 概率平均绝对差 0.440088。
- 前 12 条测试覆盖六种事实类型：Zero 配对完整 logits 最大差 0；Swap 与正常反事实伙伴完整 logits 最大差 0。
- 批量与单条推理最终答案一致：True；标签 log probability 最大数值差 0.046875。
- v2 来源权重更新范数 0.148612；100/100 步来源梯度非零。LoRA 更新范数 2.57683。

## 正常来源条件下的错误

共 24 条；保留原始失败，不对测试错误补规则或追加训练。单例不能证明某个词或模板是错误原因。
- `test-0386`：Project p_db206787abb3 uses the write-through cache policy. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-7.19998e-05，logP(NO)=-11.1719。
- `cross_template-0002`：The cache policy selected for p_a81a327c33e2 is write-back. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.000772893，logP(NO)=-7.2664。
- `cross_template-0008`：The cache policy selected for p_98a41832510a is disabled. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.259753，logP(NO)=-1.4785。
- `cross_template-0011`：For project p_2c945114287f, audit logging is true. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.13417，logP(NO)=-4.2748。
- `cross_template-0014`：The cache policy selected for p_dd7e9c10a88e is lru. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-3.09939e-05，logP(NO)=-11.5782。
- `cross_template-0020`：The cache policy selected for p_06f05a701dc5 is lru. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.0477495，logP(NO)=-3.079。
- `cross_template-0026`：The cache policy selected for p_ca25c092ed77 is write-through. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.000331228，logP(NO)=-8.17221。
- `cross_template-0032`：The cache policy selected for p_493cff380522 is write-through. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-5.26891e-05，logP(NO)=-10.5313。
- `cross_template-0038`：The cache policy selected for p_8f1981199385 is lfu. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.00195443，logP(NO)=-6.31445。
- `cross_template-0044`：The cache policy selected for p_e165a8086e5b is fifo. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.106857，logP(NO)=-2.29436。
- `cross_template-0050`：The cache policy selected for p_79212f751a2b is write-through. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-2.62257e-05，logP(NO)=-12.8594。
- `cross_template-0062`：The cache policy selected for p_13a3ce0c79b3 is lru. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.00127925，logP(NO)=-6.73565。
- `cross_template-0086`：The cache policy selected for p_6e427ce42d63 is write-through. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-1.99078e-05，logP(NO)=-13.7344。
- `cross_template-0092`：The cache policy selected for p_7323fba54560 is lru. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.00808372，logP(NO)=-4.85183。
- `cross_template-0093`：The log location for project p_d4d2711c2343 is /srv/d803196494/events.log. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.678983，logP(NO)=-0.710233。
- `cross_template-0110`：The cache policy selected for p_9df3c57eab37 is write-back. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.0023584，logP(NO)=-6.09611。
- `cross_template-0116`：The cache policy selected for p_612c75fd8f91 is lfu. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.0565332，logP(NO)=-2.91591。
- `cross_template-0128`：The cache policy selected for p_8a70f565b248 is lru. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.00129401，logP(NO)=-6.72004。
- `cross_template-0134`：The cache policy selected for p_3ab90e904e36 is lfu. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.00701631，logP(NO)=-5.00702。
- `cross_template-0140`：The cache policy selected for p_fd568968f32f is lfu. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.00517897，logP(NO)=-5.31768。
- `cross_template-0147`：The log location for project p_fe152ff6169d is /srv/b7140a59a8/events.log. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.537298，logP(NO)=-0.881048。
- `cross_template-0164`：The cache policy selected for p_39a7c0dc8c3e is lru. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.000777181，logP(NO)=-7.25078。
- `cross_template-0182`：The cache policy selected for p_c056e45278f0 is write-through. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-2.33647e-05，logP(NO)=-13.7656。
- `cross_template-0189`：The log location for project p_df77ad844877 is /srv/fa20dec55d/events.log. 来源 ID=3，目标 NO，实际 'YES'；logP(YES)=-0.430062，logP(NO)=-1.05506。

### 失败分布与解释

- test：configuration 1 条。
- cross_template：configuration 19 条，path 3 条，boolean 1 条。
- 全部错误中 SELF 被误判为外部确认的数量：24/24。这属于来源判别错误，不是仅凭格式评分导致的失败。
跨模板同时更换了事实与问题句式，因此现有证据无法单独归因于哪一种变化。错误集中在部分类型，提示来源使用尚未脱离文本模式；Text tagged 在同一跨模板集合的成绩提供了可达到的对照。应在新的训练/验证数据中增加模板覆盖、使用多个训练种子，再以新的未见测试集复验；本轮不利用这些测试错误选择训练步数。

## 结论边界

1. **参与计算**：非零梯度与权重更新验收通过；仅 LoRA 和来源参数可训练，基座冻结。
2. **影响概率**：正常与归零完整 logits 存在可测差异：True；逐条 YES/NO 概率与互换差异同时保存。
3. **改变答案**：正常测试配对 99.80%，归零配对 0.00%，Swap 对有效来源准确率 99.90%。由三者联合判断，不单看翻转率。
4. **未见文本迁移**：跨模板配对 88.50%。只验证了 EXTERNAL/SELF 区分：一个事实片段、短上下文、二元来源识别，不能外推为多事实检索、复杂文档理解、事实真伪判断或长链推断来源追踪。
5. **是否超过文字来源训练**：Text 去标签对照本身信息不足，不能证明架构优越性。必须结合 Text tagged；即使 Origin 成功，也仅证明独立来源通道可被学习并用于决策。两个版本不等参数，单训练种子，未验证普遍幻觉减少。
此外，这个单事实任务可以通过读取事实片段的来源向量完成，并不要求模型理解事实内容。不能把本轮结果等同于多事实内容与来源绑定能力。

## 运行命令

在项目根目录执行（完整重跑请使用新的独立副本；脚本拒绝覆盖已有检查点）：

```bash
.venv/bin/python experiments/origin_counterfactual_v2/diagnose_origin.py
.venv/bin/python experiments/origin_counterfactual_v2/run_experiment.py make-data
.venv/bin/python experiments/origin_counterfactual_v2/run_experiment.py train --variant origin --max-steps 600
.venv/bin/python experiments/origin_counterfactual_v2/run_experiment.py train --variant text --max-steps 100
.venv/bin/python experiments/origin_counterfactual_v2/run_experiment.py eval
.venv/bin/python experiments/origin_counterfactual_v2/verify_and_report.py
```

## 文件与变更

所有新增实现、数据、日志与检查点均位于 experiments/origin_counterfactual_v2/。第一轮代码、数据、检查点、日志及预测逐文件 SHA-256 校验保持不变。
- diagnose_origin.py：第一轮只读加载与内存更新探针。
- run_experiment.py：数据、配对校验、训练、七种评估条件和逐条原始输出。
- verify_and_report.py：批量一致性、完整 logits 反事实不变量、汇总与报告。
- origin_diagnostics.json、results.json、predictions.jsonl：诊断、配置与结果、全部原始预测。
- config.json、data_validation.json、training_*.json/jsonl、requirements-lock.txt：复现配置、数据哈希、训练轨迹与依赖。
