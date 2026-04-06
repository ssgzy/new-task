# 过程总结_orca_下一步决策

## 结论先行
- Orca 当前剩余问题更像**局部样本上的输出风格不收束**，不是 parser 不兼容，也不像统一 `ChatML wrapper / LlamaTokenizer` 损坏。
- 现阶段更建议**先把 Orca 保留为“可比较但有残余风险”的正式候选**，不要立刻恢复 canonical 主线。
- 如果一定要继续做一个最小 provisional 实验，**优先只补测 `max_new_tokens=384`**，不优先测 `320`。

## 依据
- 固定样本 v3 smoke 已恢复为 `Answer: 1.33%` 且 `valid_parse=True`。
- Orca 单模型 provisional `val_calib50` 已完成，三档结果如下：
  - `128`：`truncation_without_answer_rate=0.18`
  - `192`：`truncation_without_answer_rate=0.10`
  - `256`：`truncation_without_answer_rate=0.10`
- `256` 档剩余 5 条截断样本满足：
  - 5/5 都是 `prompt_wrapper_family=chatml`
  - 5/5 都是 `prompt_render_mode_effective=chatml`
  - 5/5 都是 `effective_tokenizer_class_name=LlamaTokenizer`
  - 5/5 输出都以 `To calculate...` / `To answer this question...` 开头，先进入长推理文本
  - 5/5 吃满 `256` token 后仍没有 `Answer:` 行，因此 `parse_error_reason=missing_answer_line`
- 非截断组 `45` 条里有 `40` 条直接以 `Answer:` 开头，同一 wrapper/tokenizer 配置并没有整体失效。
- 截断组 `prompt_length_tokens min/mean/max=958/1416.0/1633`，非截断组是 `543/1129.1/1736`，非截断组最大 prompt 反而更长，所以这 5 条不是单纯“输入太长导致输出预算不足”。

## 四类候选解释的判定
- `token 长度不够`：不是主因。因为 `192 -> 256` 截断率没有继续下降，且非截断组也能处理更长 prompt。
- `局部样本上的输出风格不收束`：**最符合现有证据**。5 条失败样本都先进入自然语言分步推理，没有及时收束到最终答案行。
- `wrapper / tokenizer 仍不稳定`：不是主要解释。因为 50 条里多数样本在同一 `chatml + LlamaTokenizer` 路径下能直接产出 `Answer:`。
- `parser 不兼容`：不是当前问题。parser 只是如实报告“没有最后答案行”，不应在本阶段为 Orca 单独改 parser。

## 是否值得补测 320 / 384
- 值得，但只能在 `outputs/provisional/` 路径下做，不得覆盖 canonical。
- 优先测 `384`，理由：
  - `192 -> 256` 已经没有降低截断率，说明如果继续验证“更长输出预算是否能接住少数长推理样本”，步长应该拉大一点，`384` 比 `320` 更有判别力。
  - 如果 `384` 仍明显高于 2% 截断口径，就可以更有把握地判定“问题主要不是 token 上限，而是模型输出风格不收束”，避免再继续小步加档浪费时间。

## 当前决策
- Orca 保留为当前正式 instruction 候选之一，但标注“可比较但有残余风险”。
- 暂时不恢复 canonical `val_calib50` / `val_screen200`。
- 若下一轮只做一个 Orca 最小动作，则在 provisional 层补 `384`，而不是先改 parser、先改 CoT、或先动 screening 门槛。

## 关联文档
- [[当前状态]]
- [[任务笔记 - FinQA 基准重建]]
- [[过程总结_协议可用性筛查]]
- [[过程总结_主表扩展与新增模型筛查]]
- [[异常与决策记录]]
- [[下一步]]
