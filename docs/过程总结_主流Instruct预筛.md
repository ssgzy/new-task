# 过程总结_主流Instruct预筛

## 目的
- 在不改变 `FinQA direct-answer` 主任务、不改 parser v1、不引入 CoT 的前提下，对新增主流 instruct/chat 模型做一轮 1 条 smoke 预筛。
- 这一轮只回答“哪些模型先进入可用输入协议、哪些模型卡在官方下载/官方入口/序列化不兼容”，不恢复 canonical 主线。

## 本轮模型
- `Mistral-7B-Instruct-v0.3`
- `Qwen2.5-7B-Instruct`
- `Llama-3.1-8B-Instruct`
- `Gemma-7B-IT`
- `Yi-1.5-6B-Chat`
- `ChatGLM3-6B`

## 输出约束
- 所有新输出只写入 `outputs/debug/` 或 `outputs/provisional/`
- 不覆盖现有 canonical `model_registry.json/csv`
- 不覆盖旧 `expansion` smoke 结果

## 当前进度
- 已确认官方 gated repo 失败：
  - `Llama-3.1-8B-Instruct`
  - `Gemma-7B-IT`
- 已完成首个新增主流模型的 smoke：
  - `Mistral-7B-Instruct-v0.3`
  - 结果是 `Answer: True / valid_parse: False / truncation_suspect: False`
  - 当前失败原因不是没有答案行，而是 `Answer:` 行里带了完整算式：`Answer: 100 * (5.99 - 3.24) / 3.24 = 86.47%`
- 已完成第二个新增主流模型的 smoke：
  - `Qwen2.5-7B-Instruct`
  - 结果是 `Answer: True / valid_parse: True / truncation_suspect: False`
  - `effective_wrapper=chat_template`
  - `effective_tokenizer_class_name=Qwen2Tokenizer`
  - 当前可记为“已进入有效输入协议”
- 已完成第三个新增主流模型的 smoke：
  - `Yi-1.5-6B-Chat`
  - 结果是 `Answer: True / valid_parse: True / truncation_suspect: False`
  - `effective_wrapper=chat_template`
  - `effective_tokenizer_class_name=TokenizersBackend`
  - 当前可记为“已进入有效输入协议”
- 已完成第四个新增主流模型的 smoke：
  - `ChatGLM3-6B`
  - 结果是 `Answer: True / valid_parse: True / truncation_suspect: False`
  - `effective_wrapper=chat_template`
  - `effective_tokenizer_class_name=ChatGLMTokenizer`
  - 实际输出：`Answer: 17.8`
  - 当前可记为“已进入有效输入协议”
- 当前已尝试模型总表已写出：
  - `outputs/debug/protocol_screening_registry/all_attempted_models_summary.json`
  - `outputs/debug/protocol_screening_registry/all_attempted_models_summary.md`
- 当前已进入有效输入协议的总模型数为 `5`：
  - `Orca-2-7B`
  - `Zephyr-7B-beta`
  - `Qwen2.5-7B-Instruct`
  - `Yi-1.5-6B-Chat`
  - `ChatGLM3-6B`

## ChatGLM3 当前状态
- `ChatGLM3-6B` 已完成官方下载，registry 已写出：
  - `outputs/metadata/provisional/model_registry.labels-chatglm3-6b.json`
- 旧的 `load_failed` 结论已被修正：
  - 真实问题不是 prompt 语义，也不是 parser，而是 `ChatGLM3-6B` 远端实现依赖旧 generation 私有接口
  - 当前 `Lion` 环境里的 `transformers=5.1.0` 不再提供这些旧私有方法
- 当前修复方式：
  - 保留官方 `build_chat_input`
  - 保留官方 `prepare_inputs_for_generation`
  - 保留官方 `process_response`
  - 在本地 smoke runner 中手动更新 `past_key_values / attention_mask / position_ids`
- 修复后结论：
  - `ChatGLM3-6B` 已通过 1 条 smoke
  - 现在可以进入“有效输入协议”集合
  - 但这一结论目前仍停留在 `outputs/debug/`，不代表已恢复 canonical 主线

## 关联文档
- [[当前状态]]
- [[任务笔记 - FinQA 基准重建]]
- [[异常与决策记录]]
- [[会话日志]]
- [[过程总结_协议可用性筛查]]
- [[过程总结_主表扩展与新增模型筛查]]
