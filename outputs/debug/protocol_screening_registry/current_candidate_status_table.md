# 当前候选池状态总表

- 更新时间：2026-04-04 HKT
- 统计范围：只基于现有 `outputs/debug/` 和 `outputs/provisional/` 结果做协议可用性收束，不修改 canonical 输出。
- 当前进入有效输入协议的模型数：`2 / 8`
- 当前正式 instruction 主表候选：`Orca-2-7B`、`Zephyr-7B-beta`
- 距离“不少于 4 个正式 instruction 主表候选”还差：`2`

| model | model_family | 当前定位 | effective wrapper / tokenizer policy | 是否进入有效输入协议 | 当前失败模式或优势 | 下一步最小动作 | 是否继续投入调试 |
|---|---|---|---|---|---|---|---|
| Orca-2-7B | explanation_distill | formal candidate | `chatml` + `LlamaTokenizer.from_pretrained`，requested_use_fast=false | 是 | 固定样本已恢复 `Answer:` 且 `valid_parse=True`；但 `val_calib50` 的 `128/192/256` 截断率仍是 `0.18/0.10/0.10`，5 条 256 截断样本都先进入长推理 | 先保留为“可比较但有残余风险”的正式候选；若继续只在 provisional 层优先补测 `384` | 是 |
| Zephyr-7B-beta | alignment_distill | formal candidate | `plain` + 默认 `AutoTokenizer`；chat_template 存在但 registry 默认不用 | 是 | v2 smoke 已稳定出现 `Answer:` 且 `valid_parse=True` | 保持正式候选，等待候选池补足后再做下一轮切换评估 | 是 |
| Lion-7B | adversarial_distillation_anchor | anchor / protocol failure case | `alpaca` + 默认 `AutoTokenizer` | 否 | 官方 Alpaca wrapper 下仍 `Answer=False / valid_parse=False / repetition_collapse=True` | 单列为历史锚点/协议失败案例；不再拉回正式主表，不重复跑同一路径 smoke | 否 |
| MiniLLM-Llama-7B | base_distilled | base baseline | `completion` + 默认 `AutoTokenizer` | 否 | completion 路径下仍未稳定输出 parser v1 需要的 `Answer:` 数值行，但方法谱系上适合作为 base baseline | 继续单列为 `base_distilled baseline`，本阶段不再调 wrapper | 否 |
| Xwin-LM-7B | alignment | expansion candidate | `vicuna` + 默认 `AutoTokenizer`；effective tokenizer=`TokenizersBackend`，`pad_token=<unk>`，`pad_token_id=0` | 否 | 1 条 smoke 连续生成 128 个 token id `0`，即 `<unk>` 重复；`decode(skip_special_tokens=True)` 后为空串 | 不重复跑同一路径 smoke；如继续，只查 Xwin 专项 tokenizer/special-token 与官方 Vicuna 序列化一致性 | 是 |
| DeepSeek-R1-Distill-Qwen-7B | reasoning_distill | expansion candidate | `chat_template` + 默认 `AutoTokenizer`；effective tokenizer=`TokenizersBackend`，fallback=`chatml` | 否 | 输出进入自然语言/`<think>` 风格长推理，128 token 内无 `Answer:` 行，`truncation_suspect=True` | 查官方 wrapper；先确认是否存在更适合 direct-answer 的官方对话模板/非思维链入口 | 是 |
| OpenR1-Distill-7B | long_reasoning_distill | expansion candidate | `plain` 为当前推荐 candidate；默认 `AutoTokenizer` -> `Qwen2Tokenizer`；chat_template 不存在，已做 pad/eos 对齐 | 否 | plain 路径只回显 `Answer: <numeric value>` 占位符，`valid_parse=False`；chatml 路径则走 `\\boxed{}` / 长推理且无 `Answer:` | 查官方 wrapper；先确认推荐序列化方式，不再继续在 plain/chatml 间盲试 | 是 |
| LaMini-LLaMA-7B | instruction_distill | pending | `alpaca` 仅占位；repo 当前不可公开解析，未加载 tokenizer | 否 | `pending_repo_id`，未跑 smoke，没有可解释协议结论 | 等待可公开访问且可复现的 repo_id；之前不下载、不 smoke、不计入正式候选 | 否 |
