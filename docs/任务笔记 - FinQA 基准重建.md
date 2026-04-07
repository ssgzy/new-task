# 任务笔记 - FinQA 基准重建

## 任务定义
- 在新项目目录下重建 FinQA benchmark 流程。
- 先数据重拉与字段核对，再冻结 prompt / parser / decode config，再进行长度校准与准入筛选。

## 本轮硬性交付
- FinQA 全量原始 split
- `protocol_v1`
- `calibration_report.csv`
- `qualification_summary.csv`

## 模型范围
### 当前正式 instruction 候选
- Orca-2-7B
- Zephyr-7B-beta

### 新增主表扩展候选
- Xwin-LM-7B
- LaMini-LLaMA-7B（当前仅保留待补 repo_id 占位，本轮不下载、不 smoke）
- DeepSeek-R1-Distill-Qwen-7B
- OpenR1-Distill-7B

### 单列对照/锚点
- Lion-7B：历史锚点 / 协议失败案例
- MiniLLM-Llama-7B：`base_distilled` baseline

## 禁止事项
- 不使用 `test1147` 做任何协议开发或筛选。
- 不默认继承旧脚本字段映射。
- 不允许临时手改 parser 适配个别模型。

## 当前执行顺序
1. 检查目录与初始化文档
2. 重拉 FinQA 并核对字段
3. 生成固定 manifest
4. 冻结 `protocol_v1`
5. 修复 benchmark 环境与 canonical 注册表
6. 先修复并验证输入接口正确性
7. 再决定是否恢复长度校准
8. 做准入筛选
9. 更新状态与日志

## 当前阻塞
- 2026-04-07 新增 6 模型预筛已启动，当前已明确：
  - `Llama-3.1-8B-Instruct`：官方 gated repo，未登录时 `snapshot_download` 直接 `401`
  - `Gemma-7B-IT`：官方 gated repo，未登录时 `snapshot_download` 直接 `401`
  - 所以这两个模型当前是“官方下载门控失败”，不是 smoke 失败
  - 本轮剩余真正需要下载并跑 smoke 的新增模型是 `Mistral-7B-Instruct-v0.3`、`Qwen2.5-7B-Instruct`、`Yi-1.5-6B-Chat`、`ChatGLM3-6B`
- `Qwen2.5-7B-Instruct` 已完成官方下载与 1 条 smoke：
  - `effective_wrapper=chat_template`
  - `effective_tokenizer_class_name=Qwen2Tokenizer`
  - `answer_appears=True / valid_parse=True / truncation_suspect=False`
  - 已进入有效输入协议
- `Yi-1.5-6B-Chat` 已完成官方下载与 1 条 smoke：
  - `effective_wrapper=chat_template`
  - `effective_tokenizer_class_name=TokenizersBackend`
  - `answer_appears=True / valid_parse=True / truncation_suspect=False`
  - 已进入有效输入协议
- 当前主流新增模型的进度变为：
  - `Mistral-7B-Instruct-v0.3`：有 `Answer:` 但输出算式，`valid_parse=False`
  - `Qwen2.5-7B-Instruct`：通过 smoke
  - `Yi-1.5-6B-Chat`：通过 smoke
  - `ChatGLM3-6B`：已下载完成，但首条 smoke 当前 `load_failed`
- `ChatGLM3-6B` 当前首条 smoke 的直接阻塞是：
  - `AttributeError: 'ChatGLMConfig' object has no attribute 'max_length'`
  - 当前更像 `trust_remote_code` / 自定义加载入口兼容问题
  - 因此现在还不能把它归类为“协议失败”或“模型能力失败”
- 当前最高优先级已从“扩展候选池 smoke”转为“协议可用性筛查后的收束分析”，但仍不恢复 canonical benchmark。
- v3 最小修复 2 模型 1 条 smoke 结论：
  - `Lion-7B -> alpaca`：官方 Alpaca-style wrapper 已生效，但仍无 `Answer:` 且重复崩塌
  - `Orca-2-7B -> chatml + LlamaTokenizer`：已恢复 `Answer: 1.33%` 且 `valid_parse=True`
- 因此当前**不建议立刻重跑全主表 canonical `val_calib50`**；若要继续，优先只做 Orca 单模型 provisional calibration，Lion 另行排查。
- expansion smoke 首轮没有新增出新的“已进入有效输入协议”模型：
  - `Xwin-LM-7B` 首轮因 partial snapshot 跳过；当前已补下载完成并补跑 1 条 smoke，但 Vicuna wrapper 下连续生成 128 个 `<unk>`，`Answer=False / valid_parse=False / truncation_suspect=True`，仍未进入有效输入协议
  - `LaMini-LLaMA-7B` 因 `pending_repo_id` 跳过
  - `DeepSeek-R1-Distill-Qwen-7B` 未出现 `Answer:`
  - `OpenR1-Distill-7B` 只回显 `Answer: <numeric value>` 占位文本，未形成合法数值
- 所以当前正式 instruction 主表候选仍只有 `Orca-2-7B` 与 `Zephyr-7B-beta`，还差 `2` 个。
- Orca 单模型 provisional `val_calib50` 已完成，但 `128/192/256` 三档的 `truncation_without_answer_rate` 分别是 `0.18 / 0.1 / 0.1`，`256` 仍高于 2% 安全口径，因此当前仍不建议恢复 canonical `val_calib50`。
- Orca `256` 档 5 条截断样本已复查：
  - 5/5 都走 `chatml + LlamaTokenizer`
  - 5/5 输出都先展开自然语言 step-by-step 推理，最终没落到 `Answer:` 行
  - 截断组 prompt 长度并不比非截断组最大值更极端，因此当前更像“个别样本触发 Orca 的长推理倾向”，而不是统一输入包装损坏
- 当前 8 个模型中只有 `Orca-2-7B` 和 `Zephyr-7B-beta` 进入有效输入协议；候选池状态总表已输出到：
  - `outputs/debug/protocol_screening_registry/current_candidate_status_table.json`
  - `outputs/debug/protocol_screening_registry/current_candidate_status_table.md`
- Orca 下一步决策已收束到 [[过程总结_orca_下一步决策]]：
  - 更建议先保留为“可比较但有残余风险”的正式候选
  - 若只做一个 provisional 最小实验，优先补 `384`，不优先补 `320`
- DeepSeek / OpenR1 下一步最小动作已收束到 [[过程总结_新增候选失败类型分类]]：
  - `DeepSeek-R1-Distill-Qwen-7B`：先查官方 wrapper
  - `OpenR1-Distill-7B`：先查官方 wrapper
- DeepSeek / OpenR1 官方 wrapper 初查已经补充到 [[过程总结_新增候选失败类型分类]]：
  - DeepSeek 官方推荐用法会显式推动 `<think>` + `\\boxed{}`，和当前无 CoT parser v1 主协议天然有张力
  - OpenR1 官方示例偏 `pipeline(messages)` user-message 入口，而不是当前手写 plain 拼接；本地 tokenizer 又没有 `chat_template`

## 当前建议方案
- 推荐优先方案：
  1. `ChatGLM3-6B` 的 `trust_remote_code` 兼容修复已经完成，当前已和 `Orca + Zephyr + Qwen2.5 + Yi` 一起组成 `5` 个已进入有效输入协议的模型。
  2. 下一步若继续补最终 6 模型名单，应优先找一个可公开访问、非 gated、方法谱系能补多样性的候选，而不是回头重跑已知失败路径。
  2. `Llama-3.1-8B-Instruct` 与 `Gemma-7B-IT` 当前先记为“官方 gated repo 未获访问权限”，不要把它们误算成协议失败样本。
  3. 再基于 DeepSeek / OpenR1 官方 wrapper 初查结论，决定是否各做一条“官方 messages 序列化”provisional 对照 smoke
  4. Orca 先保留为“可比较但有残余风险”的正式候选；如果只做一个 provisional 最小实验，则优先补 `384`
  5. `Xwin-LM-7B` 不重复跑当前 Vicuna 路径 smoke，若继续只做 tokenizer/special-token 专项核查
  6. `Lion-7B` 继续单列为历史锚点 / 协议失败案例，不拉回正式主表
  7. `MiniLLM-Llama-7B` 继续按 `base_distilled` baseline 单列保留，不再调 wrapper
  8. `LaMini-LLaMA-7B` 继续等待可公开 repo_id
- 当前明确不再重复执行：
  - 不重跑 Xwin 当前 Vicuna 路径同一条 smoke
  - 不把 Lion 拉回正式主表
  - 不继续调 MiniLLM wrapper
  - 不恢复 canonical `val_calib50`
  - 不恢复 canonical `val_screen200`
  - 不改 parser
  - 不把主协议改成 CoT
- 单模型执行仍统一走 [[组员运行说明]] 中的 provisional 流程。
- `ChatGLM3-6B` 修复后的直接结果：
  - `outputs/debug/input_smoke_mainstream/ChatGLM3-6B/smoke_mainstream.json`
  - `effective_wrapper=chat_template`
  - `effective_tokenizer_class_name=ChatGLMTokenizer`
  - `raw_generation_text=Answer: 17.8`
  - `answer_appears=True / valid_parse=True / truncation_suspect=False / repetition_collapse=False`
- 已启动 6 模型 fresh `val_calib50` calibration：
  - 模型集合：`Orca-2-7B`、`Zephyr-7B-beta`、`Qwen2.5-7B-Instruct`、`Mistral-7B-Instruct-v0.3`、`Yi-1.5-6B-Chat`、`ChatGLM3-6B`
  - 长度档位：`128 / 192 / 256`
  - run root：`outputs/provisional/calibration_runs_6model_20260407/`
  - aggregate csv：`outputs/provisional/calibration_report.6model_mainstream_20260407.csv`
  - 当前仍只写 `provisional`，不恢复 canonical
- 当前运行 checkpoint：
  - `Orca-2-7B` 三档已完成，`truncation_without_answer_rate=0.18 / 0.10 / 0.10`
  - 这与此前单模型 provisional calibration 一致，说明 fresh 6 模型 run 没有因为新 registry/新 run root 而改写 Orca 结果
  - `Zephyr-7B-beta` 三档也已完成，`truncation_without_answer_rate=0.32 / 0.14 / 0.06`
  - 这说明 `Zephyr-7B-beta` 在更长输出空间下会改善，但到 `256` 仍未达到 `2%` 安全口径
  - `Qwen2.5-7B-Instruct` 三档已完成，`truncation_without_answer_rate=0.00 / 0.00 / 0.00`
  - `Mistral-7B-Instruct-v0.3` 三档已完成，`truncation_without_answer_rate=0.00 / 0.00 / 0.00`
  - `Yi-1.5-6B-Chat` 的 `128 / 192` 已完成，都是 `0.02`
  - `Yi-1.5-6B-Chat / 256` 也已完成，仍是 `0.02`
  - `ChatGLM3-6B / 192` 与 `/256` 已完成，都是 `0.04`
  - 当前异常点在 `ChatGLM3-6B / 128`：
    - `summary.json` 与 aggregate csv 都写成 `num_examples=50`
    - 但 `predictions.jsonl` 只有 `44` 行
  - 因此这轮 6 模型 calibration 已“形式上结束”，但 aggregate csv 仍需视为 `provisional_suspect`

## 关联文档
- [[项目总览]]
- [[当前状态]]
- [[数据重拉与字段核对]]
- [[实验协议_v1]]
- [[会话日志]]
- [[异常与决策记录]]
- [[输入接口诊断]]
- [[过程总结_输入包装修复]]
- [[过程总结_主表扩展与新增模型筛查]]
- [[过程总结_协议可用性筛查]]
- [[过程总结_orca_下一步决策]]
- [[过程总结_新增候选失败类型分类]]
- [[组员运行说明]]

## 2026-04-07 最新 checkpoint：6 模型 provisional calibration 已完成
- `Orca-2-7B = 0.18 / 0.10 / 0.10`
- `Zephyr-7B-beta = 0.32 / 0.14 / 0.06`
- `Qwen2.5-7B-Instruct = 0.00 / 0.00 / 0.00`
- `Mistral-7B-Instruct-v0.3 = 0.00 / 0.00 / 0.00`
- `Yi-1.5-6B-Chat = 0.02 / 0.02 / 0.02`
- `ChatGLM3-6B = 0.46 / 0.04 / 0.04`
- `ChatGLM3-6B / 128` 的旧 44 行异常已修复，旧版本已归档。
- 当前结论：
  - `128 / 192 / 256` 三档都无法让全部 6 模型同时满足 `truncation_without_answer_rate <= 0.02`

## 2026-04-07 最新 checkpoint：6 模型 val_screen200 已冻结
- 已冻结 `screen200` 规则：
  - `max_new_tokens = 256`
  - `runtime_success >= 0.95`
  - `format_ok >= 0.80`
  - `valid_parse >= 0.60`
  - `truncation_without_answer_rate <= 0.10`
- 已补齐本轮 6 模型 registry 视图：
  - `Orca-2-7B`
  - `Zephyr-7B-beta`
  - `Qwen2.5-7B-Instruct`
  - `Mistral-7B-Instruct-v0.3`
  - `Yi-1.5-6B-Chat`
  - `ChatGLM3-6B`
- 本轮产物目录：
  - `outputs/provisional/screen200_6model_20260407/`
- 本轮实时表：
  - `outputs/provisional/screen200_6model_20260407/screen200_status_table.md`
- 下一步：
  - `Orca-2-7B` 已完成并通过冻结门槛
  - 当前在跑 `Zephyr-7B-beta / val_screen200 / 256`
  - 改为逐模型增量落盘记录

## 2026-04-07 当前 screen200 checkpoint
- `Orca-2-7B` 已完成：
  - `runtime_success = 1.000`
  - `format_ok = 0.960`
  - `valid_parse = 0.845`
  - `truncation_without_answer_rate = 0.025`
  - 结论：通过当前冻结门槛
- `Zephyr-7B-beta` 已启动：
  - 但随后按用户要求停止
  - 半截目录已归档，不计入当前活跃汇总表
