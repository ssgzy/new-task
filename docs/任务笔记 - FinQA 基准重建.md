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
  1. 先基于 DeepSeek / OpenR1 官方 wrapper 初查结论，决定是否各做一条“官方 messages 序列化”provisional 对照 smoke
  2. Orca 先保留为“可比较但有残余风险”的正式候选；如果只做一个 provisional 最小实验，则优先补 `384`
  3. `Xwin-LM-7B` 不重复跑当前 Vicuna 路径 smoke，若继续只做 tokenizer/special-token 专项核查
  4. `Lion-7B` 继续单列为历史锚点 / 协议失败案例，不拉回正式主表
  5. `MiniLLM-Llama-7B` 继续按 `base_distilled` baseline 单列保留，不再调 wrapper
  6. `LaMini-LLaMA-7B` 继续等待可公开 repo_id
- 当前明确不再重复执行：
  - 不重跑 Xwin 当前 Vicuna 路径同一条 smoke
  - 不把 Lion 拉回正式主表
  - 不继续调 MiniLLM wrapper
  - 不恢复 canonical `val_calib50`
  - 不恢复 canonical `val_screen200`
  - 不改 parser
  - 不把主协议改成 CoT
- 单模型执行仍统一走 [[组员运行说明]] 中的 provisional 流程。

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
