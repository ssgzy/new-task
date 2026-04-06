# Claude建议核查

## 核查对象
- 原文档：`Claude suggestion/问题诊断与改进建议.md`
- 关联文档：[[异常与决策记录]]、[[长度校准报告]]、[[实验协议_v1]]、[[当前状态]]

## 总结论
- 这份 Claude 分析**部分正确，但不能直接照单全收**。
- 它对“当前 runner 确实没用 `apply_chat_template()`”和“Lion / Orca 输出存在明显重复退化”这两点判断是对的。
- 但它把“一个 chat template 改动就能全局修复 Lion / Orca / Zephyr”说得太满；按当前本地 tokenizer 事实，**这个修复方案对 Lion-7B 和 Orca-2-7B 并不会自动生效**。
- 它对 CoT、TM 百分比豁免、降低 screening 门槛的建议，**更像 protocol v2/ablation 方向，不适合直接覆盖已冻结的 protocol_v1**。

## 逐条核查

### 1. “Chat Template 未应用是最高优先级根因”
- 结论：**诊断方向部分正确，修复方案不完整**。
- 证据：
  - 当前 runner 的确直接用 `prompt = build_prompt_from_record(record)`，然后 `tokenizer(prompt, return_tensors="pt")`，没有 `apply_chat_template()`。
  - 当前协议代码的 `build_prompt_from_record()` 也确实只返回 `build_plain_prompt()`。
  - 样例输出中，`Lion-7B` 和 `Orca-2-7B` 在 `128` 档都出现明显重复退化，且 `missing_answer_line + truncated_suspect = True`。
- 关键反证：
  - 实测本地 tokenizer：
    - `Lion-7B`: `has_chat_template = False`
    - `Orca-2-7B`: `has_chat_template = False`
    - `Zephyr-7B-beta`: `has_chat_template = True`
    - `MiniLLM-Llama-7B`: `has_chat_template = False`
  - 因此 Claude 文档里这段逻辑：
    - `if tokenizer.chat_template is not None: apply_chat_template(...) else: build_plain_prompt(...)`
  - **只会改变 Zephyr-7B-beta，不会改变 Lion-7B / Orca-2-7B / MiniLLM-Llama-7B**。
- 结论性判断：
  - “plain prompt 可能是退化原因之一”是合理假设。
  - “这个单处改动可让 Lion / Orca 大幅恢复”**没有被本地 tokenizer 事实支持**。
  - 若要修 Lion / Orca，必须先明确它们各自应使用的对话包装格式，并把这件事升级为一个新的 protocol/rendering 决策，而不是直接改动已冻结的 protocol_v1。

### 2. “长度冻结失败是伪问题，不需要扩到 512”
- 结论：**这个判断过于绝对，当前不能当最终结论**。
- 证据：
  - 当前 `outputs/calibration_report.csv` 里，`128 / 192 / 256` 三档确实都无法满足“全部主表模型截断率 <= 2%”。
  - 但这是在当前 protocol_v1 的 plain prompt 渲染下得到的结果。
- 判断：
  - 如果后续引入 prompt rendering v2，当前长度失败确实可能部分缓解。
  - 但在还没做 v2 smoke / calibration 前，不能直接断言“扩到 512 没必要”。
  - 所以这条更准确的表述应是：**先验证 prompt rendering 是否导致退化，再决定是否扩长 calibration 档位**。

### 3. “禁止 CoT 会系统性压低准确率”
- 结论：**研究上可能成立，但不是当前 protocol_v1 的 bug 修复项**。
- 说明：
  - 当前 [[实验协议_v1]] 明确冻结为不输出推理，只输出 `Answer: <numeric value>`。
  - 这个设计偏向“格式稳定性 + 可直接解析”的 benchmark 目标，不等价于“推理能力上限测量”。
- 建议：
  - 可以把 `cot_v1` 作为**额外 ablation / appendix 协议**。
  - 但不要在当前主线中直接把 protocol_v1 改成 CoT，否则前后结果不可比。

### 4. “gold_numeric 百分比口径有歧义，TM 应加 ×100 豁免”
- 结论：**问题识别是对的，Claude 给的 TM 修法太宽松**。
- 已知事实：
  - 我们已经确认 `qa.answer` 与 `qa.exe_ans` 在 validation 有 `623/883` 条数值不一致，百分比缩放是主要来源之一。
  - 当前 `gold_numeric` 以 `qa.answer` 标准化为准，`"93.5%" -> 93.5`。
- 为什么 Claude 的 TM 修法不能直接上：
  - `is_close(pred * 100, gold) or is_close(pred, gold * 100)` 会把“单位差异”和“真实数值差一百倍”混在一起，可能引入假阳性。
  - 当前 parser 虽然记录了 `pred_unit_type`，但 gold 侧没有显式保留 `gold_unit_type`，所以直接在 TM 双向放宽并不严谨。
- 更稳妥做法：
  - 如果要改，应升级数据 schema 和 parser/schema，显式记录 gold 与 pred 的单位类型，再做受控归一化。
  - 这属于 protocol_v2 范围，不建议直接热改 protocol_v1。

### 5. “应降低 screening 门槛，把格式失败从准入门槛里拿掉”
- 结论：**这是实验设计取舍，不是当前分析可直接判定的‘对/错’**。
- 现状：
  - 我们当前主线本来就把 `FormatOK / ValidParse / truncation_without_answer_rate` 当成准入门槛，目标是筛掉“不可稳定解析”的模型。
- 判断：
  - 如果研究问题转向“只看金融推理潜力，不看格式稳定性”，那可以重新设计门槛。
  - 但如果目标仍是“可复现、可自动解析、可比较的 student selection benchmark”，那当前门槛设计并不必然错误。

## 建议你怎么用这份 Claude 文档
- 可以采纳：
  - “优先检查 prompt rendering / chat-template 适配”这个排查方向
  - “先做每模型 1 条 smoke，再重跑 calibration”这个执行顺序
  - “当前 direct-only 协议之外，可以另做 cot_v1 ablation”这个论文补充思路
- 暂时不要直接采纳：
  - “一个 `apply_chat_template()` fallback 就能修好 Lion / Orca”
  - “不需要扩展到 512”
  - “直接给 TM 加双向 ×100 豁免”
  - “直接降低 screening 门槛”

## 建议下一步
1. 先为 Zephyr 做一个**provisional prompt rendering smoke**：plain prompt vs `apply_chat_template()` 对比。
2. 单独查 Lion / Orca 对应仓库是否有推荐 prompt 格式；如果没有 tokenizer 内置模板，则不能假装 `apply_chat_template()` 会自动解决。
3. 若 prompt rendering 确认显著影响输出，再新增 `protocol_v2_prompt_rendering`，不要直接覆盖 protocol_v1。
4. 再决定是否扩展 `max_new_tokens` 档位，或把 Lion 作为历史锚点单列。
