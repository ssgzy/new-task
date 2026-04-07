# FinBridge 评分逻辑、指标说明与报告写作指南

---

## 一、评分逻辑：Strict Match vs Relaxed Match

### 1.1 为什么需要两套评分

你的 parser v1 只认 `Answer: <纯数值>` 这种格式。但 Mistral 这样的模型会输出：
```
Answer: 100 * (5.99 - 3.24) / 3.24 = 86.47%
```
它算对了，但被 strict parser 判为 parse 失败、得分为零。

如果只报告 strict 分数，你会低估模型的真实推理能力；如果只报告 relaxed 分数，你又丢失了"模型能否遵守输出格式"这个重要维度。

**所以两套都报，它们回答的是不同的问题：**
- Strict → "模型在严格格式约束下能否正确作答？"（可部署性）
- Relaxed → "模型是否真的算对了数？"（推理能力上界）

### 1.2 Strict Match 评分逻辑（你现有的 parser v1）

```python
def strict_score(prediction_text: str, gold_numeric: float, tolerance: float = 0.005):
    """
    Strict match：只认 'Answer: <纯数值>' 格式
    返回 (em, tm, parsed_value)
    """
    # Step 1: 找最后一行 'Answer:' 开头的行
    lines = prediction_text.strip().split('\n')
    answer_line = None
    for line in reversed(lines):
        if line.strip().lower().startswith('answer:'):
            answer_line = line.strip()
            break
    
    if answer_line is None:
        return 0, 0, None  # 没有 Answer 行
    
    # Step 2: 提取 Answer: 后面的内容
    raw_value = answer_line.split(':', 1)[1].strip()
    
    # Step 3: 清洗并尝试转为数值
    cleaned = raw_value.replace(',', '').replace('$', '').replace('%', '').strip()
    
    try:
        pred = float(cleaned)
    except ValueError:
        return 0, 0, None  # 无法解析为纯数值 → valid_parse = False
    
    # Step 4: 计算 EM 和 TM
    em = 1 if pred == gold_numeric else 0
    
    # TM: tolerance match（容差匹配）
    if gold_numeric == 0:
        tm = 1 if abs(pred) <= tolerance else 0
    else:
        relative_error = abs(pred - gold_numeric) / abs(gold_numeric)
        tm = 1 if relative_error <= tolerance else 0
    
    return em, tm, pred
```

### 1.3 Relaxed Match 评分逻辑（新增）

```python
import re

def relaxed_extract(text: str):
    """
    从模型输出中宽松提取数值答案。
    策略：
    1. 如果有 Answer 行，从 Answer 行提取
    2. 在 Answer 行中，如果有 '='，取 '=' 后面最后一个数值
    3. 否则取 Answer 行中最后一个数值
    4. 如果没有 Answer 行，取整个输出的最后一个数值
    """
    lines = text.strip().split('\n')
    
    # 优先找 Answer 行
    answer_line = None
    for line in reversed(lines):
        if line.strip().lower().startswith('answer:'):
            answer_line = line.strip().split(':', 1)[1].strip()
            break
    
    target = answer_line if answer_line else text
    
    # 如果有等号，只看等号后面的部分
    if '=' in target:
        target = target.split('=')[-1]
    
    # 清洗
    target = target.replace(',', '').replace('$', '')
    
    # 提取所有数值（含负号、小数点、百分号）
    numbers = re.findall(r'-?\d+\.?\d*%?', target)
    
    if not numbers:
        return None
    
    # 取最后一个数值
    raw = numbers[-1].replace('%', '').strip()
    try:
        return float(raw)
    except ValueError:
        return None


def relaxed_score(prediction_text: str, gold_numeric: float, tolerance: float = 0.005):
    """
    Relaxed match：宽松提取最终数值
    返回 (em, tm, parsed_value)
    """
    pred = relaxed_extract(prediction_text)
    
    if pred is None:
        return 0, 0, None
    
    em = 1 if pred == gold_numeric else 0
    
    if gold_numeric == 0:
        tm = 1 if abs(pred) <= tolerance else 0
    else:
        relative_error = abs(pred - gold_numeric) / abs(gold_numeric)
        tm = 1 if relative_error <= tolerance else 0
    
    return em, tm, pred
```

### 1.4 用法示例

```python
# Mistral 的输出
text = "Answer: 100 * (5.99 - 3.24) / 3.24 = 86.47%"
gold = 86.47

strict_em, strict_tm, _ = strict_score(text, gold)
# → strict_em=0, strict_tm=0（parse 失败，因为 Answer: 后面不是纯数值）

relaxed_em, relaxed_tm, _ = relaxed_score(text, gold)
# → relaxed_em=1, relaxed_tm=1（提取到 86.47，匹配成功）

# Qwen2.5 的输出
text2 = "Answer: 86.47"
strict_em2, strict_tm2, _ = strict_score(text2, gold)
# → strict_em=1, strict_tm=1（完美匹配）
```

### 1.5 "准确率"到底是哪个指标？

在你的实验里：

- **主准确率指标 = TM（Tolerance Match）**。这是你论文 results 表格里标粗的那一列。
  - 选 TM 而不是 EM 的原因：金融数据中经常出现四舍五入差异（如 gold=17.8 vs pred=17.80），这些在实质上是正确的，EM 会误判。
- **EM（Exact Match）** 作为辅助指标报告，更严格。
- **Strict TM** = 在 strict parse 下的 tolerance match → 反映"可部署准确率"
- **Relaxed TM** = 在 relaxed parse 下的 tolerance match → 反映"推理能力上界"
- **两者的差值 (Relaxed TM - Strict TM)** = 被格式问题吃掉的准确率 → 反映 format instability 的代价

---

## 二、每个指标的详细解释

### 2.1 任务性能指标

| 指标 | 全称 | 含义 | 为什么选它 |
|------|------|------|-----------|
| **Strict EM** | Strict Exact Match | 模型输出经 strict parse 后，提取的数值与 gold 完全相等的比例 | 最严格的正确性度量，但对四舍五入差异过于敏感 |
| **Strict TM** | Strict Tolerance Match | 模型输出经 strict parse 后，提取的数值与 gold 的相对误差 ≤ 0.5% 的比例 | **主准确率指标**。容许金融数据中常见的舍入差异，是 FinQA 评估中最标准的做法 |
| **Relaxed EM** | Relaxed Exact Match | 模型输出经 relaxed parse 后的 exact match | 反映不考虑格式约束时的纯推理正确率 |
| **Relaxed TM** | Relaxed Tolerance Match | 模型输出经 relaxed parse 后的 tolerance match | **推理能力上界**。如果 Relaxed TM 远高于 Strict TM，说明模型会算但不听话 |

### 2.2 格式合规指标

| 指标 | 含义 | 为什么选它 |
|------|------|-----------|
| **format_ok_rate** | 模型输出中包含 `Answer:` 行的比例 | 反映模型是否理解了 "输出 Answer: 行" 这个指令。低 → 模型完全无视格式要求 |
| **valid_parse_rate** | `Answer:` 行能被 strict parser 成功解析为数值的比例 | 反映模型是否能精确地只输出数值。format_ok 高但 valid_parse 低 → 模型写了 Answer 行但内容不干净（像 Mistral 带算式） |
| **truncation_without_answer_rate** | 输出吃满 max_new_tokens 且没有 Answer 行的比例 | 反映模型是否倾向于生成过长的文本而不收束到最终答案。高 → 模型进入"长推理模式"停不下来 |

### 2.3 效率指标

| 指标 | 含义 | 为什么选它 |
|------|------|-----------|
| **avg_latency_ms** | 单条样本的平均推理延迟（毫秒） | 直接反映部署时的响应速度 |
| **tok_per_sec** | 每秒生成的 token 数 | 归一化的吞吐量，便于跨模型比较 |
| **peak_vram** | 推理时的 GPU 显存峰值 | 决定模型能否部署到特定硬件上 |
| **mean_output_tokens** | 平均输出 token 数 | 与推理成本直接相关；也反映模型的"啰嗦程度" |

### 2.4 错误分析指标（人工标注）

| 错误类型 | 含义 | 典型案例 |
|---------|------|---------|
| **Type G: 金融领域理解错误** | 模型误解了金融概念、比率、报告术语 | 把"revenue growth"理解为绝对值而非百分比变化 |
| **Type N: 数值推理错误** | 模型理解了问题但计算出错 | 多步计算中间步骤出错、单位混淆、选错了表格中的数字 |
| **Type F: 格式不稳定** | 模型没有按要求输出 Answer 行，或输出了无法解析的内容 | 输出完整算式、进入长推理不收束、重复生成文本 |

---

## 三、论文报告结构

### 建议的 Results 章节结构

```
5. Results

5.1 Screen200 准入结果（简要）
    - 6 个模型中 5 个通过准入，1 个（Zephyr）未通过
    - 表格：6 模型的准入指标
    - 简要说明淘汰原因

5.2 主评估结果（validation 883）
    - 核心结果表：5 模型 × (Strict TM, Relaxed TM, EM, format_ok, valid_parse)
    - Qwen2.5 >> 其他模型的主发现
    - Strict vs Relaxed 差距分析 → format instability 的量化代价

5.3 效率对比
    - 表格：5 模型 × (latency, tok/s, VRAM, mean_output_tokens)
    - 效率与准确率的 trade-off 讨论

5.4 错误分析
    - 每模型抽样标注结果
    - 三类错误的分布对比图
    - 跨模型的错误模式差异

5.5 未通过模型的定性分析
    - Zephyr 的 format compliance 失败故事
    - 对蒸馏/对齐策略选择的启示
```

---

## 四、Zephyr 的故事怎么讲

这是一个完整的、有教学价值的 negative case，可以这样给老师解释：

### 4.1 时间线

**阶段一：初始 smoke（1 条）— 看起来很好**

Zephyr-7B-beta 是我们最早通过 smoke 测试的模型之一。在 plain 模式下跑 1 条样本时，它直接输出了 `Answer: 112.24%`，valid_parse=True。当时我们认为 Zephyr 是最稳定的 direct-answer 候选。

我们还发现一个有趣的现象：Zephyr 在 plain 模式（不用 chat_template）下反而表现更好。如果启用 chat_template，Zephyr 会把 `Answer:` 当作段落标题，后面跟一大段解释文字，导致 parse 失败。这是一个"模板模式改变输出风格"的案例。

**阶段二：calibration（50 条）— 暴露问题**

当我们把规模从 1 条扩大到 50 条时，Zephyr 的表现急剧下降：

- valid_parse_rate 从 1 条 smoke 的 100% 降到 50 条的 **8–12%**
- format_ok_rate 只有 46–56%
- truncation_without_answer_rate 在 128 档高达 32%

这说明 Zephyr 在单条 smoke 上的好表现是个案，不能代表整体。在 50 条样本中，Zephyr 大量输出段落式解释而不是简洁的 `Answer: 数值`。

**阶段三：screen200（200 条）— 确认淘汰**

在 200 条正式筛选中，Zephyr 的指标进一步坐实了 calibration 的发现：

- format_ok = 0.43（门槛 0.80）
- valid_parse = 0.095（门槛 0.60）
- truncation_without_answer_rate = 0.105（门槛 0.10）

三个格式指标全部不达标。EM=0.01 意味着 200 条中只有 2 条严格正确。

### 4.2 根本原因分析

Zephyr-7B-beta 通过 Direct Distillation of LM Alignment (DPO) 从 Mistral-7B base 对齐而来。DPO 的目标是让模型生成"人类偏好的回答"——在对话场景下，这通常意味着详细、有解释、有上下文的长回答。

这恰好与我们的 direct-answer 协议冲突：我们要求模型只输出一行 `Answer: 数值`，但 DPO 训练让 Zephyr **倾向于输出详细解释**。这不是 Zephyr 的 bug，而是它的对齐目标和我们的任务需求之间的结构性矛盾。

对比之下，同样基于 DPO 的 Mistral-7B-Instruct-v0.3 在 format_ok 上达到了 1.0。区别在于 Mistral 的 instruction tuning 还包含了 SFT 阶段，可能让它更擅长遵守精确的格式指令。这提示我们：**DPO 本身不足以保证格式遵从，SFT 阶段对 format compliance 可能更关键。**

### 4.3 在论文中的定位

Zephyr 不是一个"失败品"要隐藏，而是一个核心发现的载体：

> "We find that alignment method choice significantly impacts format compliance under strict output constraints. Zephyr-7B-beta, aligned primarily through DPO, achieved only 9.5% valid parse rate on val_screen200, compared to 77–98% for models with SFT-based alignment. This suggests that preference optimization alone does not guarantee reliable instruction following for structured financial outputs."

这段话直接回应了你 proposal 里 "output-format instability" 这个错误类型，是你 error analysis 部分的重要素材。

---

## 五、模型谱系与 Proposal 对应关系

你的 proposal 标题问的是 "Can General-Purpose Distilled and Aligned LLMs Reason Effectively Over Financial Data?"，你的 5 个模型覆盖了这个问题的核心维度：

| 模型 | 参数量 | 基座 | 紧凑化方法 | 对齐方法 | Proposal 对应 |
|------|--------|------|-----------|---------|--------------|
| Orca-2-7B | 7B | Llama-2-7B | 知识蒸馏（GPT-4 解释数据） | ChatML SFT | "Orca-style" 蒸馏代表 |
| Mistral-7B-Instruct-v0.3 | 7.2B | Mistral-7B-v0.3 | 原生紧凑架构 | SFT + DPO | 紧凑架构 + 对齐代表 |
| Qwen2.5-7B-Instruct | 7.6B | Qwen2.5-7B | 原生紧凑架构 | SFT + DPO | 最新一代紧凑对齐模型代表 |
| Yi-1.5-6B-Chat | 6B | Yi-1.5-6B | 原生紧凑架构 | SFT + RLHF | RLHF 对齐路线代表 |
| ChatGLM3-6B | 6B | GLM-3 | 原生紧凑架构（非标 Prefix LM） | SFT + RLHF | 非标架构 + 对齐代表 |
| Zephyr-7B-beta | 7B | Mistral-7B-v0.1 | 原生紧凑架构 | DPO（无 SFT） | DPO-only 对齐的反面教材 |

在论文里你可以这样定位：

> "Our candidate models span two main routes for building capable compact LLMs: (1) knowledge distillation from stronger teachers (Orca-2), and (2) direct alignment of compact base models through various post-training strategies including SFT+DPO (Mistral, Qwen2.5), SFT+RLHF (Yi, ChatGLM), and DPO-only (Zephyr, used as a negative case study). This allows us to examine not only whether compact models can perform financial reasoning, but also which alignment strategies best support format-constrained numerical tasks."

---

## 六、Validation 883 跑完后，结果表的推荐格式

### 主结果表（Table 1）

```
Table 1: Financial Numerical Reasoning Performance on FinQA Validation Set (N=883)

| Model          | Params | Alignment | Strict TM | Relaxed TM | Gap   | Valid Parse | Format OK |
|----------------|--------|-----------|-----------|------------|-------|-------------|-----------|
| Qwen2.5-7B    | 7.6B   | SFT+DPO   | ???       | ???        | ???   | 0.98        | 1.00      |
| Mistral-7B     | 7.2B   | SFT+DPO   | ???       | ???        | ???   | 0.77        | 1.00      |
| Orca-2-7B      | 7B     | Distill   | ???       | ???        | ???   | 0.845       | 0.96      |
| Yi-1.5-6B      | 6B     | SFT+RLHF  | ???       | ???        | ???   | 0.71        | 0.905     |
| ChatGLM3-6B    | 6B     | SFT+RLHF  | ???       | ???        | ???   | 0.595       | 0.83      |
| Zephyr-7B†     | 7B     | DPO       | 0.02*     | ???        | ???   | 0.095       | 0.43      |

† Did not pass qualification screening; screen200 results shown for reference.
Gap = Relaxed TM - Strict TM (accuracy lost to format non-compliance)
```

### 效率表（Table 2）

```
Table 2: Inference Efficiency on FinQA Validation Set

| Model          | Latency (ms) | Tok/s | Peak VRAM (GB) | Mean Output Tokens |
|----------------|-------------|-------|----------------|-------------------|
| ...            | ...         | ...   | ...            | ...               |
```
