# FinBridge 实验方案设计

---

## 1. Proposal 理解

### 1.1 研究背景与核心问题

FinBridge 关注一个实用性很强的问题：通用蒸馏/对齐的紧凑型开源 LLM（7B–13B）能否在金融领域可靠地进行数值推理？如果不能，最小化的适配手段能否弥补差距？

核心问题可以拆成三层：(1) 现有紧凑模型在 FinQA 上的真实表现如何；(2) 失败模式是什么；(3) 轻量级适配（instruction tuning / 解释蒸馏 / 难例增强）能修复多少。

### 1.2 关键假设

| 编号 | 假设 | 可验证性 |
|------|------|----------|
| H1 | 通用蒸馏模型在 FinQA 上显著弱于 frontier 模型 | 直接对比即可验证 |
| H2 | 主要失败来自三类：领域理解、数值推理、格式不稳定 | 需要人工标注错误分类 |
| H3 | 针对性轻量适配可显著改善上述失败 | 适配前后对比 |
| H4 | FinQA 上的改善可迁移到 10-K 摘要任务 | 跨任务评估 |

### 1.3 Proposal 中不够清晰、需要补充的地方

1. **候选模型未确定**——仅列出"Orca-style, Zephyr-style, Phi-style"等族系，未给出具体 checkpoint（如 Zephyr-7B-beta vs. Zephyr-7B-alpha）。需要在实验前锁定 5–6 个具体模型及其 HuggingFace ID。
2. **Prompt 模板未定义**——"unified prompt template"是实验公平性的核心，但 proposal 未给出任何草稿。需要至少提供 2-shot 和 0-shot 两种模板并说明选择依据。
3. **FinQA 评估协议细节缺失**——"tolerance-based numerical match"的容差阈值是多少？±1%？±0.01？需明确。
4. **Teacher 模型未指定**——如果选择解释蒸馏路线，teacher 用 GPT-4、Claude 还是其他？成本预算多少？
5. **适配数据规模未估算**——"lightweight"到底是 500 条、5000 条还是 50000 条？直接影响训练成本和效果。
6. **10-K 数据集没有标准 ground truth**——ROUGE-L 对照什么参考摘要？人工写还是 frontier 模型生成？
7. **统计显著性未提及**——多个模型比较需要统计检验，proposal 未涉及。
8. **计算资源未量化**——需要多少 GPU-hours？是否有 A100/H100 访问权限？

---

## 2. 实验目标拆解

### Primary Objective

**PO1: 基准测试** — 在统一设置下评估 5–6 个紧凑模型在 FinQA 上的表现，建立可复现的 baseline。

- 指标：Answer Accuracy（精确匹配 + 容差匹配）、三类错误的频率分布

**PO2: 错误分析** — 对模型错误进行系统分类，量化领域理解 / 数值推理 / 格式不稳定各占多少。

- 指标：每类错误占总错误的比例；跨模型的错误模式一致性（Fleiss' κ）

**PO3: 轻量适配有效性** — 在选定的 student 模型上验证适配是否改善 FinQA 表现。

- 指标：适配后 Accuracy 提升幅度（绝对值 + 相对值）；三类错误比例变化

### Secondary Objective

**SO1: 跨任务迁移** — 适配模型在 10-K 摘要上是否也有改善。

- 指标：ROUGE-L、人工 factuality score（0/1 per sentence）

**SO2: 效率分析** — 各模型的推理效率对比。

- 指标：tokens/sec, latency (p50/p95), GPU memory peak

---

## 3. 实验设计

### 3.1 实验总览

本方案包含三个阶段性实验，依次进行：

- **实验 A**：紧凑模型 FinQA 基准测试 + 错误分析（对应 PO1, PO2）
- **实验 B**：轻量适配对比实验（对应 PO3）
- **实验 C**：跨任务迁移验证（对应 SO1）

### 3.2 实验 A — 基准测试与错误分析

**实验类型**：多模型离线评测 + 人工错误标注

#### 自变量
- 模型（5–6 个水平）：建议选定以下 checkpoint：
  1. Microsoft/Orca-2-7B
  2. HuggingFaceH4/zephyr-7b-beta
  3. WizardLMTeam/WizardLM-13B-V1.2
  4. microsoft/phi-2 (2.7B，作为小模型参照)
  5. mistralai/Mistral-7B-Instruct-v0.2
  6. （可选）Qwen/Qwen1.5-7B-Chat

#### 因变量
- FinQA Answer Accuracy（精确匹配）
- FinQA Answer Accuracy（容差匹配，建议容差 ±0.5% 相对误差或 ±0.01 绝对误差，取较宽者）
- 三类错误频率

#### 控制变量
- 统一 prompt 模板（建议 2-shot，含一个表格题和一个纯文本题）
- 统一解码参数：temperature=0, max_tokens=256
- 统一评估脚本和 tolerance 阈值
- 使用 FinQA 官方 test set（1,147 题）

#### 实验流程

1. **Prompt 设计**：设计统一模板，包含 system instruction（要求输出仅含最终数值答案）+ 2-shot 示例。在 validation set 上用 20 题 pilot 测试模板有效性。
2. **批量推理**：对 FinQA test set 全量推理，每个模型保存 raw output。
3. **自动评分**：编写评分脚本，先做精确字符串匹配，再做容差数值匹配。
4. **错误采样与标注**：从每个模型的错误中随机采样 100 题（若错误少于 100 则全取），由 2 名标注者独立分类为：
   - **Type G**：金融领域理解错误（概念误用、比率含义错误等）
   - **Type N**：数值推理错误（计算错误、单位混淆、多步出错等）
   - **Type F**：格式不稳定（输出含多余解释、格式不符、拒绝回答等）
   - **Type O**：其他
5. **标注一致性**：计算 Cohen's κ，目标 ≥ 0.7；不一致案例由第三人裁决。
6. **效率记录**：记录每模型在相同硬件上的 tokens/sec, latency, GPU memory。

#### 样本量

- FinQA test set 1,147 题 × 6 模型 = 约 6,882 次推理
- 错误标注：每模型 100 题 × 6 = 600 题人工标注
- 时间估算：单模型推理约 2–4 小时（A100 80GB），标注约 2 人 × 8 小时

#### 统计分析

- 模型间准确率差异：McNemar's test（两两比较）或 Cochran's Q test（多模型整体比较）
- 错误类型分布差异：Chi-square test
- 置信区间：Wilson score interval for proportions

### 3.3 实验 B — 轻量适配对比

**实验类型**：消融实验（Ablation study）

基于实验 A 的结果，选择 1 个 student 模型（选择标准：baseline 不太差 + 错误模式清晰 + 社区活跃）。

#### 自变量
- 适配方法（4 个水平）：
  - **B0**：无适配（baseline，复用实验 A 结果）
  - **B1**：直接金融 instruction tuning（用 FinQA train set 直接做 SFT）
  - **B2**：解释蒸馏（teacher 为 GPT-4o-mini 或 Claude Sonnet，生成 CoT 解释后训练 student）
  - **B3**：难例增强（基于 B0 的错误案例，让 teacher 生成变体题目 + 解答，再训练）

#### 因变量
- FinQA test Accuracy（精确 + 容差）
- 三类错误频率变化
- 训练成本（GPU-hours, API cost）

#### 控制变量
- 同一 student 模型
- LoRA rank=16, alpha=32, target modules: q_proj, v_proj
- 训练数据量统一为约 3,000 条（若某路线天然数据量不同，则下采样到 3,000 以保证公平比较；另可追加一组使用全量数据的实验作为参照）
- 训练 3 epochs, lr=2e-4, batch_size=4, gradient accumulation=8
- 评估时解码参数与实验 A 一致

#### 适配数据构造

| 路线 | 数据来源 | 构造方法 | 预估条数 |
|------|----------|----------|----------|
| B1 | FinQA train set | 直接转为 instruction format (context + question → answer) | 6,251 条（全集） |
| B2 | FinQA train set | 用 teacher API 为每题生成 CoT reasoning → (context + question → CoT + answer) | 3,000 条（采样） |
| B3 | 实验 A 错误案例 | 用 teacher 对每个错误题生成 3–5 个变体 → 混合原始训练集 | ~1,000 原始错误 × 3 变体 = 3,000 条 |

#### 实验流程

1. 从实验 A 选定 student 模型。
2. 构造 B1/B2/B3 三个训练集。
3. 分别训练三个 LoRA adapter。
4. 在 FinQA test set 上评估四组（B0–B3）。
5. 对 B1–B3 的错误再做 50 题抽样标注，观察错误类型分布是否改变。
6. 统计适配成本。

#### 统计分析

- B0 vs B1/B2/B3：McNemar's test
- 如果想比较 B1 vs B2 vs B3：使用 bootstrap resampling (n=1000) 计算 accuracy 差的 95% CI

### 3.4 实验 C — 跨任务迁移

**实验类型**：小规模对照评测

#### 设计

- 从 SEC EDGAR 收集 30 篇 10-K filing 的 Risk Factor 章节（覆盖不同行业和市值）。
- 让 B0（baseline）和 B_best（实验 B 最优适配模型）分别生成摘要。
- 参考摘要：由 GPT-4o 生成，经人工审核修正。
- 评估：
  - ROUGE-L（自动）
  - Factuality：对每个模型输出的每句话标注"supported / not supported / contradicted"（2 名标注者，30 篇 × 2 模型 = 60 份摘要）

#### 统计分析
- ROUGE-L 差异：paired t-test（30 对）
- Factuality score 差异：Wilcoxon signed-rank test

---

## 4. 风险与可行性评估

### 4.1 主要风险

| 风险 | 严重程度 | 可能性 | 缓解措施 |
|------|----------|--------|----------|
| 所有模型在 FinQA 上准确率极低（<15%），难以做有意义的比较 | 高 | 中 | 增加 few-shot 数量至 4-shot；加入 CoT 提示；若仍差，改为比较"部分正确率"或推理步骤匹配率 |
| Teacher API 成本超预算 | 中 | 中 | 优先用 GPT-4o-mini（比 GPT-4o 便宜 ~10×）；B2 路线只采样 3,000 题；设 hard cap |
| 错误标注主观性高、κ 值低 | 中 | 中 | 制定详细标注指南+10 题校准；不一致由第三人裁决 |
| LoRA 适配 3,000 条数据效果不显著 | 中 | 高 | 追加 full-data 实验组作为上界参考；同时报告学习曲线（500/1000/3000） |
| 10-K 摘要缺乏标准参考、ROUGE 不可靠 | 低 | 高 | 以人工 factuality 为主要指标，ROUGE 仅作辅助 |
| 计算资源不足 | 高 | 取决于环境 | 优先评估小模型（phi-2, 7B）；训练只用 LoRA；不做 full fine-tune |

### 4.2 最容易失败的环节

1. **Prompt 敏感性**——不同模型对 prompt 格式敏感度不同，可能需要为每个模型微调 prompt。建议在 validation set 上做小规模 prompt sensitivity 实验（3 种 prompt × 6 模型 × 50 题）。
2. **B3 难例增强的变体质量**——teacher 生成的变体可能重复或偏离原始难度分布。建议对生成的变体做人工质检（抽检 10%）。

### 4.3 替代方案

- 若所有 7B 模型表现过差：纳入 13B 或 Mixtral-8x7B 作为更强 baseline。
- 若 LoRA 无效：尝试全参数微调 phi-2（2.7B 可在单卡上 full fine-tune）。
- 若 API 预算耗尽：B2 路线退化为 B1（不用 teacher）。

---

## 5. 输出优化建议

### 5.1 需要补充的信息清单

1. ✅ 锁定 5–6 个具体模型 checkpoint（含 HuggingFace ID）
2. ✅ 设计并 pilot 测试统一 prompt 模板
3. ✅ 确定容差匹配的阈值（建议 ±0.5% 相对误差 or ±0.01 绝对值）
4. ✅ 确认可用 GPU 资源（型号、数量、可用时长）
5. ✅ 确认 API 预算（用于 teacher 生成 CoT / 变体）
6. ✅ 确定 10-K 参考摘要的来源与质检流程
7. ✅ 确定标注团队人数和标注指南

### 5.2 MVP 实验（最低可执行版本）

- **模型**：只选 3 个（phi-2, Zephyr-7B, Mistral-7B-Instruct）
- **评测**：FinQA test set，0-shot + 2-shot 两种 prompt
- **错误分析**：每模型抽样 50 题，1 人标注
- **适配**：只做 B1（直接 SFT），LoRA on Zephyr-7B，用 FinQA train set 全集
- **跨任务**：跳过
- **预计耗时**：2–3 周
- **预计 GPU**：~40 A100-hours

### 5.3 完整版实验

- 按上述实验 A + B + C 全部执行
- 含 6 模型基准、4 组适配消融、30 篇 10-K 迁移
- 预计耗时：6–8 周
- 预计 GPU：~150 A100-hours + ~$200 API 成本

---

## 6. 总结表

| 实验目标 | 方法 | 核心指标 | 预期结果 |
|----------|------|----------|----------|
| PO1: 基准表现 | 6 模型 × FinQA test set, unified prompt | Answer Accuracy (exact + tolerance) | 紧凑模型 25–45% accuracy，显著低于 frontier 模型 (>70%) |
| PO2: 错误分析 | 600 题人工标注 (2 annotators) | 三类错误比例, Cohen's κ ≥ 0.7 | 数值推理错误占 40–50%，领域理解 25–35%，格式不稳定 15–25% |
| PO3: 适配有效性 | 消融实验 B0/B1/B2/B3, LoRA | Accuracy 提升幅度, McNemar's p < 0.05 | B2 或 B3 带来 5–15pp 提升；B3 对数值推理错误改善最大 |
| SO1: 跨任务迁移 | 30 篇 10-K, baseline vs best | ROUGE-L, factuality score | 小幅改善（ROUGE-L +1–3 点），factuality 有可观测提升 |
| SO2: 效率分析 | 各模型推理 benchmark | tokens/sec, latency, GPU memory | phi-2 最快但最弱；7B 模型效率/性能平衡最好 |
