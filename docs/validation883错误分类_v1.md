# validation883错误分类_v1

## 目的
- 本轮新增的是一个后验分析脚本：
  - `scripts/validation883_error_taxonomy_v1.py`
- 它只读取现有 `validation883` 结果做错误分类，不会重跑 benchmark。
- 它不会修改：
  - prompt
  - parser v1
  - decode config
  - scoring logic
- `Strict TM` 仍然是主指标；本脚本的作用是解释 strict 失败到底更像哪一类错误。

## 输入
- 默认输入目录：
  - `outputs/provisional/validation883_assigned`
- 自动发现：
  - 所有包含 `predictions.jsonl` 的模型子目录
- 当前实际发现的模型：
  - `ChatGLM3-6B`
  - `Mistral-7B-Instruct-v0.3`
  - `Orca-2-7B`
  - `Qwen2.5-7B-Instruct`
  - `Yi-1.5-6B-Chat`

## 设计口径

### 两层分类
- 第一层：
  - `protocol_primary_bucket`
- 第二层：
  - `outcome_bucket`

### protocol_primary_bucket
- 这是协议层主错误桶，互斥。
- 当前优先级：
  1. `truncated_without_answer`
  2. `missing_answer_line`
  3. `empty_answer_line`
  4. `invalid_numeric_literal`
  5. `none`

### multiple_answer_lines_flag
- 这是非互斥布尔标记。
- 当前目的：
  - 保留“多条 `Answer:` 行”的并发症信息
  - 不强行把它塞进主错误桶

### outcome_bucket
- 这是评分层结果桶，互斥。
- 当前定义：
  - `strict_pass`
  - `strict_fail_but_relaxed_hit`
  - `strict_fail_and_relaxed_fail`

## 命令
- 本轮实际执行命令：

```bash
python scripts/validation883_error_taxonomy_v1.py
```

- 默认参数：
  - `--input-root outputs/provisional/validation883_assigned`
  - `--output-dir outputs/provisional/validation883_assigned/error_taxonomy_v1`
  - `--sample-k 15`
  - `--seed 42`

## 输出文件
- `outputs/provisional/validation883_assigned/error_taxonomy_v1/by_model_error_counts.csv`
- `outputs/provisional/validation883_assigned/error_taxonomy_v1/overall_error_counts.csv`
- `outputs/provisional/validation883_assigned/error_taxonomy_v1/sample_level_taxonomy.csv`
- `outputs/provisional/validation883_assigned/error_taxonomy_v1/manual_review_candidates.csv`
- `outputs/provisional/validation883_assigned/error_taxonomy_v1/error_taxonomy_summary.md`

## 当前运行结果

### 整体
- 总样本数：
  - `4415`
- 整体 `strict_pass_rate`：
  - `0.141110`
- 整体 `strict_fail_but_relaxed_hit_rate`：
  - `0.021971`
- 整体 `strict_fail_and_relaxed_fail_rate`：
  - `0.836920`
- 整体最高协议问题：
  - `truncated_without_answer_rate = 0.126387`
  - `invalid_numeric_literal_rate = 0.126161`
- 当前 `multiple_answer_lines_count = 0`

### 各模型的主错误画像
- `ChatGLM3-6B`
  - 主问题是 `truncated_without_answer`
  - `truncated_without_answer_rate = 0.600227`
  - `strict_fail_and_relaxed_fail_rate = 0.972820`
- `Mistral-7B-Instruct-v0.3`
  - 主问题是 `invalid_numeric_literal`
  - `invalid_numeric_literal_rate = 0.220838`
  - `strict_fail_but_relaxed_hit_rate = 0.028313`
- `Orca-2-7B`
  - 主问题是 `invalid_numeric_literal`
  - `invalid_numeric_literal_rate = 0.112118`
  - 同时存在一定 `truncated_without_answer`
- `Qwen2.5-7B-Instruct`
  - 主问题仍是 `invalid_numeric_literal`
  - 但占比很低：
    - `0.015855`
  - 说明它的 strict 失败大多不是大规模协议崩坏
- `Yi-1.5-6B-Chat`
  - 主问题是 `invalid_numeric_literal`
  - `invalid_numeric_literal_rate = 0.155153`
  - `strict_fail_but_relaxed_hit_rate = 0.053228`
  - 当前是 `5` 个模型里 relaxed 补回比例最高的一档

## manual_review_candidates 的用途
- `manual_review_candidates.csv` 只抽：
  - `strict_fail_and_relaxed_fail`
- 每个模型默认抽 `15` 条
- 当前总条数：
  - `75`
- 空列：
  - `manual_label`
  - `review_notes`
- 适合后续做：
  - 人工复核
  - 失败案例整理
  - appendix 示例收集

## 当前结论
- 这份错误分类脚本已经把“strict 失败”拆成了更具体的后验结构。
- 当前最值得报告的两点是：
  - `ChatGLM3-6B` 的失败主要是严重截断后没有答案行
  - 其它主表模型更常见的问题不是“完全没答案”，而是 `invalid_numeric_literal`
- 这有助于在报告里区分：
  - 协议彻底失控
  - 与“有答案行但 strict 不接受”的格式损失

## 关联文档
- [[评分逻辑与报告写作指南]]
- [[validation883结果字段说明]]
- [[当前状态]]
- [[任务笔记 - FinQA 基准重建]]
