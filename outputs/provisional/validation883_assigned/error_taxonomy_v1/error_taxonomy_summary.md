# validation883 error taxonomy summary

- 本文档基于 `validation883` 现有 `predictions.jsonl` 做后验错误分类。
- `Strict TM` 仍是主指标；本摘要只补充解释 strict 失败的结构。

| 模型 | strict_fail_but_relaxed_hit_rate | strict_fail_and_relaxed_fail_rate | top protocol issue by rate | top protocol issue rate |
| --- | --- | --- | --- | --- |
| ChatGLM3-6B | 0.012458 | 0.972820 | truncated_without_answer | 0.600227 |
| Mistral-7B-Instruct-v0.3 | 0.028313 | 0.805210 | invalid_numeric_literal | 0.220838 |
| Orca-2-7B | 0.014723 | 0.919592 | invalid_numeric_literal | 0.112118 |
| Qwen2.5-7B-Instruct | 0.001133 | 0.639864 | invalid_numeric_literal | 0.015855 |
| Yi-1.5-6B-Chat | 0.053228 | 0.847112 | invalid_numeric_literal | 0.155153 |
