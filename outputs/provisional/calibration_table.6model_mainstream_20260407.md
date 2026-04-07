# 6 模型 calibration 进度表（provisional）

- 快照时间：2026-04-07 HKT
- 说明：
  - `status=completed` 表示该模型该档位的 `summary.json` 已落盘，指标为正式档位结果。
  - `status=running` 表示该模型该档位仍在运行，指标为当前中间统计，不可当最终结论引用。
  - `status=pending` 表示该模型该档位尚未开始。
  - 本表只对应本轮 fresh run：
    - `outputs/provisional/calibration_runs_6model_20260407/`
    - `outputs/provisional/calibration_report.6model_mainstream_20260407.csv`

| 模型 | 档位 | 状态 | 已跑样本 | FormatOK rate | ValidParse rate | TruncationWithoutAnswer rate | 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Orca-2-7B | 128 | completed | 50 | 0.80 | 0.72 | 0.18 | 正式 summary 已落盘 |
| Orca-2-7B | 192 | completed | 50 | 0.86 | 0.78 | 0.10 | 正式 summary 已落盘 |
| Orca-2-7B | 256 | completed | 50 | 0.86 | 0.78 | 0.10 | 正式 summary 已落盘 |
| Zephyr-7B-beta | 128 | completed | 50 | 0.46 | 0.08 | 0.32 | 正式 summary 已落盘 |
| Zephyr-7B-beta | 192 | completed | 50 | 0.56 | 0.12 | 0.14 | 正式 summary 已落盘 |
| Zephyr-7B-beta | 256 | completed | 50 | 0.56 | 0.12 | 0.06 | 正式 summary 已落盘 |
| Qwen2.5-7B-Instruct | 128 | completed | 50 | 1.00 | 0.96 | 0.00 | 正式 summary 已落盘 |
| Qwen2.5-7B-Instruct | 192 | completed | 50 | 1.00 | 0.96 | 0.00 | 正式 summary 已落盘 |
| Qwen2.5-7B-Instruct | 256 | completed | 50 | 1.00 | 0.96 | 0.00 | 正式 summary 已落盘 |
| Mistral-7B-Instruct-v0.3 | 128 | completed | 50 | 1.00 | 0.74 | 0.00 | 正式 summary 已落盘 |
| Mistral-7B-Instruct-v0.3 | 192 | completed | 50 | 1.00 | 0.74 | 0.00 | 正式 summary 已落盘 |
| Mistral-7B-Instruct-v0.3 | 256 | completed | 50 | 1.00 | 0.74 | 0.00 | 正式 summary 已落盘 |
| Yi-1.5-6B-Chat | 128 | completed | 50 | 0.90 | 0.74 | 0.02 | 正式 summary 已落盘 |
| Yi-1.5-6B-Chat | 192 | completed | 50 | 0.90 | 0.74 | 0.02 | 正式 summary 已落盘 |
| Yi-1.5-6B-Chat | 256 | completed | 50 | 0.90 | 0.74 | 0.02 | 正式 summary 已落盘 |
| ChatGLM3-6B | 128 | completed | 50 | 0.52 | 0.34 | 0.46 | 已单独重跑并对齐 `predictions/summary` |
| ChatGLM3-6B | 192 | completed | 50 | 0.90 | 0.62 | 0.04 | 正式 summary 已落盘 |
| ChatGLM3-6B | 256 | completed | 50 | 0.90 | 0.62 | 0.04 | 正式 summary 已落盘 |

## 当前可见结论
- `Orca-2-7B` 三档都已经完成，且 `192 / 256` 的 `truncation_without_answer_rate` 都是 `0.10`。
- `Zephyr-7B-beta` 三档都已经完成，`truncation_without_answer_rate` 分别是 `0.32 / 0.14 / 0.06`，虽然随长度增大明显下降，但 `256` 仍高于 `2%` 安全口径。
- `Qwen2.5-7B-Instruct` 三档都已经完成，且三档的 `truncation_without_answer_rate` 都是 `0.00`。
- `Mistral-7B-Instruct-v0.3` 三档都已经完成，且三档的 `truncation_without_answer_rate` 都是 `0.00`。
- `Yi-1.5-6B-Chat` 三档都已经完成，`truncation_without_answer_rate=0.02 / 0.02 / 0.02`。
- `ChatGLM3-6B` 三档都已经完成，`truncation_without_answer_rate=0.46 / 0.04 / 0.04`。
- `ChatGLM3-6B / 128` 的旧 44 行异常版本已归档到：
  - `outputs/provisional/archive/chatglm3-6b-max128-suspect-20260407/`
- 当前所有 completed 档位均满足：
  - `predictions.jsonl` 行数等于 `summary.json.num_examples`
- 结论：
  - 本轮 6 模型 fresh provisional calibration 已完整结束
  - `128 / 192 / 256` 三档里仍没有一个能让全部 6 模型同时满足 `truncation_without_answer_rate <= 0.02`
