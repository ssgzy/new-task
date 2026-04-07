# val_screen200 进度表

- 生成时间：2026-04-07 21:08:40 HKT
- 冻结参数：`max_new_tokens=256`
- 冻结门槛： `runtime_success >= 0.95`, `format_ok >= 0.80`, `valid_parse >= 0.60`, `truncation_without_answer_rate <= 0.10`

| 模型 | 状态 | 通过冻结门槛 | RuntimeSuccess | FormatOK | ValidParse | Truncation | EM | TM | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Orca-2-7B | ok | yes | 1.000 | 0.960 | 0.845 | 0.025 | 0.030 | 0.075 |  |
| Zephyr-7B-beta | ok | no | 1.000 | 0.430 | 0.095 | 0.105 | 0.010 | 0.020 |  |
| Mistral-7B-Instruct-v0.3 | running_no_summary | pending | - | - | - | - | - | - | predictions_lines=65 |
| Qwen2.5-7B-Instruct | pending | pending | - | - | - | - | - | - |  |
| Yi-1.5-6B-Chat | pending | pending | - | - | - | - | - | - |  |
| ChatGLM3-6B | pending | pending | - | - | - | - | - | - |  |
