# val_screen200 进度表

- 生成时间：2026-04-07 22:21:32 HKT
- 冻结参数：`max_new_tokens=256`
- 冻结门槛： `runtime_success >= 0.95`, `format_ok >= 0.80`, `valid_parse >= 0.60`, `truncation_without_answer_rate <= 0.10`

| 模型 | 状态 | 通过冻结门槛 | RuntimeSuccess | FormatOK | ValidParse | Truncation | EM | TM | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Orca-2-7B | ok | yes | 1.000 | 0.960 | 0.845 | 0.025 | 0.030 | 0.075 |  |
| Zephyr-7B-beta | ok | no | 1.000 | 0.430 | 0.095 | 0.105 | 0.010 | 0.020 |  |
| Mistral-7B-Instruct-v0.3 | ok | yes | 1.000 | 1.000 | 0.770 | 0.000 | 0.095 | 0.155 |  |
| Qwen2.5-7B-Instruct | ok | yes | 1.000 | 1.000 | 0.980 | 0.000 | 0.250 | 0.335 |  |
| Yi-1.5-6B-Chat | ok | yes | 1.000 | 0.905 | 0.710 | 0.005 | 0.050 | 0.085 |  |
| ChatGLM3-6B | ok | no | 1.000 | 0.830 | 0.595 | 0.045 | 0.020 | 0.040 |  |
