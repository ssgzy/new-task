# val_screen200 最终汇总（6 模型，provisional）

- 生成时间：2026-04-07 22:30 HKT
- 冻结参数：`max_new_tokens = 256`
- 冻结门槛：
  - `runtime_success >= 0.95`
  - `format_ok >= 0.80`
  - `valid_parse >= 0.60`
  - `truncation_without_answer_rate <= 0.10`

## 结果总表

| 模型 | RuntimeSuccess | FormatOK | ValidParse | Truncation | EM | TM | 是否通过 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Orca-2-7B | 1.000 | 0.960 | 0.845 | 0.025 | 0.030 | 0.075 | yes |
| Zephyr-7B-beta | 1.000 | 0.430 | 0.095 | 0.105 | 0.010 | 0.020 | no |
| Mistral-7B-Instruct-v0.3 | 1.000 | 1.000 | 0.770 | 0.000 | 0.095 | 0.155 | yes |
| Qwen2.5-7B-Instruct | 1.000 | 1.000 | 0.980 | 0.000 | 0.250 | 0.335 | yes |
| Yi-1.5-6B-Chat | 1.000 | 0.905 | 0.710 | 0.005 | 0.050 | 0.085 | yes |
| ChatGLM3-6B | 1.000 | 0.830 | 0.595 | 0.045 | 0.020 | 0.040 | no |

## 当前结论

- 通过 `screen200` 冻结门槛的模型共有 `4` 个：
  - `Orca-2-7B`
  - `Mistral-7B-Instruct-v0.3`
  - `Qwen2.5-7B-Instruct`
  - `Yi-1.5-6B-Chat`
- 未通过的模型共有 `2` 个：
  - `Zephyr-7B-beta`
  - `ChatGLM3-6B`

## 未通过原因简述

- `Zephyr-7B-beta`
  - `format_ok = 0.430` 明显低于门槛
  - `valid_parse = 0.095` 明显低于门槛
  - `truncation_without_answer_rate = 0.105` 略高于门槛
- `ChatGLM3-6B`
  - `runtime_success / format_ok / truncation` 都达标
  - 仅 `valid_parse = 0.595` 比门槛 `0.60` 低 `0.005`

## 下一步建议

- 当前可以进入下一阶段候选池的模型：
  - `Orca-2-7B`
  - `Mistral-7B-Instruct-v0.3`
  - `Qwen2.5-7B-Instruct`
  - `Yi-1.5-6B-Chat`
- `Zephyr-7B-beta` 当前不建议继续作为正式主表模型推进。
- `ChatGLM3-6B` 如需继续争取，可单列为“边界失败模型”，因为它只差 `valid_parse` 的极小幅度。

