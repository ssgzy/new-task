# validation883 最终汇总

- 生成时间：2026-04-08 17:18 HKT
- 运行目录：`outputs/provisional/validation883_assigned/`
- 主准确率指标：`Strict TM`
- 说明：
  - strict 为主线结果
  - relaxed 仅作为补充上界
  - `relaxed_gap_tm = relaxed_tm - strict_tm`

| 模型 | Strict TM | Strict EM | Relaxed TM | Gap | FormatOK | ValidParse | Truncation | AvgLatency(ms) | tok/s | MeanOutputTokens | 当前定位 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 0.359003 | 0.225368 | 0.360136 | 0.001133 | 0.997735 | 0.981880 | 0.000000 | 3369.95 | 2.4163 | 8.14 | 正式主表 |
| Mistral-7B-Instruct-v0.3 | 0.166478 | 0.093998 | 0.194790 | 0.028313 | 0.998867 | 0.778029 | 0.000000 | 4628.89 | 2.7585 | 12.77 | 正式主表 |
| Yi-1.5-6B-Chat | 0.099660 | 0.054360 | 0.152888 | 0.053228 | 0.898075 | 0.742922 | 0.005663 | 4341.07 | 4.2706 | 18.54 | 正式主表 |
| Orca-2-7B | 0.065685 | 0.036240 | 0.080408 | 0.014723 | 0.950170 | 0.838052 | 0.026048 | 5180.55 | 4.6251 | 23.96 | 正式主表 |
| ChatGLM3-6B | 0.014723 | 0.007928 | 0.027180 | 0.012458 | 0.380521 | 0.253681 | 0.600227 | 21112.13 | 7.5357 | 159.09 | 失败案例 / 附录 |

## 当前结论

- 按 `Strict TM` 排序：
  - `Qwen2.5-7B-Instruct`
  - `Mistral-7B-Instruct-v0.3`
  - `Yi-1.5-6B-Chat`
  - `Orca-2-7B`
  - `ChatGLM3-6B`
- 当前正式主表建议保留：
  - `Qwen2.5-7B-Instruct`
  - `Mistral-7B-Instruct-v0.3`
  - `Yi-1.5-6B-Chat`
  - `Orca-2-7B`
- `ChatGLM3-6B` 在 `validation883` 上出现显著退化：
  - `format_ok = 0.380521`
  - `valid_parse = 0.253681`
  - `truncation_without_answer_rate = 0.600227`
  - 因此当前不再保留其 `boundary-pass candidate` 叙述
