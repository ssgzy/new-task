# Chart-Ready 导出说明

- `chart_accuracy.csv`：画主结果图，推荐柱状图
- `chart_format.csv`：画格式/可解析/截断三指标对照图
- `chart_efficiency.csv`：画延迟、吞吐、显存、输出长度图
- `chart_screen_vs_validation.csv`：画 `screen200` 与 `validation883` 对照，特别适合讲 `ChatGLM3-6B` 的退化

## 推荐图
- 图 1：`chart_accuracy.csv` 中的 `strict_tm`
- 图 2：`chart_accuracy.csv` 中的 `strict_tm` vs `relaxed_tm`
- 图 3：`chart_format.csv` 中的 `format_ok / valid_parse / truncation_without_answer_rate`
- 图 4：`chart_efficiency.csv` 中的 `avg_latency_ms` 或 `peak_vram_gb`
- 图 5：`chart_screen_vs_validation.csv` 中 `ChatGLM3-6B` 的 `screen200_valid_parse` vs `validation883_valid_parse`
