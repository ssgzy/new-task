# validation883结果字段说明

## 用途
- 本文档用于解释 `outputs/provisional/validation883_assigned/validation883_merged_results.csv` 中每一列的含义。
- 这张表是当前 `5` 个模型在 `validation883` 上的合并结果表，适合直接做论文结果表、组会汇报和 `NotebookLM` 图表输入。

## 表头总览
- `模型`
- `runtime_success`
- `format_ok`
- `valid_parse`
- `strict_em`
- `strict_tm`
- `relaxed_em`
- `relaxed_tm`
- `relaxed_gap_tm`
- `truncation_without_answer_rate`
- `avg_latency_ms`
- `tok_per_sec`
- `peak_vram`
- `mean_output_tokens`

## 每一列是什么意思

### 模型
- 含义：
  - 当前这一行对应的模型名称。
- 当前表中共有 `5` 个模型：
  - `Qwen2.5-7B-Instruct`
  - `Mistral-7B-Instruct-v0.3`
  - `Yi-1.5-6B-Chat`
  - `Orca-2-7B`
  - `ChatGLM3-6B`

### runtime_success
- 含义：
  - 在 `883` 条样本上，运行层面成功完成并产生输出的比例。
- 取值解释：
  - `1.0` 表示全部样本都成功跑完，没有中途 runtime crash。
  - 低于 `1.0` 表示有样本在推理过程中失败、报错或没有产出可记录结果。
- 这次结果里怎么读：
  - `5` 个模型都是 `1.0`，说明这轮 `validation883` 在运行层面都完整结束了。
- 注意：
  - 这个指标只说明“程序有没有跑完”，不说明答案质量。

### format_ok
- 含义：
  - 输出里是否出现了符合协议的 `Answer:` 行。
- 取值解释：
  - 越接近 `1` 越好。
  - 高说明模型愿意按协议写出最终答案行。
  - 低说明模型常常不肯收束到 `Answer:`，或者输出风格明显偏离协议。
- 这次结果里怎么读：
  - `Qwen2.5 = 0.997735`，几乎每条都按协议落到 `Answer:`。
  - `Mistral = 0.998867`，也非常稳定。
  - `Yi = 0.898075`，多数能落到 `Answer:`，但明显不如前两者稳定。
  - `Orca = 0.950170`，整体还可以，但仍有一部分样本没落到答案行。
  - `ChatGLM3 = 0.380521`，大多数样本根本没有形成可接受的 `Answer:` 行。

### valid_parse
- 含义：
  - 严格 parser 能否把 `Answer:` 行解析成合法纯数值。
- 取值解释：
  - 越接近 `1` 越好。
  - 它比 `format_ok` 更严格，因为不只是“写了 `Answer:`”，还要求后面真的是可评分数值。
- 这次结果里怎么读：
  - `Qwen2.5 = 0.981880`，几乎所有答案行都能被 strict parser 正常解析。
  - `Mistral = 0.778029`，说明它虽然经常写 `Answer:`，但不少答案行格式还不够干净。
  - `Yi = 0.742922`，也存在明显格式损失。
  - `Orca = 0.838052`，比 `Yi` 和 `Mistral` 更稳一些，但仍不是高稳定协议模型。
  - `ChatGLM3 = 0.253681`，说明即使有输出，真正能进入 strict 评分的比例也很低。
- 和 `format_ok` 的关系：
  - `format_ok` 高但 `valid_parse` 明显低，往往意味着模型“愿意写答案行，但写得不够规整”。

### strict_em
- 含义：
  - 在 strict parser 成功的前提下，预测答案与 gold 完全一致的比例。
- 取值解释：
  - 这是最严格的命中标准。
  - 只要数值格式、精度或表达方式稍有差异，就可能不算命中。
- 这次结果里怎么读：
  - `Qwen2.5 = 0.225368`，说明它有较高的完全精确命中能力。
  - `Mistral = 0.093998`
  - `Yi = 0.054360`
  - `Orca = 0.036240`
  - `ChatGLM3 = 0.007928`
- 报告里怎么用：
  - 它适合作为保守下界，不适合作为主排名指标。

### strict_tm
- 含义：
  - 在 strict parser 成功的前提下，预测答案与 gold 在容差内匹配的比例。
- 取值解释：
  - 这是当前项目的主准确率指标。
  - 比 `strict_em` 更适合金融数值问答，因为允许合理的数值容差。
- 这次结果里怎么读：
  - `Qwen2.5 = 0.359003`
  - `Mistral = 0.166478`
  - `Yi = 0.099660`
  - `Orca = 0.065685`
  - `ChatGLM3 = 0.014723`
- 当前排序：
  - `Qwen2.5 > Mistral > Yi > Orca > ChatGLM3`
- 报告里怎么用：
  - 这是主表排序和主结论最应该看的列。

### relaxed_em
- 含义：
  - 用 relaxed 提取规则后，预测值与 gold 完全相等的比例。
- 取值解释：
  - 它是宽松版本的 EM。
  - 用于估计“模型可能算对了，但 strict 格式没有完全交对卷”的情况。
- 这次结果里怎么读：
  - `Qwen2.5 = 0.225368`，与 `strict_em` 几乎一样，说明几乎没有额外隐藏收益。
  - `Mistral = 0.105323`，比 strict 更高，说明有一部分结果是“算对但 strict 格式损失”。
  - `Yi = 0.083805`
  - `Orca = 0.044168`
  - `ChatGLM3 = 0.019253`

### relaxed_tm
- 含义：
  - 用 relaxed 提取规则后，预测值与 gold 在容差内匹配的比例。
- 取值解释：
  - 它不是主指标，而是“推理能力上界”的补充视角。
- 这次结果里怎么读：
  - `Qwen2.5 = 0.360136`
  - `Mistral = 0.194790`
  - `Yi = 0.152888`
  - `Orca = 0.080408`
  - `ChatGLM3 = 0.027180`
- 如何理解：
  - 如果 `relaxed_tm` 只比 `strict_tm` 高一点，说明问题主要不是格式。
  - 如果 `relaxed_tm` 比 `strict_tm` 高很多，说明模型存在明显“算到了但没按 strict 方式交卷”的情况。

### relaxed_gap_tm
- 含义：
  - `relaxed_tm - strict_tm`
- 取值解释：
  - 越大，说明被格式问题、答案落点问题、收束问题吃掉的准确率越多。
- 这次结果里怎么读：
  - `Qwen2.5 = 0.001133`
  - `Mistral = 0.028313`
  - `Yi = 0.053228`
  - `Orca = 0.014723`
  - `ChatGLM3 = 0.012458`
- 结论：
  - `Qwen2.5` 的 strict 和 relaxed 几乎重合，说明它主要反映真实能力，不太受格式损失影响。
  - `Yi` 的 gap 最大，说明它有不少“潜在算对但格式没交好”的样本。
  - `Mistral` 也有较明显格式损失。

### truncation_without_answer_rate
- 含义：
  - 输出疑似被截断，而且直到结尾仍没有有效 `Answer:` 行的比例。
- 取值解释：
  - 越低越好。
  - 高说明模型经常先长篇展开，最终在 `max_new_tokens=256` 内没收束到可评分答案。
- 这次结果里怎么读：
  - `Qwen2.5 = 0.000000`
  - `Mistral = 0.000000`
  - `Yi = 0.005663`
  - `Orca = 0.026048`
  - `ChatGLM3 = 0.600227`
- 结论：
  - `Qwen2.5` 和 `Mistral` 的输出长度控制最稳。
  - `Orca` 仍有一部分样本会进入长推理后不收束。
  - `ChatGLM3` 的主要失败点之一就是严重截断。

### avg_latency_ms
- 含义：
  - 单样本平均延迟，单位毫秒。
- 取值解释：
  - 越低越快。
- 这次结果里怎么读：
  - `Qwen2.5 = 3369.95 ms`
  - `Mistral = 4628.89 ms`
  - `Yi = 4341.07 ms`
  - `Orca = 5180.55 ms`
  - `ChatGLM3 = 21112.13 ms`
- 结论：
  - `Qwen2.5` 既是最强结果，也是平均延迟最低的一档。
  - `ChatGLM3` 的平均延迟远高于其它模型。

### tok_per_sec
- 含义：
  - 每秒生成 token 数。
- 取值解释：
  - 越高越快。
- 这次结果里怎么读：
  - `Qwen2.5 = 2.4163`
  - `Mistral = 2.7585`
  - `Yi = 4.2706`
  - `Orca = 4.6251`
  - `ChatGLM3 = 7.5357`
- 注意：
  - 这个指标不能单独看。
  - 例如 `ChatGLM3` 的 `tok/s` 最高，但它输出极长且大面积跑偏，所以整体延迟仍然最差。

### peak_vram
- 含义：
  - 本轮运行峰值显存占用，单位是字节。
- 取值解释：
  - 越低越省资源。
- 这次结果里怎么读：
  - `Qwen2.5 = 15232224512`
  - `Mistral = 14496533760`
  - `Yi = 12123144960`
  - `Orca = 13477529088`
  - `ChatGLM3 = 14306105856`
- 实际口径：
  - 可以粗略看成 `12GB - 15GB` 这一档。
  - `Yi` 是这组里峰值显存最低的。

### mean_output_tokens
- 含义：
  - 平均每条样本生成了多少输出 token。
- 取值解释：
  - 越低通常说明模型越容易快速收束到最终答案。
  - 过高往往意味着解释过长、格式不收束或重复展开。
- 这次结果里怎么读：
  - `Qwen2.5 = 8.14`
  - `Mistral = 12.77`
  - `Yi = 18.54`
  - `Orca = 23.96`
  - `ChatGLM3 = 159.09`
- 结论：
  - `Qwen2.5` 的输出最短也最稳。
  - `Orca` 和 `Yi` 输出更长，说明更容易进入解释或铺垫。
  - `ChatGLM3` 输出异常长，这和它的高截断率、低格式稳定性是一致的。

## 这张表应该怎么读

### 先看哪几列
- 第一优先：
  - `strict_tm`
- 第二优先：
  - `format_ok`
  - `valid_parse`
  - `truncation_without_answer_rate`
- 第三优先：
  - `relaxed_tm`
  - `relaxed_gap_tm`
- 最后再看：
  - `avg_latency_ms`
  - `tok_per_sec`
  - `peak_vram`
  - `mean_output_tokens`

### 为什么主排序看 strict_tm
- 因为这轮 benchmark 的目标不是“模型脑子里会不会算”，而是“在冻结协议下，它能不能交出可评分的正确答案”。
- `strict_tm` 同时要求：
  - 有答案行
  - 能被 strict parser 解析
  - 数值与 gold 匹配

### relaxed_tm 该怎么讲
- 它不应该替代主结果。
- 它更像一条补充说明：
  - 如果模型 `relaxed_tm` 明显高于 `strict_tm`，说明模型部分能力被格式问题掩盖了。
- 本轮最典型的是：
  - `Yi`
  - `Mistral`

## 当前五个模型一句话解读
- `Qwen2.5-7B-Instruct`
  - 当前最强且最稳，准确率、格式稳定性和截断控制都最好。
- `Mistral-7B-Instruct-v0.3`
  - 主结果第二，但存在一定 strict 格式损失，relaxed 提示其潜在能力略高于 strict 表现。
- `Yi-1.5-6B-Chat`
  - 主结果第三，格式和收束性明显弱于 `Qwen2.5`/`Mistral`，但 relaxed gap 较大，说明有一部分被格式吃掉。
- `Orca-2-7B`
  - 能跑通且协议上基本可用，但准确率和截断控制都明显不如前三者。
- `ChatGLM3-6B`
  - 在 `validation883` 上出现明显格式和截断退化，不再适合作为正式主表模型。

## 报告中怎么引用这张表
- 如果老师只看一眼：
  - 看 `strict_tm` 排名
- 如果老师会追问“是不是格式问题造成的”：
  - 看 `relaxed_tm` 和 `relaxed_gap_tm`
- 如果老师会追问“为什么 ChatGLM3 掉这么多”：
  - 看 `format_ok`、`valid_parse`、`truncation_without_answer_rate`、`mean_output_tokens`
- 如果老师会追问“部署和复现成本”：
  - 看 `avg_latency_ms`、`tok_per_sec`、`peak_vram`

## 关联文件
- `outputs/provisional/validation883_assigned/validation883_merged_results.csv`
- `outputs/provisional/validation883_assigned/validation883_merged_results.md`
- `outputs/provisional/validation883_assigned/validation883_final_summary.md`
- [[评分逻辑与报告写作指南]]
- [[当前主表建议]]
- [[组会汇报资料清单]]
