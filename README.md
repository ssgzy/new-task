# 项目总览

## 项目基本信息
- 项目名称：CDS527 Group Assignment
- 项目根目录：`/Users/sam/Documents/Documents - sam的MacBook Pro/LU课程资料/CDS527 Big Data Analytics Language Models/CDS527 group project`
- 执行环境：conda `CDS527`（Python 3.10.19，PySpark 3.1.2，JupyterLab 4.5.3）
- 当前核心输入：
  - `Group_Project_2026_T2 .docx`（作业说明）
  - `GAI Declaration Sheet (sample).docx`
  - `Google.docx`（Task 2 素材）
  - `SMILE Twitter Emotion Dataset.docx`（数据说明）
  - `smile-annotations-final.csv`（Task 1 数据，1299 行）
  - `PySpark.png`（官方实验大纲图）

## 导航
- [[当前状态]]
- [[实验协议]]
- [[任务笔记 - Task1 内容梳理]]
- [[数据说明]]
- [[结果索引]]
- [[问题与坑点]]
- [[下一步]]
- [[项目规范 - 执行流程]]
- [[任务笔记 - 阅读作业要求]]
- [[常用命令]]
- [[会话日志]]

## 当前目标
- **Task 1 Notebook 主线实验全部完成** ✅
- 待完成：`Group_X.report.docx`（Task 2 case study）、`Group_X.present.pptx`、`Group_X.gai.docx`、`work_distribution.docx`

## 执行总纲（Step 进度）
| Step | 内容 | 状态 |
|------|------|------|
| 1 | 项目现状检查 | ✅ 完成 |
| 2 | 文档体系梳理与补齐 | ✅ 完成 |
| 3 | 环境检查 | ✅ 完成 |
| 4 | 数据读取与审查 | ✅ 完成 |
| 5 | 固定实验协议 | ✅ 完成 |
| 6 | Baseline（TF-IDF + LR，macro-F1=0.2337） | ✅ 完成 |
| 7 | 模型比较（Section 2，最优 CNB=0.3332） | ✅ 完成 |
| 8 | 表示方法比较（Section 3，固定 LR 设置下 BOW 系列无差异） | ✅ 完成 |
| 9 | 改进方法与深度分析（Section 4，最优 LR+Weight rp=0.5，macro-F1=0.3453） | ✅ 完成 |

## 最终实验结果摘要
| Section | 最佳配置 | macro-F1 |
|---------|---------|---------|
| S1 Baseline | TF-IDF unigram + LR | 0.2337 |
| S2 Model Comparison | TF-IDF unigram + CNB | 0.3332 |
| S3 Repr Comparison | 固定 LR 下 BOW 系列无差异 | 0.2337 |
| **S4 Improvement** | **LR + 逆频率类别权重 (rp=0.5)** | **0.3453** ← 全项目最优 |

## 当前目录结构
- `文档/`：Obsidian 主笔记区
- `日志/`：过程日志（`会话日志.md`）
- `输出/figures/`：10 张可视化图（fig1–fig10）
- `输出/reports/`：全部文本报告（EDA + S1–S4）
- `输出/data/`：3 个 CSV（S2/S3/S4 结果）
- `工作区/Group_X.code.ipynb`：Task 1 主 Notebook（Section 0–9 全部完成）
- `tmp/docs/`：DOCX 临时提取结果
