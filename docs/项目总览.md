# 项目总览

## 项目目标
- 在全新项目目录中重建 FinQA benchmark 流程，避免继承旧项目结构、旧脚本假设和旧实验残留。
- 交付四个硬性结果：FinQA 全量原始 split、`protocol_v1`、`calibration_report.csv`、`qualification_summary.csv`。
- 全程同步维护 Obsidian 文档、日志、命令记录和结果索引，确保结果可追踪。

## 当前工作目录
- 项目根目录：`/Users/sam/Documents/Documents - sam的MacBook Pro/论文📑/Distillation/new task`
- Python 环境：`conda` 的 `Lion`

## 目录结构
- `docs/`：说明性文档与实验协议
- `logs/`：会话日志与日期日志
- `outputs/`：报表、汇总结果、后续 benchmark 输出
- `data/`：FinQA 原始数据、标准化 manifest、后续缓存
- `scripts/`：数据处理、协议实现、评测脚本

## 核心文档导航
- 状态入口：[[当前状态]]
- 本轮任务：[[任务笔记 - FinQA 基准重建]]
- 实验协议：[[实验协议_v1]]
- 数据记录：[[数据重拉与字段核对]]
- 输入接口诊断：[[输入接口诊断]]
- 输入包装修复过程总结：[[过程总结_输入包装修复]]
- 长度校准：[[长度校准报告]]
- 准入筛选：[[准入筛选总结]]
- 模型状态：[[模型注册表]]
- 组员运行：[[组员运行说明]]
- validation883 组员运行：[[组员运行说明_validation883]]
- 模型谱系：[[模型谱系与主表角色说明]]
- 评分与写作：[[评分逻辑与报告写作指南]]
- 主表建议：[[当前主表建议]]
- 异常与决策：[[异常与决策记录]]
- Claude 建议核查：[[Claude建议核查]]
- 日志入口：[[会话日志]]
- 常用命令：[[常用命令]]
- 下一步：[[下一步]]

## 模型分组
### 正式主表（参与 student selection）
- Lion-7B
- Orca-2-7B
- Zephyr-7B-beta
- MiniLLM-Llama-7B

### 扩展表（appendix / supplementary）
- DeepSeek-R1-Distill-Qwen-7B
- OpenR1-Distill-7B

## 强约束
- 严禁使用 `test1147` 做任何模型筛选、prompt 调整、token 长度选择、parser 修补或 student selection。
- 先做字段核对，不默认相信旧脚本或旧 parser。
- 冻结统一 prompt / parser / decode config，不为单个模型写特殊规则。

## 非 markdown 资料
- `Lion-7B 之后公开 7B 蒸馏模型与论文梳理.pdf`：当前目录中已有的外部参考 PDF，后续如读取会在 [[会话日志]] 和相关任务 note 中记录用途。
