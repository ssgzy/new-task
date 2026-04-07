# 组员运行说明_validation883

## 目标
- 这份文档面向组员，目标是：
  - 拉代码
  - 建环境
  - 下载或检查自己负责的模型
  - 直接在终端运行 `validation883`
- 原则：
  - 尽量不要改代码
  - 不要改 parser / prompt / decode config
  - 结果默认写到 `provisional`
  - 不污染 canonical 目录

## 一次性准备

### 1. 拉代码
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd Distillation/new\ task
```

### 2. 创建并激活 conda 环境
```bash
conda create -n Lion python=3.11.14 -y
conda activate Lion
```

### 3. 安装依赖
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-validation883.txt
```

## 检查模型 cache / registry

### 1. 只检查本地是否已有模型
```bash
python scripts/ensure_candidate_models.py --labels "Qwen2.5-7B-Instruct" --local-only --provisional
```

### 2. 如果本地没有，再下载
```bash
python scripts/ensure_candidate_models.py --labels "Qwen2.5-7B-Instruct" --download-missing --provisional
```

### 3. 查看生成的 provisional registry
```bash
cat outputs/metadata/provisional/model_registry.labels-qwen2-5-7b-instruct.json
```

## 直接运行 validation883

### 兼容旧 shell wrapper 的默认命令
```bash
bash scripts/run_assigned_model_v1.sh --label "Qwen2.5-7B-Instruct" --download-model
```

### 推荐命令：使用项目 wrapper
```bash
python scripts/run_validation883_assigned_v1.py --label "Qwen2.5-7B-Instruct" --resume-existing
```

### 如果模型还没下载，可让 wrapper 一起下载
```bash
python scripts/run_validation883_assigned_v1.py --label "Qwen2.5-7B-Instruct" --download-missing --resume-existing
```

### 结果目录
- 该命令会把结果写到：
  - `outputs/provisional/validation883_assigned/Qwen2.5-7B-Instruct/`
- 其中至少包括：
  - `summary.json`
  - `predictions.jsonl`
  - `report.csv`
- 说明：
  - `scripts/run_assigned_model_v1.sh` 现在默认也会落到同一条 `validation883` 路径
  - 只有显式加 `--legacy-screen-only` 才会回退到旧的 `screen200` 流程

## 如需手动运行底层 benchmark

### 1. 先确保 provisional registry 已生成
```bash
python scripts/ensure_candidate_models.py --labels "Qwen2.5-7B-Instruct" --provisional
```

### 2. 再运行底层脚本
```bash
python scripts/run_finqa_local_benchmark_v1.py \
  --model-label "Qwen2.5-7B-Instruct" \
  --model-path "/Users/<YOU>/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<SNAPSHOT_ID>" \
  --manifest data/manifests/validation883.jsonl \
  --output-jsonl outputs/provisional/validation883_assigned/Qwen2.5-7B-Instruct/predictions.jsonl \
  --summary-json outputs/provisional/validation883_assigned/Qwen2.5-7B-Instruct/summary.json \
  --max-new-tokens 256 \
  --resume-existing
```

## 如果要复查 screen200

### 单模型 screen200
```bash
python scripts/run_qualification_v1.py \
  --labels "Qwen2.5-7B-Instruct" \
  --model-registry outputs/metadata/provisional/model_registry.labels-qwen2-5-7b-instruct.json \
  --run-root outputs/provisional/screen200_member_check \
  --report-csv outputs/provisional/screen200_member_check/reports/qwen2-5-7b-instruct.csv \
  --max-new-tokens 256 \
  --resume \
  --screen-only \
  --provisional
```

## 不要改的内容
- 不要改 `parser v1`
- 不要改 `prompt system/user`
- 不要改 `max_new_tokens = 256`
- 不要改 `temperature / top_p / do_sample`
- 不要把结果写到 canonical `outputs/qualification_summary.csv`
- 不要手改 `model_registry.json` 的 canonical 版本

## 常见故障排查

### 1. 模型下载失败
- 先确认网络和 Hugging Face 登录状态。
- gated repo 需要先在 Hugging Face 网页上同意 license。
- 已知 gated：
  - `Llama-3.1-8B-Instruct`
  - `Gemma-7B-IT`

### 2. 断点续跑
- 直接在原命令后追加或保留：
```bash
--resume-existing
```
- 如果是 `run_qualification_v1.py`，使用：
```bash
--resume
```

### 3. 输出目录不一致
- validation883 请统一写到：
  - `outputs/provisional/validation883_assigned/<模型名>/`
- screen200 请统一写到：
  - `outputs/provisional/<你自己的run_root>/`

### 4. ChatGLM3 相关问题
- `ChatGLM3-6B` 需要项目里现有的 `trust_remote_code` / generation compatibility patch。
- 不要自行改成别的 loader 路径。
- 直接使用项目脚本，不要自己另写 notebook loader。

### 5. Orca tokenizer 问题
- `Orca-2-7B` 当前固定使用 slow tokenizer 语义。
- 环境里需要：
  - `protobuf==3.20.3`
  - `sentencepiece==0.1.99`
  - `tiktoken==0.12.0`

## GitHub 协作说明

### 如何把项目上传到 GitHub
```bash
git init
git remote add origin <YOUR_GITHUB_REPO_URL>
git add .
git commit -m "init finqa benchmark workspace"
git push -u origin main
```

### 组员如何 clone 后直接运行
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd Distillation/new\ task
conda activate Lion
python scripts/run_validation883_assigned_v1.py --label "Qwen2.5-7B-Instruct" --resume-existing
```

### 哪些文件不要改
- `scripts/finqa_protocol_v1.py`
- `scripts/run_finqa_local_benchmark_v1.py`
- `scripts/run_qualification_v1.py`
- `scripts/relaxed_scoring.py`
- `docs/实验协议_v1.md`

### 结果应回传到哪些目录
- 组员只负责一个模型时，统一回传：
  - `outputs/provisional/validation883_assigned/<模型名>/summary.json`
  - `outputs/provisional/validation883_assigned/<模型名>/predictions.jsonl`
  - `outputs/provisional/validation883_assigned/<模型名>/report.csv`

### 建议的单模型命令
```bash
python scripts/run_validation883_assigned_v1.py --label "Yi-1.5-6B-Chat" --resume-existing
```

### 如果组员跑完了，应交回哪些文件
- `summary.json`
- `predictions.jsonl`
- `report.csv`
- 终端日志

### 保存终端日志的建议命令
```bash
python scripts/run_validation883_assigned_v1.py --label "Yi-1.5-6B-Chat" --resume-existing 2>&1 | tee outputs/provisional/validation883_assigned/Yi-1.5-6B-Chat/run.log
```

## 关联文档
- [[实验协议_v1]]
- [[评分逻辑与报告写作指南]]
- [[准入筛选总结]]
- [[当前主表建议]]
- [[模型注册表]]
