# 微博“躺平 / 摆烂 / 佛系”语义研究项目

将原始微博 CSV 整理为可分析语料，并提供围绕“躺平 / 摆烂 / 佛系”的抽样、预标注、分类训练与语义变化分析流程。

这个仓库面向研究和分析工作流，不是简单的关键词计数脚本。它要解决的问题是：同样包含“躺平”“摆烂”“佛系”的微博，哪些是在表达可解释的态度、评价或语义场景，哪些只是标题、口头禅、交易黑话、剧情设定或顺带提及。和只看词频的做法相比，这个仓库把数据整理、人工审核、监督分类和下游语义分析串成了一条可重复执行的流程。当前状态为 `experimental / research workflow`；其中 `broad` 分析链相对更成熟。

## Features

- 用统一入口 [`main.py`](./main.py) 把 `raw/` 下的微博 CSV 转成可复用的 parquet 语料，并维护增量状态。
- 生成适合人工审核的抽样数据，并支持用本地或 API 模型做预标注草稿。
- 支持单标签和 `broad / strict` 双标签 BERT 分类训练。
- 支持将分类器批量应用到全量语料，并继续进入 `07-10` 的 broad 分析链。
- 提供主题建模、关键词语义邻近、概念漂移和报告型 readouts。
- 将语义分桶规则和人工覆盖项放在 [`bert/config/`](./bert/config/) 下，便于后续人工调整。

## Quick Start

环境要求：

- Python 3.11+（仓库默认使用根目录 `.venv`）
- 首次运行需要安装 [`requirements.txt`](./requirements.txt)
- Windows 用户请同时阅读 [`WINDOWS_SETUP.md`](./WINDOWS_SETUP.md)

最短路径是先把主流程跑通：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python main.py status
.venv/bin/python main.py run
.venv/bin/python main.py export-csv text
```

最小可运行结果：

- `main.py run` 会把 `raw/` 中可识别的 CSV 处理到 [`data/processed/text_dedup/`](./data/processed/text_dedup/)
- `main.py export-csv text` 会导出 [`data/exports/text_dedup.csv`](./data/exports/)

如果你只想先确认主流程能不能跑，看 [`USER_MANUAL.md`](./USER_MANUAL.md)。如果你已经有审核数据，想直接训练或分析，看 [`bert/README.md`](./bert/README.md)。

## Installation

macOS / Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

仓库默认约定：

- macOS / Linux 使用 `.venv/bin/python`
- Windows 使用 `.\.venv\Scripts\python.exe`
- 不向全局 Python 环境安装依赖

## Usage

### 1. 运行主流程

查看状态：

```bash
.venv/bin/python main.py status
```

增量处理 `raw/`：

```bash
.venv/bin/python main.py run
```

全量重建主流程产物：

```bash
.venv/bin/python main.py full
```

导出 CSV 供人工检查：

```bash
.venv/bin/python main.py export-csv
.venv/bin/python main.py export-csv merged
.venv/bin/python main.py export-csv text
```

主流程的正式输入是 [`raw/`](./raw/)，主要输出在：

- [`data/processed/merged_dedup/`](./data/processed/merged_dedup/)
- [`data/processed/preprocessed/`](./data/processed/preprocessed/)
- [`data/processed/text_dedup/`](./data/processed/text_dedup/)
- [`data/reports/`](./data/reports/)
- [`data/state/`](./data/state/)

### 2. 抽样、预标注与训练

从主流程产物抽样：

```bash
.venv/bin/python bert/01_stratified_sampling.py \
  --input "data/processed/text_dedup/*.parquet" \
  --output "bert/data/sample.csv" \
  --n 6000 \
  --report_path "bert/data/sampling_report.json"
```

生成 LLM 预标注草稿：

```bash
cp bert/llm_label_local.example.toml bert/llm_label_local.toml
.venv/bin/python bert/02_llm_label_local.py --config bert/llm_label_local.toml
```

将审核表整理为单标签训练输入：

```bash
.venv/bin/python bert/03_normalize_labels.py \
  --input_csv "bert/data/labeled.csv" \
  --output_csv "bert/data/labeled_binary.csv"
```

训练单标签分类器：

```bash
.venv/bin/python bert/04_train_bert_classifier.py \
  --input_csv "bert/data/labeled_binary.csv" \
  --output_dir "bert/artifacts/tangping_bert"
```

训练 `broad / strict` 双标签分类器：

```bash
.venv/bin/python bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/labeled.csv" \
  --base_output_dir "bert/artifacts/dual_label_run"
```

将模型批量应用到全量 parquet：

```bash
.venv/bin/python bert/06_predict_bert_classifier.py \
  --model_dir "bert/artifacts/tangping_bert/best_model" \
  --input_pattern "data/processed/text_dedup/*.parquet" \
  --output_dir "data/processed/text_dedup_predicted_broad"
```

### 3. 运行 broad 分析链

构建 broad 分析底表：

```bash
.venv/bin/python bert/07_build_broad_analysis_base.py \
  --input_pattern "data/processed/text_dedup_predicted_broad/*.parquet" \
  --output_path "bert/artifacts/broad_analysis/analysis_base.parquet"
```

主题建模：

```bash
.venv/bin/python bert/08_topic_model_bertopic.py \
  --input_path "bert/artifacts/broad_analysis/analysis_base.parquet" \
  --output_dir "bert/artifacts/broad_analysis/topic_model_BAAI"
```

关键词语义分析：

```bash
.venv/bin/python bert/09_keyword_semantic_analysis.py \
  --input_path "bert/artifacts/broad_analysis/analysis_base.parquet" \
  --output_dir "bert/artifacts/broad_analysis/semantic_analysis"
```

整理 `09` 的报告型输出：

```bash
.venv/bin/python bert/09_prepare_semantic_midterm.py \
  --semantic_dir "bert/artifacts/broad_analysis/semantic_analysis" \
  --bucket_rules_path "bert/config/semantic_bucket_rules.json" \
  --bucket_overrides_path "bert/config/semantic_bucket_overrides.csv"
```

概念漂移分析：

```bash
.venv/bin/python bert/10_concept_drift_analysis.py \
  --output_dir "bert/artifacts/broad_analysis/drift_analysis"
```

代表性输出包括：

- [`bert/artifacts/broad_analysis/analysis_base.parquet`](./bert/artifacts/broad_analysis/)
- 主题建模默认输出目录 `bert/artifacts/broad_analysis/topic_model_BAAI/`
- [`bert/artifacts/broad_analysis/semantic_analysis/`](./bert/artifacts/broad_analysis/semantic_analysis)
- [`bert/artifacts/broad_analysis/semantic_analysis/readouts/`](./bert/artifacts/broad_analysis/semantic_analysis/readouts)
- [`bert/artifacts/broad_analysis/drift_analysis/`](./bert/artifacts/broad_analysis/drift_analysis)

更完整的参数说明、输入列识别规则和输出文件解释见 [`bert/README.md`](./bert/README.md)。

## Project Structure

```text
result/
├── raw/                              原始微博 CSV 输入
├── data/
│   ├── processed/                    主流程产物
│   │   ├── merged_dedup/
│   │   ├── preprocessed/
│   │   ├── text_dedup/
│   │   └── text_dedup_predicted_broad/
│   ├── exports/                      导出的 CSV
│   ├── reports/                      主流程运行报告
│   └── state/                        增量状态
├── scripts/
│   └── pipeline/                     主流程脚本
├── bert/
│   ├── data/                         抽样表、审核表等工作文件
│   ├── artifacts/                    训练与分析输出
│   ├── config/                       停用词、语义分桶、覆盖规则
│   ├── lib/                          BERT 与分析公共代码
│   ├── scripts/                      辅助脚本
│   └── README.md                     下游流程说明
├── models/                           本地模型目录
├── main.py                           主流程统一入口
├── USER_MANUAL.md                    主流程说明
├── WINDOWS_SETUP.md                  Windows 运行说明
└── README.md
```

## Configuration

### 数据与路径约定

- 原始数据放在 [`raw/`](./raw/)
- 主流程产物默认写入 [`data/processed/`](./data/processed/)
- 抽样与审核工作文件放在 [`bert/data/`](./bert/data/)
- 训练与分析产物放在 [`bert/artifacts/`](./bert/artifacts/)

### LLM 预标注配置

- 模板文件：[`bert/llm_label_local.example.toml`](./bert/llm_label_local.example.toml)
- 建议复制为本地配置文件后再运行：

```bash
cp bert/llm_label_local.example.toml bert/llm_label_local.toml
```

- 预标注脚本支持 `qwen_openai`、`openai_compatible`、`ollama`
- 如果不走配置文件，脚本会尝试从环境变量读取 API key；仓库文档中已出现的变量包括：
  - `DASHSCOPE_API_KEY`
  - `QWEN_API_KEY`
  - `OPENAI_API_KEY`
  - `FIXER_API_KEY`

### 语义分析配置

- 主题停用词：[`bert/config/topic_stopwords.txt`](./bert/config/topic_stopwords.txt)
- 语义分桶规则：[`bert/config/semantic_bucket_rules.json`](./bert/config/semantic_bucket_rules.json)
- 人工覆盖项：[`bert/config/semantic_bucket_overrides.csv`](./bert/config/semantic_bucket_overrides.csv)
- `09` 的报告型输出会额外生成覆盖模板，便于人工校正

### 本地模型

- 本地模型目录在 [`models/`](./models/)
- 训练与预测脚本支持 `cpu`、`cuda`、`mps` 设备参数
- Windows 环境、CUDA 相关说明见 [`WINDOWS_SETUP.md`](./WINDOWS_SETUP.md)

## Workflow

```text
raw CSV
  -> main.py run/full
  -> data/processed/text_dedup
  -> bert/01_stratified_sampling.py
  -> bert/02_llm_label_local.py
  -> 人工审核
  -> bert/03_normalize_labels.py (单标签时)
  -> bert/04_train_bert_classifier.py 或 bert/05_train_dual_label_classifier.py
  -> bert/06_predict_bert_classifier.py
  -> bert/07_build_broad_analysis_base.py
  -> bert/08_topic_model_bertopic.py
  -> bert/09_keyword_semantic_analysis.py
  -> bert/09_prepare_semantic_midterm.py
  -> bert/10_concept_drift_analysis.py
```

边界上可以把仓库理解成两条任务线：

- `broad`：保留关键词承担可解释语义的文本，适合做主题、语义场景和概念漂移分析
- `strict`：只保留更直接表达现实态度、行为方式或评价框架的文本，边界更窄

当前更成熟的分析出口是 `broad` 链。

## Examples

- 主流程说明：[`USER_MANUAL.md`](./USER_MANUAL.md)
- BERT 与分析流程：[`bert/README.md`](./bert/README.md)
- Windows 运行说明：[`WINDOWS_SETUP.md`](./WINDOWS_SETUP.md)
- 抽样示例：[`bert/data/sample_6000.csv`](./bert/data/sample_6000.csv)
- 标注工作文件示例：[`bert/data/sample_1.xlsx`](./bert/data/sample_1.xlsx)
- `08` 当前操作说明：[`bert/08_topic_model_bertopic_当前操作说明.md`](./bert/08_topic_model_bertopic_当前操作说明.md)

## FAQ

### 这是一个可以直接拿来做微博爬取的仓库吗？

不是。这个仓库假定你已经有本地 CSV，并把它们放在 [`raw/`](./raw/) 下。首页重点是语料整理、标注、分类和分析。

### 只跑 `main.py run` 能得到什么？

你会得到可继续抽样和分析的 parquet 语料，核心目录是 [`data/processed/text_dedup/`](./data/processed/text_dedup/)。

### `02_llm_label_local.py` 的输出能直接当最终标签吗？

不建议。这个脚本产出的 `labeled.csv` 是人工审核前的草稿，不应直接当作训练真值。

### `07-10` 依赖哪一步的输出？

默认依赖 [`bert/artifacts/broad_analysis/analysis_base.parquet`](./bert/artifacts/broad_analysis/) 或它上游的 [`data/processed/text_dedup_predicted_broad/`](./data/processed/text_dedup_predicted_broad/)。

### 支持哪些设备？

训练与预测脚本支持 `cpu`、`cuda`、`mps`。Windows 上的环境差异请看 [`WINDOWS_SETUP.md`](./WINDOWS_SETUP.md)。

### 输出文件太多时该先看什么？

主流程先看 [`data/reports/`](./data/reports/)。`09` 语义分析优先看 [`bert/artifacts/broad_analysis/semantic_analysis/readouts/`](./bert/artifacts/broad_analysis/semantic_analysis/readouts) 下的 `semantic_keyword_overview.csv`、`semantic_context_trajectory.csv`、`semantic_context_shift_summary.csv`。

## Roadmap

- TODO: 补充更完整的公开复现实验说明
- TODO: 明确数据与大体量产物的发布边界

## Contributing

欢迎通过 Issue 说明问题、缺陷或研究需求。提交 PR 时建议保持单一改动主题；如果改动会影响流程、目录约定或输出解释，请同步更新：

- [`README.md`](./README.md)
- [`USER_MANUAL.md`](./USER_MANUAL.md)
- [`bert/README.md`](./bert/README.md)
- [`WINDOWS_SETUP.md`](./WINDOWS_SETUP.md)

## License

[MIT](./LICENSE)

## Acknowledgements

本仓库依赖并围绕以下开源工具组织流程：

- [Transformers](https://github.com/huggingface/transformers)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [BERTopic](https://github.com/MaartenGr/BERTopic)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [jieba](https://github.com/fxsjy/jieba)
- [pandas](https://github.com/pandas-dev/pandas)
- [pyarrow](https://github.com/apache/arrow)
