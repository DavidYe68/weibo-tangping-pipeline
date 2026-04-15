# 微博“躺平 / 摆烂 / 佛系”语义研究项目

本项目用于构建和分析一个围绕“躺平 / 摆烂 / 佛系”的微博语料库。完整流程包括：

1. 从原始 CSV 中整理语料。
2. 对微博文本做去重和预处理。
3. 分层抽样并进行人工审核。
4. 训练单标签或双标签 BERT 分类器。
5. 将分类器应用到全量语料。
6. 对研究语料做主题、关键词语义邻域和概念漂移分析。

项目的核心目标不是简单统计关键词出现次数，而是尽量识别这些词在微博正文中是否承担了与研究相关的语义功能，并进一步分析它们在不同时间段中的语义分布和变化。

## 研究对象

当前流程默认围绕以下关键词组织：

- `躺平`
- `摆烂`
- `佛系`

原始数据中的文件路径需要能够反映关键词，项目会从目录结构中提取 `keyword` 元信息。

## 项目结构

可以把整个项目理解成五层：

```text
result/
├── raw/                         原始微博 CSV
├── data/                        主流程生成的中间数据与导出结果
│   ├── processed/               parquet 主产物
│   ├── exports/                 导出的 CSV
│   ├── reports/                 运行报告
│   └── state/                   增量处理状态
├── scripts/pipeline/            原始数据整理与预处理脚本
├── bert/                        抽样、标注、训练、预测、分析脚本
│   ├── 01_stratified_sampling.py
│   ├── 02_llm_label_local.py
│   ├── 03_normalize_labels.py
│   ├── 04_train_bert_classifier.py
│   ├── 05_train_dual_label_classifier.py
│   ├── 06_predict_bert_classifier.py
│   ├── 07_build_broad_analysis_base.py
│   ├── 08_topic_model_bertopic.py
│   ├── 09_keyword_semantic_analysis.py
│   ├── 10_concept_drift_analysis.py
│   ├── lib/                     BERT 与分析阶段的公共函数
│   ├── data/                    抽样表、审核表等人工处理中间文件
│   └── artifacts/               模型、预测结果、分析结果
├── main.py                      主流程统一入口
├── README.md                    项目首页说明
├── bert/README.md               BERT 与 07-10 分析说明
├── USER_MANUAL.md               更完整的使用手册
└── WINDOWS_SETUP.md             Windows 环境配置说明
```

如果按实验顺序来理解：

- `raw/` 是输入。
- [`main.py`](/Users/apple/Local/fdurop/code/result/main.py) 和 [`scripts/pipeline/`](/Users/apple/Local/fdurop/code/result/scripts/pipeline) 负责把原始 CSV 变成可抽样的 `data/processed/text_dedup/`。
- [`bert/01_stratified_sampling.py`](/Users/apple/Local/fdurop/code/result/bert/01_stratified_sampling.py) 到 [`bert/05_train_dual_label_classifier.py`](/Users/apple/Local/fdurop/code/result/bert/05_train_dual_label_classifier.py) 负责抽样、预标注、人工审核后的训练。
- [`bert/06_predict_bert_classifier.py`](/Users/apple/Local/fdurop/code/result/bert/06_predict_bert_classifier.py) 负责把训练好的模型打到全量语料。
- [`bert/07_build_broad_analysis_base.py`](/Users/apple/Local/fdurop/code/result/bert/07_build_broad_analysis_base.py) 到 [`bert/10_concept_drift_analysis.py`](/Users/apple/Local/fdurop/code/result/bert/10_concept_drift_analysis.py) 负责研究分析。

## 环境准备

仓库根目录默认使用 `.venv` 作为 Python 虚拟环境。

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要在 Windows + NVIDIA 环境上训练或预测，还需要安装对应 CUDA 版本的 PyTorch。可参考 [`WINDOWS_SETUP.md`](/Users/apple/Local/fdurop/code/result/WINDOWS_SETUP.md)。

## 原始数据约定

原始数据放在 `raw/` 下，脚本会递归扫描满足下面模式的 CSV：

`.../csv/{keyword}/**/*.csv`

例如：

- `raw/1/csv/躺平/2024/1/1.csv`
- `raw/2/csv/%23佛系%23/2025/6/8.csv`

项目会从路径中提取：

- `keyword`
- `source_file`
- 可能的年月信息

## 主流程：原始数据到可抽样语料

统一入口是 [`main.py`](/Users/apple/Local/fdurop/code/result/main.py)。

常用命令：

```bash
python main.py run
python main.py full
python main.py status
python main.py export-csv
python main.py export-csv merged
python main.py export-csv text
```

### `run`

增量处理原始数据，主要完成三件事：

1. 合并原始 CSV 并按 `id` 去重。
2. 对新增文本做清洗和规范化。
3. 按 `cleaned_text` 进一步做文本去重。

### `full`

从头重建整个主流程产物。适用于：

- 清洗逻辑变化后需要重算。
- 状态文件失效。
- 需要重新生成全部 parquet。

### `status`

查看当前原始文件索引、去重结果、预处理产物和最近一次运行状态。

### 主流程关键输出

- `data/processed/merged_dedup/`
- `data/processed/preprocessed/`
- `data/processed/text_dedup/`

其中 `data/processed/text_dedup/` 是后续抽样、训练和分析最常用的输入。

## 实验流程

### 01. 分层抽样

从 `text_dedup` 中抽取人工审核样本。默认会结合时间、关键词和文本长度做分层。

```bash
python bert/01_stratified_sampling.py \
  --input "data/processed/text_dedup/*.parquet" \
  --output "bert/data/sample_review.csv" \
  --n 6000 \
  --report_path "bert/data/sampling_report.json"
```

常见输出：

- `bert/data/sample_review.csv`
- `bert/data/sampling_report.json`

### 02. LLM 预标注

对抽样语料进行预标注。这里的输出只是待审核草稿，不能直接视为最终训练数据。

```bash
python bert/02_llm_label_local.py \
  --input "bert/data/sample_review.csv" \
  --output "bert/data/sample_prelabel.csv" \
  --report_path "bert/data/labeling_report.json"
```

运行前如果需要本地配置，可复制：

```bash
cp bert/llm_label_local.example.toml bert/llm_label_local.toml
```

随后应进行人工审核，形成最终可训练的数据表。

### 03. 标签整理

如果你的训练任务是单标签二分类，可以先用 `03` 把审核结果统一整理成标准二值标签：

```bash
python bert/03_normalize_labels.py \
  --input_csv "bert/data/reviewed.csv" \
  --output_csv "bert/data/reviewed_binary.csv"
```

这一步主要服务于单标签训练；如果你已经准备好了 `broad / strict` 两列，可以直接进入 `05`。

### 04. 单标签 BERT 训练

适用于只有一套二分类标签的情况。

```bash
python bert/04_train_bert_classifier.py \
  --input_csv "bert/data/reviewed_binary.csv" \
  --output_dir "bert/artifacts/single_label_run"
```

### 05. 双标签 BERT 训练

适用于同时存在 `broad` 和 `strict` 两套标签的情况。脚本会分别训练两套模型，并输出并排对照结果。

```bash
python bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_part1.csv" "bert/data/reviewed_part2.csv" \
  --base_output_dir "bert/artifacts/dual_label_run"
```

如果你想把某些文件固定留作测试集：

```bash
python bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_pool.csv" \
  --test_path "bert/data/reviewed_holdout.csv" \
  --base_output_dir "bert/artifacts/dual_label_holdout"
```

常见输出：

- `bert/artifacts/.../broad/`
- `bert/artifacts/.../strict/`
- `bert/artifacts/.../compare/`
- `bert/artifacts/.../inspect/`
- `bert/artifacts/.../run_overview.md`

更细的训练说明见 [`bert/README.md`](/Users/apple/Local/fdurop/code/result/bert/README.md)。

### 06. 全量预测

训练完成后，可以把模型应用到 `text_dedup` 的全量语料上。

如果后续要进入 `07-10`，推荐直接使用 `broad` 模型，并把输出目录写成 `data/processed/text_dedup_predicted_broad`：

```bash
python bert/06_predict_bert_classifier.py \
  --model_dir "bert/artifacts/dual_label_run/broad/best_model" \
  --input_pattern "data/processed/text_dedup/*.parquet" \
  --output_dir "data/processed/text_dedup_predicted_broad" \
  --device cuda
```

常见输出列包括：

- `pred_label`
- `pred_label_text`
- `pred_prob_1`
- `pred_prob_0`
- `pred_confidence`

### 07. 构建分析底表

将 `06` 的预测结果整理为统一分析表，默认只保留预测为正样本的语料。脚本会同时规范化时间、关键词和 IP 属地；缺失 IP 不会被丢弃，而是统一记为 `UNKNOWN_IP`。

```bash
python bert/07_build_broad_analysis_base.py
```

如果你在 `06` 中使用了其他输出目录，则需要同步修改 `--input_pattern`。

输出位置：

- `bert/artifacts/broad_analysis/analysis_base.parquet`
- `bert/artifacts/broad_analysis/analysis_base_report.json`

其中 `analysis_base_report.json` 会额外汇总：

- `rows_by_ip`
- `rows_by_period_and_ip`
- 缺失 IP 的数量和占比

### 08. 主题模型分析

在 `07` 的底表上运行 BERTopic，生成文档级主题分配、主题词表，以及围绕 `topic / 时间 / IP` 的多组占比表。脚本支持 embedding checkpoint，意外中断后可以用 `--resume` 续跑。

```bash
python bert/08_topic_model_bertopic.py
```

常用变体：

```bash
python bert/08_topic_model_bertopic.py \
  --device cuda \
  --resume
```

重点输出：

- `bert/artifacts/broad_analysis/topic_model/document_topics.parquet`
- `bert/artifacts/broad_analysis/topic_model/topic_info.csv`
- `bert/artifacts/broad_analysis/topic_model/topic_terms.csv`
- `bert/artifacts/broad_analysis/topic_model/topic_share_by_period.csv`
- `bert/artifacts/broad_analysis/topic_model/topic_share_by_period_and_keyword.csv`
- `bert/artifacts/broad_analysis/topic_model/topic_share_by_ip.csv`
- `bert/artifacts/broad_analysis/topic_model/topic_share_by_period_and_ip.csv`
- `bert/artifacts/broad_analysis/topic_model/topic_share_by_period_and_ip_and_keyword.csv`
- `bert/artifacts/broad_analysis/topic_model/topic_model_summary.json`

### 09. 关键词语义分析

对 `躺平 / 摆烂 / 佛系` 在不同时间段内的共现词和语义邻域进行分析。运行时会输出分阶段日志，便于区分分词、共现统计和 embedding 排序分别耗时多少。

```bash
python bert/09_keyword_semantic_analysis.py
```

重点输出：

- `bert/artifacts/broad_analysis/semantic_analysis/keyword_cooccurrence.csv`
- `bert/artifacts/broad_analysis/semantic_analysis/keyword_semantic_neighbors.csv`
- `bert/artifacts/broad_analysis/semantic_analysis/tokenized_analysis_base.parquet`

### 10. 概念漂移分析

比较相邻时间段之间的共现结构、语义邻域和主题分布变化。除了总体和按关键词比较，也会输出按 IP、按 `IP + 关键词` 的 topic 漂移结果。

```bash
python bert/10_concept_drift_analysis.py
```

重点输出：

- `bert/artifacts/broad_analysis/drift_analysis/keyword_collocation_drift.csv`
- `bert/artifacts/broad_analysis/drift_analysis/keyword_neighbor_drift.csv`
- `bert/artifacts/broad_analysis/drift_analysis/topic_drift_by_keyword.csv`
- `bert/artifacts/broad_analysis/drift_analysis/topic_share_change_by_keyword.csv`
- `bert/artifacts/broad_analysis/drift_analysis/topic_drift_overall.csv`
- `bert/artifacts/broad_analysis/drift_analysis/topic_share_change_overall.csv`
- `bert/artifacts/broad_analysis/drift_analysis/topic_drift_by_ip.csv`
- `bert/artifacts/broad_analysis/drift_analysis/topic_share_change_by_ip.csv`
- `bert/artifacts/broad_analysis/drift_analysis/topic_drift_by_ip_and_keyword.csv`
- `bert/artifacts/broad_analysis/drift_analysis/topic_share_change_by_ip_and_keyword.csv`

## 推荐的完整顺序

如果你要从原始数据一路跑到分析结果，推荐按下面的顺序执行：

1. `python main.py run`
2. `python bert/01_stratified_sampling.py`
3. `python bert/02_llm_label_local.py`
4. 人工审核
5. 视任务选择 `python bert/03_normalize_labels.py` 和 `python bert/04_train_bert_classifier.py`，或直接运行 `python bert/05_train_dual_label_classifier.py`
6. `python bert/06_predict_bert_classifier.py`
7. `python bert/07_build_broad_analysis_base.py`
8. `python bert/08_topic_model_bertopic.py`
9. `python bert/09_keyword_semantic_analysis.py`
10. `python bert/10_concept_drift_analysis.py`

## 仓库中默认不包含的内容

为避免仓库过大，以下内容通常不上传到 GitHub：

- `raw/`
- `data/`
- `archive/`
- `logs/`
- 本地虚拟环境

也就是说，这个仓库主要保存代码、配置和实验脚本；大体积数据和运行产物需要单独同步。

## 参考文档

- [`bert/README.md`](/Users/apple/Local/fdurop/code/result/bert/README.md)
- [`USER_MANUAL.md`](/Users/apple/Local/fdurop/code/result/USER_MANUAL.md)
- [`WINDOWS_SETUP.md`](/Users/apple/Local/fdurop/code/result/WINDOWS_SETUP.md)
