# BERT 工作流说明

`bert/` 现在按更贴近实际研究流程的方式组织：

- LLM 标注只是初筛草稿，后续必须人工审核。
- 训练脚本不再假设固定文件名、固定来源或固定实验协议。
- 你手上有多少份已审核 CSV/XLSX 都可以，是否合并、是否单独留作测试、哪些只进训练集，都由命令行参数决定。

## 运行前先确认环境

默认使用仓库根目录下的 `.venv`。

- 如果你已经 `source .venv/bin/activate`，下面示例里的 `python3` 可以直接照抄。
- 如果你是 Codex、Claude Code、Cursor 之类的代理工具，建议直接写成 `.venv/bin/python`，避免误用系统环境。

## 核心原则

1. `02_llm_label_local.py` 只负责生成预标注，不负责替代人工判断。
2. 进入训练阶段的数据，默认都应当是“人工复核过”的 CSV/XLSX。
3. `04_train_bert_classifier.py` 负责单标签训练。
4. `05_train_dual_label_classifier.py` 负责 `broad` / `strict` 双标签训练。
5. `06_predict_bert_classifier.py` 负责把训练好的模型批量打到全量 parquet 上。
6. `07` 到 `10` 是“全量 broad 语义分析链路”，默认围绕 `broad` 预测结果展开。

## 目录概览

```text
bert/
├── 01_stratified_sampling.py
├── 02_llm_label_local.py
├── 03_normalize_labels.py
├── 04_train_bert_classifier.py
├── 05_train_dual_label_classifier.py
├── 06_predict_bert_classifier.py
├── 07_build_broad_analysis_base.py
├── 08_topic_model_bertopic.py
├── 09_keyword_semantic_analysis.py
├── 10_concept_drift_analysis.py
├── lib/             训练、预测和分析阶段的公共模块
├── scripts/         辅助脚本
├── data/            抽样表、审核表等人工处理中间文件
└── artifacts/       模型、评估结果、预测结果、分析结果
```

如果按实验阶段来理解：

- `01-03`：样本抽取、预标注、标签整理。
- `04-05`：模型训练与评估。
- `06`：全量预测。
- `07-10`：主题、语义邻域和时间漂移分析。

## 最常见的真实流程

### 1. 从 parquet 抽样

```bash
python3 bert/01_stratified_sampling.py \
  --input "data/processed/text_dedup/*.parquet" \
  --output "bert/data/sample_review.csv" \
  --n 6000 \
  --report_path "bert/data/sampling_report.json"
```

### 2. 用 LLM 做预标注

```bash
python3 bert/02_llm_label_local.py \
  --input "bert/data/sample_review.csv" \
  --output "bert/data/sample_prelabel.csv" \
  --report_path "bert/data/labeling_report.json"
```

说明：

- 这一步的输出只能当“待审核草稿”。
- 请先人工审核，再把结果拿去训练。

### 3. 单标签训练

如果你的审核结果只有一套标签，例如 `label` / `tangping_related` / `tangping_related_label`：

```bash
python3 bert/04_train_bert_classifier.py \
  --input_csv "bert/data/reviewed_a.csv" "bert/data/reviewed_b.csv" \
  --output_dir "bert/artifacts/single_label_run"
```

如果你想显式指定哪个文件只进训练、哪个文件单独留作测试：

```bash
python3 bert/04_train_bert_classifier.py \
  --train_csv "bert/data/reviewed_train_extra.csv" \
  --input_csv "bert/data/reviewed_pool_a.csv" "bert/data/reviewed_pool_b.csv" \
  --test_csv "bert/data/reviewed_holdout.csv" \
  --output_dir "bert/artifacts/single_label_holdout"
```

这里的规则是：

- `--input_csv`：这些文件会先合并，再随机切成 train/val/test。
- `--train_csv` / `--train_only_csv`：这些文件只进训练集。
- `--val_csv`：这些文件只进验证集。
- `--test_csv`：这些文件只进测试集。

### 4. 双标签训练

如果你的人工审核表里同时有 `broad` 和 `strict` 两列：

```bash
python3 bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_part1.csv" "bert/data/reviewed_part2.csv" \
  --base_output_dir "bert/artifacts/dual_label_run"
```

如果你想把某一份文件固定留作测试，另一些只进训练：

```bash
python3 bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_pool_a.csv" "bert/data/reviewed_pool_b.csv" \
  --train_path "bert/data/reviewed_manual_boost.csv" \
  --test_path "bert/data/reviewed_external_test.csv" \
  --base_output_dir "bert/artifacts/dual_label_holdout"
```

规则和 `04` 一样：

- `--input_path`：合并后随机切分。
- `--train_path`：只进训练集。
- `--val_path`：只进验证集。
- `--test_path`：只进测试集。

如果你想“分开测试多个来源”，最直接的做法就是多跑几次 `05`，每次把不同来源放到 `--test_path`。

### 关于固定 split 参数

如果你传了 `--train_path`、`--val_path`、`--test_path` 这类参数：

- `--input_path` 里的文件仍然先按 `val_size` / `test_size` 自己完成随机切分。
- 这些固定 split 的文件是在切分完成后再追加到指定 split。
- 所以“`input_path` 这批数据内部的切分比例”不会变。
- 变化的是“最终总数据集”的整体比例，因为你额外加了只进某个 split 的样本。

## 列名约定

### 文本列

脚本会自动识别这些常见列名：

- `cleaned_text`
- `cleaned_text_with_emoji`
- `text_raw`
- `微博正文`
- `text`
- `content`

如果自动识别不到，显式传 `--text_col`。

### 单标签列

`04_train_bert_classifier.py` 会优先识别：

- `label`
- `tangping_related`
- `tangping_related_label`
- `broad`
- `strict`

如果你想强制指定，传 `--label_col`。

### 双标签列

`05_train_dual_label_classifier.py` 默认要求：

- `broad`
- `strict`

如果列名不同，可以传：

- `--broad_col`
- `--strict_col`

## 输出产物

`04` 和 `05` 的输出都会落在你指定的 `output_dir` / `base_output_dir` 下，常见文件包括：

- `train_split.csv`
- `val_split.csv`
- `test_split.csv`
- `metrics.json`
- `training_history.json`
- `test_predictions.csv`
- `test_misclassified.csv`
- `best_model/`

双标签训练额外会生成一套分层目录，建议按这个顺序看：

- `run_overview.md`
- `shared/shared_split_dataset.csv`
- `shared/shared_split_manifest.json`
- `compare/test_predictions_side_by_side.csv`
- `compare/test_misclassified_side_by_side.csv`
- `inspect/summary.md`
- `inspect/diagnosis/label_diagnosis.csv`
- `inspect/review/top_fp_*.csv`
- `inspect/review/top_fn_*.csv`

## 进入 07-10 之前

如果你现在已经在 Windows 机器上完成了：

1. 主流程产出 `data/processed/text_dedup/*.parquet`
2. 人工审核
3. `05_train_dual_label_classifier.py`

那么接下来建议先跑一次 `06`，把 `broad` 模型打到全量语料上。

### 5. 用 broad 模型做全量预测

典型命令：

```bash
python3 bert/06_predict_bert_classifier.py \
  --model_dir "bert/artifacts/dual_label_run/broad/best_model" \
  --input_pattern "data/processed/text_dedup/*.parquet" \
  --output_dir "data/processed/text_dedup_predicted_broad" \
  --device cuda
```

这里建议显式把输出目录写成 `data/processed/text_dedup_predicted_broad`，因为 `07_build_broad_analysis_base.py` 默认就是从这里读。

`06` 常见输出列包括：

- `pred_label`
- `pred_label_text`
- `pred_prob_1`
- `pred_prob_0`
- `pred_confidence`

如果你在 `06` 里用了默认输出目录 `data/processed/text_dedup_predicted/`，也没问题，只要在 `07` 里同步改 `--input_pattern`。

## 07-10 分析链路

### 6. `07_build_broad_analysis_base.py`

作用：

- 读取 `06` 的预测结果
- 规范化文本列、时间列、关键词列
- 自动识别并规范化 IP 属地列；缺失 IP 会统一记为 `UNKNOWN_IP`
- 默认只保留 `pred_label == 1` 的正样本
- 生成后续 `08` / `09` 共用的分析底表

默认命令：

```bash
python3 bert/07_build_broad_analysis_base.py
```

如果你的预测文件放在别的位置：

```bash
python3 bert/07_build_broad_analysis_base.py \
  --input_pattern "data/processed/text_dedup_predicted/*.parquet" \
  --output_path "bert/artifacts/broad_analysis/analysis_base.parquet"
```

常用可选参数：

- `--include_negative`：连负样本也保留
- `--min_confidence 0.8`：只保留置信度足够高的样本
- `--text_col` / `--time_col` / `--keyword_col` / `--ip_col`：强制指定列名

重点输出：

- `bert/artifacts/broad_analysis/analysis_base.parquet`
- `bert/artifacts/broad_analysis/analysis_base_report.json`

其中 `analysis_base_report.json` 会额外给出：

- `rows_by_ip`
- `rows_by_period_and_ip`
- `missing_ip_rows_after_filter`
- `missing_ip_rate_after_filter`

### 7. `08_topic_model_bertopic.py`

作用：

- 在 `07` 生成的分析底表上做 BERTopic
- 输出文档级 topic 结果、topic 词表，以及 `topic / 时间 / IP` 三个维度的占比表
- 缺失 IP 不会被丢掉，而是作为 `UNKNOWN_IP` 单独保留
- 支持 embedding 断点续跑，避免中途打断后从头编码

默认命令：

```bash
python3 bert/08_topic_model_bertopic.py
```

常用变体：

```bash
python3 bert/08_topic_model_bertopic.py \
  --time_granularity quarter \
  --min_topic_size 50 \
  --top_n_words 15 \
  --device cuda \
  --resume \
  --save_model
```

常用可选参数：

- `--device auto|cpu|cuda|mps`：控制 sentence-transformers 的编码设备
- `--resume`：如果已有 embedding checkpoint，则直接续跑
- `--checkpoint_dir`：自定义 checkpoint 目录
- `--ip_col`：手动指定 IP 列名

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

其中最适合直接做比较的是这三张表：

- `topic_share_by_period.csv`：只看时间维度
- `topic_share_by_period_and_ip.csv`：比较不同 IP 在各时间段的 topic 分布
- `topic_share_by_period_and_ip_and_keyword.csv`：在 `关键词 + 时间 + IP` 的细粒度下比较 topic 分布

### 8. `09_keyword_semantic_analysis.py`

作用：

- 对每个关键词在不同时间段做共现词分析
- 用 embedding 对候选词再排序，得到 semantic neighbors
- 脚本会输出分阶段进度，便于判断是在分词、统计还是 embedding 阶段

默认命令：

```bash
python3 bert/09_keyword_semantic_analysis.py
```

常用变体：

```bash
python3 bert/09_keyword_semantic_analysis.py \
  --time_granularity quarter \
  --min_doc_freq 10 \
  --top_k_terms 80 \
  --device cuda \
  --top_k_neighbors 30
```

重点输出：

- `bert/artifacts/broad_analysis/semantic_analysis/keyword_cooccurrence.csv`
- `bert/artifacts/broad_analysis/semantic_analysis/keyword_semantic_neighbors.csv`
- `bert/artifacts/broad_analysis/semantic_analysis/tokenized_analysis_base.parquet`
- `bert/artifacts/broad_analysis/semantic_analysis/semantic_analysis_summary.json`

### 9. `10_concept_drift_analysis.py`

作用：

- 比较相邻时间段的共现词变化
- 比较相邻时间段的 semantic neighbors 变化
- 比较相邻时间段的 topic share 变化
- 除了按关键词和总体比较，也会额外输出按 IP、按 `IP + 关键词` 的 topic 漂移结果

默认命令：

```bash
python3 bert/10_concept_drift_analysis.py
```

常用变体：

```bash
python3 bert/10_concept_drift_analysis.py \
  --time_granularity quarter \
  --top_n 30
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
- `bert/artifacts/broad_analysis/drift_analysis/drift_analysis_summary.json`

## Windows 机器上的注意事项

`08` / `09` 会额外依赖：

- `bertopic`
- `sentence-transformers`
- `jieba`

它们已经写在根目录 [`requirements.txt`](../requirements.txt) 里，但第一次运行 embedding 模型时，通常还会下载模型权重。

如果你的 Windows 机器联网：

- 直接在已安装 `requirements.txt` 的环境里运行即可。

如果你的 Windows 机器离线：

- 先把 embedding 模型缓存好，或者把模型目录拷到本地。
- 运行 `08` / `09` 时把 `--embedding_model` 指到本地目录。
- 再加上 `--local_files_only`，避免脚本尝试联网下载。

## 最后建议

更稳妥的习惯是：

1. `01` 抽样。
2. `02` 预标注。
3. 人工审核。
4. 审核后的多个 CSV/XLSX 直接喂给 `04` 或 `05`。
5. 用 `05` 产出的 `broad/best_model` 跑 `06` 全量预测。
6. 再顺序跑 `07`、`08`、`09`、`10`。
7. 需要换测试集或换分析口径时，只改命令参数，不改脚本逻辑。

这套方式比维护一堆固定案例更稳，也更符合真实研究流程。
