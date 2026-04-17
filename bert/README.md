# BERT 工作流说明

这份文档只讲 `bert/` 目录下的下游流程：

- 抽样
- LLM 预标注
- 标签整理
- 单标签 / 双标签训练
- 全量预测
- `07-10` 的 broad 分析链

如果你还没有把 `raw/` 处理成 `data/processed/text_dedup/`，先去看根目录的 [`USER_MANUAL.md`](../USER_MANUAL.md)。

## 先别急着跑命令，先理解这条流程在干什么

这一套流程不是“拿到数据以后直接训练模型”，中间其实有一个很重要的人工判断环节。

更贴近真实工作的顺序是：

1. 从全量语料里抽一批样本
2. 用 LLM 先打一个预标注草稿
3. 人工审核，把边界真正定下来
4. 再用审核后的数据训练分类器
5. 把训练好的模型打到全量语料
6. 在全量预测结果上继续做主题、语义和漂移分析

这里最容易踩的坑只有一个：

`02_llm_label_local.py` 只是帮你省初筛时间，不是替你做研究判断。

如果没有人工复核，后面的训练质量会很不稳。

## 环境约定

仓库默认使用根目录下的 `.venv`：

- macOS / Linux：`.venv/bin/python`
- Windows PowerShell：`.\.venv\Scripts\python.exe`

下面示例统一按 macOS / Linux 写。Windows 下只要替换解释器路径。

## 这条流程里每个脚本负责什么

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
├── scripts/
├── lib/
├── data/
└── artifacts/
```

可以粗暴理解成：

- `01-03`：准备样本和标签
- `04-06`：训练模型并打到全量数据
- `07-10`：把 broad 预测结果变成能读、能讲、能汇报的分析结果

## 推荐的真实工作顺序

如果你是第一次接这条流程，建议按下面走：

1. `01` 抽样
2. `02` 预标注
3. 人工审核
4. 单标签任务走 `03 -> 04`
5. 双标签任务直接走 `05`
6. 跑 `06` 做全量预测
7. 跑 `07`
8. 跑 `08`
9. 跑 `09`
10. 跑 `10`

如果你的目标只是训练一个分类器，其实跑到 `06` 就可以停。

如果你的目标是做 broad 语义分析，重点就会落在 `07-10`。

## `bert/data/` 和 `bert/artifacts/` 怎么理解

### `bert/data/`

这个目录更像“人工处理工作台”。

常见内容包括：

- 抽样结果
- 预标注结果
- 人工审核表
- 合并后的审核表

### `bert/artifacts/`

这个目录更像“模型和分析产物仓库”。

建议长期保持下面这种结构：

```text
bert/artifacts/
├── runs/
│   ├── dual_label/
│   └── single_label/
└── broad_analysis/
    ├── topic_model_BAAI/
    ├── semantic_analysis/
    ├── drift_analysis/
    ├── overview/
    ├── snapshots/
    └── legacy/
```

为什么这样分：

- `runs/` 放训练 run，便于回看模型、指标和测试集
- `broad_analysis/` 放分析结果，便于后面做整理、比较和汇报
- `snapshots/` 放一次性批次，避免当前主结果被冲乱
- `legacy/` 放历史残留，省得和现在的标准产物混在一起

仓库里已经带了整理脚本：

```bash
.venv/bin/python bert/scripts/organize_artifacts.py
```

如果你发现 `artifacts/` 目录越跑越乱，这个脚本会有用。

## 1. 从 parquet 抽样：`01_stratified_sampling.py`

这一步是在做什么：

- 从 `data/processed/text_dedup/*.parquet` 里抽一批样本
- 让后面的标注集不要全被某一类文本占满
- 给人工审核准备一个起点

默认输入一般是：

- `data/processed/text_dedup/*.parquet`

常用命令：

```bash
.venv/bin/python bert/01_stratified_sampling.py \
  --input "data/processed/text_dedup/*.parquet" \
  --output "bert/data/sample.csv" \
  --n 6000 \
  --report_path "bert/data/sampling_report.json"
```

跑完之后，最值得看的通常是：

- `bert/data/sample.csv`
- `bert/data/sampling_report.json`

什么时候你需要回头改这一步：

- 样本量明显不够
- 某些关键词或时期占比失衡
- 你想针对特定研究问题重新抽一批更偏的样本

## 2. 用 LLM 做预标注：`02_llm_label_local.py`

这一步的作用很明确：

- 先给样本打一版“机器草稿”
- 减少人工从零开始标的时间

但还是那句话：

这一步不是最终标签。

常用命令：

```bash
.venv/bin/python bert/02_llm_label_local.py \
  --input "bert/data/sample.csv" \
  --output "bert/data/labeled.csv" \
  --report_path "bert/data/labeling_report.json"
```

最稳妥的做法，是先复制一份配置模板：

```bash
cp bert/llm_label_local.example.toml bert/llm_label_local.toml
.venv/bin/python bert/02_llm_label_local.py --config bert/llm_label_local.toml
```

运行前最好确认三件事：

1. provider 配置是可用的
2. model 名称是你本机或账号上真能调用的
3. API key 或本地服务已经准备好

如果你不用配置文件，也可以用环境变量，比如：

- `DASHSCOPE_API_KEY`
- `QWEN_API_KEY`
- `OPENAI_API_KEY`
- `FIXER_API_KEY`

如果你走本地 `ollama`：

- 通常不需要 API key
- 但本地服务要先启动
- 模型也要提前拉好

## 3. 标签整理：`03_normalize_labels.py`

这一步不是每次都要跑。

它主要服务于单标签任务，也就是你想把审核结果整理成标准二值标签的时候。

常用命令：

```bash
.venv/bin/python bert/03_normalize_labels.py \
  --input_csv "bert/data/labeled.csv" \
  --output_csv "bert/data/labeled_binary.csv"
```

什么时候要用：

- 你的审核表是单标签二分类
- 列名和格式还不够统一

什么时候可以跳过：

- 你的审核表已经明确有 `broad` 和 `strict` 两列
- 你准备直接做双标签训练

如果人工审核分散在多个 XLSX，可以先合并：

```bash
.venv/bin/python bert/scripts/merge_xlsx_annotations.py \
  bert/data/review_part1.xlsx bert/data/review_part2.xlsx \
  -o bert/data/review_merged.xlsx \
  --report-json bert/data/review_merged.report.json
```

## 4. 单标签训练：`04_train_bert_classifier.py`

这个脚本适合你只有一套标签的时候。

比如：

- `label`
- `tangping_related`
- `tangping_related_label`

常用命令：

```bash
.venv/bin/python bert/04_train_bert_classifier.py \
  --input_csv "bert/data/reviewed_a.csv" "bert/data/reviewed_b.csv" \
  --output_dir "bert/artifacts/runs/single_label/single_label_run"
```

如果你想固定一部分测试集，不想让脚本随机切进去：

```bash
.venv/bin/python bert/04_train_bert_classifier.py \
  --train_csv "bert/data/reviewed_train_extra.csv" \
  --input_csv "bert/data/reviewed_pool_a.csv" "bert/data/reviewed_pool_b.csv" \
  --test_csv "bert/data/reviewed_holdout.csv" \
  --output_dir "bert/artifacts/runs/single_label/single_label_holdout"
```

怎么理解这些参数：

- `--input_csv`：合并后再随机切 train/val/test
- `--train_csv` / `--train_only_csv`：只进训练集
- `--val_csv`：只进验证集
- `--test_csv`：只进测试集

## 5. 双标签训练：`05_train_dual_label_classifier.py`

如果你的人工审核表里已经有：

- `broad`
- `strict`

那通常直接走这个脚本。

常用命令：

```bash
.venv/bin/python bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_part1.csv" "bert/data/reviewed_part2.csv" \
  --base_output_dir "bert/artifacts/runs/dual_label/dual_label_run"
```

如果你想固定留一份测试集：

```bash
.venv/bin/python bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_pool_a.csv" "bert/data/reviewed_pool_b.csv" \
  --train_path "bert/data/reviewed_manual_boost.csv" \
  --test_path "bert/data/reviewed_external_test.csv" \
  --base_output_dir "bert/artifacts/runs/dual_label/dual_label_holdout"
```

怎么理解这些 split 参数：

- `--input_path`：合并后随机切分
- `--train_path`：只进训练集
- `--val_path`：只进验证集
- `--test_path`：只进测试集

要注意的一点是：

固定 split 文件不会改变 `--input_path` 那批数据内部的切分逻辑，而是在随机切分之后再追加到指定 split。

所以：

- 内部随机切分比例不变
- 最终总数据集比例会变

这是正常现象。

## 文本列和标签列怎么识别

### 文本列

脚本通常会自动识别这些常见列名：

- `cleaned_text`
- `cleaned_text_with_emoji`
- `text_raw`
- `微博正文`
- `text`
- `content`

如果自动识别不到，再显式传 `--text_col`。

一般建议：

- 分类训练和预测优先 `cleaned_text`
- 如果你真的想保留 emoji 的语义影响，再考虑 `cleaned_text_with_emoji`

### 单标签列

`04_train_bert_classifier.py` 会优先识别：

- `label`
- `tangping_related`
- `tangping_related_label`
- `broad`
- `strict`

如果不想让脚本自动猜，直接传 `--label_col`。

### 双标签列

`05_train_dual_label_classifier.py` 默认要求：

- `broad`
- `strict`

如果列名不同，可以传：

- `--broad_col`
- `--strict_col`

## 训练输出怎么看

`04` 和 `05` 的输出目录里，常见会有这些东西：

- `train_split.csv`
- `val_split.csv`
- `test_split.csv`
- `metrics.json`
- `training_history.json`
- `test_predictions.csv`
- `test_misclassified.csv`
- `best_model/`

如果是双标签训练，还会多出一套更方便排查的内容。比较值得先看的有：

- `run_overview.md`
- `shared/shared_split_dataset.csv`
- `shared/shared_split_manifest.json`
- `compare/test_predictions_side_by_side.csv`
- `compare/test_misclassified_side_by_side.csv`
- `inspect/summary.md`
- `inspect/diagnosis/label_diagnosis.csv`

实际看结果时，我更建议这样读：

1. 先看 `metrics.json`，判断模型大体行不行
2. 再看 `test_misclassified.csv`，确认错在哪里
3. 如果是双标签任务，再看 side-by-side 结果，比较 `broad` 和 `strict` 哪个更不稳

## 6. 全量预测：`06_predict_bert_classifier.py`

这一步是在做什么：

- 把你训练好的模型打到全量 parquet
- 让后面的分析不再只依赖抽样数据

如果你后面准备进入 `07-10`，建议把输出目录显式写成：

- `data/processed/text_dedup_predicted_broad`

这样最顺手。

常用命令：

```bash
.venv/bin/python bert/06_predict_bert_classifier.py \
  --model_dir "bert/artifacts/runs/dual_label/dual_label_run/broad/best_model" \
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

如果你只是想保留正样本，也可以看脚本的 `--only_positive` 参数。

## 进入 `07-10` 之前，先确认这三件事

1. 主流程已经产出 `data/processed/text_dedup/*.parquet`
2. 训练数据是人工复核过的
3. 你已经有一套靠谱的 broad 预测结果

这三件事缺一个，后面的 broad 分析都会变得很虚。

## 7. 构建 broad 分析底表：`07_build_broad_analysis_base.py`

这一步的任务是把预测结果整理成一个干净、统一、后面都能吃的分析底表。

它会做的事包括：

- 读取 `06` 的预测结果
- 规范化文本列、时间列、关键词列
- 识别并规范化 IP 属地列
- 默认只保留 `pred_label == 1` 的正样本
- 生成 `08` 和 `09` 共用的底表

默认命令：

```bash
.venv/bin/python bert/07_build_broad_analysis_base.py
```

如果你的预测目录不在默认位置：

```bash
.venv/bin/python bert/07_build_broad_analysis_base.py \
  --input_pattern "data/processed/text_dedup_predicted/*.parquet" \
  --output_path "bert/artifacts/broad_analysis/analysis_base.parquet"
```

常用可选参数：

- `--include_negative`
- `--min_confidence 0.8`
- `--text_col`
- `--time_col`
- `--keyword_col`
- `--ip_col`

重点输出：

- `bert/artifacts/broad_analysis/analysis_base.parquet`
- `bert/artifacts/broad_analysis/analysis_base_report.json`

## 8. 主题建模：`08_topic_model_bertopic.py`

这一步更适合回答：

- broad 语料里有哪些主题
- 各主题在时间上怎么变化
- 不同关键词、不同地区上，主题占比怎么变

默认命令：

```bash
.venv/bin/python bert/08_topic_model_bertopic.py
```

常用变体：

```bash
.venv/bin/python bert/08_topic_model_bertopic.py \
  --time_granularity quarter \
  --min_topic_size 50 \
  --umap_n_neighbors 30 \
  --top_n_words 15 \
  --device cuda \
  --resume \
  --save_model
```

如果你希望主题更适合汇报，而不是切得特别碎，可以从更稳一点的参数开始：

```bash
.venv/bin/python bert/08_topic_model_bertopic.py \
  --min_topic_size 300 \
  --hdbscan_min_samples 60 \
  --umap_n_neighbors 80 \
  --nr_topics 12 \
  --outlier_reduction_strategy c-tf-idf+distributions \
  --outlier_reduction_threshold 0.05 \
  --resume
```

常见输出：

- `topic_model_summary.json`
- `readouts/topic_info.csv`
- `readouts/topic_overview.csv`
- `readouts/topic_terms.csv`
- `readouts/topic_share_by_period.csv`
- `viz_inputs/document_topics.parquet`

如果你第一次看 `08` 的结果，建议优先看：

1. `topic_overview.csv`
2. `topic_info.csv`
3. `topic_share_by_period.csv`

这三个文件最容易先看出“主题大概长什么样”和“时间上怎么变”。不过它们各自解决的问题不一样，最好别混着看。

### `topic_overview.csv` 里会有什么

这个文件适合当第一眼总览表。它不是最原始的 BERTopic 输出，而是脚本整理过的一张“主题摘要表”。

通常会看到这些信息：

- `topic_id`：主题编号
- `topic_label_machine`：模型自动生成的主题名
- `topic_label_zh`：预留给人工补中文标签的列，默认可能是空的
- `topic_label_display`：展示时优先使用的主题名
- `topic_count`：这个主题一共覆盖多少条文本
- `share_of_all_docs_pct`：它占全部分析文本的比例
- `share_of_clustered_docs_pct`：它占非离群主题文本的比例
- `top_terms`：这个主题最能代表它的一串关键词
- `peak_period`：这个主题最活跃的时间段
- `peak_doc_count` / `peak_doc_share_pct`：它在峰值时间段有多少文本、占当期多少比例
- `dominant_keyword`：在这个主题里最常见的研究关键词，比如更偏“躺平”还是“摆烂”
- `dominant_keyword_share_within_topic_pct`：这个关键词在该主题内部占比多高

怎么理解这张表：

- 如果你想快速知道“这轮主题模型大概切出了哪些块”，先看它
- 如果你想挑几个重点主题做人工命名或中期汇报，也先看它
- 如果某个主题 `topic_count` 很大，但 `top_terms` 很杂，通常说明这个主题还需要继续清理或合并

### `topic_info.csv` 里会有什么

这个文件更接近 BERTopic 原始结果，是一张更底层的主题信息表。

通常会看到这些信息：

- `Topic`：主题编号
- `Count`：该主题包含的文本数
- `Name`：BERTopic 自动生成的主题名称，通常是若干关键词拼接出来的
- `Representation`、`Representative_Docs` 一类字段：主题代表词或代表文本
- `topic_label_machine`：脚本补上的机器标签列
- `topic_label_zh`：脚本预留的人工中文标签列

这张表更适合做这些事：

- 检查 BERTopic 原始命名是不是靠谱
- 回看某个主题的代表文本到底像不像它的名字
- 区分正常主题和离群项

要特别注意：

- `Topic = -1` 一般表示离群文本，也就是没有稳定归进某个主题的内容
- 所以你看主题数量时，最好把 `-1` 和正常主题分开理解

### `topic_share_by_period.csv` 里会有什么

这个文件是时间分布表，用来回答“某个主题在哪些时间段最明显”。

通常会看到这些信息：

- `period_label`：时间段标签，比如按月或按季度
- `topic_id`
- `topic_label`
- `doc_count`：这个时间段里，该主题有多少条文本
- `doc_share`：这个主题在该时间段内部占比多少

怎么理解：

- `doc_count` 看的是绝对量，适合看某段时间讨论到底多不多
- `doc_share` 看的是相对占比，适合看“这一时期的话题重心是不是更偏向这个主题”

这张表最适合回答：

- 哪个主题在哪个时期达到峰值
- 某个主题是短期爆发，还是长期持续
- 时间变化是“总量一起涨”，还是“结构比例发生了变化”

如果你看到某个主题在某段时间 `doc_count` 很高，但 `doc_share` 没明显上升，往往说明那段时间整体文本都变多了，不一定是这个主题单独变得更重要。

### 一个实用的阅读顺序

如果你是第一次打开 `08` 的结果，推荐这样读：

1. 先看 `topic_overview.csv`，建立“这轮主题大盘长什么样”的感觉
2. 再看 `topic_info.csv`，确认每个主题的名字、代表词和代表文本到底靠不靠谱
3. 最后看 `topic_share_by_period.csv`，判断这些主题是怎么随时间变化的

这样读下来，通常就能比较稳地回答三个问题：

- 这轮模型切出了哪些主要主题
- 每个主题大概在说什么
- 它们是在哪些时间段变强或变弱的

## 9. 关键词语义分析：`09_keyword_semantic_analysis.py`

这一步更偏“关键词附近都在和谁一起出现，它们的语义邻居是什么”。

它会做的事：

- 为每个关键词、每个时间段提取可解释的临近词候选
- 先做共现词统计
- 再用 embedding 重排，得到更稳定的 semantic neighbors

默认命令：

```bash
.venv/bin/python bert/09_keyword_semantic_analysis.py
```

常用变体：

```bash
.venv/bin/python bert/09_keyword_semantic_analysis.py \
  --time_granularity quarter \
  --min_doc_freq 10 \
  --top_k_terms 80 \
  --device cuda \
  --top_k_neighbors 30
```

重点输出：

- `semantic_analysis_summary.json`
- `viz_inputs/keyword_cooccurrence.csv`
- `viz_inputs/keyword_semantic_neighbors.csv`
- `viz_inputs/tokenized_analysis_base.parquet`

如果你想把 `09` 的结果整理成更适合阅读和汇报的表，再接着跑：

```bash
.venv/bin/python bert/09_prepare_semantic_midterm.py \
  --semantic_dir "bert/artifacts/broad_analysis/semantic_analysis"
```

这一步会生成一批更容易直接读的结果，比如：

- `semantic_keyword_overview.csv`
- `semantic_context_trajectory.csv`
- `semantic_context_shift_summary.csv`
- `semantic_period_shortlist.csv`
- `semantic_bucket_override_template.csv`
- `semantic_midterm_notes.md`

如果你准备读 `09`，最推荐的顺序是：

1. 先看 `semantic_keyword_overview.csv`
2. 再看 `semantic_context_trajectory.csv`
3. 最后用 `semantic_context_shift_summary.csv` 提炼结论

不过这几份文件不是同一种东西。有的是总览表，有的是时间轨迹表，有的是人工修正模板。分开理解会更清楚。

### `semantic_keyword_overview.csv` 里会有什么

这个文件最适合当 `09` 的第一入口。它会把每个关键词下比较值得保留的代表词先挑出来，方便你先建立整体印象。

通常会看到这些信息：

- `keyword`：对应的研究关键词
- `term`：被保留下来的代表词
- `midterm_rank`：这个词在该关键词内部的大致优先级
- `term_doc_freq`：这个词命中了多少文本
- `lift`：它和该关键词绑定得紧不紧
- `embedding_similarity`：它是否得到了语义邻居结果的支持
- `midterm_score`：脚本综合频次、区分度和语义支持后算出来的排序分
- `theme_bucket` / `context_bucket`：这个词最终被归到哪个主题桶、语境桶
- `example_1_text`、`example_2_text`：代表文本，方便你回头看真实语境

这张表最适合回答：

- 每个关键词大概有哪些代表用法
- 哪些词更值得拿来命名语义簇
- 哪些词虽然常见，但语义上其实不稳定

如果你想先抓“这个关键词现在主要在什么语境里被使用”，先看它。

### `semantic_context_trajectory.csv` 里会有什么

这个文件是时间轨迹表。它不是盯着单个词，而是把同一时间段、同一个 `context_bucket` 下的词聚合起来看。

通常会看到这些信息：

- `keyword`
- `period_label`：时间段标签，比如月度或季度
- `context_bucket`：这个时间段里占优势的语义簇
- `doc_count_in_keyword`：该关键词在这一时间段总共覆盖多少文本
- `context_term_count`：这个语义簇里保留下来多少个代表词
- `context_term_doc_freq_sum`：这些代表词的总文档频次
- `context_midterm_score_sum`：这个语义簇累计的综合分数
- `context_doc_freq_share`：它按词频在该时间段里占多大比重
- `context_score_share`：它按综合分在该时间段里占多大比重
- `lead_terms`：这个时间段最能代表该语义簇的几组词

这张表最适合回答：

- 某个语义簇是在什么时候开始变强的
- 某段时间里哪个语义簇占主导
- 语义变化到底是某几个词偶然冒头，还是同一类语境整体变强了

如果你做时间分析，`context_score_share` 往往比单看词频更稳一些，因为它不只是机械累加出现次数。

### `semantic_context_shift_summary.csv` 里会有什么

这个文件是轨迹表的浓缩版。它不把每个时间段都展开，而是把一个语义簇从头到尾的变化压缩成几项摘要。

通常会看到这些信息：

- `keyword`
- `context_bucket`
- `period_count`：这个语义簇一共持续了多少期
- `first_period` / `latest_period`：最早和最新出现在哪个时间段
- `first_context_score_share` / `latest_context_score_share`：最早和最新时占比多少
- `score_share_change`：从最早到最新，整体是增强还是减弱
- `peak_period`：它最强的时间段
- `peak_context_score_share`：峰值时占比多少
- `avg_context_score_share`：整个观察期里的平均强度
- `representative_terms_over_time`：这个语义簇在不同时间段里出现过的代表词摘要

这张表最适合做结论提炼，比如：

- 哪些语义簇是长期稳定存在的
- 哪些只是某一段时间突然爆出来的
- 哪些语义簇在后期明显增强或衰退

如果你在写中期汇报或章节总结，这张表通常最好用。

### `semantic_period_shortlist.csv` 里会有什么

这个文件更像“按时间段展开的回查表”。它保留的是每个关键词、每个时间段里值得继续读的代表词。

通常会看到这些信息：

- `keyword`
- `period_label`
- `term`
- `midterm_rank`
- `term_doc_freq`
- `theme_bucket` / `context_bucket`
- `example_1_text`、`example_2_text`

它最适合用在这些时候：

- 你已经发现某个时间段有变化，想回头问“那时候到底冒出了哪些词”
- 你想解释为什么某一时期某个语义簇突然上升
- 你需要从摘要表回到更具体的词和文本例子

所以它更像“展开细看”的入口，不是最先读的总览表。

### `semantic_bucket_override_template.csv` 里会有什么

这个文件不是直接拿来写结论的，而是拿来改桶的。

通常会看到这些信息：

- `keyword`
- `period_label`
- `term`
- `auto_context_bucket` / `context_bucket`
- `auto_theme_bucket` / `theme_bucket`
- `override_context_bucket`
- `override_theme_bucket`
- `enabled`
- `note`
- `example_1_text`

怎么用它：

1. 先看脚本自动分的桶对不对
2. 把明显分错的行挑出来
3. 填到 `override_context_bucket` 或 `override_theme_bucket`
4. 再复制到 `bert/config/semantic_bucket_overrides.csv`
5. 重跑整理脚本

它更像一个修正规则的工作台。

### `semantic_midterm_notes.md` 里会有什么

这是脚本自动写出来的一份导读，偏向“把这一轮 `09` 的结果快速讲给人听”。

里面通常会有：

- 这轮候选词和保留词的大致数量
- 读取顺序建议
- `09` 的原理摘要
- 主要噪声来源
- 各关键词的总体语义摘要
- 语义簇时间变化的简短概括
- 如果要做人工修正，应该从哪里下手

这份文件适合：

- 你隔了一段时间回来，需要快速找回上下文
- 你想先看一版自动生成的读法，再决定进哪个 CSV 深挖
- 你要先把这轮结果讲给别人听

### 一个更实用的阅读顺序

如果你第一次打开 `09` 的整理结果，推荐这样读：

1. 先看 `semantic_keyword_overview.csv`，建立“每个关键词主要有哪些语义簇”的感觉
2. 再看 `semantic_context_trajectory.csv`，确认这些语义簇在时间上是怎么起伏的
3. 用 `semantic_context_shift_summary.csv` 把时间变化压缩成可以直接写进结论的摘要
4. 如果想解释某个时间段为什么会变，再回头看 `semantic_period_shortlist.csv`
5. 如果发现自动分桶不理想，再用 `semantic_bucket_override_template.csv` 做人工修正
6. `semantic_midterm_notes.md` 可以当这一整包结果的导读

## 语义分桶规则怎么调

`09_prepare_semantic_midterm.py` 背后用到两类规则：

- `bert/config/semantic_bucket_rules.json`
- `bert/config/semantic_bucket_overrides.csv`

怎么理解：

- `rules` 是自动分桶规则
- `overrides` 是人工修正表

比较实用的工作方式通常是：

1. 先跑一次自动整理
2. 从 `semantic_bucket_override_template.csv` 里挑出明显不对的项
3. 复制到 `semantic_bucket_overrides.csv`
4. 再重跑一遍

这样比一开始就手工全修要省力很多。

## 10. 概念漂移分析：`10_concept_drift_analysis.py`

这一步是在看“相邻时间段之间到底变了什么”。

它会比较：

- 共现词变化
- semantic neighbors 变化
- topic share 变化

而且不只是总体比较，也会额外输出：

- 按关键词
- 按 IP
- 按 `IP + 关键词`

默认命令：

```bash
.venv/bin/python bert/10_concept_drift_analysis.py
```

常用变体：

```bash
.venv/bin/python bert/10_concept_drift_analysis.py \
  --time_granularity quarter \
  --top_n 30
```

重点输出：

- `drift_analysis_summary.json`
- `readouts/keyword_collocation_drift.csv`
- `readouts/keyword_neighbor_drift.csv`
- `readouts/topic_drift_by_keyword.csv`
- `readouts/topic_drift_overall.csv`
- `viz_inputs/topic_share_change_by_keyword.csv`

如果你是第一次看 `10`，通常先看：

1. `topic_drift_overall.csv`
2. `topic_drift_by_keyword.csv`
3. `keyword_neighbor_drift.csv`

## `07-10` 的输出目录怎么放

当前建议固定成这一套：

- `08` 的主结果放 `bert/artifacts/broad_analysis/topic_model_BAAI/`
- `09` 的主结果放 `bert/artifacts/broad_analysis/semantic_analysis/`
- `10` 的主结果放 `bert/artifacts/broad_analysis/drift_analysis/`

每个目录内部再统一分两层：

- `readouts/`：给人直接看
- `viz_inputs/`：给程序和可视化继续调用

如果你跑的是截断月份、临时批次或者 overnight 版本，建议先写到：

- `bert/artifacts/broad_analysis/snapshots/<group>/<run_tag>/`

这样不会把当前主结果冲掉。

## 主题太碎的时候怎么办

这不是 bug，很多时候是参数太松。

一个更稳妥的处理顺序通常是：

1. 先把聚类收紧，减少碎片 topic
2. 再做 outlier 回填
3. 最后再把主题数压到汇报口径

实践上可以先试：

- `--min_topic_size 300`
- `--hdbscan_min_samples 50~100`
- `--umap_n_neighbors 50~100`
- `--nr_topics 8~15`

如果你一上来就强行把主题压到很少，结果反而可能更难解释。

## 什么时候该扩停用词表

如果你发现这些东西反复跑进 topic 头部，通常就该考虑扩停用词了：

- 粉圈词
- 游戏词
- 二手交易词
- 编号和纯数字
- 节日和问候语

对应文件通常是：

- `bert/config/topic_stopwords.txt`

## `07 -> 08` 之间要不要加规则过滤

如果你的研究重点是“躺平 / 摆烂 / 佛系”的社会语义，而不是交易黑话、抽卡帖或游戏流量帖，那确实值得考虑在 `07` 和 `08` 之间加一层轻过滤。

例如：

- `佛系收`
- `佛系出`
- `中转`
- `抽卡`
- `代肝`
- `黑市`

这些内容往往会在主题里造成很强的噪声。

## Windows 上要注意什么

`08` / `09` 会额外依赖：

- `bertopic`
- `sentence-transformers`
- `jieba`

它们已经写在根目录 `requirements.txt` 里，但第一次运行通常还会下载模型权重。

如果 Windows 机器离线：

- 先把 embedding 模型缓存好
- 或者直接把模型目录拷到本地
- 运行时加 `--embedding_model <本地目录>`
- 再加 `--local_files_only`

更细的 Windows 说明请看 [`../WINDOWS_SETUP.md`](../WINDOWS_SETUP.md)。

## 最后给一个最实用的判断标准

如果你不知道现在该跑到哪一步，先问自己：

- 我现在是在“准备标签”吗？
- 我现在是在“训练模型”吗？
- 我现在是在“读全量语义结果”吗？

对应关系是：

- 准备标签：看 `01-03`
- 训练模型：看 `04-06`
- 读 broad 结果：看 `07-10`

这样会比按文件名硬记流程更不容易乱。
