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

## 这条流程里哪些是入口，哪些是支撑层

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
├── 09_prepare_semantic_midterm.py
├── 10_concept_drift_analysis.py
├── 08_topic_model_bertopic_当前操作说明.md
├── config/
├── scripts/
├── lib/
├── data/
└── artifacts/
```

更接近当前代码实际架构的理解方式是：

- `01-10`：真正的阶段入口脚本
- `09_prepare_semantic_midterm.py`：不是额外研究支线，而是 `09` 结果进入“可读 / 可汇报 / 可人工修正”状态的整理层
- `config/`：停用词、语义分桶规则、人工覆盖项，属于流程本体的一部分
- `lib/`：训练、预测、broad-analysis 布局和共享逻辑
- `scripts/`：合并标注、整理产物、overnight 运行、topic compare 之类的辅助脚本
- `data/`：抽样表、审核表、示例工作文件
- `artifacts/`：训练输出和 broad-analysis 输出

一句话说：

- `01-06` 是“从样本到模型”
- `07-09_prepare` 是“从 broad 预测到可读结果”
- `10` 是“基于 `08` 和 `09` 结果继续做漂移比较”

## 先分清两层目录：默认工作目录 vs 归档后的 canonical 布局

现在 `bert/` 里最容易让人混淆的，不是脚本顺序，而是“脚本默认写到哪里”和“长期整理后希望它长成什么样”不是一回事。

### 第一层：脚本默认工作目录

不额外传参时，当前代码的默认落盘位置大致是：

- `04_train_bert_classifier.py` -> `bert/artifacts/tangping_bert`
- `05_train_dual_label_classifier.py` -> `bert/artifacts/dual_label_run`
- `06_predict_bert_classifier.py` -> `data/processed/text_dedup_predicted`
- `07` -> `bert/artifacts/broad_analysis/analysis_base.parquet`
- `08` -> `bert/artifacts/broad_analysis/topic_model_BAAI`
- `09` -> `bert/artifacts/broad_analysis/semantic_analysis`
- `10` -> `bert/artifacts/broad_analysis/drift_analysis`

这些路径反映的是“脚本现在怎么工作”，不是最终归档口径。

### 第二层：整理后的 canonical 布局

如果你显式指定输出目录，或者后面跑：

```bash
.venv/bin/python bert/scripts/organize_artifacts.py
```

更适合长期保存和回看的结构是：

```text
bert/artifacts/
├── runs/
│   ├── single_label/
│   └── dual_label/
└── broad_analysis/
    ├── topic_model_BAAI/
    ├── semantic_analysis/
    ├── drift_analysis/
    ├── overview/
    ├── snapshots/
    └── legacy/
```

这两层不要混着理解：

- 文档里提到的 `runs/...` 是长期归档口径
- 脚本帮助信息里的默认值，才是当前无参运行的真实行为
- 仓库里像 `topic_model_compare/` 这样的目录，属于分析实验或参数比较目录，不是 `08 -> 09 -> 10` 的标准交接目录

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
10. 如果你要直接阅读 `09`、做中期整理或准备汇报，再跑 `09_prepare_semantic_midterm.py`
11. 跑 `10`

如果你的目标只是训练一个分类器，其实跑到 `06` 就可以停。

如果你的目标是做 broad 语义分析，重点就会落在 `07-09_prepare + 10`。

## `bert/data/` 和 `bert/artifacts/` 怎么理解

### `bert/data/`

这个目录更像“人工处理工作台”。

常见内容包括：

- 抽样结果
- 预标注结果
- 人工审核表
- 合并后的审核表

这里还要分清一件事：

- 脚本默认会生成像 `sample.csv`、`labeled.csv`、`labeled_binary.csv` 这样的工作文件
- 但仓库里当前保留的示例文件，可能是 `sample_6000.csv` 和若干 `.xlsx`

也就是说：

- “脚本默认文件名”
- “你当前手上正在审核的文件名”

不一定相同，这不是异常。

### `bert/artifacts/`

这个目录更像“模型和分析产物仓库”，但它有两种状态：

- 脚本刚跑完时的工作目录状态
- 经过整理后的 canonical 状态

如果你现在只是想把流程跑通，优先看“脚本默认工作目录”。

如果你现在是在整理长期结果、准备归档或准备给别人接手，再看 `runs/` / `broad_analysis/` 这套 canonical 布局。

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
  --output_dir "bert/artifacts/tangping_bert"
```

如果你想固定一部分测试集，不想让脚本随机切进去：

```bash
.venv/bin/python bert/04_train_bert_classifier.py \
  --train_csv "bert/data/reviewed_train_extra.csv" \
  --input_csv "bert/data/reviewed_pool_a.csv" "bert/data/reviewed_pool_b.csv" \
  --test_csv "bert/data/reviewed_holdout.csv" \
  --output_dir "bert/artifacts/tangping_bert_holdout"
```

怎么理解这些参数：

- `--input_csv`：合并后再随机切 train/val/test
- `--train_csv` / `--train_only_csv`：只进训练集
- `--val_csv`：只进验证集
- `--test_csv`：只进测试集

这里再强调一次：

- `bert/artifacts/tangping_bert` 是当前脚本默认工作目录
- 如果你想从一开始就按长期归档结构保存，也可以显式写成 `bert/artifacts/runs/single_label/<run_name>`
- 但那是你主动指定后的路径，不是脚本默认值

## 5. 双标签训练：`05_train_dual_label_classifier.py`

如果你的人工审核表里已经有：

- `broad`
- `strict`

那通常直接走这个脚本。

常用命令：

```bash
.venv/bin/python bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_part1.csv" "bert/data/reviewed_part2.csv" \
  --base_output_dir "bert/artifacts/dual_label_run"
```

如果你想固定留一份测试集：

```bash
.venv/bin/python bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_pool_a.csv" "bert/data/reviewed_pool_b.csv" \
  --train_path "bert/data/reviewed_manual_boost.csv" \
  --test_path "bert/data/reviewed_external_test.csv" \
  --base_output_dir "bert/artifacts/dual_label_holdout"
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

结构上要注意：

- `05` 不是 `04` 的下一步
- 它是另一条训练分支
- 真实分叉是：单标签走 `03 -> 04 -> 06`，双标签走“已审核表 -> 05 -> 取其中的 broad 或 strict 模型去跑 06”

如果你想按长期归档布局保存双标签结果，可以显式写成：

- `bert/artifacts/runs/dual_label/<run_name>`

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
  --model_dir "bert/artifacts/dual_label_run/broad/best_model" \
  --input_pattern "data/processed/text_dedup/*.parquet" \
  --output_dir "data/processed/text_dedup_predicted_broad" \
  --device cuda
```

如果你不显式传 `--output_dir`：

- `06` 默认会写到 `data/processed/text_dedup_predicted`
- `07` 默认却是从 `data/processed/text_dedup_predicted_broad/*.parquet` 继续读

所以：

- 想直接接上 `07`，就不要省这一个参数

常见输出列包括：

- `pred_label`
- `pred_label_text`
- `pred_prob_1`
- `pred_prob_0`
- `pred_confidence`

如果你只是想保留正样本，也可以看脚本的 `--only_positive` 参数。

## 进入 `07-09_prepare + 10` 之前，先确认这三件事

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

从架构上说，`07` 是 broad-analysis 的真正分叉点。

它后面不是简单的：

- `07 -> 08 -> 09 -> 10`

而是：

- `07 -> 08`
- `07 -> 09`
- `08 + 09 -> 10`

也就是说：

- `08` 和 `09` 都直接吃 `analysis_base.parquet`
- `10` 才是第一次同时依赖 `08` 和 `09` 结果的步骤

## 8. 主题建模：`08_topic_model_bertopic.py`

这一步更适合回答：

- broad 语料里有哪些主题
- 各主题在时间上怎么变化
- 不同关键词、不同地区上，主题占比怎么变

如果你想按当前代码的真实执行顺序理解 `08`，直接看：

- [`08_topic_model_bertopic_当前操作说明.md`](08_topic_model_bertopic_当前操作说明.md)

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
  --outlier_reduction_strategy c-tf-idf+distributions \
  --outlier_reduction_threshold 0.05 \
  --resume
```

这里的意思是先把原始聚类和 outlier 回填做稳，再看是否需要额外压主题数。
对于 broad 语料，不建议一上来就固定压到很小的 `nr_topics`。

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

这一部分负责从 `07` 生成的 broad 分析底表中提取关键词周围的高价值语义线索，并将结果整理成可阅读、可复核、可继续修正的文件。它在整个 `07-10` 链条中的位置，是把“关键词相关文本”转成“关键词在不同时间段里与哪些表达共同出现，这些表达最终落入哪些语境桶”的结构化结果。后续的 `10` 会继续利用这里的共现和邻居结果做漂移比较。

### 输入与主要依赖

主脚本 [09_keyword_semantic_analysis.py](/Users/apple/Local/fdurop/code/result/bert/09_keyword_semantic_analysis.py) 默认读取：

- `bert/artifacts/broad_analysis/analysis_base.parquet`

该文件由 `07_build_broad_analysis_base.py` 生成，提供文本内容、标准化关键词标签和时间字段，是 09 的唯一底表输入。

脚本运行时还会用到：

- `bert/config/topic_stopwords.txt`：分词后的停用词表
- `jieba`：中文分词
- `sentence-transformers`：语义邻居重排

整理脚本 [09_prepare_semantic_midterm.py](/Users/apple/Local/fdurop/code/result/bert/09_prepare_semantic_midterm.py) 继续读取 09 主脚本的结果，并额外使用：

- `bert/config/semantic_midterm_noise_terms.txt`：仅作用于 09 整理阶段的噪声词
- `bert/config/semantic_bucket_rules.json`：自动分桶规则
- `bert/config/semantic_bucket_overrides.csv`：人工覆盖表

### 关键处理过程

`09_keyword_semantic_analysis.py` 的处理过程分为三步。

第一步是关键词过滤和时间切分。脚本从 `analysis_base.parquet` 中保留目标关键词对应的文本，生成 `period_label`，并对文本做分词、去停用词和 token set 构建。[09_keyword_semantic_analysis.py](/Users/apple/Local/fdurop/code/result/bert/09_keyword_semantic_analysis.py#L357)

第二步是共现词打分。脚本按 `keyword + period_label` 统计候选词，计算文档频次、词频、PMI 和 lift，保留每个关键词、每个时间段中排序靠前的候选项。[09_keyword_semantic_analysis.py](/Users/apple/Local/fdurop/code/result/bert/09_keyword_semantic_analysis.py#L403)

第三步是语义邻居重排。脚本对每个关键词及其候选词编码，计算 embedding similarity，并保留每组候选中的高相似度词项。[09_keyword_semantic_analysis.py](/Users/apple/Local/fdurop/code/result/bert/09_keyword_semantic_analysis.py#L435)

`09_prepare_semantic_midterm.py` 在此基础上继续做整理工作。它先将共现词表和邻居表合并成统一候选池，再根据噪声规则、频次门槛、语义支持和规则桶生成中期整理候选表。随后回查 `tokenized_analysis_base.parquet`，补充代表文本、文本重复度和命中统计，在此基础上生成总体 shortlist、分期 shortlist、语义轨迹表、漂移摘要表以及人工修正工作表。[09_prepare_semantic_midterm.py](/Users/apple/Local/fdurop/code/result/bert/09_prepare_semantic_midterm.py#L1059)

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

这里最好把两件事分开记：

- `09_keyword_semantic_analysis.py`：生成机器侧的 `viz_inputs/`
- `09_prepare_semantic_midterm.py`：把 09 结果整理成真正给人读的 `readouts/`

如果你只跑前者，不跑后者，`09` 这一段其实还没进入最适合阅读的状态。

### 主脚本输出

`09_keyword_semantic_analysis.py` 的直接产物位于 `bert/artifacts/broad_analysis/semantic_analysis/`：

- `semantic_analysis_summary.json`：本轮运行的参数、输入路径和输出路径摘要
- `viz_inputs/tokenized_analysis_base.parquet`：完成关键词过滤、时间切分和分词后的 09 专用底表
- `viz_inputs/keyword_cooccurrence.csv`：按 `keyword + period_label` 生成的共现词候选表
- `viz_inputs/keyword_semantic_neighbors.csv`：基于共现候选继续做 embedding 重排后的邻居表

其中，`keyword_cooccurrence.csv` 和 `keyword_semantic_neighbors.csv` 是整理阶段的主要输入；`tokenized_analysis_base.parquet` 会在整理阶段回查代表文本、命中次数和文本重复度。

### 整理阶段输出

`09_prepare_semantic_midterm.py` 将主脚本结果整理为四组文件，统一写入 `readouts/`。

`01_start_here/` 放第一次阅读时优先查看的结果：

- `semantic_midterm_notes.md`：导读、噪声概况和关键词摘要
- `semantic_keyword_overview.csv`：ALL 时段的总体 shortlist，保留每个关键词的代表词、分桶结果和示例文本
- `semantic_context_trajectory.csv`：按 `keyword + period_label + context_bucket` 聚合后的时间轨迹表
- `semantic_context_shift_summary.csv`：由轨迹表进一步压缩得到的变化摘要

`02_period_detail/` 放分期细表：

- `semantic_period_shortlist.csv`：每个关键词、每个时间段保留的代表词
- `semantic_period_overview.csv`：对 `semantic_period_shortlist.csv` 的 period 级汇总
- `semantic_context_bucket_summary.csv`：对总体 shortlist 的 context bucket 摘要

`03_workbench/` 放人工复核和规则修正所需文件：

- `semantic_midterm_candidates.csv`：整理阶段生成的总候选表，包含自动保留标记、分桶结果、排序分和示例文本
- `semantic_bucket_override_template.csv`：从 shortlist 中抽出的人工改桶模板
- `semantic_midterm_coding_template.csv`：便于进一步人工筛选的工作表
- `semantic_noise_diagnostics.csv`：各类自动剔除原因的统计摘要

`99_meta/` 放运行元信息：

- `semantic_midterm_summary.json`：本轮整理的行数统计和参数摘要
- `semantic_midterm_operation_log.md`：生成日志

### 文件依赖关系

09 的文件依赖关系可以概括为下面这条链：

- `analysis_base.parquet`
- `tokenized_analysis_base.parquet`
- `keyword_cooccurrence.csv`
- `keyword_semantic_neighbors.csv`
- `semantic_midterm_candidates.csv`
- `semantic_keyword_overview.csv` / `semantic_period_shortlist.csv`
- `semantic_context_trajectory.csv`
- `semantic_context_shift_summary.csv`

其中：

- `semantic_midterm_candidates.csv` 是整理阶段的候选总表，后续 shortlist、工作台文件和摘要表都从这里继续生成。[09_prepare_semantic_midterm.py](/Users/apple/Local/fdurop/code/result/bert/09_prepare_semantic_midterm.py#L1070)
- `semantic_keyword_overview.csv` 来自 `ALL` 时段候选的二次筛选和重排，是总体层面的核心读物。[09_prepare_semantic_midterm.py](/Users/apple/Local/fdurop/code/result/bert/09_prepare_semantic_midterm.py#L1074)
- `semantic_period_shortlist.csv` 来自非 `ALL` 时段候选的分期筛选，是时间段解释的主要入口。[09_prepare_semantic_midterm.py](/Users/apple/Local/fdurop/code/result/bert/09_prepare_semantic_midterm.py#L1074)
- `semantic_context_trajectory.csv` 基于 period 级候选按 `context_bucket` 聚合生成。[09_prepare_semantic_midterm.py](/Users/apple/Local/fdurop/code/result/bert/09_prepare_semantic_midterm.py#L1129)
- `semantic_context_shift_summary.csv` 由轨迹表继续汇总得到，是用于汇报和章节写作的摘要结果。[09_prepare_semantic_midterm.py](/Users/apple/Local/fdurop/code/result/bert/09_prepare_semantic_midterm.py#L1131)

### 主要文件用途

`semantic_keyword_overview.csv` 包含总体 shortlist。文件中保留了 `term_doc_freq`、`lift`、`embedding_similarity`、`midterm_score`、`theme_bucket`、`context_bucket` 以及代表文本，适合用于命名语义簇、检查总体分桶和确认示例文本是否可靠。

`semantic_context_trajectory.csv` 汇总每个时间段中各个语义桶的强度，包含 `context_term_count`、`context_term_doc_freq_sum`、`context_midterm_score_sum`、`context_doc_freq_share`、`context_score_share` 和 `lead_terms`，适合用于观察语义桶的时间变化。

`semantic_context_shift_summary.csv` 从轨迹表中提取 `first_period`、`latest_period`、`score_share_change`、`peak_period`、`peak_context_score_share` 和 `representative_terms_over_time`，是后续汇报中最稳定的摘要来源之一。

`semantic_period_shortlist.csv` 展开到 `keyword + period_label + term` 粒度，保留分期代表词、示例文本和分桶结果，用于追查某一时期为什么出现特定语义变化。

`semantic_bucket_override_template.csv` 和 `bert/config/semantic_bucket_overrides.csv` 对应，前者负责收集需要人工修正的行，后者负责在下一轮整理时生效。

`semantic_noise_diagnostics.csv` 用于检查噪声来源，例如自变体、ASCII 词、平台招募词或手工噪声词是否占比过高。停用词或规则调整通常从这张表开始。

### 建议阅读顺序

第一次查看 09 结果时，优先顺序如下：

1. `readouts/01_start_here/semantic_midterm_notes.md`
2. `readouts/01_start_here/semantic_keyword_overview.csv`
3. `readouts/01_start_here/semantic_context_trajectory.csv`
4. `readouts/01_start_here/semantic_context_shift_summary.csv`
5. `readouts/02_period_detail/semantic_period_shortlist.csv`

前四个文件足以建立总体判断；`semantic_period_shortlist.csv` 用于回查某一时间段的具体词项。只有在分桶需要修正、readout 噪声偏高或候选数量明显异常时，才需要继续进入 `03_workbench/`。

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

## `07-09_prepare + 10` 的输出目录怎么放

当前建议固定成这一套：

- `08` 的主结果放 `bert/artifacts/broad_analysis/topic_model_BAAI/`
- `09` 的主结果放 `bert/artifacts/broad_analysis/semantic_analysis/`
- `10` 的主结果放 `bert/artifacts/broad_analysis/drift_analysis/`

补一句实际情况：

- 仓库里如果同时出现 `topic_model_compare/`，那通常表示参数比较或实验目录
- 它不是 `10` 默认读取的 canonical `08` 输出目录

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
- 先不设 `--nr_topics`，先看原始聚类质量

如果你一上来就强行把主题压到很少，结果反而可能更难解释，甚至把多个语义场硬并到一个大 topic 里。

## 什么时候该扩停用词表

如果你发现这些东西反复跑进 topic 头部，通常就该考虑扩停用词了：

- 平台口头禅和连接词
- 编号、页码、纯数字
- 节日问候和模板短句
- 明显的平台活动词或运营词

对应文件通常是：

- `bert/config/topic_stopwords.txt`

## `07 -> 08` 之间要不要加规则过滤

如果你的研究设计明确不想研究某些语义场，比如交易帖、抽卡帖或特定垂类社区内容，那可以在 `07` 和 `08` 之间加一层轻过滤。

但如果你跑的是 broad 口径，就不要默认把游戏、饭圈、股票这类内容当噪声删掉；它们往往本身就是 broad 语义扩散的一部分。

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
