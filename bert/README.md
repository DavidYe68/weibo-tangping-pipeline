# BERT 工作流说明

本手册负责 `bert/` 目录下的完整下游流程：抽样、预标注、标签整理、训练、全量预测，以及 `07-10` 的 broad 分析链。主流程 `raw/ -> data/processed/` 请看 [USER_MANUAL.md](/Users/apple/Local/fdurop/code/result/USER_MANUAL.md)。

## 运行前先确认环境

默认使用仓库根目录下的 `.venv`。

- macOS / Linux：`.venv/bin/python`
- Windows PowerShell：`.\.venv\Scripts\python.exe`

下面示例统一采用 macOS / Linux 写法；Windows 下把 `.venv/bin/python` 替换为 `.\.venv\Scripts\python.exe` 即可。

## 核心原则

1. `02_llm_label_local.py` 只负责生成预标注草稿，不负责替代人工判断。
2. 进入训练阶段的数据，默认都应当是人工复核过的 CSV/XLSX。
3. `04_train_bert_classifier.py` 负责单标签训练。
4. `05_train_dual_label_classifier.py` 负责 `broad / strict` 双标签训练。
5. `06_predict_bert_classifier.py` 负责把训练好的模型批量打到全量 parquet 上。
6. `07-10` 是围绕 broad 预测结果展开的分析链。

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
└── artifacts/       模型、评估结果、预测结果、分析结果（建议按 runs/ 和 broad_analysis/ 分层）
```

## artifacts 目录约定

为了避免训练 run、分析中间结果和历史可视化堆在同一层，建议把 `bert/artifacts/` 固定整理成下面这个结构：

```text
bert/artifacts/
├── runs/
│   ├── dual_label/      dual-label 训练 run
│   └── single_label/    single-label 训练 run
└── broad_analysis/      07-11 分析链的标准输出
    ├── overview/        只保留“先看什么”的浓缩表和 manifest
    ├── snapshots/       带日期或一次性批处理的快照输出
    └── legacy/          已淘汰或重复的历史输出
```

补充约定：

- `04` / `05` 的新训练产物，优先直接写到 `bert/artifacts/runs/...`
- `07`-`11` 的分析产物，继续放在 `bert/artifacts/broad_analysis/`
- `broad_analysis/README.md` + `broad_analysis/overview/` 是默认阅读入口；原始明细继续保留在各自目录
- BERTopic 主结果目录优先使用 `bert/artifacts/broad_analysis/topic_model_BAAI/`
- 带日期后缀或 overnight 的分析批次，优先放进 `bert/artifacts/broad_analysis/snapshots/`
- 历史遗留的旧版 topic 可视化或按 embedding 名额外分出的目录，统一挪到 `bert/artifacts/broad_analysis/legacy/`

仓库里带了一个可重复执行的整理脚本：

```bash
.venv/bin/python bert/scripts/organize_artifacts.py
```

说明：

- 这个脚本会把顶层的训练 run 归档到 `runs/`
- 会把 dated / overnight 的 broad-analysis 输出归档到 `snapshots/`
- 会把旧版或重复的 broad-analysis 输出归到 `legacy/`
- 如果 `topic_visualization/` 里只有一层模型名子目录，会把 bundle 摊平回标准位置
- 脚本默认值里仍保留了部分旧路径名以兼容老命令；如果你想从一开始就保持整洁，训练时请显式把输出目录写到 `runs/`

## 最常见的真实流程

### 1. 从 parquet 抽样

默认输入是 `data/processed/text_dedup/*.parquet`，默认输出是 `bert/data/sample.csv`。

```bash
.venv/bin/python bert/01_stratified_sampling.py \
  --input "data/processed/text_dedup/*.parquet" \
  --output "bert/data/sample.csv" \
  --n 6000 \
  --report_path "bert/data/sampling_report.json"
```

### 2. 用 LLM 做预标注

默认输入是 `bert/data/sample.csv`，默认输出是 `bert/data/labeled.csv`。

```bash
.venv/bin/python bert/02_llm_label_local.py \
  --input "bert/data/sample.csv" \
  --output "bert/data/labeled.csv" \
  --report_path "bert/data/labeling_report.json"
```

说明：

- 这一步的输出只能当“待审核草稿”。
- 如果你不改参数，`02` 的默认输出可以直接接 `03` 的默认输入。
- 请先人工审核，再把结果拿去训练。

运行前建议先把 provider 配清楚：

```bash
cp bert/llm_label_local.example.toml bert/llm_label_local.toml
.venv/bin/python bert/02_llm_label_local.py --config bert/llm_label_local.toml
```

补充说明：

- 如果 `bert/llm_label_local.toml` 存在，`02` 会默认读取它；最稳妥的做法是复制示例模板后，把 `[labeler]` 和 `[fixer]` 明确改成你本机实际可用的 provider / model。
- 如果你不用配置文件，也可以直接传参数，或者准备环境变量：`DASHSCOPE_API_KEY`、`QWEN_API_KEY`、`OPENAI_API_KEY`、`FIXER_API_KEY`。
- 如果你走本地 `ollama`，通常不需要 API key，但要先确保本地服务已经启动，对应模型已经拉好。

### 3. 标签整理

如果你的训练任务是单标签二分类，可以先把审核结果整理成标准二值标签：

```bash
.venv/bin/python bert/03_normalize_labels.py \
  --input_csv "bert/data/labeled.csv" \
  --output_csv "bert/data/labeled_binary.csv"
```

这一步主要服务于单标签训练；如果你已经准备好了 `broad / strict` 两列，可以直接进入 `05`。

如果人工审核被拆成多份 XLSX，可以先合并再训练：

```bash
.venv/bin/python bert/scripts/merge_xlsx_annotations.py \
  bert/data/review_part1.xlsx bert/data/review_part2.xlsx \
  -o bert/data/review_merged.xlsx \
  --report-json bert/data/review_merged.report.json
```

说明：

- 第一个 XLSX 会被当作主模板。
- 合并后的 `review_merged.xlsx` 可以直接继续喂给 `04` 或 `05`。

### 4. 单标签训练

如果你的审核结果只有一套标签，例如 `label` / `tangping_related` / `tangping_related_label`：

```bash
.venv/bin/python bert/04_train_bert_classifier.py \
  --input_csv "bert/data/reviewed_a.csv" "bert/data/reviewed_b.csv" \
  --output_dir "bert/artifacts/runs/single_label/single_label_run"
```

如果你想显式指定哪个文件只进训练、哪个文件单独留作测试：

```bash
.venv/bin/python bert/04_train_bert_classifier.py \
  --train_csv "bert/data/reviewed_train_extra.csv" \
  --input_csv "bert/data/reviewed_pool_a.csv" "bert/data/reviewed_pool_b.csv" \
  --test_csv "bert/data/reviewed_holdout.csv" \
  --output_dir "bert/artifacts/runs/single_label/single_label_holdout"
```

规则：

- `--input_csv`：这些文件会先合并，再随机切成 train/val/test。
- `--train_csv` / `--train_only_csv`：这些文件只进训练集。
- `--val_csv`：这些文件只进验证集。
- `--test_csv`：这些文件只进测试集。

### 5. 双标签训练

如果你的人工审核表里同时有 `broad` 和 `strict` 两列：

```bash
.venv/bin/python bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_part1.csv" "bert/data/reviewed_part2.csv" \
  --base_output_dir "bert/artifacts/runs/dual_label/dual_label_run"
```

如果你想把某一份文件固定留作测试，另一些只进训练：

```bash
.venv/bin/python bert/05_train_dual_label_classifier.py \
  --input_path "bert/data/reviewed_pool_a.csv" "bert/data/reviewed_pool_b.csv" \
  --train_path "bert/data/reviewed_manual_boost.csv" \
  --test_path "bert/data/reviewed_external_test.csv" \
  --base_output_dir "bert/artifacts/runs/dual_label/dual_label_holdout"
```

规则和 `04` 一样：

- `--input_path`：合并后随机切分。
- `--train_path`：只进训练集。
- `--val_path`：只进验证集。
- `--test_path`：只进测试集。

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

补充约定：

- BERT 训练 / 预测默认优先 `cleaned_text`，适合把 emoji 压成统一占位的二分类场景
- `07`-`10` 分析链也默认优先 `cleaned_text`；如果你想实验保留 emoji 文本对主题词的影响，再显式传 `--text_col cleaned_text_with_emoji`

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

如果你已经完成：

1. 主流程产出 `data/processed/text_dedup/*.parquet`
2. 人工审核
3. `05_train_dual_label_classifier.py`

那么接下来建议先跑一次 `06`，把 `broad` 模型打到全量语料上。

### 6. 用 broad 模型做全量预测

`06_predict_bert_classifier.py` 的默认输出目录是 `data/processed/text_dedup_predicted/`，但如果后续要进入 `07-10`，建议显式写成 `data/processed/text_dedup_predicted_broad`，这样能直接对齐 `07` 的默认输入。

```bash
.venv/bin/python bert/06_predict_bert_classifier.py \
  --model_dir "bert/artifacts/runs/dual_label/dual_label_run/broad/best_model" \
  --input_pattern "data/processed/text_dedup/*.parquet" \
  --output_dir "data/processed/text_dedup_predicted_broad" \
  --device cuda
```

`06` 常见输出列包括：

- `pred_label`
- `pred_label_text`
- `pred_prob_1`
- `pred_prob_0`
- `pred_confidence`

## 07-10 分析链路

### 7. `07_build_broad_analysis_base.py`

作用：

- 读取 `06` 的预测结果
- 规范化文本列、时间列、关键词列
- 自动识别并规范化 IP 属地列；缺失 IP 会统一记为 `UNKNOWN_IP`
- 默认只保留 `pred_label == 1` 的正样本
- 生成后续 `08` / `09` 共用的分析底表
- 默认沿用仓库的文本列自动识别顺序，通常会选到 `cleaned_text`

默认命令：

```bash
.venv/bin/python bert/07_build_broad_analysis_base.py
```

如果你的预测文件放在别的位置：

```bash
.venv/bin/python bert/07_build_broad_analysis_base.py \
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

### 8. `08_topic_model_bertopic.py`

作用：

- 在 `07` 生成的分析底表上做 BERTopic
- 输出文档级 topic 结果、topic 词表，以及 `topic / 时间 / IP` 三个维度的占比表
- 缺失 IP 不会被丢掉，而是作为 `UNKNOWN_IP` 单独保留
- 支持 embedding 和 UMAP 降维结果断点续跑，避免中途打断后从头编码或重跑降维
- 默认启用 `multilingual + jieba` 的中文主题提词，避免 topic 标签被英文/代码碎片主导

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

常用可选参数：

- `--device auto|cpu|cuda|mps`：控制 sentence-transformers 的编码设备
- `--embedding_model`：指定模型名或本地目录；默认改为 `BAAI/bge-small-zh-v1.5`
- `--topic_language multilingual|english`：BERTopic 的语言模式；中文语料建议保持默认 `multilingual`
- `--topic_tokenizer jieba|default`：topic 提词的分词方式；默认 `jieba`
- `--topic_stopwords_path`：topic 提词停用词表，默认 `bert/config/topic_stopwords.txt`
- `--topic_token_min_length`：`jieba` 提词时保留的最短 token 长度，默认 `2`
- `--umap_n_neighbors`：显式控制 UMAP 邻居数，默认 `30`
- `--local_files_only`：只从本地加载 embedding 资源
- `--calculate_probabilities`：显式计算“每篇文档 x 全部 topic”的完整概率矩阵；非常吃内存，默认关闭
- `--resume`：如果已有 embedding / UMAP checkpoint，则尽量从 checkpoint 继续
- `--checkpoint_dir`：自定义 checkpoint 目录
- `--umap_low_memory / --no-umap_low_memory`：控制 UMAP 低内存模式，默认开启
- `--hdbscan_core_dist_n_jobs`：控制 HDBSCAN 并行度；更小的值更省内存
- `--ip_col`：手动指定 IP 列名

重点输出：

- `bert/artifacts/broad_analysis/topic_model_BAAI/document_topics.parquet`
- `bert/artifacts/broad_analysis/topic_model_BAAI/topic_info.csv`
- `bert/artifacts/broad_analysis/topic_model_BAAI/topic_overview.csv`
- `bert/artifacts/broad_analysis/topic_model_BAAI/topic_terms.csv`
- `bert/artifacts/broad_analysis/topic_model_BAAI/topic_share_by_period.csv`
- `bert/artifacts/broad_analysis/topic_model_BAAI/topic_share_by_period_and_keyword.csv`
- `bert/artifacts/broad_analysis/topic_model_BAAI/topic_share_by_ip.csv`
- `bert/artifacts/broad_analysis/topic_model_BAAI/topic_share_by_period_and_ip.csv`
- `bert/artifacts/broad_analysis/topic_model_BAAI/topic_share_by_period_and_ip_and_keyword.csv`
- `bert/artifacts/broad_analysis/topic_model_BAAI/topic_model_summary.json`

补充：

- `topic_info.csv` 会额外预留 `topic_label_machine` 和 `topic_label_zh` 两列，便于后续手工补中文主题标签。
- `topic_overview.csv` 会把主题规模、峰值时间、dominant keyword 和 top terms 预先整理好，适合直接接中期展示或人工 topic 编码。

### 9. `09_keyword_semantic_analysis.py`

作用：

- 对每个关键词在不同时间段做共现词分析
- 用 embedding 对候选词再排序，得到 semantic neighbors
- 脚本会输出分阶段进度，便于判断是在分词、统计还是 embedding 阶段

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

- `bert/artifacts/broad_analysis/semantic_analysis/keyword_cooccurrence.csv`
- `bert/artifacts/broad_analysis/semantic_analysis/keyword_semantic_neighbors.csv`
- `bert/artifacts/broad_analysis/semantic_analysis/tokenized_analysis_base.parquet`
- `bert/artifacts/broad_analysis/semantic_analysis/semantic_analysis_summary.json`

补充：

- `08` / `09` 默认共用停用词表 `bert/config/topic_stopwords.txt`
- `09` 的默认 embedding 也改为 `BAAI/bge-small-zh-v1.5`，和 `08` 保持一致

如果你发现 `09` 的原始词表太脏，不适合直接拿去讲中期，可以在已有 `09` 输出上再跑一层“报告整理”：

```bash
.venv/bin/python bert/09_prepare_semantic_midterm.py \
  --semantic_dir "bert/artifacts/broad_analysis/semantic_analysis"
```

这一步不会重跑重型 embedding，只会读取 `09` 的现成结果并生成：

- `midterm_bundle/semantic_keyword_overview.csv`：每个关键词总体上更适合进入中期报告的词
- `midterm_bundle/semantic_period_shortlist.csv`：按月筛过一轮的 lead terms
- `midterm_bundle/semantic_midterm_coding_template.csv`：人工编码模板，附代表原文
- `midterm_bundle/semantic_noise_diagnostics.csv`：被自动判成噪声的词和原因
- `midterm_bundle/semantic_midterm_notes.md`：面向中期汇报的阅读说明

### 10. `10_concept_drift_analysis.py`

作用：

- 比较相邻时间段的共现词变化
- 比较相邻时间段的 semantic neighbors 变化
- 比较相邻时间段的 topic share 变化
- 除了按关键词和总体比较，也会额外输出按 IP、按 `IP + 关键词` 的 topic 漂移结果

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

### 11. `11_visualize_topic_outputs.py`

作用：

- 直接读取 `08` 产出的 BERTopic 结果，生成单一 `topic_dashboard.html`
- 用五个标签页组织中期展示叙事：总览、关键词画像、时间演化、地域分布、主题浏览
- 支持从 `08` 的 `topic_model_summary.json` 自动对齐输入路径，避免手动重复敲多条 CSV 路径
- 支持展示层过滤：`--min_topic_size` 和 `--noise_pattern` 只影响 `11` 输出，不改动 `08` 的原始 CSV

适用场景：

- 如果你要讲“BERTopic 聚出了什么、三个关键词分别落到哪些主题、主题如何随时间变化、哪些省份更集中”，直接跑 `11` 就行。
- 如果你已经有 `08` 的 summary，优先用 `--from_summary`，这样最不容易路径对错。

默认命令：

```bash
.venv/bin/python bert/11_visualize_topic_outputs.py \
  --from_summary "bert/artifacts/broad_analysis/topic_model_BAAI/topic_model_summary.json"
```

常用变体：

```bash
.venv/bin/python bert/11_visualize_topic_outputs.py \
  --from_summary "bert/artifacts/broad_analysis/topic_model_BAAI/topic_model_summary.json" \
  --top_n_topics 30 \
  --top_n_terms 12 \
  --min_topic_size 300 \
  --noise_pattern "佛系收|佛系出|中转|抽卡|代肝|黑市|挂卡|求扩"
```

重点输出：

- `bert/artifacts/broad_analysis/topic_visuals/topic_dashboard.html`
- `bert/artifacts/broad_analysis/topic_visuals/topic_display_table.csv`
- `bert/artifacts/broad_analysis/topic_visuals/topic_visualization_summary.json`

补充：

- `topic_dashboard.html` 是唯一入口，不再默认拆成多份附图 HTML。
- 总览页会直接给出 Topic -1 占比、Top 主题与长尾桶、关键词体量饼图、每期文档量折线。
- 关键词画像页保留“躺平 / 摆烂 / 佛系”的对照，同时把结果收紧成更适合汇报的堆叠图 + 雷达图。
- 时间演化页除了重点主题折线，还新增“时段 × 主题”热力图。
- 地域分布页保留省级热力图，并支持切换到头部主题。
- 主题浏览页会给出可排序的主题卡片：label、文档数、峰值时间、top terms、sparkline、关键词构成条。
- `topic_display_table.csv` 是汇报用的展示主题清单，不会覆盖 `08` 的底表。
- 图表使用 ECharts CDN 资源，打开 HTML 时需要能访问对应脚本地址。
- 如果你已经在 `topic_info.csv` 里补了 `topic_label_zh`，图里会优先使用人工中文标签；否则会回退到 `topic_terms.csv` 的前几个词。

## 输出目录规范

推荐把 `08` 和 `11` 的输出目录拆开管理：

- `08` 的原始主题模型结果放在 `bert/artifacts/broad_analysis/topic_model/<run_tag>/`
- `11` 的展示结果放在 `bert/artifacts/broad_analysis/topic_visuals/<run_tag>/`

例如：

```bash
.venv/bin/python bert/08_topic_model_bertopic.py \
  --output_dir "bert/artifacts/broad_analysis/topic_model/bge_small_v15"

.venv/bin/python bert/11_visualize_topic_outputs.py \
  --from_summary "bert/artifacts/broad_analysis/topic_model/bge_small_v15/topic_model_summary.json"
```

补充说明：

- 如果 `--from_summary` 指向的是规范目录 `topic_model/<run_tag>/topic_model_summary.json`，`11` 默认会把输出落到对应的 `topic_visuals/<run_tag>/`
- 如果 `--from_summary` 指向的是历史平铺目录，例如 `topic_model_BAAI/topic_model_summary.json`，`11` 默认会把输出落到同级的 `topic_visuals/`
- 历史遗留目录如 `topic_model_BAAI`、`topic_visuals_BAAI`、`topic_interpretability_BAAI` 建议归档到 `bert/artifacts/broad_analysis/legacy/` 或按 `<run_tag>` 重新整理
- 新版 `11` 的 `topic_dashboard.html` 已覆盖旧版 `topic_prevalence`、`topic_keyword_alignment`、`topic_evolution_heatmap`、`topic_term_detail`、`topic_wordclouds` 的主要信息点

## 主题清理建议

下面这些是建议，不是默认行为。它们会改变主题结果，建议先单独试一轮，再决定是否固化到主流程。

### A. 先从训练参数收敛主题数

对 280 万量级语料来说，`--min_topic_size 30` 往往偏小，容易切出大量碎片 topic。比较稳妥的起点：

- `--min_topic_size 300`
- `--nr_topics 60` 或 `auto`
- `--umap_n_neighbors 50`

### B. 扩停用词表

建议优先检查这些噪声来源是否反复进入头部 topic，再决定是否写进 `bert/config/topic_stopwords.txt`：

| 噪声类型 | 代表 token | 可考虑加入停用词 |
|---|---|---|
| 粉圈 / 艺人 | 饭圈、爱豆、人名 | `王鹤` `宋茜` `杨幂` `王一博` `张泽禹` `丁程鑫` `柳智敏` |
| 游戏 / 手游 | 玩家、战队、角色 | `第五人格` `光遇` `sky` `xyg` `estar` `ag` `原神` `暖暖` |
| 二手交易 / 应援 | 周边、抽卡、代肝 | `佛系收` `佛系出` `中转站` `周边` `代肝` `黑市` `挂卡` `抽卡` `应援` |
| 纯数字 / 编号 | 年份、页码、计量 | `2023` `2024` `2025` `2026` `p1` `p2` `p3` `p4` `500w` `100w` |
| 日历 / 天气 | 节日、天气、问候 | `新年快乐` `生日快乐` `下雨天` `周末` `国庆` `五一` `双十` |

### C. 在 `07 -> 08` 之间加规则过滤

如果研究重点是“躺平 / 摆烂 / 佛系”话语，而不是交易黑话或游戏流量，建议在 `07` 产物进入 `08` 之前做一次轻量过滤：

- 命中 `(佛系收|佛系出|中转|抽卡|代肝|黑市|挂卡|求扩)` 的帖子，直接剔除或只抽样一部分进入 topic model
- 关键词最好作为独立 token 判断，而不是纯子串匹配；中文场景可以先分词，再判断 token 是否命中

### D. 训练后降噪

如果不想重跑 embedding，可以优先考虑 BERTopic 自带的后处理：

- `topic_model.reduce_outliers(docs, topics, strategy="embeddings")`
- `topic_model.reduce_topics(docs, nr_topics=50)`
- 对明显重复的簇再做手工 `merge_topics`

### E. Seeded BERTopic

如果你最关心的是“躺平 / 摆烂 / 佛系”三类姿态，而不是让生活流 topic 自然冒出来，可以考虑显式 seed：

```python
BERTopic(
    seed_topic_list=[
        ["躺平", "内卷", "加班", "不想努力", "摸鱼"],
        ["摆烂", "破罐破摔", "随便", "不想学", "放弃"],
        ["佛系", "随缘", "无所谓", "淡定", "看开"],
    ]
)
```

这样更适合研究导向的命题，但代价是主题结构会更“带假设”。

## Windows 机器上的注意事项

`08` / `09` 会额外依赖：

- `bertopic`
- `sentence-transformers`
- `jieba`

它们已经写在根目录 [`requirements.txt`](/Users/apple/Local/fdurop/code/result/requirements.txt) 里，但第一次运行 embedding 模型时，通常还会下载模型权重。

如果你的 Windows 机器联网：

- 直接在已安装 `requirements.txt` 的环境里运行即可。

如果你的 Windows 机器离线：

- 先把 embedding 模型缓存好，或者把模型目录拷到本地。
- 运行 `08` / `09` 时把 `--embedding_model` 指到本地目录。
- 再加上 `--local_files_only`，避免脚本尝试联网下载。

## 最后建议

更稳妥的习惯是：

1. 先把主流程跑到 `data/processed/text_dedup/`。
2. 用 `01` 抽样。
3. 用 `02` 生成预标注草稿。
4. 人工审核。
5. 单标签任务走 `03 -> 04`，双标签任务直接走 `05`。
6. 用 `05` 产出的 `broad/best_model` 跑 `06` 全量预测。
7. 再顺序跑 `07`、`08`、`09`、`10`。
8. 需要换测试集或换分析口径时，只改命令参数，不改脚本逻辑。
