# BERT 子目录说明

这个 `bert/` 目录是一套围绕“躺平 / 摆烂 / 佛系”微博语料构建的本地工作流，覆盖了：

1. 从大规模去重后的微博 parquet 中抽样。
2. 用 LLM 做“相关 / 无关”初筛标注。
3. 把人工或 LLM 标注统一归一成二分类标签。
4. 训练中文 BERT 分类器。
5. 用训练好的模型回灌整批 parquet 数据。
6. 针对 `sample_6000_labeled.xlsx` 同时训练 `broad` / `strict` 两套标准。
7. 合并多人或多批次的 Excel 标注结果。

如果你只想快速跑通一遍，建议按下面顺序使用：

`01_stratified_sampling.py` -> `02_llm_label_local.py` -> `03_normalize_labels.py` -> `04_train_bert_classifier.py` -> `05_predict_bert_classifier.py`

如果你已经有人工标注好的 `sample_6000_labeled.xlsx`，则可以直接使用：

`06_train_sample_6000_dual.py`

## 目录结构

```text
bert/
├── 01_stratified_sampling.py
├── 02_llm_label_local.py
├── 03_normalize_labels.py
├── 04_train_bert_classifier.py
├── 05_predict_bert_classifier.py
├── 06_train_sample_6000_dual.py
├── llm_label_local.example.toml
├── data/
│   ├── sample_6000.csv
│   ├── sampling_report.json
│   ├── sample_1.xlsx
│   ├── sample_2.xlsx
│   ├── sample_foxi.xlsx
│   ├── sample_foxi_manuel_added.xlsx
│   ├── sample_6000_labeled.xlsx
│   └── sample_6000_labeled.merge_report.json
├── scripts/
│   └── merge_xlsx_annotations.py
└── artifacts/
    └── ... 训练输出、评估结果、模型目录
```

## 环境要求

推荐环境：

- Python 3.11 及以上
- 已安装 PyTorch、Transformers、pandas、numpy、scikit-learn、requests、openpyxl、pyarrow、tqdm

参考安装：

```bash
python3 -m pip install pandas numpy scikit-learn requests openpyxl pyarrow tqdm torch transformers
```

说明：

- `02_llm_label_local.py` 使用了 `tomllib`，因此更适合 Python 3.11+。
- `04` 和 `05` 默认会从 Hugging Face 加载模型；如果你是离线环境，需要提前把模型下载到本地并配合 `--local_files_only` 使用。
- 训练和预测支持 `cpu` / `cuda` / `mps`，默认 `--device auto`。

## 数据与产物说明

### `data/` 下的现有文件

- `sample_6000.csv`
  - 6000 条抽样结果。
  - 当前实际列包括：`id`、`cleaned_text`、`cleaned_text_with_emoji`、`text_raw`、`发布时间`、`话题`、`keyword`、`转发数`、`评论数`、`点赞数`、`ip`、`source_file`。
  - 适合作为 LLM 初筛或人工标注输入。

- `sampling_report.json`
  - 抽样报告。
  - 记录样本来源、分层维度、每个 strata 的总体量与采样量。

- `sample_1.xlsx` / `sample_2.xlsx` / `sample_foxi.xlsx`
  - 多份 Excel 标注源文件。
  - 用于合并成统一标注表。

- `sample_foxi_manuel_added.xlsx`
  - 额外手工补充的 Excel 文件。
  - 当前脚本不会自动读取它，只有你在命令里显式传入时才会参与合并或后续处理。

- `sample_6000_labeled.xlsx`
  - 已合并完成的主标注表。
  - 注意它的第一行实际上像“嵌入式表头”，`04_train_bert_classifier.py` 已经专门兼容了这种格式。
  - 训练脚本会从其中识别 `cleaned_text`、`broad`、`strict` 等列。

- `sample_6000_labeled.merge_report.json`
  - Excel 合并报告。
  - 记录合并来源、更新单元格数、追加行数、冲突情况。

### `artifacts/` 下的常见输出

训练脚本和双标准训练脚本会在这里写出结果，常见内容包括：

- `best_model/`
- `metrics.json`
- `training_history.json`
- `train_config.json`
- `train_split.csv`
- `val_split.csv`
- `test_split.csv`
- `test_predictions.csv`
- `test_misclassified.csv`
- `summary.json`
- `test_predictions_side_by_side.csv`

## 工作流概览

### 方案 A：从原始 parquet 到批量预测

1. 用 `01_stratified_sampling.py` 从去重后的 parquet 做分层抽样。
2. 用 `02_llm_label_local.py` 对样本做 LLM 初筛。
3. 用 `03_normalize_labels.py` 统一标签格式。
4. 用 `04_train_bert_classifier.py` 训练分类器。
5. 用 `05_predict_bert_classifier.py` 对整批 parquet 回灌预测。

### 方案 B：直接用已整理的 Excel 训练两套标准

1. 准备 `sample_6000_labeled.xlsx`，确保有 `broad` 和 `strict` 两列。
2. 使用 `06_train_sample_6000_dual.py`。
3. 查看 `bert/artifacts/<输出目录>/summary.json` 和 side-by-side 结果。

### 方案 C：合并多人标注的 Excel

1. 用 `scripts/merge_xlsx_annotations.py` 把多个 xlsx 合并成一个主表。
2. 再用 `06` 或 `04` 做训练。

## 各脚本详细说明

## 1. `01_stratified_sampling.py`

文件：[`01_stratified_sampling.py`](/Users/apple/Local/fdurop/code/result/bert/01_stratified_sampling.py)

### 作用

从 parquet 数据里按分层策略抽取固定数量样本，输出一个 CSV，并附带分层报告。

它会尽量自动识别：

- 时间列：`发布时间`、`created_at`、`publish_time`、`timestamp`
- 关键词列：`keyword`、`hit_keyword`、`query_keyword`
- 文本列：`cleaned_text`、`cleaned_text_with_emoji`、`text_raw`、`微博正文` 等

默认分层维度：

- 月份
- keyword
- 文本长度桶

文本长度桶固定为：

- `<=20`
- `21-60`
- `61-140`
- `>140`

### 默认输入输出

- 输入：`data/processed/text_dedup/*.parquet`
- 输出：`bert/data/sample.csv`
- 报告：`bert/data/sampling_report.json`

### 常用参数

- `--input`
  - parquet glob 路径。
- `--output`
  - 抽样结果 CSV 路径。
- `--n`
  - 目标样本数，默认 `6000`。
- `--seed`
  - 随机种子。
- `--k_min`
  - 对样本量足够大的 strata 至少抽 1 条；设为 `0` 可关闭。
- `--text_col`
  - 强制指定文本列。
- `--report_path`
  - 抽样报告路径。

### 示例

从去重 parquet 中抽 6000 条：

```bash
python3 bert/01_stratified_sampling.py \
  --input "data/processed/text_dedup/*.parquet" \
  --output "bert/data/sample_6000.csv" \
  --n 6000 \
  --seed 42 \
  --k_min 20 \
  --report_path "bert/data/sampling_report.json"
```

如果你的文本列不是自动识别候选之一：

```bash
python3 bert/01_stratified_sampling.py \
  --input "data/processed/text_dedup/*.parquet" \
  --output "bert/data/sample_custom.csv" \
  --text_col "content"
```

### 输出内容

- 抽样 CSV：保留原始列，不额外写入内部辅助列。
- 报告 JSON：记录
  - 使用了哪些分层维度
  - 实际识别到的时间列和关键词列
  - 每个 strata 的总体量和抽样量
  - 抽样后的分布统计

### 适用场景

- 构造人工标注样本。
- 构造 LLM 初筛样本。
- 避免样本被某些月份、关键词或长度段过度主导。

## 2. `02_llm_label_local.py`

文件：[`02_llm_label_local.py`](/Users/apple/Local/fdurop/code/result/bert/02_llm_label_local.py)

### 作用

对 CSV 或 parquet 中的文本逐条调用 LLM，做“是否应保留进入躺平研究语料”的二分类初筛。

脚本内部做了几件事：

- 自动检测文本列。
- 调用 labeler 模型生成 JSON 结果。
- 结果不合法时，调用 fixer 模型修复 JSON。
- 支持断点续跑。
- 支持按批保存 checkpoint。
- 最终输出带标注列的新 CSV / parquet，以及统计报告。

### 判定目标

输出不是细分类，而是“相关 / 无关”初筛。最终核心字段包括：

- `tangping_related_label`
  - `相关` / `无关`
- `tangping_related`
  - 二值字段
- `exclusion_type`
  - 无关原因
- `confidence`
  - `高` / `中` / `低`
- `llm_reason`
  - 简短判断依据
- `llm_raw`
  - 原始 LLM 输出，可选
- `llm_fixed_raw`
  - 修复后的 JSON 原文，可选

### 支持的数据格式

- 输入：`.csv`、`.parquet`
- 输出：`.csv`、`.parquet`

### 配置方式

有两种常用方式：

1. 全部通过命令行参数传入。
2. 用 `--config` 读取 TOML 配置。

默认会寻找：

- `bert/llm_label_local.toml`

仓库里提供了模板：

- [`llm_label_local.example.toml`](/Users/apple/Local/fdurop/code/result/bert/llm_label_local.example.toml)

### 常用参数

- `--config`
  - TOML 配置文件路径。
- `--input`
  - 输入 CSV / parquet。
- `--output`
  - 输出 CSV / parquet。
- `--report_path`
  - 统计报告路径。
- `--text_col`
  - 强制指定文本列。
- `--labeler_provider`
  - `qwen_openai`、`openai_compatible`、`ollama`。
- `--labeler_base_url`
  - labeler 的 API 地址。
- `--labeler_api_key`
  - labeler 的 key。
- `--labeler_model`
  - labeler 模型名。
- `--fixer_provider`
  - fixer 提供方，默认可用 `ollama`。
- `--fixer_base_url`
  - fixer API 地址。
- `--fixer_api_key`
  - fixer key。
- `--base_url`
  - Ollama 默认地址，默认 `http://localhost:11434`。
- `--fixer_model`
  - 修 JSON 用的模型名。
- `--max_chars`
  - 每条文本截断长度，默认 `800`。
- `--temperature`
  - 温度。
- `--timeout`
  - 请求超时秒数。
- `--fix_json`
  - 是否启用 fixer。
- `--save_raw`
  - 是否保留原始模型输出。
- `--save_fixed_raw`
  - 是否保留 fixer 输出。
- `--save_every`
  - 每处理多少条写一次 checkpoint。
- `--max_workers`
  - 并发线程数。
- `--request_retries`
  - 请求重试次数。
- `--retry_backoff_sec`
  - 退避秒数。

### 断点续跑规则

如果输出文件已经存在，脚本会尝试复用之前的结果，但需要满足：

- 输出文件能成功读取。
- 输入行数和输出行数一致。
- 如果存在 `id` 列，则输入和输出的 `id` 顺序一致。

满足这些条件时，脚本只会处理还没完成的行。

### 示例 1：使用 Qwen 做主标注，Ollama 做 JSON 修复

```bash
python3 bert/02_llm_label_local.py \
  --input "bert/data/sample_6000.csv" \
  --output "bert/data/labeled.csv" \
  --report_path "bert/data/labeling_report.json" \
  --labeler_provider qwen_openai \
  --labeler_model qwen-coder-plus-latest \
  --fixer_provider ollama \
  --fixer_model qwen3:8b \
  --max_workers 2 \
  --save_every 100
```

### 示例 2：直接使用 TOML 模板

```bash
python3 bert/02_llm_label_local.py \
  --config "bert/llm_label_local.example.toml"
```

### 示例 3：全部走本地 Ollama

```bash
python3 bert/02_llm_label_local.py \
  --input "bert/data/sample_6000.csv" \
  --output "bert/data/labeled.csv" \
  --labeler_provider ollama \
  --labeler_model "gpt-oss:20b" \
  --fixer_provider ollama \
  --fixer_model "qwen3:8b" \
  --base_url "http://localhost:11434"
```

### 输出内容

- 标注结果文件：在原始数据基础上追加上面提到的标注列。
- 报告 JSON：记录
  - 文本列名
  - provider / model / base_url
  - 恢复条数 `resumed`
  - fixer 触发次数
  - parse / schema 失败次数
  - 标签分布与排除类型分布

### 什么时候用它

- 你想先让 LLM 做一轮高召回初筛。
- 你需要为后续人工复核或 BERT 训练准备初始标签。
- 你需要可恢复、可批量保存的本地标注流水线。

## 3. `03_normalize_labels.py`

文件：[`03_normalize_labels.py`](/Users/apple/Local/fdurop/code/result/bert/03_normalize_labels.py)

### 作用

把各种格式的“相关 / 无关”标签统一成标准二分类数值：

- `1` 表示相关
- `0` 表示无关

脚本会自动识别标签来源列，优先候选包括：

- `tangping_related_label`
- `tangping_related`
- `label`

### 可识别的标签别名

正类示例：

- `1`
- `true`
- `yes`
- `relevant`
- `positive`
- `相关`
- `有关`

负类示例：

- `0`
- `false`
- `no`
- `irrelevant`
- `negative`
- `无关`
- `不相关`

### 默认输入输出

- 输入：`bert/data/labeled.csv`
- 输出：`bert/data/labeled_binary.csv`
- 报告：`bert/data/labeled_binary_report.json`

### 常用参数

- `--input_csv`
- `--output_csv`
- `--report_path`
- `--label_col`

### 示例

```bash
python3 bert/03_normalize_labels.py \
  --input_csv "bert/data/labeled.csv" \
  --output_csv "bert/data/labeled_binary.csv" \
  --report_path "bert/data/labeled_binary_report.json"
```

如果要强制指定来源标签列：

```bash
python3 bert/03_normalize_labels.py \
  --input_csv "bert/data/labeled.csv" \
  --label_col "tangping_related_label"
```

### 输出内容

输出 CSV 除了保留原列，还会新增或重写：

- `label_raw`
- `label`
- `label_text`
- `tangping_related_label`
- `tangping_related`

如果原表已有 `tangping_related_label` 或 `tangping_related`，脚本还会保留：

- `tangping_related_label_raw`
- `tangping_related_raw`

### 报错特点

如果有无法识别的标签值，脚本会直接报错，并给出若干示例值，方便你先清洗数据再训练。

## 4. `04_train_bert_classifier.py`

文件：[`04_train_bert_classifier.py`](/Users/apple/Local/fdurop/code/result/bert/04_train_bert_classifier.py)

### 作用

训练一个中文 BERT 二分类模型，用于判断微博文本是否“与躺平语义研究相关”。

脚本负责：

- 读取 CSV 或 Excel 数据。
- 自动识别文本列和标签列。
- 自动把标签统一到 `0/1`。
- 按随机分层或预定义 split 列切分 train / val / test。
- 训练 Transformers 分类器。
- 保存最佳模型、指标、测试集预测结果和错分样本。

### 支持输入格式

- `.csv`
- `.xlsx`
- `.xls`

### 标签列自动识别

优先检查：

- `label`
- `tangping_related`
- `tangping_related_label`
- `broad`
- `strict`

其中脚本把 `2` 也视作负类 `0`，这对某些历史标注表很有用。

### 文本列自动识别

优先检查：

- `cleaned_text`
- `cleaned_text_with_emoji`
- `text_raw`
- `微博正文`
- `text`
- `content`
- `body`
- `message`
- `post_text`
- `desc`
- `description`
- `title`

### 默认输入输出

- 输入：`bert/data/labeled_binary.csv`
- 输出目录：`bert/artifacts/tangping_bert`

### 主要参数

- `--input_csv`
  - 训练数据，CSV 或 Excel。
- `--output_dir`
  - 输出目录。
- `--model_name_or_path`
  - 基础模型，默认 `bert-base-chinese`。
- `--text_col`
  - 指定文本列。
- `--label_col`
  - 指定标签列。
- `--split_col`
  - 若数据中已带 `train/val/test` 列，可直接复用。
- `--sheet_name`
  - Excel sheet 名。
- `--max_length`
- `--batch_size`
- `--epochs`
- `--learning_rate`
- `--weight_decay`
- `--warmup_ratio`
- `--max_grad_norm`
- `--val_size`
- `--test_size`
- `--positive_threshold`
- `--seed`
- `--device`
- `--local_files_only`

### 示例 1：使用归一化后的 CSV 训练

```bash
python3 bert/04_train_bert_classifier.py \
  --input_csv "bert/data/labeled_binary.csv" \
  --output_dir "bert/artifacts/tangping_bert" \
  --model_name_or_path "bert-base-chinese" \
  --epochs 3 \
  --batch_size 16 \
  --max_length 128
```

### 示例 2：直接读取 Excel 的 `broad` 列训练

```bash
python3 bert/04_train_bert_classifier.py \
  --input_csv "bert/data/sample_6000_labeled.xlsx" \
  --label_col "broad" \
  --text_col "cleaned_text" \
  --output_dir "bert/artifacts/sample_6000_broad"
```

### 示例 3：离线加载本地模型

```bash
python3 bert/04_train_bert_classifier.py \
  --input_csv "bert/data/labeled_binary.csv" \
  --model_name_or_path "/path/to/local/model" \
  --local_files_only \
  --device mps
```

### 输出文件说明

在 `output_dir` 下会生成：

- `best_model/`
  - 验证集 `f1` 最优时的模型与 tokenizer。
- `training_history.json`
  - 每个 epoch 的训练损失和验证指标。
- `metrics.json`
  - 训练配置、数据切分规模、验证集和测试集指标。
- `train_config.json`
  - 本次运行的关键信息。
- `train_split.csv` / `val_split.csv` / `test_split.csv`
  - 训练、验证、测试数据切分结果。
- `test_predictions.csv`
  - 测试集逐条预测结果。
- `test_misclassified.csv`
  - 测试集中预测错误的样本。

### 训练逻辑特点

- 使用 `AdamW` 优化器。
- 使用线性 warmup 调度器。
- 以验证集 `f1` 作为最佳模型选择依据。
- 最后会重新加载 `best_model` 对 val / test 评估。

### 注意事项

- 清洗后有效样本数少于 20 会直接报错。
- 每个类别至少需要 2 条样本，否则没法做分层切分。
- 如果使用 `--split_col`，该列只能包含 `train`、`val`、`test` 或其常见别名。

## 5. `05_predict_bert_classifier.py`

文件：[`05_predict_bert_classifier.py`](/Users/apple/Local/fdurop/code/result/bert/05_predict_bert_classifier.py)

### 作用

用 `04` 训练好的模型，对一批 parquet 文件进行预测，并把预测结果写回新的 parquet 文件。

### 默认输入输出

- 模型目录：`bert/artifacts/tangping_bert/best_model`
- 输入：`data/processed/text_dedup/*.parquet`
- 输出目录：`data/processed/text_dedup_predicted`

### 主要参数

- `--model_dir`
  - 精调后模型目录，必须包含模型和 tokenizer。
- `--input_pattern`
  - parquet glob。
- `--output_dir`
  - 预测后 parquet 保存目录。
- `--text_col`
  - 强制指定文本列。
- `--batch_size`
- `--max_length`
- `--positive_threshold`
- `--device`
- `--local_files_only`
- `--only_positive`
  - 只保留预测为正类的行。

### 示例 1：批量预测并保留全部行

```bash
python3 bert/05_predict_bert_classifier.py \
  --model_dir "bert/artifacts/tangping_bert/best_model" \
  --input_pattern "data/processed/text_dedup/*.parquet" \
  --output_dir "data/processed/text_dedup_predicted"
```

### 示例 2：只导出预测为相关的微博

```bash
python3 bert/05_predict_bert_classifier.py \
  --model_dir "bert/artifacts/tangping_bert/best_model" \
  --input_pattern "data/processed/text_dedup/*.parquet" \
  --output_dir "data/processed/text_dedup_positive" \
  --only_positive
```

### 输出内容

每个输出 parquet 会在原始列基础上新增：

- `pred_label`
- `pred_label_text`
- `pred_prob_1`
- `pred_prob_0`
- `model_dir`

同时在输出目录下生成：

- `prediction_summary.json`

该 summary 会记录：

- 输入 glob
- 模型目录
- 每个文件的样本数
- 各文件正负类数量
- 汇总 totals

## 6. `06_train_sample_6000_dual.py`

文件：[`06_train_sample_6000_dual.py`](/Users/apple/Local/fdurop/code/result/bert/06_train_sample_6000_dual.py)

### 作用

专门针对 `sample_6000_labeled.xlsx` 这类同时包含 `broad` 和 `strict` 两套标签标准的数据，分别训练两套模型，并生成对照报告。

它本质上是 `04_train_bert_classifier.py` 的封装器，额外做了三件事：

1. 从同一份数据中生成共享 train / val / test 切分。
2. 分别训练 `broad` 和 `strict` 两个模型。
3. 合并两份测试结果，输出 side-by-side 对照表。

### 默认输入输出

- 输入：`bert/data/sample_6000_labeled.xlsx`
- 输出目录：`bert/artifacts/sample_6000`

### 共享切分策略

为了让 `broad` 和 `strict` 的评估可以横向比较，脚本会尽量让两套模型使用同一批 train / val / test 样本。

内部优先尝试以下分层策略：

1. `broad_norm + strict_norm` 联合分层
2. 只按 `broad_norm` 分层
3. 只按 `strict_norm` 分层
4. 完全随机切分

因此即使标签分布不理想，它也会自动回退。

### 主要参数

- `--input_path`
- `--base_output_dir`
- `--model_name_or_path`
- `--text_col`
- `--sheet_name`
- `--max_length`
- `--batch_size`
- `--epochs`
- `--learning_rate`
- `--weight_decay`
- `--warmup_ratio`
- `--max_grad_norm`
- `--val_size`
- `--test_size`
- `--positive_threshold`
- `--seed`
- `--device`
- `--local_files_only`

### 示例

```bash
python3 bert/06_train_sample_6000_dual.py \
  --input_path "bert/data/sample_6000_labeled.xlsx" \
  --base_output_dir "bert/artifacts/sample_6000" \
  --model_name_or_path "bert-base-chinese" \
  --text_col "cleaned_text" \
  --epochs 2 \
  --batch_size 16
```

### 输出目录结构

运行后，`base_output_dir` 下通常会有：

- `shared_split_dataset.csv`
  - 写入了共享切分信息的数据集。
- `shared_split_manifest.json`
  - 记录共享切分策略、可用样本数、丢弃样本数等。
- `broad/`
  - 一套完整的 `04` 训练输出。
- `strict/`
  - 另一套完整的 `04` 训练输出。
- `test_predictions_combined.csv`
  - 两套结果纵向拼接。
- `test_misclassified_combined.csv`
  - 两套结果中的所有错分样本。
- `test_predictions_side_by_side.csv`
  - 同一条测试样本在 `broad` / `strict` 两个模型下的预测对照。
- `test_misclassified_side_by_side.csv`
  - 任一标准下错分的样本对照表。
- `summary.json`
  - 总汇总报告。

### 适合什么时候用

- 你既关心宽松口径，也关心严格口径。
- 你希望对同一批样本横向比较两套标注标准训练出的模型。
- 你需要快速定位“某条样本在 broad 下对，在 strict 下错”这类差异。

## 7. `scripts/merge_xlsx_annotations.py`

文件：[`scripts/merge_xlsx_annotations.py`](/Users/apple/Local/fdurop/code/result/bert/scripts/merge_xlsx_annotations.py)

### 作用

把多个 xlsx 标注文件合并到一个主 workbook 中。

脚本特点：

- 第一个输入文件被当作主模板。
- 可自动识别单行表头、双行表头、无表头三种布局。
- 通过 `id` 或 `cleaned_text` 匹配同一行。
- 对 `broad` / `strict` 字段，把 `0` 和 `2` 视为可兼容的负类。
- 如果目标单元格为空，会优先填充。
- 如果值不同且不能安全覆盖，会记录冲突。

### 命令格式

```bash
python3 bert/scripts/merge_xlsx_annotations.py <输入1.xlsx> <输入2.xlsx> ... \
  -o <输出.xlsx> \
  --report-json <报告.json>
```

### 示例

把三份标注表合并成主表：

```bash
python3 bert/scripts/merge_xlsx_annotations.py \
  bert/data/sample_1.xlsx \
  bert/data/sample_2.xlsx \
  bert/data/sample_foxi.xlsx \
  -o bert/data/sample_6000_labeled.xlsx \
  --report-json bert/data/sample_6000_labeled.merge_report.json
```

### 输出内容

- 输出 xlsx
  - 合并后的主标注表。
- 可选报告 JSON
  - 追加了多少行
  - 更新了多少单元格
  - 是否存在冲突
  - 每个源文件贡献了多少更新

### 适用场景

- 多人并行标注后回收统一结果。
- 一部分文件有双表头、一部分文件没有标准表头时做兼容合并。

## 8. `llm_label_local.example.toml`

文件：[`llm_label_local.example.toml`](/Users/apple/Local/fdurop/code/result/bert/llm_label_local.example.toml)

### 作用

这是 `02_llm_label_local.py` 的配置模板，适合把常用参数固定下来，减少每次命令行输入。

当前模板内容大意：

- 输入：`bert/data/sample.csv`
- 输出：`bert/data/labeled.csv`
- 报告：`bert/data/labeling_report.json`
- `max_chars = 800`
- `temperature = 0.0`
- `save_every = 100`
- `max_workers = 1`
- labeler 默认走 `ollama`，模型为 `gpt-oss:20b`
- fixer 默认走 `ollama`，模型为 `qwen3:8b`

### 示例

```bash
python3 bert/02_llm_label_local.py \
  --config bert/llm_label_local.example.toml
```

如果你想作为正式本地配置使用，最常见的做法是复制一份为：

```text
bert/llm_label_local.toml
```

然后把 API key、模型名和输入输出路径改成你自己的。

## 常见使用范例

## 范例 1：从大盘数据重新抽样并训练

```bash
python3 bert/01_stratified_sampling.py \
  --input "data/processed/text_dedup/*.parquet" \
  --output "bert/data/sample_6000.csv" \
  --n 6000

python3 bert/02_llm_label_local.py \
  --input "bert/data/sample_6000.csv" \
  --output "bert/data/labeled.csv" \
  --report_path "bert/data/labeling_report.json"

python3 bert/03_normalize_labels.py \
  --input_csv "bert/data/labeled.csv" \
  --output_csv "bert/data/labeled_binary.csv"

python3 bert/04_train_bert_classifier.py \
  --input_csv "bert/data/labeled_binary.csv" \
  --output_dir "bert/artifacts/tangping_bert"
```

## 范例 2：直接用现成 Excel 训练 broad / strict

```bash
python3 bert/06_train_sample_6000_dual.py \
  --input_path "bert/data/sample_6000_labeled.xlsx" \
  --base_output_dir "bert/artifacts/sample_6000"
```

## 范例 3：先合并人工标注，再做双标准训练

```bash
python3 bert/scripts/merge_xlsx_annotations.py \
  bert/data/sample_1.xlsx \
  bert/data/sample_2.xlsx \
  bert/data/sample_foxi.xlsx \
  -o bert/data/sample_6000_labeled.xlsx \
  --report-json bert/data/sample_6000_labeled.merge_report.json

python3 bert/06_train_sample_6000_dual.py \
  --input_path "bert/data/sample_6000_labeled.xlsx" \
  --base_output_dir "bert/artifacts/sample_merge"
```

## 常见问题

### 1. 训练脚本读 Excel 时，为什么列名看起来不对？

`sample_6000_labeled.xlsx` 这类文件的第一行本身像一行“伪表头”。`04_train_bert_classifier.py` 已经实现了嵌入式表头识别逻辑，会自动把第一行提升成真正列名，所以一般不需要手动改表。

### 2. 为什么 `02_llm_label_local.py` 没有从头重新跑？

因为它支持断点续跑。如果输出文件已存在，而且行数和 `id` 能对上，脚本会自动跳过已完成样本。

### 3. `04` 和 `06` 的区别是什么？

- `04` 是通用训练脚本，适用于单标签标准。
- `06` 是双标准封装器，专门处理 `broad` / `strict` 两套标签，并生成对照结果。

### 4. 什么时候用 `--local_files_only`？

当模型和 tokenizer 已经在本地，或者你在离线环境运行时，就要加这个参数。

### 5. `05_predict_bert_classifier.py` 为什么只支持 parquet？

因为它的定位是“批量回灌到处理后的语料库”，默认面向 `data/processed/text_dedup/*.parquet` 这类中间产物，而不是训练表。

## 建议的维护习惯

- 新增训练实验时，把输出写到新的 `bert/artifacts/<实验名>/` 目录，避免覆盖旧模型。
- 保留 `metrics.json`、`summary.json` 和 `test_misclassified*.csv`，它们是回顾实验最有用的文件。
- 如果做多人标注，优先统一 `id` 和 `cleaned_text`，这样 `merge_xlsx_annotations.py` 的匹配最稳。
- 如果要长期使用 LLM 标注流程，建议维护一份正式的 `bert/llm_label_local.toml`，而不是每次手敲参数。
