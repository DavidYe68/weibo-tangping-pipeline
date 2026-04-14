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

## 目录概览

```text
bert/
├── 01_stratified_sampling.py
├── 02_llm_label_local.py
├── 03_normalize_labels.py
├── 04_train_bert_classifier.py
├── 05_train_dual_label_classifier.py
├── 06_predict_bert_classifier.py
├── lib/
├── scripts/
└── data/
```

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

## 最后建议

更稳妥的习惯是：

1. `01` 抽样。
2. `02` 预标注。
3. 人工审核。
4. 审核后的多个 CSV/XLSX 直接喂给 `04` 或 `05`。
5. 需要换测试集时，只改命令参数，不改脚本逻辑。

这套方式比维护一堆固定案例更稳，也更符合真实研究流程。
