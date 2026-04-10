# AI_attitude 风格统一流水线

这个仓库现在适合作为“代码仓库”同步到 GitHub，用来在另一台机器上继续跑主流程、抽样、标注和 BERT 训练/预测。

## 仓库包含什么

- 主流程入口：`main.py`
- 数据流水线：`scripts/pipeline/`
- 抽样与标注：`bert/01_*`、`bert/02_*`、`bert/03_*`
- BERT 训练与预测：`bert/04_*`、`bert/05_*`
- BERT 公共模块：`bert/lib/`
- 依赖清单：`requirements.txt`

## 仓库不包含什么

为了避免 GitHub 仓库过大，下面这些目录和文件默认不上传：

- `AI/`
- `raw/`
- `archive/`
- `data/`
- `logs/`
- `preprocessed.zip`

也就是说，GitHub 负责同步代码；真正的大数据文件需要你另外同步到 Windows 机器。

## 原始数据目录

原始数据放在 `raw/` 下，支持任意子目录，只要满足下面的路径模式即可识别：

`.../csv/{keyword}/**/*.csv`

例如：

- `raw/1/csv/躺平/2024/1/1.csv`
- `raw/2/csv/%23佛系%23/2025/6/8.csv`

## 常用命令

统一入口是 [`main.py`](main.py)。

```bash
python main.py run
python main.py full
python main.py status
python main.py export-csv
python main.py export-csv merged
python main.py export-csv text
```

## 直接试训 sample_6000 标注集

如果你已经把标注文件放在 `bert/data/sample_6000_labeled.xlsx`，可以直接分别训练 `broad` 和 `strict` 两套 BERT：

```bash
python3 bert/05_train_sample_6000_dual.py --local_files_only
```

输出目录默认是：

- `bert/artifacts/sample_6000/broad/`
- `bert/artifacts/sample_6000/strict/`
- `bert/artifacts/sample_6000/shared_split_dataset.csv`
- `bert/artifacts/sample_6000/test_predictions_side_by_side.csv`
- `bert/artifacts/sample_6000/test_misclassified_side_by_side.csv`

每套都会产出：

- `train_split.csv`
- `val_split.csv`
- `test_split.csv`
- `metrics.json`
- `training_history.json`
- `test_predictions.csv`
- `test_misclassified.csv`
- `best_model/`

其中基目录下额外会产出一组方便排查的问题清单：

- `test_predictions_side_by_side.csv`：同一批测试样本里 `broad` / `strict` 的预测结果横向对齐
- `test_misclassified_side_by_side.csv`：只保留测试集里至少有一边预测错误的样本
- `inspect/summary.md`：一眼看“先查哪类问题、先开哪个文件”
- `inspect/label_diagnosis.csv`：按标签给出当前更像 `FP` 问题还是 `FN` 问题
- `inspect/metrics_overview.csv`：把关键 val/test 指标压平成一张表
- `inspect/error_summary.csv`：按 `FP/FN` 汇总错误数量
- `inspect/side_by_side_error_summary.csv`：看两套标准是一起错，还是只有一边错
- `inspect/top_fp_*.csv` / `inspect/top_fn_*.csv`：按问题类型拆开的重点错例，适合直接人工排查

补充说明：

- `bert/04_train_bert_classifier.py`、`bert/05_train_sample_6000_dual.py`、`bert/06_eval_by_source.py`、`bert/07_predict_bert_classifier.py` 现在主要负责 CLI 参数和流程编排。
- 共享的数据读取、标签归一化、切分、训练与预测实现已经统一抽到 `bert/lib/`，后续加新的评估协议时不需要再在多个脚本里复制逻辑。

## Windows + NVIDIA 4060

如果你准备在另一台 Windows 机器上继续运行，先看 [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md)。

简版流程：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python main.py status
```

## 标注配置

`bert/02_llm_label_local.py` 默认会尝试读取 `bert/llm_label_local.toml`。

出于安全考虑，仓库只保留示例文件：

- `bert/llm_label_local.example.toml`

使用时复制一份本地配置，再填入你自己的 key，或者直接使用环境变量。

## 参考文档

- [`USER_MANUAL.md`](USER_MANUAL.md)
- [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md)
