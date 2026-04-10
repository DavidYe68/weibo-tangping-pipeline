# 完整使用手册

本项目的代码入口、目录约定和使用方式已经统一到 `AI_attitude` 风格，统一后的原则如下：

- 唯一源数据目录：`raw/`
- 历史源数据归档目录：`archive/raw_data_legacy/`
- 历史报告归档目录：`archive/reports_legacy/`
- 唯一主流程入口：`main.py`
- 唯一内部处理目录：`data/processed/`
- 唯一状态目录：`data/state/`
- 唯一报告目录：`data/reports/`
- 唯一导出目录：`data/exports/`
- 内部存储统一优先使用 `parquet`

## 1. 目录结构

```text
result/
├── raw/                         # 源数据
├── archive/
│   └── raw_data_legacy/         # 历史源数据归档
│   └── reports_legacy/          # 历史报告归档
├── data/
│   ├── bert/
│   │   ├── sample.csv
│   │   ├── labeled.csv
│   │   ├── labeled_binary.csv
│   │   └── *.json
│   ├── processed/
│   │   ├── merged_dedup/
│   │   ├── preprocessed/
│   │   └── text_dedup/
│   ├── state/
│   ├── reports/
│   └── exports/
├── scripts/
│   ├── pipeline/
│   │   ├── s01_core.py
│   │   ├── s02_merge.py
│   │   ├── s03_dedup.py
│   │   └── s04_preprocess.py
├── bert/
│   ├── 01_stratified_sampling.py
│   └── 02_llm_label_local.py
├── main.py
├── README.md
└── USER_MANUAL.md
```

## 2. 源数据要求

主流程从 `raw/` 递归扫描 CSV。

可识别模式：

`.../csv/{keyword}/**/*.csv`

例如：

- `raw/1/csv/躺平/2024/01/01.csv`
- `raw/2/csv/佛系/2025/06/08.csv`
- `raw/结果文件/csv/chatgpt/2025/7/31.csv`

识别逻辑：

- `csv` 目录的下一层作为 `keyword`
- 如果后续目录中存在年、月，脚本会提取它们用于排序
- 文件名数字部分会作为日排序参考

## 3. 主流程概念

主流程分三层结果：

1. `merged_dedup`
2. `preprocessed`
3. `text_dedup`

含义如下：

- `merged_dedup`: 从 `raw/` 读取原始 CSV，补充 `keyword` 和 `source_file`，按 `id` 增量去重后的主表
- `preprocessed`: 对新增去重数据做文本清洗后的结果
- `text_dedup`: 对预处理结果按 `cleaned_text` 再做增量文本去重后的结果

## 4. 统一命令入口

统一入口文件是 [main.py](main.py)。

### 4.1 增量运行

```bash
python main.py
python main.py run
```

作用：

1. 扫描 `raw/` 下所有可识别 CSV
2. 读取 `data/state/raw_manifest.json`
3. 只处理新增或变更文件
4. 对新增数据按 `id` 去重
5. 对新增去重数据做预处理
6. 对新增预处理结果按 `cleaned_text` 去重
7. 更新状态文件和报告

适用场景：

- 新增了原始 CSV
- 修改了部分 raw 文件
- 想在现有处理结果基础上继续追加

### 4.2 全量重建

```bash
python main.py full
```

作用：

- 清理当前项目生成的 processed/state/reports/exports 产物
- 从 `raw/` 全量重新计算

会重建的目录/文件：

- `data/processed/*`
- `data/state/*`
- `data/reports/*`
- `data/exports/*`

不会动的内容：

- `raw/`
- 仓库中的代码文件

适用场景：

- 修改了清洗逻辑
- 状态文件不可信
- 希望完全重算历史数据

### 4.3 查看状态

```bash
python main.py status
```

输出信息包括：

- 已索引 raw 文件数
- `merged_dedup` 行数与分片数
- `preprocessed` 分片数
- `text_dedup` 行数与分片数
- 最近一次运行模式和结束时间
- 关键路径

### 4.4 导出 CSV

```bash
python main.py export-csv
python main.py export-csv merged
python main.py export-csv text
```

导出位置：

- `data/exports/merged_dedup.csv`
- `data/exports/text_dedup.csv`

说明：

- 主流程内部存储仍然是 parquet
- CSV 只用于交换、人工查看或兼容下游工具

## 5. 脚本分组

主流程相关脚本集中放在 `scripts/pipeline/`：

- [scripts/pipeline/s01_core.py](scripts/pipeline/s01_core.py)
- [scripts/pipeline/s02_merge.py](scripts/pipeline/s02_merge.py)
- [scripts/pipeline/s03_dedup.py](scripts/pipeline/s03_dedup.py)
- [scripts/pipeline/s04_preprocess.py](scripts/pipeline/s04_preprocess.py)

例如：

```bash
python scripts/pipeline/s02_merge.py run
python scripts/pipeline/s03_dedup.py status
python scripts/pipeline/s04_preprocess.py export-csv text
```

## 6. 内部数据说明

### 6.1 merged_dedup

目录：

- `data/processed/merged_dedup/part-*.parquet`

默认列：

- `id`
- `微博正文`
- `发布时间`
- `话题`
- `keyword`
- `转发数`
- `评论数`
- `点赞数`
- `ip`
- `source_file`

说明：

- 这是按 `id` 去重后的主表
- `source_file` 记录原始文件相对路径

### 6.2 preprocessed

目录：

- `data/processed/preprocessed/part-*.parquet`

默认列：

- `id`
- `cleaned_text`
- `cleaned_text_with_emoji`
- `text_raw`
- `发布时间`
- `话题`
- `keyword`
- `转发数`
- `评论数`
- `点赞数`
- `ip`
- `source_file`

说明：

- `cleaned_text`：emoji 统一替换成 `[emoji]`
- `cleaned_text_with_emoji`：emoji 转成文字描述

### 6.3 text_dedup

目录：

- `data/processed/text_dedup/part-*.parquet`

说明：

- 基于 `preprocessed` 的 `cleaned_text` 做增量文本去重
- 这是后续采样与标注最常用的输入

## 7. 状态文件说明

### 7.1 raw_manifest.json

路径：

- `data/state/raw_manifest.json`

作用：

- 记录每个 raw CSV 的 `size`、`mtime_ns`、`keyword`
- 用于判断哪些文件发生了变化

### 7.2 id_hashes.txt

路径：

- `data/state/id_hashes.txt`

作用：

- 保存已出现 `id` 的哈希
- 用于增量 `id` 去重

### 7.3 text_hashes.txt

路径：

- `data/state/text_hashes.txt`

作用：

- 保存已出现 `cleaned_text` 的哈希
- 用于增量文本去重

### 7.4 pipeline_last_run.json

路径：

- `data/reports/pipeline_last_run.json`

作用：

- 记录最近一次运行模式、处理文件数、写出分片、累计行数等

## 8. 下游脚本

### 8.1 bert/01_stratified_sampling.py

命令：

```bash
python bert/01_stratified_sampling.py
```

默认行为：

- 从 `data/processed/text_dedup/*.parquet` 读取
- 输入默认视为已经完成文本去重，不会在抽样阶段再次去重
- 按月份、keyword、文本长度分层抽样
- 输出 `bert/data/sample.csv`
- 同时输出 `bert/data/sampling_report.json`

常用参数：

```bash
python bert/01_stratified_sampling.py --n 6000
python bert/01_stratified_sampling.py --seed 42
python bert/01_stratified_sampling.py --k_min 0
python bert/01_stratified_sampling.py --input "data/processed/text_dedup/*.parquet"
python bert/01_stratified_sampling.py --output bert/data/sample.csv
python bert/01_stratified_sampling.py --report_path bert/data/sampling_report.json
```

说明：

- 自动识别文本列时只会匹配内置候选列名；如果没有匹配到，会直接报错并要求显式传入 `--text_col`
- `--k_min 0` 表示关闭“满足阈值的分层至少抽 1 条”的保底逻辑
- 当 `--n` 很小而符合保底条件的分层太多时，可能无法满足“总样本数精确等于 n”，这时可以提高 `--k_min` 或增大 `--n`

### 8.2 bert/02_llm_label_local.py

命令：

```bash
python bert/02_llm_label_local.py
```

默认行为：

- 默认从 `bert/data/sample.csv` 读取
- 主标注模型默认走 Qwen 兼容 OpenAI API
- JSON 修复器默认仍走本地 Ollama
- 默认会读取 `bert/llm_label_local.toml` 配置文件（如存在）
- 仓库内默认只保留示例文件 `bert/llm_label_local.example.toml`
- 默认输出 `bert/data/labeled.csv`
- 同时输出 `bert/data/labeling_report.json`

常用参数：

```bash
python bert/02_llm_label_local.py
python bert/02_llm_label_local.py --input bert/data/sample.csv --output bert/data/labeled.csv
python bert/02_llm_label_local.py --report_path bert/data/labeling_report.json
export DASHSCOPE_API_KEY=your_key_here
python bert/02_llm_label_local.py --labeler_model qwen-coder-plus-latest
python bert/02_llm_label_local.py --labeler_api_key your_key_here
python bert/02_llm_label_local.py --labeler_base_url https://dashscope.aliyuncs.com/compatible-mode/v1
python bert/02_llm_label_local.py --fixer_model qwen3:8b
python bert/02_llm_label_local.py --base_url http://localhost:11434
cp bert/llm_label_local.example.toml bert/llm_label_local.toml
python bert/02_llm_label_local.py --config bert/llm_label_local.toml
python bert/02_llm_label_local.py --fixer_provider qwen_openai --fixer_model qwen-plus-latest --fixer_api_key your_key_here
```

配置文件示例：

```toml
input = "bert/data/sample.csv"
output = "bert/data/labeled.csv"
report_path = "bert/data/labeling_report.json"

[labeler]
provider = "qwen_openai"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen-coder-plus-latest"

[fixer]
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:8b"
```

输出列：

- `tangping_related_label`：`相关` 或 `无关`
- `tangping_related`：二值标签，`1` 表示相关，`0` 表示无关
- `exclusion_type`：无关时的排除类别
- `confidence`：高 / 中 / 低
- `llm_reason`：简短理由

进度显示：

- 脚本会输出 `stage=load_input`、`stage=detect_text_col`、`stage=labeling_start`、`stage=write_output`、`stage=write_report`
- 安装了 `tqdm` 时会显示逐条标注进度条

## 9. 迁移后的注意事项

这次迁移后，以下旧结构已经删除或停止使用：

- 旧的 `step_01` 到 `step_07`
- 旧的 `count_*` 脚本
- 旧的 `bert_classify/` 训练预测结构
- 旧的 `project_io.py`

以下旧目录如果还存在，只是历史产物保留，不再参与主流程：

- 根目录 `preprocessed/`
- 根目录 `outputs/`
- `archive/reports_legacy/`
- 根目录 `logs/`

当前唯一应依赖的主流程目录是：

- `raw/`
- `data/processed/`
- `data/state/`
- `data/reports/`
- `data/exports/`

## 10. 推荐使用顺序

### 10.1 数据流水线

```bash
python main.py run
python main.py status
python main.py export-csv text
```

运行 `main.py` 时会显示 `discovered raw_files`、`target raw_files`、`processed files`、`flushing parquet buffers` 等阶段进度。

### 10.2 抽样与标注

```bash
python bert/01_stratified_sampling.py
python bert/02_llm_label_local.py --input bert/data/sample.csv --output bert/data/labeled.csv
```

### 10.3 全量重建

```bash
python main.py full
python main.py status
```

## 11. 常见问题

### 11.1 为什么主流程不再从 `data/` 里的旧 CSV 开始？

因为项目现在已经和 `AI_attitude` 完全统一，唯一源数据入口是 `raw/`。

### 11.2 为什么内部结果全部写到 `data/processed/`？

这是统一后的标准结构，便于维护增量状态和统一导出。

### 11.3 为什么根目录还有 `outputs/`、`preprocessed/`？

那是迁移前保留下来的历史产物。统一后的主流程不再使用它们，也不需要再向这些目录写入。

### 11.4 什么情况下用 `run`，什么情况下用 `full`？

- 平时追加新数据：`run`
- 修改清洗逻辑或怀疑状态不一致：`full`

### 11.5 是否要求 parquet 依赖？

是。统一后的主流程内部存储就是 parquet。
