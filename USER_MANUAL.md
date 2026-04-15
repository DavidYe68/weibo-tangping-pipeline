# 主流程使用手册

本手册只负责说明项目的主流程，也就是从 `raw/` 原始 CSV 到 `data/processed/`、`data/state/`、`data/reports/`、`data/exports/` 的这一段。抽样、预标注、训练、预测和 `07-10` 分析链请看 [bert/README.md](/Users/apple/Local/fdurop/code/result/bert/README.md)。

## 1. 目录约定

当前主流程依赖的核心目录如下：

- 唯一源数据目录：`raw/`
- 主流程入口：`main.py`
- 主流程处理目录：`data/processed/`
- 主流程状态目录：`data/state/`
- 主流程报告目录：`data/reports/`
- 主流程导出目录：`data/exports/`
- 抽样和标注数据目录：`bert/data/`
- 训练和分析产物目录：`bert/artifacts/`

目录结构可以粗看成：

```text
result/
├── raw/                         源数据
├── archive/
│   ├── raw_data_legacy/         历史源数据归档
│   └── reports_legacy/          历史报告归档
├── data/
│   ├── processed/
│   │   ├── merged_dedup/
│   │   ├── preprocessed/
│   │   └── text_dedup/
│   ├── state/
│   ├── reports/
│   └── exports/
├── scripts/
│   └── pipeline/
├── bert/
│   ├── data/
│   ├── artifacts/
│   └── README.md
├── main.py
└── USER_MANUAL.md
```

说明：

- `bert/data/` 和 `bert/artifacts/` 不属于主流程内部目录，但会消费主流程生成的 `text_dedup`。
- `archive/` 下的内容主要是历史归档，不参与当前正式流程。

## 2. 命令写法

本仓库默认使用根目录下的 `.venv`。下面示例统一采用 macOS / Linux 写法：

```bash
.venv/bin/python ...
```

如果你在 Windows PowerShell，请把它替换为：

```powershell
.\.venv\Scripts\python.exe ...
```

## 3. 源数据要求

主流程从 `raw/` 递归扫描 CSV，识别模式如下：

`.../csv/{keyword}/**/*.csv`

例如：

- `raw/1/csv/躺平/2024/01/01.csv`
- `raw/2/csv/佛系/2025/06/08.csv`
- `raw/结果文件/csv/chatgpt/2025/7/31.csv`

识别逻辑：

- `csv` 目录的下一层作为 `keyword`
- 如果后续目录中存在年、月，脚本会提取它们用于排序
- 文件名数字部分会作为日排序参考

原始数据路径最好能稳定反映关键词来源，否则后续 `keyword` 元信息会不完整。

## 4. 主流程三层产物

主流程会依次生成三层结果：

1. `merged_dedup`
2. `preprocessed`
3. `text_dedup`

含义如下：

- `merged_dedup`：从 `raw/` 读取原始 CSV，补充 `keyword` 和 `source_file`，按 `id` 增量去重后的主表
- `preprocessed`：对新增去重数据做文本清洗后的结果
- `text_dedup`：对预处理结果按 `cleaned_text` 再做增量文本去重后的结果

其中 `data/processed/text_dedup/` 是后续抽样、训练和分析最常用的输入。

## 5. 统一命令入口

统一入口文件是 [`main.py`](/Users/apple/Local/fdurop/code/result/main.py)。

### 5.1 增量运行

```bash
.venv/bin/python main.py
.venv/bin/python main.py run
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

### 5.2 全量重建

```bash
.venv/bin/python main.py full
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
- `bert/data/`
- `bert/artifacts/`

适用场景：

- 修改了清洗逻辑
- 状态文件不可信
- 希望完全重算历史数据

### 5.3 查看状态

```bash
.venv/bin/python main.py status
```

输出信息包括：

- 已索引 raw 文件数
- `merged_dedup` 行数与分片数
- `preprocessed` 分片数
- `text_dedup` 行数与分片数
- 最近一次运行模式和结束时间
- 关键路径

### 5.4 导出 CSV

```bash
.venv/bin/python main.py export-csv
.venv/bin/python main.py export-csv merged
.venv/bin/python main.py export-csv text
```

导出位置：

- `data/exports/merged_dedup.csv`
- `data/exports/text_dedup.csv`

说明：

- 主流程内部存储仍然是 parquet
- CSV 只用于交换、人工查看或兼容下游工具

## 6. 主流程相关脚本

主流程相关脚本集中放在 [`scripts/pipeline/`](/Users/apple/Local/fdurop/code/result/scripts/pipeline)：

- [scripts/pipeline/s01_core.py](/Users/apple/Local/fdurop/code/result/scripts/pipeline/s01_core.py)
- [scripts/pipeline/s02_merge.py](/Users/apple/Local/fdurop/code/result/scripts/pipeline/s02_merge.py)
- [scripts/pipeline/s03_dedup.py](/Users/apple/Local/fdurop/code/result/scripts/pipeline/s03_dedup.py)
- [scripts/pipeline/s04_preprocess.py](/Users/apple/Local/fdurop/code/result/scripts/pipeline/s04_preprocess.py)

一般情况下优先使用 [`main.py`](/Users/apple/Local/fdurop/code/result/main.py)。只有在你明确想从某个阶段单独查看或调试时，才需要直接跑这些分组脚本。

## 7. 内部数据说明

### 7.1 `merged_dedup`

目录：

- `data/processed/merged_dedup/part-*.parquet`

常见列：

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

### 7.2 `preprocessed`

目录：

- `data/processed/preprocessed/part-*.parquet`

常见列：

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

### 7.3 `text_dedup`

目录：

- `data/processed/text_dedup/part-*.parquet`

说明：

- 基于 `preprocessed` 的 `cleaned_text` 做增量文本去重
- 这是后续抽样、标注、训练和 broad 分析链的标准输入

## 8. 状态文件说明

### 8.1 `raw_manifest.json`

路径：

- `data/state/raw_manifest.json`

作用：

- 记录每个 raw CSV 的 `size`、`mtime_ns`、`keyword`
- 用于判断哪些文件发生了变化

### 8.2 `id_hashes.txt`

路径：

- `data/state/id_hashes.txt`

作用：

- 保存已出现 `id` 的哈希
- 用于增量 `id` 去重

### 8.3 `text_hashes.txt`

路径：

- `data/state/text_hashes.txt`

作用：

- 保存已出现 `cleaned_text` 的哈希
- 用于增量文本去重

### 8.4 `pipeline_last_run.json`

路径：

- `data/reports/pipeline_last_run.json`

作用：

- 记录最近一次运行模式、处理文件数、写出分片、累计行数等

## 9. 和下游流程的衔接

主流程结束后，后续一般从这里开始：

1. 用 [`bert/01_stratified_sampling.py`](/Users/apple/Local/fdurop/code/result/bert/01_stratified_sampling.py) 从 `data/processed/text_dedup/*.parquet` 抽样
2. 用 [`bert/02_llm_label_local.py`](/Users/apple/Local/fdurop/code/result/bert/02_llm_label_local.py) 生成预标注草稿
3. 人工审核样本
4. 用 `04` 或 `05` 训练模型
5. 用 `06` 对全量 parquet 做预测
6. 顺序运行 `07-10`

这些步骤的详细命令、参数和输出见 [bert/README.md](/Users/apple/Local/fdurop/code/result/bert/README.md)。

## 10. 迁移后的注意事项

以下旧结构已经删除或停止使用：

- 旧的 `step_01` 到 `step_07`
- 旧的 `count_*` 脚本
- 旧的 `bert_classify/` 训练预测结构
- 旧的 `project_io.py`

以下旧目录如果还存在，只是历史产物保留，不再参与当前主流程：

- 根目录 `preprocessed/`
- 根目录 `outputs/`
- `archive/reports_legacy/`
- 根目录 `logs/`

## 11. 常见问题

### 11.1 什么情况下用 `run`，什么情况下用 `full`？

- 平时追加新数据：`run`
- 修改清洗逻辑或怀疑状态不一致：`full`

### 11.2 为什么内部结果都写到 `data/processed/`？

这是当前项目使用的标准结构，便于维护增量状态、处理结果和统一导出。

### 11.3 为什么主流程不再从 `data/` 里的旧 CSV 开始？

因为当前主流程默认只从 `raw/` 读取原始数据，其他历史目录中的旧 CSV 已不再作为正式输入入口。

### 11.4 为什么根目录还有 `outputs/`、`preprocessed/`？

那是早期实验阶段保留下来的历史产物。当前主流程不再使用它们，也不需要再向这些目录写入。

### 11.5 是否要求 parquet 依赖？

是。当前主流程内部存储就是 parquet。
