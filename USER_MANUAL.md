# 主流程使用手册

这份文档只讲主流程，也就是把 `raw/` 里的原始微博 CSV 整理成后续可抽样、可训练、可分析的标准数据。

如果你关心的是抽样、预标注、训练和 `07-10` 分析链，请直接看 [`bert/README.md`](./bert/README.md)。

## 这份手册解决什么问题

主流程听起来像“数据预处理”，但它其实做了三件很具体的事：

1. 统一把散落在 `raw/` 里的 CSV 找出来
2. 把原始微博整理成字段比较稳定的 parquet
3. 用增量状态避免每次都从头重跑

跑完之后，最关键的产物是：

- `data/processed/text_dedup/`

后面的抽样、训练和 broad 分析，基本都从这里接。

## 先理解主流程在干什么

主流程不是“一口气把所有数据处理到底”，而是分三层往前走：

1. `merged_dedup`
2. `preprocessed`
3. `text_dedup`

它们分别代表：

### `merged_dedup`

把 `raw/` 中识别到的 CSV 合并起来，补上来源信息，再按 `id` 去重。

可以把它理解成：

- “哪些微博进来了”
- “原始字段大致长什么样”
- “每条微博来自哪个源文件”

### `preprocessed`

在 `merged_dedup` 的基础上做文本清洗。

这一步主要是在做：

- 统一文本列
- 处理 emoji
- 保留后面训练和分析真正会用到的文本内容

### `text_dedup`

按清洗后的文本再去一次重。

原因很简单：微博里经常有不同 `id` 但正文几乎一样的内容。如果不做文本去重，后面抽样和分析会被重复文本拖偏。

一句话总结：

- `merged_dedup` 解决“同一条微博别重复算两次”
- `text_dedup` 解决“同一段话别重复算很多次”

## 目录约定

主流程真正会碰到的目录主要是这些：

- `raw/`：唯一正式输入目录
- `data/processed/`：处理结果
- `data/state/`：增量状态
- `data/reports/`：运行报告
- `data/exports/`：导出的 CSV
- `scripts/pipeline/`：主流程脚本
- `main.py`：统一入口

大致结构如下：

```text
result/
├── raw/
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
├── main.py
└── USER_MANUAL.md
```

补一句：

- `bert/data/` 和 `bert/artifacts/` 会用到主流程产物，但不属于主流程本身
- `archive/` 里的内容一般是历史归档，不参与当前正式流程

## 原始数据怎么放

主流程会递归扫描 `raw/` 下的 CSV，识别模式大致是：

`.../csv/{keyword}/**/*.csv`

例如：

- `raw/1/csv/躺平/2024/01/01.csv`
- `raw/2/csv/佛系/2025/06/08.csv`
- `raw/结果文件/csv/chatgpt/2025/7/31.csv`

脚本会按下面的规则理解路径：

- `csv` 的下一层目录，当作 `keyword`
- 后面的年、月目录如果存在，会被提取出来参与排序
- 文件名里的数字会被当作日期排序参考

这意味着一件事：

原始文件路径最好别太随意。哪怕文件内容一样，路径里如果没有稳定的关键词层级，后面补出来的 `keyword` 元信息就会不完整。

## 统一命令入口

统一入口是 [`main.py`](./main.py)。

仓库默认使用根目录下的 `.venv`：

- macOS / Linux：`.venv/bin/python`
- Windows PowerShell：`.\.venv\Scripts\python.exe`

下面示例统一按 macOS / Linux 写。Windows 下只需要把解释器路径替换掉。

## 四个最常用命令

### 1. 增量运行：`run`

```bash
.venv/bin/python main.py
.venv/bin/python main.py run
```

这是平时最常用的命令。

它会做这些事：

1. 扫描 `raw/` 下所有能识别的 CSV
2. 读取 `data/state/raw_manifest.json`
3. 找出新增或修改过的文件
4. 对新增数据按 `id` 去重
5. 做文本清洗
6. 按 `cleaned_text` 再做文本去重
7. 更新状态文件和运行报告

什么时候用：

- 新加了原始 CSV
- 改了少量原始文件
- 想在现有结果上继续追加

什么时候别急着用它：

- 你改了清洗逻辑
- 你怀疑旧状态文件已经不可信
- 你想让整个历史结果重新按最新规则生成

这时候更适合用 `full`。

### 2. 全量重建：`full`

```bash
.venv/bin/python main.py full
```

这个命令会把主流程自己的核心产物和状态清掉，然后从 `raw/` 全量重算。

会重建的内容：

- `data/processed/merged_dedup/`
- `data/processed/preprocessed/`
- `data/processed/text_dedup/`
- `data/state/raw_manifest.json`
- `data/state/id_hashes.txt`
- `data/state/text_hashes.txt`
- `data/reports/pipeline_last_run.json`
- `data/exports/*.csv`

不会动的内容：

- `raw/`
- 仓库代码
- `bert/data/`
- `bert/artifacts/`
- `data/processed/text_dedup_predicted*/`
- `data/processed/` 里其他不属于主流程三层产物的目录

什么时候用：

- 你改了清洗逻辑
- 状态文件可能坏了
- 历史数据要按新规则重算

要注意的一点：

`full` 只负责主流程重建，不会自动帮你重跑后面的预测和分析。也就是说，如果你希望 `06-10` 的结果和这次重建保持一致，需要自己再跑一次下游脚本。

### 3. 查看状态：`status`

```bash
.venv/bin/python main.py status
```

这个命令适合在真正开跑前先看一眼。

通常会告诉你：

- 当前索引了多少 raw 文件
- `merged_dedup` 有多少行、多少分片
- `preprocessed` 和 `text_dedup` 的分片情况
- 最近一次运行模式和结束时间
- 关键路径现在指向哪里

如果你不确定仓库是不是已经处理过一轮，先跑这个最稳。

### 4. 导出 CSV：`export-csv`

```bash
.venv/bin/python main.py export-csv
.venv/bin/python main.py export-csv merged
.venv/bin/python main.py export-csv text
```

导出位置：

- `data/exports/merged_dedup.csv`
- `data/exports/text_dedup.csv`

这一步的意义不是替代 parquet，而是：

- 方便人工查看
- 方便发给别的工具
- 方便和不支持 parquet 的下游流程对接

如果你只是继续在本项目里跑脚本，优先还是用 parquet。

## 主流程相关脚本

主流程脚本集中在 [`scripts/pipeline/`](./scripts/pipeline/)：

- [`scripts/pipeline/s01_core.py`](./scripts/pipeline/s01_core.py)
- [`scripts/pipeline/s02_merge.py`](./scripts/pipeline/s02_merge.py)
- [`scripts/pipeline/s03_dedup.py`](./scripts/pipeline/s03_dedup.py)
- [`scripts/pipeline/s04_preprocess.py`](./scripts/pipeline/s04_preprocess.py)

大多数情况下，直接用 `main.py` 就够了。

只有在这些场景下，才建议直接进脚本层：

- 你要调试某个阶段
- 你想确认某个处理环节到底做了什么
- 你在改主流程代码，需要单独验证一个步骤

## 产物分别长什么样

### `data/processed/merged_dedup/`

常见列包括：

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

你可以把它理解成“按 `id` 去重后的主表”。

这里最值得注意的是：

- `source_file` 记录原始文件相对路径
- `keyword` 主要来自目录结构，而不是正文自动猜测

### `data/processed/preprocessed/`

常见列包括：

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

最关键的两个文本列：

- `cleaned_text`：emoji 会统一替换成 `[emoji]`
- `cleaned_text_with_emoji`：emoji 会转成文字描述

如果你后面是做分类任务，通常先用 `cleaned_text`。

### `data/processed/text_dedup/`

这是后面最常被当作标准输入的目录。

它的意思是：

- 基于 `preprocessed` 的 `cleaned_text` 再做一轮去重
- 尽量避免重复正文在抽样和分析里反复出现

如果你只记住一个目录，记住这个就行。

## 状态文件是干什么的

主流程能增量运行，靠的就是 `data/state/` 里的这些文件。

### `raw_manifest.json`

路径：

- `data/state/raw_manifest.json`

作用：

- 记录每个 raw CSV 的 `size`、`mtime_ns`、`keyword`
- 用来判断哪些文件发生了变化

### `id_hashes.txt`

路径：

- `data/state/id_hashes.txt`

作用：

- 保存已经见过的 `id` 哈希
- 用来做增量 `id` 去重

### `text_hashes.txt`

路径：

- `data/state/text_hashes.txt`

作用：

- 保存已经见过的 `cleaned_text` 哈希
- 用来做增量文本去重

### `pipeline_last_run.json`

路径：

- `data/reports/pipeline_last_run.json`

作用：

- 记录最近一次运行模式、处理文件数、写出分片和累计行数

如果你在排查“为什么这次没有处理新文件”，优先看 `raw_manifest.json` 和 `pipeline_last_run.json`。

## 主流程和下游怎么衔接

主流程跑完之后，后面一般这么接：

1. 用 `bert/01_stratified_sampling.py` 从 `data/processed/text_dedup/*.parquet` 抽样
2. 用 `bert/02_llm_label_local.py` 生成预标注草稿
3. 人工审核
4. 用 `04` 或 `05` 训练模型
5. 用 `06` 对全量 parquet 做预测
6. 顺序运行 `07-10`

这一段的细节请看 [`bert/README.md`](./bert/README.md)。

## 常见情况和处理建议

### 我平时到底该用 `run` 还是 `full`？

- 大多数日常追加：`run`
- 改过清洗逻辑、怀疑状态脏了：`full`

如果你开始犹豫，先问自己一句：

“我想追加新数据，还是想把旧结果推倒重来？”

前者用 `run`，后者用 `full`。

### 为什么项目内部主要用 parquet，不直接用 CSV？

因为这里的主流程和下游脚本本来就是按 parquet 组织的。CSV 主要拿来交换、人工看、或者给别的工具兼容。

### 为什么主流程不再从旧目录里的 CSV 开始？

因为现在正式入口就是 `raw/`。其他历史目录里的 CSV 就算还在，也不应该再被当成新的正式输入。

### 根目录还有 `outputs/`、`preprocessed/`、`logs/`，是不是现在还在用？

通常不是。它们更多是早期实验或历史遗留。当前正式主流程主要围绕 `raw/` 和 `data/`。

### 发现结果不对，先查哪里？

建议按这个顺序查：

1. 先跑 `main.py status`
2. 看 `data/reports/pipeline_last_run.json`
3. 看 `data/state/raw_manifest.json`
4. 抽一两个 parquet 分片确认字段和行数
5. 如果你改过处理逻辑，再考虑跑 `main.py full`

这样排查通常比直接重跑更快。
