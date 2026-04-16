# 微博“躺平 / 摆烂 / 佛系”语义研究项目

本仓库用于构建和分析一个围绕“躺平 / 摆烂 / 佛系”的微博语料库。项目把大规模文本处理、人工审核、双标签分类和后续语义分析串成一条可复用的研究流程，目标不是简单统计关键词出现次数，而是尽量识别这些词在微博正文中是否承担了与研究相关的语义功能，并进一步分析它们在不同时间段中的语义分布和变化。

## 研究关注

项目当前围绕三组关键词组织语料：

- `躺平`
- `摆烂`
- `佛系`

核心问题包括：

1. 这些词在微博中进入了哪些语义场景。
2. 这些词的主题板块、共现词和语义邻域如何变化。
3. 当这些词成为现实行为方式或评价框架时，文本表达了怎样的心态与立场。
4. 相关表达在时间和地域层面呈现出怎样的分布差异。

## 两套语料边界

项目同时构造两套语料，以承接两类分析任务。

### 宽松版语料（broad）

宽松版语料服务于概念漂移、主题结构和语义重心演化分析。这里保留的是在正文中承担可解释语义的用法，包括：

- 字面“躺平”、休息、放松等原始用法
- 日常摆烂、临时松懈、生活态度表达
- “佛系”作为风格、状态、做事方式的表达
- 股票、游戏、饭圈等场景中的语义转义
- 对群体、组织、机构的评价性使用

宽松版会排除纯标签、纯标题、纯宣传、小说剧情设定、广告资源帖以及缺乏正文语义的碎片文本。

### 严格版语料（strict）

严格版语料服务于社会心态、立场和评价框架研究。这里保留的是把“躺平 / 摆烂 / 佛系”作为现实行为方式或评价框架核心来表达的文本，例如：

- 低投入、放弃争取、被动接受、自保、减少竞争
- 对他人、群体、机构、组织“不作为”或“摆烂”的评价
- 围绕这些标签本身展开的现实讨论

严格版不收录字面休息、佛系性格描述、交易黑话、标题口号、剧情设定，以及只是顺带提及关键词、但关键词不构成全文心态核心的文本。

## 仓库结构

```text
result/
├── raw/                         原始微博 CSV
├── data/                        主流程生成的中间数据与导出结果
│   ├── processed/               parquet 主产物
│   ├── exports/                 导出的 CSV
│   ├── reports/                 运行报告
│   └── state/                   增量处理状态
├── scripts/pipeline/            原始数据整理与预处理脚本
├── bert/                        抽样、标注、训练、预测、分析脚本
│   ├── data/                    抽样表、审核表等人工处理中间文件
│   └── artifacts/               模型、预测结果、分析结果
├── main.py                      主流程统一入口
├── README.md                    项目入口说明
├── USER_MANUAL.md               主流程手册
├── bert/README.md               抽样、训练与分析手册
└── WINDOWS_SETUP.md             Windows 环境差异说明
```

如果按工作流来理解：

1. [`main.py`](/Users/apple/Local/fdurop/code/result/main.py) 和 [`scripts/pipeline/`](/Users/apple/Local/fdurop/code/result/scripts/pipeline) 负责把原始 CSV 变成可抽样的 `data/processed/text_dedup/`。
2. [`bert/01_stratified_sampling.py`](/Users/apple/Local/fdurop/code/result/bert/01_stratified_sampling.py) 到 [`bert/05_train_dual_label_classifier.py`](/Users/apple/Local/fdurop/code/result/bert/05_train_dual_label_classifier.py) 负责抽样、预标注、人工审核后的训练。
3. [`bert/06_predict_bert_classifier.py`](/Users/apple/Local/fdurop/code/result/bert/06_predict_bert_classifier.py) 负责把训练好的模型打到全量语料。
4. [`bert/07_build_broad_analysis_base.py`](/Users/apple/Local/fdurop/code/result/bert/07_build_broad_analysis_base.py) 到 [`bert/10_concept_drift_analysis.py`](/Users/apple/Local/fdurop/code/result/bert/10_concept_drift_analysis.py) 负责 broad 语料的分析链路。

## 快速开始

仓库默认使用根目录下的 `.venv`。下面示例优先使用显式解释器路径：

- macOS / Linux：`.venv/bin/python`
- Windows PowerShell：`.\.venv\Scripts\python.exe`

首次创建环境：

macOS / Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

最短工作流：

1. 运行主流程，把 `raw/` 处理为 `data/processed/text_dedup/`
2. 从 `text_dedup` 抽样并生成预标注草稿
3. 人工审核样本
4. 用审核后的 CSV/XLSX 训练单标签或双标签模型
5. 对全量语料做预测
6. 顺序运行 `07-10` 的 broad 分析链

常用命令示例：

```bash
.venv/bin/python main.py status
.venv/bin/python main.py run
.venv/bin/python bert/01_stratified_sampling.py
.venv/bin/python bert/02_llm_label_local.py
.venv/bin/python bert/05_train_dual_label_classifier.py --help
.venv/bin/python bert/06_predict_bert_classifier.py --help
```

Windows 下把 `.venv/bin/python` 替换为 `.\.venv\Scripts\python.exe` 即可。

补充提醒：

- `bert/02_llm_label_local.py` 在首次运行前，通常需要先准备 `bert/llm_label_local.toml` 或相应环境变量/API key；直接裸跑很可能卡在 provider 配置上。
- 推荐先复制 `bert/llm_label_local.example.toml` 再修改；具体写法见 [bert/README.md](/Users/apple/Local/fdurop/code/result/bert/README.md) 和 [WINDOWS_SETUP.md](/Users/apple/Local/fdurop/code/result/WINDOWS_SETUP.md)。

## 文档导航

- [USER_MANUAL.md](/Users/apple/Local/fdurop/code/result/USER_MANUAL.md)：`raw/ -> data/processed/` 主流程、状态文件、导出逻辑
- [bert/README.md](/Users/apple/Local/fdurop/code/result/bert/README.md)：`01-10` 抽样、预标注、训练、预测和 broad 分析链
- [WINDOWS_SETUP.md](/Users/apple/Local/fdurop/code/result/WINDOWS_SETUP.md)：Windows + NVIDIA 环境差异、GPU 与离线模型说明

## 当前代码侧重点

当前仓库里已经落地的分析脚本主要围绕 broad 语料展开，即：

- `07_build_broad_analysis_base.py`
- `08_topic_model_bertopic.py`
- `09_keyword_semantic_analysis.py`
- `10_concept_drift_analysis.py`

strict 语料主要用于更严格的样本边界控制和下游社会心态研究入口；如果你只看现成脚本，优先从 broad 链路理解仓库结构会更直接。

## 仓库中通常不包含的内容

为避免仓库过大，以下内容通常不上传到 GitHub：

- `raw/`
- `data/`
- `archive/`
- `logs/`
- 本地虚拟环境

也就是说，这个仓库主要保存代码、配置和实验脚本；大体积数据和运行产物需要单独同步。
