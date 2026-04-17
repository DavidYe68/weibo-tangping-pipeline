# 微博“躺平 / 摆烂 / 佛系”语义研究项目

这个仓库不是拿关键词做简单计数，而是想尽量回答一个更实际的问题：

同样写了“躺平”“摆烂”“佛系”的微博，哪些是真的在表达某种心态、立场和评价，哪些只是口头禅、标题、交易黑话、剧情设定，或者顺手一提。

所以整个项目被拆成两段：

1. 主流程：先把 `raw/` 里的原始 CSV 整理成能继续处理的标准语料。
2. 下游流程：再抽样、预标注、人工审核、训练分类器，并把模型打到全量语料上做分析。

这份 `README.md` 只做总览和导航。具体怎么跑、哪里容易出问题，请看对应子文档。

## 这个项目在研究什么

当前语料围绕三组关键词展开：

- `躺平`
- `摆烂`
- `佛系`

更具体一点，项目主要关心这些事：

- 这些词在微博里到底落进了哪些语义场景
- 它们是日常口语、生活方式、评价框架，还是某种特定圈层黑话
- 不同时间段里，语义重点有没有变化
- 在地域和时间维度上，哪些表达在扩张，哪些在收缩

## 两套语料边界

为了后面分析不打架，项目把语料边界分成两套。

### `broad`：宽口径

这套语料是给“概念怎么漂移、语义怎么分布”这种问题用的。只要关键词在正文里确实承担了某种可解释的语义，通常都可以留下来。

比如：

- 字面“躺平”、休息、放松
- 日常“摆烂”、松懈、状态不好
- “佛系”作为风格、做事方式、相处态度
- 股票、游戏、饭圈等语境里的转义用法
- 对群体、组织、机构的评价性使用

但像下面这些，一般会排掉：

- 只有标题和口号，没有正文内容
- 纯广告、资源帖、抽卡帖
- 小说剧情设定
- 只带了关键词，但正文没有实际语义

### `strict`：严口径

这套语料是给“社会心态、现实行为方式、评价框架”这种更窄的问题用的。重点是：关键词是不是这条微博真正的意思核心。

比如：

- 低投入、少争取、被动接受、自保
- 对个人、组织、机构“不作为”“摆烂”的评价
- 围绕“躺平 / 摆烂 / 佛系”本身展开的现实讨论

一般不会收：

- 字面休息
- 单纯性格描述
- 股票黑话、交易黑话
- 口号式标题
- 只是顺带提及关键词的文本

一句话理解：

- `broad` 更适合看“这个词都被怎么用”
- `strict` 更适合看“这个词是不是在表达一种现实态度或评价”

## 仓库怎么读

如果第一次进这个仓库，建议按这个顺序看：

1. 先看这份 `README.md`，知道项目分几段、文档怎么分工
2. 再看 [`USER_MANUAL.md`](./USER_MANUAL.md)，把 `raw/ -> data/processed/` 这一段跑通
3. 如果要做抽样、训练、预测和分析，再看 [`bert/README.md`](./bert/README.md)
4. 如果你在 Windows 上跑，补看 [`WINDOWS_SETUP.md`](./WINDOWS_SETUP.md)

## 文档分工

### 根目录 `README.md`

这份文档只回答三个问题：

- 这个项目是干什么的
- 仓库分成哪几段
- 你下一步该看哪份文档

### `USER_MANUAL.md`

这份文档只讲主流程，也就是：

- `raw/` 里的 CSV 怎么被识别
- `main.py run / full / status / export-csv` 分别干什么
- `data/processed/`、`data/state/`、`data/reports/` 里会产生什么
- 平时该什么时候增量跑，什么时候全量重建

### `bert/README.md`

这份文档只讲下游流程，也就是：

- 从 `text_dedup` 抽样
- LLM 预标注
- 人工审核
- 单标签 / 双标签训练
- 全量预测
- `07-10` 的 broad 分析链

### `WINDOWS_SETUP.md`

这份文档只讲 Windows 和默认环境不一样的地方：

- 虚拟环境
- CUDA 版 PyTorch
- 本地或离线模型
- Windows 下最常见的命令和报错场景

## 仓库结构

```text
result/
├── raw/                         原始微博 CSV
├── data/
│   ├── processed/               主流程生成的 parquet
│   ├── exports/                 导出的 CSV
│   ├── reports/                 运行报告
│   └── state/                   增量处理状态
├── scripts/pipeline/            主流程脚本
├── bert/                        抽样、标注、训练、预测、分析
│   ├── data/                    抽样表、审核表等中间文件
│   ├── artifacts/               模型、预测结果、分析结果
│   └── README.md
├── main.py                      主流程统一入口
├── README.md
├── USER_MANUAL.md
└── WINDOWS_SETUP.md
```

如果按实际工作来理解：

- `main.py` 负责把原始数据整理成可抽样的标准输入
- `bert/01-06` 负责从样本走到全量预测
- `bert/07-10` 负责 broad 语料的后续分析

## 最短上手路径

仓库默认使用根目录下的 `.venv`。

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

最短命令链：

```bash
.venv/bin/python main.py status
.venv/bin/python main.py run
.venv/bin/python bert/01_stratified_sampling.py
.venv/bin/python bert/02_llm_label_local.py
.venv/bin/python bert/05_train_dual_label_classifier.py --help
.venv/bin/python bert/06_predict_bert_classifier.py --help
```

Windows 下把 `.venv/bin/python` 换成 `.\.venv\Scripts\python.exe`。

## 一条完整流程长什么样

实际工作里，最常见的是这条线：

1. 把原始微博 CSV 放到 `raw/`
2. 跑 `main.py run`，得到 `data/processed/text_dedup/`
3. 用 `bert/01_stratified_sampling.py` 抽样
4. 用 `bert/02_llm_label_local.py` 做预标注草稿
5. 人工审核
6. 用 `04` 或 `05` 训练分类器
7. 用 `06` 把模型打到全量语料
8. 继续跑 `07-10` 做 broad 分析

如果你现在只是想确认主流程是否正常，读 `USER_MANUAL.md` 就够了。

如果你现在已经有审核过的标注表，直接去读 `bert/README.md` 会更省时间。

## 现在仓库里最成熟的是哪一段

当前已经比较成体系的是这条 broad 分析链：

- `07_build_broad_analysis_base.py`
- `08_topic_model_bertopic.py`
- `09_keyword_semantic_analysis.py`
- `10_concept_drift_analysis.py`

也就是说：

- `strict` 更像边界更严的研究入口
- `broad` 是目前最适合直接往下接分析脚本的一套结果

## 仓库里通常不会带什么

为了避免仓库太大，下面这些内容通常不会上传：

- `raw/`
- `data/`
- `archive/`
- `logs/`
- `.venv/`

所以 GitHub 上一般只有代码和文档。真正要跑的时候，数据和历史产物需要你自己同步。

## 你现在该去哪份文档

- 想处理原始数据：看 [`USER_MANUAL.md`](./USER_MANUAL.md)
- 想做抽样、训练、预测、分析：看 [`bert/README.md`](./bert/README.md)
- 想在 Windows + NVIDIA 上跑：看 [`WINDOWS_SETUP.md`](./WINDOWS_SETUP.md)
