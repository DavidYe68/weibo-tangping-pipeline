# 08 主题建模当前操作说明

这份说明只讲一件事：

**`bert/08_topic_model_bertopic.py` 现在到底是怎么跑出来结果的。**

不讲历史版本，不讲“原来是什么、后来改成什么”，只讲当前代码实际执行的流程、输入、参数、输出，以及你现在看到的那些结果表是怎么来的。

---

## 1. 这一步的定位

`08_topic_model_bertopic.py` 是 broad 分析链里的主题建模步骤。

它接收 `07_build_broad_analysis_base.py` 生成的 broad 底表，然后完成下面这些事：

1. 读取并筛选 broad 分析文本
2. 把文本编码成 embedding
3. 用 UMAP 降维
4. 用 HDBSCAN 聚类
5. 让 BERTopic 提取每个 topic 的代表词
6. 可选地对 `-1` outlier 做再分配
7. 生成一组适合后续阅读、汇报、可视化的输出表

所以你可以把 `08` 理解成：

**把 broad 语料切成主题，并把“主题是什么、各主题占多少、什么时候更明显、在哪些关键词/IP 下更突出”这些信息整理出来。**

---

## 2. 输入是什么

默认输入是：

`bert/artifacts/broad_analysis/analysis_base.parquet`

这个表必须至少有下面这些列：

- `analysis_text`：要做主题建模的文本
- `keyword_normalized`：规范化后的关键词，比如 `躺平` / `摆烂` / `佛系`
- `publish_time`：时间列

IP 列不是硬性必需项。

如果有原始地区列，脚本会尝试自动识别并规范化；如果没有，脚本也能继续跑，只是会把全部记录统一标成缺失 IP。

脚本默认参数里，对应关系是：

- `--text_col analysis_text`
- `--keyword_col keyword_normalized`
- `--time_col publish_time`
- `--ip_col` 默认自动检测

如果这些列名不一致，脚本会直接报错，不会静默替你猜。

---

## 3. 最基本的运行方式

最基本的命令是：

```bash
.venv/bin/python bert/08_topic_model_bertopic.py
```

这会使用当前代码里的默认配置：

- 输入：`bert/artifacts/broad_analysis/analysis_base.parquet`
- 输出：`bert/artifacts/broad_analysis/topic_model_BAAI`
- 关键词：`躺平 摆烂 佛系`
- 文本列：`analysis_text`
- 时间粒度：`month`
- embedding 模型：`BAAI/bge-small-zh-v1.5`
- `min_topic_size=30`
- `top_n_words=10`
- `nr_topics=None`
- `topic_tokenizer=jieba`
- `topic_stopwords_path=bert/config/topic_stopwords.txt`
- `topic_token_min_length=2`
- `umap_n_neighbors=30`
- `umap_low_memory=True`
- `calculate_probabilities=False`
- `hdbscan_min_samples=None`，此时会自动等于 `min_topic_size`
- `outlier_reduction_strategy=none`

也就是说，**默认跑法本身并不会做 outlier 再分配**。

---

## 4. 如果你要复现当前 broad 里常见的对比跑法

你现在最近在看的那批 broad topic compare 结果，并不是默认参数直接跑出来的，而是用 `08` 的当前能力、配上更激进的聚类参数和 outlier reduction 跑出来的。

比如你现在一直在看的 `O_outlier`，对应的是这种思路：

```bash
.venv/bin/python bert/08_topic_model_bertopic.py \
  --input_path bert/artifacts/broad_analysis/analysis_base.parquet \
  --output_dir bert/artifacts/broad_analysis/topic_model_compare/O_outlier \
  --min_topic_size 500 \
  --hdbscan_min_samples 25 \
  --umap_n_neighbors 120 \
  --outlier_reduction_strategy c-tf-idf+distributions \
  --outlier_reduction_threshold 0.0 \
  --resume
```

这条命令表达的是：

- 先把聚类做得更粗一些：`min_topic_size=500`
- 同时单独把 `hdbscan_min_samples` 调低到 `25`，避免 outlier 太多
- 用更大的 `umap_n_neighbors=120`，让局部邻域更宽
- 聚类完成后，再用 BERTopic 官方的 `reduce_outliers`
- 先跑 `c-tf-idf`，再对剩余 `-1` 跑 `distributions`
- 因为 `--resume` 开着，所以 embedding 和降维都可以复用 checkpoint

如果你看的是 `topic_model_compare/` 下面那些子目录，本质上都是 **同一个 `08` 脚本**，只是换了参数和输出目录。

---

## 5. 当前代码的实际执行流程

下面按代码的真实顺序讲。

### 5.1 读取底表并检查必需列

脚本先读取 `--input_path` 指向的表。

然后立刻检查：

- 关键词列在不在
- 文本列在不在
- 时间列在不在

如果缺任何一个，直接报错。

接着它会自动识别地区列，并统一补出：

- `ip_normalized`
- `ip_missing`

所以后面按 IP 做分布统计时，脚本不依赖你手动先把 IP 清洗好。

---

### 5.2 先做关键词过滤，再去掉空文本

这一步不是对全量底表直接跑 BERTopic，而是先把目标关键词筛出来。

当前逻辑是：

1. 把 `keyword_col` 里属于 `--keywords` 的行保留
2. 把文本列空值补成空字符串
3. 去掉去空格后仍然为空的文本

所以最终进入 BERTopic 的，并不是分析底表的全部记录，而是：

**属于指定关键词集合、且文本不为空的那部分。**

默认关键词集合是：

- `躺平`
- `摆烂`
- `佛系`

---

### 5.3 生成 `period_label`

脚本会把 `publish_time` 转成一个按粒度截断后的时间标签，列名固定叫：

`period_label`

具体长什么样，取决于 `--time_granularity`：

- `month` -> `2023-07`
- `quarter` -> `2023Q3`
- `year` -> `2023`

后面所有按时间的 share 表，都是基于这个 `period_label` 算出来的。

---

### 5.4 为 resume/checkpoint 计算数据指纹

脚本不会只看“文件名一样不一样”来判断能不能续跑。

它会从下面这些列生成一个 fingerprint：

- 文本列
- 关键词列
- 时间列
- `ip_normalized`

这个 fingerprint 会写进：

`<output_dir>/checkpoints/checkpoint_manifest.json`

这样 `--resume` 时，脚本能判断：

- 现在这批文档是不是和上一次完全一致
- embedding 模型是不是同一个
- UMAP 配置是不是同一个

只有这些都对得上，才会真的复用旧 checkpoint。

---

### 5.5 embedding：先尝试复用，不行就重新编码

当前 embedding 模型默认是：

`BAAI/bge-small-zh-v1.5`

执行逻辑是：

1. 如果 `--resume` 开着、embedding checkpoint 存在、指纹一致、而且 checkpoint 记录的 embedding 模型也和当前请求一致  
   -> 直接加载 `document_embeddings.npy`
2. 否则  
   -> 用 sentence-transformers 重新编码全部文本

编码后的结果会保存到：

`<output_dir>/checkpoints/document_embeddings.npy`

所以 embedding 是这一步里最早被 checkpoint 化的部分。

设备选择逻辑是：

- 明确指定 `cpu/cuda/mps` 就照用
- `auto` 时按下面顺序找：
  1. CUDA
  2. MPS
  3. CPU

---

### 5.6 topic term 提取：当前默认用中文 tokenizer

这一步很重要，因为它直接影响你看到的 `top_terms`。

当前默认不是 sklearn 的英文 token 逻辑，而是：

`topic_tokenizer=jieba`

脚本会构造一个 `CountVectorizer`，里面挂的是自定义的 `ChineseTopicTokenizer`。

这个 tokenizer 现在的行为是：

1. 优先用 `jieba.lcut`
2. 如果环境里没有 `jieba`，就退回脚本里的 CJK fallback 分词
3. 读入 `bert/config/topic_stopwords.txt`
4. 过滤停用词
5. 过滤纯数字 token，比如 `10`
6. 过滤页码类 token，比如 `p1`
7. 过滤长度小于 `topic_token_min_length` 的 token
8. 过滤不匹配中文/英文/数字/下划线 token 规则的碎片

所以你现在看到的 topic term 提取，不是裸 BERTopic，也不是默认英文 CountVectorizer，而是：

**`jieba + stopwords + numeric/page token 过滤 + 最小长度约束`**

如果你显式写：

```bash
--topic_tokenizer default
```

那脚本才会退回默认的 CountVectorizer 行为。

---

### 5.7 当前 BERTopic 模型是怎么组起来的

当前代码不是完全交给 BERTopic 自己内部默认配置，而是显式构造了几个关键组件。

#### 5.7.1 UMAP

如果环境里装了 `umap-learn`，脚本会显式创建：

- `n_neighbors = args.umap_n_neighbors`
- `n_components = 5`
- `min_dist = 0.0`
- `metric = cosine`
- `low_memory = args.umap_low_memory`
- `random_state = args.seed`
- `verbose = args.umap_verbose`

所以当前 `08` 的降维空间固定是 **5 维 cosine UMAP**。

#### 5.7.2 HDBSCAN

如果环境里装了 `hdbscan`，脚本会显式创建：

- `min_cluster_size = min_topic_size`
- `min_samples = hdbscan_min_samples`  
  如果你没单独给，就等于 `min_topic_size`
- `metric = euclidean`
- `cluster_selection_method = eom`
- `core_dist_n_jobs = args.hdbscan_core_dist_n_jobs`
- `prediction_data = args.calculate_probabilities`

所以当前 outlier 多不多、topic 碎不碎，最关键的三个旋钮就是：

- `min_topic_size`
- `hdbscan_min_samples`
- `umap_n_neighbors`

#### 5.7.3 BERTopic 本体

最后 BERTopic 本体接进去的是：

- `language`
- `embedding_model=None`
- `umap_model=显式构造的 UMAP`
- `hdbscan_model=显式构造的 HDBSCAN`
- `vectorizer_model=中文 CountVectorizer`
- `min_topic_size`
- `top_n_words`
- `nr_topics`
- `low_memory`
- `calculate_probabilities`

这里有一个关键点：

**embedding 是脚本自己先算好的，BERTopic 本体并不负责重新编码文本。**

---

### 5.8 降维也支持 checkpoint

embedding 之后，脚本还会单独对降维做 checkpoint。

如果下面这些都满足：

- `--resume`
- `reduced_embeddings.npy` 存在
- `dimensionality_reduction_model.pkl` 存在
- fingerprint 一致
- reducer signature 一致
- embedding 模型一致

那脚本会直接复用：

- `reduced_embeddings.npy`
- `dimensionality_reduction_model.pkl`

否则就重新跑 UMAP，并把结果重新保存。

也就是说，当前 `08` 的 resume 是两层：

1. embedding 级别
2. 降维级别

这也是为什么大语料重跑时，`--resume` 很有价值。

---

### 5.9 聚类和 topic 提取是怎么做的

当前代码不是直接调用一个最外层的 `fit_transform(texts)` 完事，而是把 BERTopic 的几个内部步骤拆开了：

1. 用已经降维好的向量做聚类
2. 如果没有 `nr_topics`，先按频次重排 topic 编号
3. 提取 topic representation
4. 如果设置了 `nr_topics`，再做 topic reduction
5. 保存 representative docs
6. 映射概率

这意味着：

- 如果你设了 `nr_topics=auto` 或某个整数  
  -> BERTopic 会在初始聚类后再做合并
- 如果你没设  
  -> 保留原始聚类主题

---

### 5.10 当前 outlier reduction 是怎么工作的

默认值是：

`--outlier_reduction_strategy none`

也就是说，默认**不做** outlier reduction。

但当前脚本已经支持这些策略：

- `none`
- `c-tf-idf`
- `distributions`
- `embeddings`
- `probabilities`
- `c-tf-idf+distributions`

具体逻辑是：

1. 先拿到初始 topic 分配
2. 统计初始 `-1` 数量
3. 如果策略不是 `none`，就调用 BERTopic 的 `reduce_outliers`
4. 如果你选的是 `c-tf-idf+distributions`  
   -> 先跑一遍 `c-tf-idf`  
   -> 只对还剩下的 `-1` 再跑一遍 `distributions`
5. 如果发生了 topic 变更，就调用 `topic_model.update_topics(...)`

有两个你现在特别该记住的点：

#### 第一，`probabilities` 策略只有在 `--calculate_probabilities` 开着时才能用

因为这个策略需要完整的 document-topic probability matrix。

#### 第二，outlier 被重新分配后，脚本会把这些文档的 `topic_probability` 设成 `NaN`

原因不是 bug，而是：

**BERTopic 重新分配了 topic，但不会同时给出新的 HDBSCAN 概率。**

所以当前代码的处理是：

- `topic_id` 和 `topic_label` 更新为新 topic
- `topic_probability` 留空

这是故意这样做的。

---

## 6. `doc_topics` 这张底层文档表是怎么来的

聚类完成后，脚本先生成一张文档级结果表。

它本质上是在筛过关键词、去过空文本的 `filtered` 表上，追加：

- `topic_id`
- `topic_probability`
- `topic_label`

其中：

- `topic_id < 0` 表示 outlier
- `topic_label` 来自 `topic_info` 的 `topic_label_machine`
- 如果 label 缺失，就回退到 `Name`
- 再不行就回退成 `Topic {id}`

这张文档级表会保存到：

`viz_inputs/document_topics.parquet`

后面几乎所有 share 表，都是从这张表 groupby 出来的。

---

## 7. 当前 08 会生成哪些表，以及它们是怎么算出来的

### 7.1 `topic_info.csv`

路径：

`readouts/topic_info.csv`

来源：

- `topic_model.get_topic_info()`

脚本会再补两列：

- `topic_label_machine`
- `topic_label_zh`

其中：

- `topic_label_machine` 默认取 BERTopic 的 `Name`
- `topic_label_zh` 只是预留人工中文命名，默认空着

这张表最接近 BERTopic 原始输出。

---

### 7.2 `topic_terms.csv`

路径：

`readouts/topic_terms.csv`

来源：

- 对每个正常 topic 调 `topic_model.get_topic(topic_id)`
- 再 flatten 成长表

这张表是你看每个 topic 的 term rank、term score 时最底层的来源。

---

### 7.3 `topic_share_by_period.csv`

路径：

`readouts/topic_share_by_period.csv`

算法：

1. 先只保留 `topic_id >= 0` 的文档
2. 按 `period_label + topic_id + topic_label` 分组计数
3. 得到 `doc_count`
4. 再除以该 period 下全部正常 topic 文档总数
5. 得到 `doc_share`

所以它回答的是：

**在某个时间段内部，各 topic 分别占了多大比例。**

---

### 7.4 `topic_share_by_period_and_keyword.csv`

路径：

`readouts/topic_share_by_period_and_keyword.csv`

算法：

1. 先保留正常 topic 文档
2. 按 `keyword + period + topic` 分组计数
3. 用同一个 `keyword + period` 下的总量做分母
4. 算出 `doc_share`

它回答的是：

**同一关键词内部，某个时期最突出的 topic 是什么。**

---

### 7.5 `topic_share_by_ip.csv`

路径：

`viz_inputs/topic_share_by_ip.csv`

算法和 period 版一样，只是把时间换成地区：

1. 先保留正常 topic 文档
2. 按 `ip_normalized + topic` 分组
3. 计算每个 IP 内部的 topic 占比

它回答的是：

**在不同地区内部，topic 结构怎么分布。**

---

### 7.6 `topic_share_by_period_and_ip.csv`

路径：

`viz_inputs/topic_share_by_period_and_ip.csv`

这个是：

**每个时期、每个 IP 内部的 topic 占比。**

分组维度是：

- `period_label`
- `ip_normalized`
- `topic_id`
- `topic_label`

---

### 7.7 `topic_share_by_period_and_ip_and_keyword.csv`

路径：

`viz_inputs/topic_share_by_period_and_ip_and_keyword.csv`

这个是最细粒度的一张 share 表：

- 关键词
- 时间
- IP
- topic

都一起分开算。

它适合做更细的切片分析，但不适合第一眼总览。

---

### 7.8 `topic_overview.csv`

路径：

`readouts/topic_overview.csv`

这张表不是 BERTopic 直接给的，而是脚本自己整理出来的“阅读友好版”总表。

它用到的输入是：

- `topic_info.csv`
- `topic_terms.csv`
- `topic_share_by_period.csv`
- `topic_share_by_period_and_keyword.csv`

当前代码里，它是这样组出来的：

1. 从 `topic_info` 取正常 topic
2. 从 `topic_terms` 拼出每个 topic 的前 10 个词
3. 从 `topic_share_by_period` 里找该 topic 的峰值时间段
4. 从 `topic_share_by_period_and_keyword` 里找该 topic 的 dominant keyword
5. 再补出：
   - `share_of_all_docs_pct`
   - `share_of_clustered_docs_pct`
   - `topic_label_display`
   - `peak_doc_count`
   - `peak_doc_share_pct`
   - `dominant_keyword_share_within_topic_pct`

这里要特别注意：

**当前 `topic_overview.csv` 里的 `peak_period` 是按绝对量 `doc_count` 优先选的。**

排序规则是：

1. `doc_count` 降序
2. `doc_share` 降序
3. `period_label` 升序

所以你在 `08` 原始结果里看到的 peak，是“绝对量峰值”，不是“相对占比峰值”。

---

### 7.9 `topic_model_summary.json`

路径：

`topic_model_summary.json`

这是当前这次运行的元信息摘要。

它会记录：

- 输入输出路径
- 文档量
- clustered 文档量
- outlier 文档量与比例
- topic 数
- embedding 模型
- 设备
- tokenizer
- stopwords 路径
- UMAP 参数
- HDBSCAN 参数
- `nr_topics`
- 初始 outlier 规模
- outlier reduction 策略与阈值
- checkpoint 路径
- 各类输出表路径

这张 json 很重要，因为它相当于这次 run 的“配置快照 + 结果摘要”。

---

## 8. 当前输出目录结构

对任意一个 `--output_dir`，当前 `08` 会生成：

```text
<output_dir>/
├── topic_model_summary.json
├── checkpoints/
│   ├── filtered_documents.parquet
│   ├── document_embeddings.npy
│   ├── reduced_embeddings.npy
│   ├── dimensionality_reduction_model.pkl
│   └── checkpoint_manifest.json
├── readouts/
│   ├── topic_info.csv
│   ├── topic_overview.csv
│   ├── topic_terms.csv
│   ├── topic_share_by_period.csv
│   └── topic_share_by_period_and_keyword.csv
└── viz_inputs/
    ├── document_topics.parquet
    ├── topic_share_by_ip.csv
    ├── topic_share_by_period_and_ip.csv
    ├── topic_share_by_period_and_ip_and_keyword.csv
    └── model/               # 只有 --save_model 时才会有
```

---

## 9. 现在看 08 的结果，推荐的顺序

如果你现在要看某一轮 `08` 的结果，最稳的顺序是：

1. 先看 `topic_model_summary.json`  
   确认这轮到底用了什么参数、outlier 有多少
2. 再看 `topic_overview.csv`  
   建立对整盘 topic 的第一印象
3. 再看 `topic_info.csv`  
   检查命名和代表文本到底靠不靠谱
4. 然后看 `topic_share_by_period.csv`  
   判断这些 topic 的时间变化
5. 如果要做更细切片，再看：
   - `topic_share_by_period_and_keyword.csv`
   - `topic_share_by_ip.csv`
   - `topic_share_by_period_and_ip.csv`

---

## 10. 现在最容易误解的几个点

### 10.1 `Topic = -1` 不是一个正常主题

它表示 outlier，也就是 HDBSCAN 没把这些文本稳定归进任何一个 cluster。

---

### 10.2 `topic_overview.csv` 不是 BERTopic 原生表

它是脚本在 `topic_info + topic_terms + share` 基础上重新拼出来的摘要表。

---

### 10.3 `peak_period` 在 08 原始输出里默认是绝对量峰值

不是相对占比峰值。

如果你想看“某主题在哪个时期最占结构比例”，要么自己改读 `doc_share`，要么像我们后面做 macro merge 那样再做一层派生表。

---

### 10.4 outlier reduction 之后的 `topic_probability` 可能是空的

这不是丢数据，而是因为重新分配后的概率 BERTopic 没有重算。

---

### 10.5 `resume` 不是无脑续跑

只有 fingerprint 和 reducer signature 对得上，checkpoint 才会被真正复用。

---

## 11. 一句话总结

当前的 `08_topic_model_bertopic.py` 做的是：

**从 broad 分析底表中筛出目标关键词文本，先算 embedding，再用 UMAP + HDBSCAN + BERTopic 做主题提取，可选地把 outlier 再分配回 topic，最后输出一组面向阅读、时间分析、关键词切片、IP 切片的结果表。**

如果你接下来要继续写汇报，最常用的三张表仍然是：

1. `topic_model_summary.json`
2. `readouts/topic_overview.csv`
3. `readouts/topic_share_by_period.csv`

它们基本已经能回答：

- 这轮切出了什么主题
- 每个主题大概是什么
- 它们在什么时候更强
