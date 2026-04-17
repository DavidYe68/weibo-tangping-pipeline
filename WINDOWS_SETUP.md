# Windows 运行说明

这份文档只讲 Windows 环境和仓库默认写法不一样的地方。

如果你想知道项目整体在做什么，请先看 [`README.md`](./README.md)。

如果你想知道主流程怎么跑，请看 [`USER_MANUAL.md`](./USER_MANUAL.md)。

如果你想做抽样、训练、预测和 `07-10` 分析，请看 [`bert/README.md`](./bert/README.md)。

## 什么时候需要看这份文档

下面这些情况，建议直接看这份：

- 你在 Windows 机器上第一次配环境
- 你准备用 NVIDIA GPU 跑训练或预测
- 你准备离线跑 embedding 模型
- 你照着根目录文档执行，但命令路径在 Windows 下不对

## 先说结论

Windows 下最常见的差异，其实就三件事：

1. Python 解释器路径不一样
2. 如果要用 GPU，需要额外确认 CUDA 版 PyTorch
3. 第一次跑 embedding 或 BERT 相关脚本时，模型下载和缓存更容易出问题

只要先把这三件事弄对，后面命令基本和其他平台一样。

## 环境准备

建议先准备：

- `Python 3.11`
- `Git`
- 最新的 NVIDIA 驱动

如果你打算跑训练或大批量预测，建议直接在 PowerShell 里操作，少一点路径和权限问题。

## 创建虚拟环境并安装依赖

```powershell
git clone <你的仓库地址>
cd result
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你习惯先激活环境，也可以这样：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在这个仓库里，非交互命令更推荐直接写完整解释器路径，也就是：

```powershell
.\.venv\Scripts\python.exe ...
```

这样最省心。

## GPU 什么时候需要额外处理

如果你只是跑主流程：

- `main.py run`
- `main.py full`
- `main.py status`

通常不需要额外折腾 CUDA。

如果你要跑这些脚本，GPU 会更有价值：

- `bert/04_train_bert_classifier.py`
- `bert/05_train_dual_label_classifier.py`
- `bert/06_predict_bert_classifier.py`
- `bert/08_topic_model_bertopic.py`
- `bert/09_keyword_semantic_analysis.py`

尤其是训练和 embedding 相关步骤，差别会比较明显。

## 安装支持 CUDA 的 PyTorch

如果你的机器是 NVIDIA 显卡，并且你想让 PyTorch 真正走 GPU，可以再安装 CUDA 版 wheel：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

装完之后，先别急着跑训练，先检查一遍：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果最后输出是 `True`，说明 PyTorch 已经能看到 GPU。

如果不是 `True`，常见原因通常是：

- 显卡驱动太旧
- 装成了 CPU 版 PyTorch
- 机器虽然有 NVIDIA，但当前环境没有正确加载驱动

## 数据要自己同步

GitHub 仓库通常不会包含这些大目录：

- `raw/`
- `data/`
- `archive/`
- `logs/`
- `.venv/`

所以在 Windows 电脑上，最常见的真实情况是：

- 代码拉下来了
- 但数据和历史产物并不在仓库里

这时候你还需要自己同步：

- 如果只想重新跑主流程，至少要把 `raw/` 带过来
- 如果想接着训练或预测，可能还要带 `bert/data/`
- 如果想直接接分析结果，可能还要带 `bert/artifacts/` 或 `data/processed/`

## Windows 下最常用命令

### 主流程

查看状态：

```powershell
.\.venv\Scripts\python.exe main.py status
```

增量处理：

```powershell
.\.venv\Scripts\python.exe main.py run
```

全量重建：

```powershell
.\.venv\Scripts\python.exe main.py full
```

### 下游训练与预测

双标签训练：

```powershell
.\.venv\Scripts\python.exe bert/05_train_dual_label_classifier.py --device cuda
```

全量预测：

```powershell
.\.venv\Scripts\python.exe bert/06_predict_bert_classifier.py --device cuda
```

BERTopic / 语义分析：

```powershell
.\.venv\Scripts\python.exe bert/08_topic_model_bertopic.py --device cuda
.\.venv\Scripts\python.exe bert/09_keyword_semantic_analysis.py --device cuda
```

如果想看完整参数，直接在对应命令后加 `--help`。

## 本地预标注配置

如果你要跑 `bert/02_llm_label_local.py`，一般先从示例配置文件开始最省事：

```powershell
Copy-Item bert/llm_label_local.example.toml bert/llm_label_local.toml
```

然后把自己的 provider、model、API key 配进去。

如果你不想写配置文件，也可以临时用环境变量：

```powershell
$env:DASHSCOPE_API_KEY="your_key_here"
.\.venv\Scripts\python.exe bert/02_llm_label_local.py
```

如果你走的是本地 `ollama`，通常不需要 API key，但要先确认本地服务和模型都已经就绪。

## 离线模型怎么处理

`08_topic_model_bertopic.py` 和 `09_keyword_semantic_analysis.py` 第一次运行时，通常会下载 embedding 模型。

如果 Windows 机器不能联网，建议提前准备本地模型目录，然后这样跑：

- 把模型缓存好，或者直接拷贝模型目录
- 用 `--embedding_model <本地目录>` 指向它
- 再加 `--local_files_only`

这一步很重要。否则脚本会默认尝试联网加载，离线环境下容易卡住或直接报错。

## 常见情况

### 主流程能跑，但训练脚本很慢

先看是不是还在 CPU 上跑。很多时候不是代码慢，而是根本没用上 GPU。

### `--device cuda` 传了还是报错

通常先查两件事：

1. `torch.cuda.is_available()` 到底是不是 `True`
2. 你装的是不是 CUDA 版 PyTorch

### 第一次跑 `08` / `09` 卡住

很常见，尤其是第一次下载模型时。先确认：

- 当前机器能不能联网
- 模型是不是已经在本地缓存
- 是否需要 `--local_files_only`

### Windows 下路径老是写错

最简单的做法是统一用这一种格式：

```powershell
.\.venv\Scripts\python.exe some_script.py
```

不要在同一个会话里一会儿用激活环境的 `python`，一会儿又用系统 Python，容易把依赖装乱。

## 推荐阅读顺序

1. 先读 [`README.md`](./README.md)，知道仓库结构和文档分工
2. 再读 [`USER_MANUAL.md`](./USER_MANUAL.md)，先把主流程跑通
3. 最后按 [`bert/README.md`](./bert/README.md) 去跑抽样、训练、预测和分析
