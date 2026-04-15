# Windows + NVIDIA 4060 运行说明

这个文档只说明 Windows 环境下和仓库默认写法不同的部分：虚拟环境、CUDA 版 PyTorch、离线模型和常用命令。主流程说明见 [USER_MANUAL.md](/Users/apple/Local/fdurop/code/result/USER_MANUAL.md)，抽样、训练和 `07-10` 分析链见 [bert/README.md](/Users/apple/Local/fdurop/code/result/bert/README.md)。

## 1. 准备环境

- 安装 `Python 3.11`
- 安装 `Git`
- 安装最新的 NVIDIA 驱动

建议在 PowerShell 中执行。

## 2. 创建虚拟环境并安装依赖

```powershell
git clone <你的仓库地址>
cd result
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你习惯先激活虚拟环境，也可以：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. 安装支持 CUDA 的 PyTorch

如果你要跑 `bert/04_train_bert_classifier.py`、`bert/05_train_dual_label_classifier.py`、`bert/06_predict_bert_classifier.py`，或者后续的 embedding 分析脚本，可以再安装 GPU 版 PyTorch：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

安装后先检查 GPU 是否可用：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出 `True`，说明 4060 已经可以被 PyTorch 调用。

## 4. 同步数据

GitHub 仓库通常不会包含下面这些大文件/目录：

- `raw/`
- `data/`
- `archive/`
- `logs/`
- 本地虚拟环境 `.venv/`

所以你需要另外把自己真正要用的数据同步到 Windows 电脑。常见做法：

- 只传代码到 GitHub，然后用移动硬盘或网盘拷贝 `raw/`
- 如果你想直接继续训练或预测，也一并拷贝 `bert/data/`、`bert/artifacts/` 或 `data/processed/`

## 5. 常用命令

查看主流程状态：

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

如果你想看完整参数，请直接运行对应脚本的 `--help`。

## 6. 本地标注配置

如果你要跑 `bert/02_llm_label_local.py`，请复制示例配置文件并填写自己的 key：

```powershell
Copy-Item bert/llm_label_local.example.toml bert/llm_label_local.toml
```

也可以不写入文件，直接使用环境变量：

```powershell
$env:DASHSCOPE_API_KEY="your_key_here"
.\.venv\Scripts\python.exe bert/02_llm_label_local.py
```

## 7. 离线模型说明

`08_topic_model_bertopic.py` 和 `09_keyword_semantic_analysis.py` 第一次运行时，通常会下载 embedding 模型权重。

如果你的 Windows 机器离线：

- 先把模型缓存好，或者把模型目录拷到本地
- 运行脚本时使用 `--embedding_model <本地目录>`
- 再加上 `--local_files_only`

## 8. 推荐阅读顺序

1. 先读 [README.md](/Users/apple/Local/fdurop/code/result/README.md) 了解仓库入口和文档分工
2. 再读 [USER_MANUAL.md](/Users/apple/Local/fdurop/code/result/USER_MANUAL.md) 处理 `raw/ -> data/processed/`
3. 最后按 [bert/README.md](/Users/apple/Local/fdurop/code/result/bert/README.md) 跑抽样、训练、预测和 `07-10`
