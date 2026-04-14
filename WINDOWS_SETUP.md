# Windows + NVIDIA 4060 运行说明

这个仓库现在默认作为“代码仓库”上传 GitHub，不包含大体积数据、虚拟环境和运行产物。

如果你要在另一台 Windows 电脑继续跑，推荐按下面做：

## 1. 准备环境

- 安装 `Python 3.11`
- 安装 `Git`
- 安装最新的 NVIDIA 驱动

建议在 PowerShell 中执行。

## 2. 拉取代码并创建虚拟环境

```powershell
git clone <你的仓库地址>
cd result
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 安装支持 CUDA 的 PyTorch

如果你要跑 `bert/04_train_bert_classifier.py` 或 `bert/06_predict_bert_classifier.py`，再安装 GPU 版 PyTorch：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

安装后先检查 GPU 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出 `True`，说明 4060 已经可以被 PyTorch 调用。

## 4. 同步数据

GitHub 仓库不会包含下面这些大文件/目录：

- `raw/`
- `data/`
- `archive/`
- `logs/`
- `preprocessed.zip`
- 本地虚拟环境 `AI/`

所以你需要另外把自己真正要用的数据同步到 Windows 电脑。常见做法：

- 只传代码到 GitHub，然后用移动硬盘/网盘拷贝 `raw/`
- 如果你想直接继续训练或预测，也一并拷贝 `bert/data/` 或 `data/processed/`

## 5. 常用命令

查看主流程状态：

```powershell
python main.py status
```

增量处理：

```powershell
python main.py run
```

全量重建：

```powershell
python main.py full
```

BERT 训练：

```powershell
python bert/04_train_bert_classifier.py --device cuda
```

BERT 预测：

```powershell
python bert/06_predict_bert_classifier.py --device cuda
```

## 6. 本地标注配置

如果你要跑 `bert/02_llm_label_local.py`，请复制示例配置文件并填写自己的 key：

```powershell
Copy-Item bert/llm_label_local.example.toml bert/llm_label_local.toml
```

也可以不写入文件，直接使用环境变量：

```powershell
$env:DASHSCOPE_API_KEY="your_key_here"
python bert/02_llm_label_local.py
```
