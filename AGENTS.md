# Agent Guide

This repository uses the virtual environment at the repo root: `.venv`.

## Python / pip rule

- Always prefer `.venv/bin/python` and `.venv/bin/pip` on macOS/Linux.
- On Windows, use `.venv\Scripts\python.exe` and `.venv\Scripts\pip.exe`.
- Do not install dependencies into the global Python environment.
- If a command in the docs uses `python` or `pip`, assume it should run inside `.venv`.

## First-run setup

If `.venv` does not exist yet:

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Preferred command style

Use explicit interpreter paths when running scripts non-interactively:

macOS/Linux:

```bash
.venv/bin/python main.py status
.venv/bin/python main.py run
.venv/bin/python bert/04_train_bert_classifier.py --help
.venv/bin/python bert/05_train_dual_label_classifier.py --help
.venv/bin/python bert/06_predict_bert_classifier.py --help
```

Windows:

```powershell
.\.venv\Scripts\python.exe main.py status
.\.venv\Scripts\python.exe main.py run
.\.venv\Scripts\python.exe bert/04_train_bert_classifier.py --help
.\.venv\Scripts\python.exe bert/05_train_dual_label_classifier.py --help
.\.venv\Scripts\python.exe bert/06_predict_bert_classifier.py --help
```

## Project notes

- Main pipeline entry: `main.py`
- BERT workflow details: `bert/README.md`
- Windows setup: `WINDOWS_SETUP.md`
- The worktree may contain local user changes; avoid destructive git commands unless explicitly requested.
