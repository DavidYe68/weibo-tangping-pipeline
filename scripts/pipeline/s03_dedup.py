#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主流程分组脚本：dedup 视角入口。
建议优先使用 `python main.py run|full|status`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline.s01_core import export_csv, get_status, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="主流程分组脚本（dedup 视角）")
    parser.add_argument("command", nargs="?", choices=["run", "full", "status", "export-csv"], default="run")
    parser.add_argument("target", nargs="?", choices=["all", "merged", "text"], default="all")
    args = parser.parse_args()

    if args.command == "status":
        payload = get_status()
    elif args.command == "full":
        payload = run_pipeline(full_rebuild=True)
    elif args.command == "export-csv":
        payload = export_csv(args.target)
    else:
        payload = run_pipeline(full_rebuild=False)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
