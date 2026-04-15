#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目统一入口：
- run         增量处理（默认）
- full        全量重建
- status      查看状态
- export-csv  按需导出 CSV
"""

from __future__ import annotations

import argparse
import json
import sys

from scripts.pipeline.s01_core import export_csv, get_status, run_pipeline


def emit_cli_progress(message: str) -> None:
    print(f"[main] {message}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="主流程入口：处理 raw/，维护 data/processed/ 与 data/state/，并按需导出 CSV。",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["run", "full", "status", "export-csv"],
        default="run",
        help="run=增量处理，full=全量重建，status=查看状态，export-csv=导出 CSV；默认 run。",
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=["all", "merged", "text"],
        default="all",
        help="仅用于 export-csv：all=全部，merged=merged_dedup，text=text_dedup；默认 all。",
    )
    args = parser.parse_args()

    if args.command == "status":
        emit_cli_progress("collecting status")
        payload = get_status()
    elif args.command == "full":
        emit_cli_progress("running full rebuild")
        payload = run_pipeline(full_rebuild=True)
    elif args.command == "export-csv":
        emit_cli_progress(f"exporting csv target={args.target}")
        payload = export_csv(args.target)
    else:
        emit_cli_progress("running incremental pipeline")
        payload = run_pipeline(full_rebuild=False)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
