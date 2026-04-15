#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from lib.io_utils import ensure_parent, save_json
from lib.labels import NORMALIZE_LABEL_CANDIDATES, detect_label_column, normalize_label_value


def emit(message: str) -> None:
    print(f"[normalize] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 labeled.csv 中的审核标签整理成单标签训练可用的二值标签。"
    )
    parser.add_argument(
        "--input_csv",
        default="bert/data/labeled.csv",
        help="输入带标签的 CSV 路径；默认 bert/data/labeled.csv。",
    )
    parser.add_argument(
        "--output_csv",
        default="bert/data/labeled_binary.csv",
        help="整理后的输出 CSV 路径；默认 bert/data/labeled_binary.csv。",
    )
    parser.add_argument(
        "--report_path",
        default="bert/data/labeled_binary_report.json",
        help="整理报告 JSON 路径；默认 bert/data/labeled_binary_report.json。",
    )
    parser.add_argument(
        "--label_col",
        default=None,
        help="可选，显式指定来源标签列；不传则自动识别常见候选列。",
    )
    return parser.parse_args()


def build_report(
    df: pd.DataFrame,
    label_source: str,
    invalid_rows: pd.DataFrame,
) -> Dict[str, Any]:
    binary_counts = df["label"].value_counts(dropna=False).sort_index().to_dict()
    text_counts = df["label_text"].value_counts(dropna=False).to_dict()
    report: Dict[str, Any] = {
        "rows_total": int(len(df)),
        "label_source": label_source,
        "binary_counts": {str(k): int(v) for k, v in binary_counts.items()},
        "label_text_counts": {str(k): int(v) for k, v in text_counts.items()},
        "invalid_rows": int(len(invalid_rows)),
    }
    if not invalid_rows.empty:
        example_columns = [column for column in ["id", label_source] if column in invalid_rows.columns]
        sample_rows = invalid_rows[example_columns].head(20).to_dict(orient="records")
        report["invalid_examples"] = sample_rows
    return report


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    report_path = Path(args.report_path)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    emit(f"Loading {input_csv}")
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    label_source = detect_label_column(
        df,
        args.label_col,
        candidates=NORMALIZE_LABEL_CANDIDATES,
        treat_two_as_negative=False,
        source_name="CSV",
    )
    emit(f"Using label source column: {label_source}")

    normalized = df[label_source].map(lambda value: normalize_label_value(value, treat_two_as_negative=False))
    invalid_mask = normalized.isna()
    invalid_rows = df.loc[invalid_mask].copy()

    if invalid_mask.any():
        example_values = list(
            pd.Series(df.loc[invalid_mask, label_source].astype("string").unique()).dropna().head(10)
        )
        raise ValueError(
            f"Found {int(invalid_mask.sum())} rows with unrecognized labels in '{label_source}'. "
            f"Examples: {example_values}"
        )

    working = df.copy()
    working["label_raw"] = working[label_source]
    working["label"] = normalized.astype(int)
    working["label_text"] = working["label"].map({1: "相关", 0: "无关"})

    if "tangping_related_label" in working.columns:
        working["tangping_related_label_raw"] = working["tangping_related_label"]
    if "tangping_related" in working.columns:
        working["tangping_related_raw"] = working["tangping_related"]

    working["tangping_related_label"] = working["label"]
    working["tangping_related"] = working["label"]

    ensure_parent(output_csv)
    ensure_parent(report_path)

    report = build_report(working, label_source, invalid_rows)

    working.to_csv(output_csv, index=False, encoding="utf-8-sig")
    save_json(report_path, report)

    emit(f"Wrote normalized CSV to {output_csv}")
    emit(f"Wrote report to {report_path}")
    emit(
        "Counts after normalization: "
        + ", ".join(
            f"{label}={count}"
            for label, count in sorted(report["binary_counts"].items(), key=lambda item: item[0])
        )
    )


if __name__ == "__main__":
    main()
