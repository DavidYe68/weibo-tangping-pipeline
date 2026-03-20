#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


LABEL_CANDIDATES = ["tangping_related_label", "tangping_related", "label"]
POSITIVE_ALIASES = {"1", "1.0", "true", "yes", "y", "relevant", "positive", "相关", "有关"}
NEGATIVE_ALIASES = {"0", "0.0", "false", "no", "n", "irrelevant", "negative", "无关", "不相关"}


def emit(message: str) -> None:
    print(f"[normalize] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize tangping labels into binary 1/0 values for BERT training."
    )
    parser.add_argument(
        "--input_csv",
        default="data/bert/labeled.csv",
        help="Input labeled CSV path.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/bert/labeled_binary.csv",
        help="Output CSV path after label normalization.",
    )
    parser.add_argument(
        "--report_path",
        default="data/bert/labeled_binary_report.json",
        help="Normalization report JSON path.",
    )
    parser.add_argument(
        "--label_col",
        default=None,
        help="Optional source label column. If omitted, auto-detect from common candidates.",
    )
    return parser.parse_args()


def normalize_label_value(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if value in (0, 1):
            return int(value)

    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in POSITIVE_ALIASES:
        return 1
    if lowered in NEGATIVE_ALIASES:
        return 0
    return None


def detect_label_column(df: pd.DataFrame, forced: Optional[str]) -> str:
    if forced:
        if forced not in df.columns:
            raise ValueError(f"Label column '{forced}' not found in CSV.")
        return forced

    best_col = None
    best_valid = -1
    for column in LABEL_CANDIDATES:
        if column not in df.columns:
            continue
        normalized = df[column].map(normalize_label_value)
        valid_count = int(normalized.notna().sum())
        if valid_count > best_valid:
            best_col = column
            best_valid = valid_count

    if best_col is None or best_valid <= 0:
        raise ValueError(
            "Could not auto-detect a usable label column. "
            "Please pass --label_col explicitly."
        )
    return best_col


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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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

    label_source = detect_label_column(df, args.label_col)
    emit(f"Using label source column: {label_source}")

    normalized = df[label_source].map(normalize_label_value)
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

    df = df.copy()
    df["label_raw"] = df[label_source]
    df["label"] = normalized.astype(int)
    df["label_text"] = df["label"].map({1: "相关", 0: "无关"})

    if "tangping_related_label" in df.columns:
        df["tangping_related_label_raw"] = df["tangping_related_label"]
    if "tangping_related" in df.columns:
        df["tangping_related_raw"] = df["tangping_related"]

    df["tangping_related_label"] = df["label"]
    df["tangping_related"] = df["label"]

    ensure_parent(output_csv)
    ensure_parent(report_path)

    report = build_report(df, label_source, invalid_rows)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

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
