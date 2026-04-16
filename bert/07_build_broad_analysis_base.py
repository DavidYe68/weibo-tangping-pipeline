#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from lib.analysis_utils import (
    DEFAULT_ANALYSIS_KEYWORDS,
    load_tabular_files,
    prepare_analysis_frame,
    resolve_emit,
    save_dataframe,
)
from lib.io_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized broad-analysis base table from prediction outputs."
    )
    parser.add_argument(
        "--input_pattern",
        default="data/processed/text_dedup_predicted_broad/*.parquet",
        help="Glob pattern for broad prediction outputs. CSV and parquet are supported.",
    )
    parser.add_argument(
        "--output_path",
        default="bert/artifacts/broad_analysis/analysis_base.parquet",
        help="Path for the normalized analysis base table.",
    )
    parser.add_argument(
        "--report_path",
        default="bert/artifacts/broad_analysis/analysis_base_report.json",
        help="Path for the build summary report.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=list(DEFAULT_ANALYSIS_KEYWORDS),
        help="Canonical keywords to keep. Defaults to 躺平 摆烂 佛系.",
    )
    parser.add_argument(
        "--text_col",
        default=None,
        help=(
            "Optional forced text column name. By default it follows the repo text-column "
            "auto-detection order, which usually resolves to cleaned_text."
        ),
    )
    parser.add_argument("--time_col", default=None, help="Optional forced time column name.")
    parser.add_argument("--keyword_col", default=None, help="Optional forced keyword column name.")
    parser.add_argument("--ip_col", default=None, help="Optional forced IP column name.")
    parser.add_argument(
        "--prediction_label_col",
        default="pred_label",
        help="Prediction label column used to keep positive predictions.",
    )
    parser.add_argument(
        "--include_negative",
        action="store_true",
        help="Keep negative predictions too. Default behavior keeps only positive rows.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=None,
        help="Optional lower bound on pred_prob_1 or pred_confidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emit = resolve_emit("analysis-base", None)

    df, files = load_tabular_files(args.input_pattern, emit=emit)
    analysis_df, metadata = prepare_analysis_frame(
        df,
        text_col=args.text_col,
        time_col=args.time_col,
        keyword_col=args.keyword_col,
        ip_col=args.ip_col,
        keywords=args.keywords,
        positive_label_col=args.prediction_label_col,
        positive_only=not args.include_negative,
        min_confidence=args.min_confidence,
    )

    output_path = Path(args.output_path)
    report_path = Path(args.report_path)
    save_dataframe(analysis_df, output_path)

    report = {
        "input_pattern": args.input_pattern,
        "input_files": [str(Path(path).resolve()) for path in files],
        "output_path": str(output_path.resolve()),
        "selected_keywords": metadata["selected_keywords"],
        "rows_by_keyword": {
            str(keyword): int(count)
            for keyword, count in analysis_df["keyword_normalized"].value_counts(dropna=False).sort_index().to_dict().items()
        },
        "rows_by_period": {
            str(period): int(count)
            for period, count in analysis_df["year_month"].fillna("NA").value_counts().sort_index().to_dict().items()
        },
        "rows_by_ip": {
            str(ip_value): int(count)
            for ip_value, count in analysis_df["ip_normalized"].value_counts(dropna=False).sort_index().to_dict().items()
        },
        "rows_by_period_and_ip": [
            {
                "year_month": str(period),
                "ip_normalized": str(ip_value),
                "row_count": int(count),
            }
            for (period, ip_value), count in analysis_df.groupby(["year_month", "ip_normalized"], dropna=False).size().sort_index().items()
        ],
        "metadata": metadata,
    }
    save_json(report_path, report)

    emit(f"Saved analysis base to {output_path}")
    emit(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
