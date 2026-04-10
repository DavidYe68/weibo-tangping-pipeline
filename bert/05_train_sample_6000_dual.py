#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from lib.data_utils import detect_text_column, load_training_dataframe
from lib.io_utils import save_json
from lib.labels import normalize_label_value
from lib.reporting import build_metrics_snapshot, write_dual_run_inspect_artifacts
from lib.splits import create_shared_splits
from lib.training import TrainClassifierConfig, run_training


def emit(message: str) -> None:
    print(f"[dual-train] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train separate BERT classifiers for broad and strict labels from sample_6000_labeled.xlsx."
    )
    parser.add_argument(
        "--input_path",
        default="bert/data/sample_6000_labeled.xlsx",
        help="Excel dataset path.",
    )
    parser.add_argument(
        "--base_output_dir",
        default="bert/artifacts/sample_6000",
        help="Base directory for broad/strict model outputs.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-chinese",
        help="HF model name or local path for the base encoder.",
    )
    parser.add_argument("--text_col", default="cleaned_text", help="Text column name.")
    parser.add_argument("--sheet_name", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs for each label standard.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the positive class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load model/tokenizer from local files.",
    )
    return parser.parse_args()


def build_train_config(args: argparse.Namespace, label_col: str, input_path: Path, output_dir: Path) -> TrainClassifierConfig:
    return TrainClassifierConfig(
        input_csv=str(input_path),
        output_dir=str(output_dir),
        model_name_or_path=args.model_name_or_path,
        text_col=args.text_col,
        label_col=label_col,
        split_col="__dual_split",
        sheet_name=args.sheet_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        val_size=args.val_size,
        test_size=args.test_size,
        positive_threshold=args.positive_threshold,
        seed=args.seed,
        device=args.device,
        local_files_only=bool(args.local_files_only),
    )


def prepare_shared_dataset(args: argparse.Namespace, base_output_dir: Path) -> tuple[Path, Dict[str, Any]]:
    input_path = Path(args.input_path)
    df = load_training_dataframe(input_path, args.sheet_name)
    if df.empty:
        raise ValueError("Input dataset is empty.")

    text_col = detect_text_column(df, args.text_col, source_name="dataset")
    for label_col in ("broad", "strict"):
        if label_col not in df.columns:
            raise ValueError(f"Required label column '{label_col}' not found in dataset.")

    working = df.copy()
    working["__dual_row_id"] = np.arange(len(working), dtype=np.int64)
    working[text_col] = working[text_col].fillna("").astype(str).str.strip()
    working["broad_norm"] = working["broad"].map(lambda value: normalize_label_value(value, treat_two_as_negative=True))
    working["strict_norm"] = working["strict"].map(lambda value: normalize_label_value(value, treat_two_as_negative=True))

    valid_mask = (
        (working[text_col] != "")
        & working["broad_norm"].notna()
        & working["strict_norm"].notna()
    )
    usable_df = working.loc[valid_mask].copy().reset_index(drop=True)
    if len(usable_df) < 20:
        raise ValueError("Too few rows with valid broad/strict labels after cleaning.")

    split_frames, split_strategy = create_shared_splits(
        df=usable_df,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        emit=emit,
    )

    split_parts: List[pd.DataFrame] = []
    split_sizes: Dict[str, int] = {}
    for split_name, split_df in split_frames.items():
        tagged = split_df.copy()
        tagged["__dual_split"] = split_name
        split_parts.append(tagged)
        split_sizes[split_name] = int(len(tagged))

    shared_df = pd.concat(split_parts, ignore_index=True)
    shared_df = shared_df.drop(columns=["broad_norm", "strict_norm"])
    shared_input_path = base_output_dir / "shared_split_dataset.csv"
    shared_input_path.parent.mkdir(parents=True, exist_ok=True)
    shared_df.to_csv(shared_input_path, index=False, encoding="utf-8-sig")

    split_manifest = {
        "shared_input_path": str(shared_input_path.resolve()),
        "text_col": text_col,
        "split_col": "__dual_split",
        "split_strategy": split_strategy,
        "total_rows": int(len(df)),
        "usable_rows": int(len(usable_df)),
        "dropped_rows": int(len(df) - len(usable_df)),
        "split_sizes": split_sizes,
    }
    manifest_path = base_output_dir / "shared_split_manifest.json"
    save_json(manifest_path, split_manifest)
    return shared_input_path, split_manifest


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    return pd.read_csv(predictions_path)


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").fillna("false").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def build_side_by_side_predictions(
    broad_df: pd.DataFrame,
    strict_df: pd.DataFrame,
    text_col: str,
) -> pd.DataFrame:
    row_key = "__dual_row_id"
    if row_key not in broad_df.columns or row_key not in strict_df.columns:
        raise ValueError("Combined comparison requires '__dual_row_id' in both prediction files.")

    base_columns = [column for column in [row_key, "__dual_split", "id", text_col] if column in broad_df.columns]
    base_df = broad_df[base_columns].copy()

    compare_columns = [
        "gold_label",
        "gold_label_text",
        "pred_label",
        "pred_label_text",
        "pred_prob_1",
        "pred_prob_0",
        "pred_confidence",
        "is_error",
        "error_type",
    ]
    broad_view = broad_df[[row_key] + compare_columns].rename(
        columns={column: f"broad_{column}" for column in compare_columns}
    )
    strict_view = strict_df[[row_key] + compare_columns].rename(
        columns={column: f"strict_{column}" for column in compare_columns}
    )

    merged = base_df.merge(broad_view, on=row_key, how="inner").merge(strict_view, on=row_key, how="inner")
    merged["has_error"] = coerce_bool_series(merged["broad_is_error"]) | coerce_bool_series(merged["strict_is_error"])
    return merged


def save_combined_reports(base_output_dir: Path, text_col: str) -> Dict[str, str]:
    broad_predictions = load_predictions(base_output_dir / "broad" / "test_predictions.csv")
    strict_predictions = load_predictions(base_output_dir / "strict" / "test_predictions.csv")

    combined_predictions = pd.concat([broad_predictions, strict_predictions], ignore_index=True)
    combined_predictions_path = base_output_dir / "test_predictions_combined.csv"
    combined_predictions.to_csv(combined_predictions_path, index=False, encoding="utf-8-sig")

    combined_errors = combined_predictions[coerce_bool_series(combined_predictions["is_error"])].copy()
    combined_errors_path = base_output_dir / "test_misclassified_combined.csv"
    combined_errors.to_csv(combined_errors_path, index=False, encoding="utf-8-sig")

    side_by_side = build_side_by_side_predictions(broad_predictions, strict_predictions, text_col)
    side_by_side_path = base_output_dir / "test_predictions_side_by_side.csv"
    side_by_side.to_csv(side_by_side_path, index=False, encoding="utf-8-sig")

    side_by_side_errors = side_by_side[coerce_bool_series(side_by_side["has_error"])].copy()
    side_by_side_errors_path = base_output_dir / "test_misclassified_side_by_side.csv"
    side_by_side_errors.to_csv(side_by_side_errors_path, index=False, encoding="utf-8-sig")

    return {
        "combined_predictions_path": str(combined_predictions_path.resolve()),
        "combined_misclassified_path": str(combined_errors_path.resolve()),
        "side_by_side_predictions_path": str(side_by_side_path.resolve()),
        "side_by_side_misclassified_path": str(side_by_side_errors_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    base_output_dir = Path(args.base_output_dir)
    summary_path = base_output_dir / "summary.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_path}")

    base_output_dir.mkdir(parents=True, exist_ok=True)
    shared_input_path, shared_split_manifest = prepare_shared_dataset(args, base_output_dir)

    summary: Dict[str, Any] = {
        "input_path": str(input_path.resolve()),
        "base_output_dir": str(base_output_dir.resolve()),
        "shared_split": shared_split_manifest,
        "runs": {},
    }

    for label_col in ("broad", "strict"):
        output_dir = base_output_dir / label_col
        emit(f"Training {label_col} model -> {output_dir}")
        result = run_training(
            build_train_config(args, label_col, shared_input_path, output_dir),
            emit=lambda message, prefix=label_col: emit(f"[{prefix}] {message}"),
        )
        summary["runs"][label_col] = {
            "output_dir": str(output_dir.resolve()),
            "metrics_path": result["metrics_path"],
            "metrics_snapshot": build_metrics_snapshot(result["metrics"]),
            "best_model_dir": result["best_model_dir"],
            "test_predictions_path": result["test_predictions_path"],
            "test_misclassified_path": result["test_misclassified_path"],
        }

    combined_report_paths = save_combined_reports(base_output_dir, shared_split_manifest["text_col"])
    summary["combined_reports"] = combined_report_paths
    summary["inspect_reports"] = write_dual_run_inspect_artifacts(
        base_output_dir,
        experiment_name=base_output_dir.name,
        text_col=shared_split_manifest["text_col"],
    )
    save_json(summary_path, summary)
    emit(f"Dual-training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
