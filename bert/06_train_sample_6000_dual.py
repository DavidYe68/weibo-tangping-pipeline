#!/usr/bin/env python3
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def build_train_command(args: argparse.Namespace, label_col: str, output_dir: Path) -> List[str]:
    script_path = Path(__file__).with_name("04_train_bert_classifier.py")
    command = [
        sys.executable,
        str(script_path),
        "--input_csv",
        str(Path(args.input_path)),
        "--output_dir",
        str(output_dir),
        "--model_name_or_path",
        args.model_name_or_path,
        "--text_col",
        args.text_col,
        "--label_col",
        label_col,
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--val_size",
        str(args.val_size),
        "--test_size",
        str(args.test_size),
        "--positive_threshold",
        str(args.positive_threshold),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.sheet_name:
        command.extend(["--sheet_name", args.sheet_name])
    if args.local_files_only:
        command.append("--local_files_only")
    return command


def load_train_module() -> Any:
    script_path = Path(__file__).with_name("04_train_bert_classifier.py")
    spec = importlib.util.spec_from_file_location("train_bert_classifier", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_with_optional_stratify(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
    stratify_values: pd.Series | None,
) -> Dict[str, pd.DataFrame]:
    working = df.copy()
    key_column = "__dual_split_key"
    if stratify_values is not None:
        working[key_column] = stratify_values.astype(str)

    train_df, temp_df = train_test_split(
        working,
        test_size=val_size + test_size,
        stratify=working[key_column] if stratify_values is not None else None,
        random_state=seed,
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[key_column] if stratify_values is not None else None,
        random_state=seed,
    )

    frames = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    for name, frame in frames.items():
        if key_column in frame.columns:
            frames[name] = frame.drop(columns=[key_column])
    return frames


def create_shared_splits(df: pd.DataFrame, val_size: float, test_size: float, seed: int) -> tuple[Dict[str, pd.DataFrame], str]:
    strategies = [
        ("pair", df["broad_norm"].astype(str) + "_" + df["strict_norm"].astype(str)),
        ("broad", df["broad_norm"]),
        ("strict", df["strict_norm"]),
        ("random", None),
    ]

    last_error: Exception | None = None
    for strategy_name, stratify_values in strategies:
        try:
            frames = split_with_optional_stratify(df, val_size, test_size, seed, stratify_values)
            return frames, strategy_name
        except ValueError as exc:
            last_error = exc
            emit(f"Shared split fallback: strategy={strategy_name} unavailable ({exc})")

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to create shared splits.")


def prepare_shared_dataset(args: argparse.Namespace, base_output_dir: Path) -> tuple[Path, Dict[str, Any]]:
    train_module = load_train_module()
    input_path = Path(args.input_path)
    df = train_module.load_training_dataframe(input_path, args.sheet_name)
    if df.empty:
        raise ValueError("Input dataset is empty.")

    text_col = train_module.detect_text_column(df, args.text_col)
    for label_col in ("broad", "strict"):
        if label_col not in df.columns:
            raise ValueError(f"Required label column '{label_col}' not found in dataset.")

    working = df.copy()
    working["__dual_row_id"] = np.arange(len(working), dtype=np.int64)
    working[text_col] = working[text_col].fillna("").astype(str).str.strip()
    working["broad_norm"] = working["broad"].map(train_module.normalize_label_value)
    working["strict_norm"] = working["strict"].map(train_module.normalize_label_value)

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
    manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


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
        command = build_train_command(args, label_col, output_dir)
        command[command.index("--input_csv") + 1] = str(shared_input_path)
        command.extend(["--split_col", "__dual_split"])
        emit(f"Training {label_col} model -> {output_dir}")
        subprocess.run(command, check=True)
        metrics_path = output_dir / "metrics.json"
        summary["runs"][label_col] = {
            "output_dir": str(output_dir.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "metrics": load_metrics(metrics_path),
        }

    combined_report_paths = save_combined_reports(base_output_dir, shared_split_manifest["text_col"])
    summary["combined_reports"] = combined_report_paths
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(f"Dual-training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
