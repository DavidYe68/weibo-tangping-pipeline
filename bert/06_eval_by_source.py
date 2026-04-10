#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from lib.data_utils import detect_text_column, load_training_dataframe
from lib.io_utils import save_json
from lib.labels import normalize_label_value
from lib.reporting import build_metrics_snapshot, write_dual_run_inspect_artifacts, write_eval_collection_inspect_artifacts
from lib.training import TrainClassifierConfig, run_training


def emit(message: str) -> None:
    print(f"[eval-by-source] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate broad/strict models by held-out source while keeping manual foxi data train-only."
    )
    parser.add_argument("--sample1_path", default="bert/data/sample_1.xlsx", help="Path to sample_1 labels.")
    parser.add_argument("--sample2_path", default="bert/data/sample_2.xlsx", help="Path to sample_2 labels.")
    parser.add_argument("--foxi_path", default="bert/data/sample_foxi.xlsx", help="Path to real foxi labels.")
    parser.add_argument(
        "--manual_path",
        default="bert/data/sample_foxi_manuel_added.xlsx",
        help="Path to manual foxi augmentation labels.",
    )
    parser.add_argument(
        "--base_output_dir",
        default="bert/artifacts/eval_by_source",
        help="Base directory for source-held-out evaluation outputs.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-chinese",
        help="HF model name or local path for the base encoder.",
    )
    parser.add_argument("--text_col", default="cleaned_text", help="Preferred text column name.")
    parser.add_argument("--sheet_name", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs for each label standard.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation split ratio taken from the real-data training pool.",
    )
    parser.add_argument(
        "--foxi_test_size",
        type=float,
        default=0.2,
        help="Held-out test ratio for the real foxi dataset.",
    )
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
    parser.add_argument(
        "--skip_manual",
        action="store_true",
        help="Ignore manual foxi augmentation and evaluate using only real-data pools.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["sample_1_test", "sample_2_test", "foxi_test"],
        default=["sample_1_test", "sample_2_test", "foxi_test"],
        help="Subset of source-held-out experiments to run.",
    )
    return parser.parse_args()


def load_labeled_source(path: Path, source_name: str, text_col_hint: str, sheet_name: Optional[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = load_training_dataframe(path, sheet_name)
    if df.empty:
        raise ValueError(f"Input dataset is empty: {path}")

    resolved_text_col = detect_text_column(df, text_col_hint, source_name=path.name)
    for label_col in ("broad", "strict"):
        if label_col not in df.columns:
            raise ValueError(f"Required label column '{label_col}' not found in {path}")

    working = df.copy()
    working["__eval_text"] = working[resolved_text_col].fillna("").astype(str).str.strip()
    working["broad_norm"] = working["broad"].map(lambda value: normalize_label_value(value, treat_two_as_negative=True))
    working["strict_norm"] = working["strict"].map(lambda value: normalize_label_value(value, treat_two_as_negative=True))
    working["__source_name"] = source_name
    working["__source_path"] = str(path.resolve())

    valid_mask = (
        (working["__eval_text"] != "")
        & working["broad_norm"].notna()
        & working["strict_norm"].notna()
    )
    usable = working.loc[valid_mask].copy().reset_index(drop=True)
    if usable.empty:
        raise ValueError(f"No usable rows with valid text/broad/strict labels in {path}")
    return usable


def split_with_fallback(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if test_size <= 0 or test_size >= 1:
        raise ValueError(f"{label}: split ratio must be between 0 and 1.")

    strategies = [
        ("pair", df["broad_norm"].astype(int).astype(str) + "_" + df["strict_norm"].astype(int).astype(str)),
        ("broad", df["broad_norm"].astype(int)),
        ("strict", df["strict_norm"].astype(int)),
        ("random", None),
    ]

    last_error: Exception | None = None
    for strategy_name, stratify_values in strategies:
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                stratify=stratify_values if stratify_values is not None else None,
                random_state=seed,
            )
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True), strategy_name
        except ValueError as exc:
            last_error = exc
            emit(f"{label}: strategy={strategy_name} unavailable ({exc})")

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{label}: failed to create split.")


def build_train_config(args: argparse.Namespace, label_col: str, input_path: Path, output_dir: Path) -> TrainClassifierConfig:
    return TrainClassifierConfig(
        input_csv=str(input_path),
        output_dir=str(output_dir),
        model_name_or_path=args.model_name_or_path,
        text_col="__eval_text",
        label_col=label_col,
        split_col="__eval_split",
        sheet_name=args.sheet_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        val_size=args.val_size,
        test_size=args.foxi_test_size,
        positive_threshold=args.positive_threshold,
        seed=args.seed,
        device=args.device,
        local_files_only=bool(args.local_files_only),
    )


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
    row_key = "__eval_row_id"
    if row_key not in broad_df.columns or row_key not in strict_df.columns:
        raise ValueError("Combined comparison requires '__eval_row_id' in both prediction files.")

    base_columns = [column for column in [row_key, "__eval_split", "__source_name", "id", text_col] if column in broad_df.columns]
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


def save_combined_reports(base_output_dir: Path) -> Dict[str, str]:
    broad_predictions = pd.read_csv(base_output_dir / "broad" / "test_predictions.csv")
    strict_predictions = pd.read_csv(base_output_dir / "strict" / "test_predictions.csv")

    combined_predictions = pd.concat([broad_predictions, strict_predictions], ignore_index=True)
    combined_predictions_path = base_output_dir / "test_predictions_combined.csv"
    combined_predictions.to_csv(combined_predictions_path, index=False, encoding="utf-8-sig")

    combined_errors = combined_predictions[coerce_bool_series(combined_predictions["is_error"])].copy()
    combined_errors_path = base_output_dir / "test_misclassified_combined.csv"
    combined_errors.to_csv(combined_errors_path, index=False, encoding="utf-8-sig")

    side_by_side = build_side_by_side_predictions(broad_predictions, strict_predictions, "__eval_text")
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


def prepare_experiment_dataset(
    experiment_name: str,
    train_real_pool: pd.DataFrame,
    test_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    args: argparse.Namespace,
    base_output_dir: Path,
) -> tuple[Path, Dict[str, Any]]:
    train_real_df, val_real_df, val_strategy = split_with_fallback(
        train_real_pool,
        args.val_size,
        args.seed,
        label=f"{experiment_name}/val",
    )

    parts: List[pd.DataFrame] = []
    split_sizes: Dict[str, int] = {}

    for split_name, frame in (("train", train_real_df), ("val", val_real_df), ("test", test_df)):
        tagged = frame.copy()
        tagged["__eval_split"] = split_name
        parts.append(tagged)
        split_sizes[split_name] = int(len(tagged))

    manual_rows = 0
    if not manual_df.empty:
        manual_tagged = manual_df.copy()
        manual_tagged["__eval_split"] = "train"
        parts.append(manual_tagged)
        manual_rows = int(len(manual_tagged))
        split_sizes["train"] += manual_rows

    combined = pd.concat(parts, ignore_index=True)
    combined["__eval_row_id"] = range(len(combined))
    combined = combined.drop(columns=["broad_norm", "strict_norm"])

    output_dir = base_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "shared_split_dataset.csv"
    combined.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    manifest = {
        "experiment_name": experiment_name,
        "dataset_path": str(dataset_path.resolve()),
        "split_col": "__eval_split",
        "text_col": "__eval_text",
        "val_strategy": val_strategy,
        "real_train_pool_rows": int(len(train_real_pool)),
        "manual_train_only_rows": manual_rows,
        "test_rows": int(len(test_df)),
        "split_sizes": split_sizes,
        "source_breakdown": {
            split_name: {
                source: int(count)
                for source, count in combined.loc[combined["__eval_split"] == split_name, "__source_name"]
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            }
            for split_name in ("train", "val", "test")
        },
    }
    save_json(output_dir / "shared_split_manifest.json", manifest)
    return dataset_path, manifest


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    sample1_df = load_labeled_source(Path(args.sample1_path), "sample_1", args.text_col, args.sheet_name)
    sample2_df = load_labeled_source(Path(args.sample2_path), "sample_2", args.text_col, args.sheet_name)
    foxi_df = load_labeled_source(Path(args.foxi_path), "foxi_real", args.text_col, args.sheet_name)
    manual_df = pd.DataFrame()
    if not args.skip_manual:
        manual_path = Path(args.manual_path)
        if manual_path.exists():
            manual_df = load_labeled_source(manual_path, "foxi_manual", args.text_col, args.sheet_name)
        else:
            emit(f"Manual augmentation file not found, continuing without it: {manual_path}")

    foxi_train_df, foxi_test_df, foxi_strategy = split_with_fallback(
        foxi_df,
        args.foxi_test_size,
        args.seed,
        label="foxi_real/test",
    )

    all_experiments = [
        ("sample_1_test", pd.concat([sample2_df, foxi_train_df], ignore_index=True), sample1_df),
        ("sample_2_test", pd.concat([sample1_df, foxi_train_df], ignore_index=True), sample2_df),
        ("foxi_test", pd.concat([sample1_df, sample2_df, foxi_train_df], ignore_index=True), foxi_test_df),
    ]
    selected_experiments = set(args.experiments)
    experiments = [item for item in all_experiments if item[0] in selected_experiments]
    if not experiments:
        raise ValueError("No experiments selected.")

    overall_summary: Dict[str, Any] = {
        "base_output_dir": str(base_output_dir.resolve()),
        "foxi_real_split": {
            "strategy": foxi_strategy,
            "train_rows": int(len(foxi_train_df)),
            "test_rows": int(len(foxi_test_df)),
        },
        "manual_policy": "train_only" if not args.skip_manual else "ignored",
        "selected_experiments": [name for name, _, _ in experiments],
        "experiments": {},
    }

    for experiment_name, train_real_pool, test_df in experiments:
        emit(f"Preparing experiment {experiment_name}")
        dataset_path, manifest = prepare_experiment_dataset(
            experiment_name,
            train_real_pool,
            test_df,
            manual_df,
            args,
            base_output_dir,
        )
        experiment_dir = base_output_dir / experiment_name
        experiment_summary: Dict[str, Any] = {
            "dataset_manifest": manifest,
            "runs": {},
        }

        for label_col in ("broad", "strict"):
            output_dir = experiment_dir / label_col
            emit(f"Training {experiment_name}/{label_col} -> {output_dir}")
            result = run_training(
                build_train_config(args, label_col, dataset_path, output_dir),
                emit=lambda message, prefix=f"{experiment_name}/{label_col}": emit(f"[{prefix}] {message}"),
            )
            experiment_summary["runs"][label_col] = {
                "output_dir": str(output_dir.resolve()),
                "metrics_path": result["metrics_path"],
                "metrics_snapshot": build_metrics_snapshot(result["metrics"]),
                "best_model_dir": result["best_model_dir"],
                "test_predictions_path": result["test_predictions_path"],
                "test_misclassified_path": result["test_misclassified_path"],
            }

        experiment_summary["combined_reports"] = save_combined_reports(experiment_dir)
        experiment_summary["inspect_reports"] = write_dual_run_inspect_artifacts(
            experiment_dir,
            experiment_name=experiment_name,
            text_col="__eval_text",
            source_col="__source_name",
        )
        save_json(experiment_dir / "summary.json", experiment_summary)
        overall_summary["experiments"][experiment_name] = experiment_summary

    overall_summary["inspect_reports"] = write_eval_collection_inspect_artifacts(base_output_dir, overall_summary)
    save_json(base_output_dir / "summary.json", overall_summary)
    emit(f"Source-held-out evaluation summary saved to {base_output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
