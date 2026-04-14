#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from lib.collection_utils import load_text_collection_frame
from lib.data_utils import drop_optional_training_metadata
from lib.io_utils import save_json
from lib.labels import detect_label_column, normalize_label_value


def emit(message: str) -> None:
    print(f"[train] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, and test a BERT-style classifier for tangping relevance."
    )
    parser.add_argument(
        "--input_csv",
        nargs="*",
        default=None,
        help="One or more CSV/XLSX files to merge before training. If omitted, falls back to bert/data/labeled_binary.csv when available.",
    )
    parser.add_argument(
        "--train_csv",
        nargs="*",
        default=None,
        help="CSV/XLSX files forced into the training split.",
    )
    parser.add_argument(
        "--train_only_csv",
        nargs="*",
        default=None,
        help="Alias of --train_csv for train-only data.",
    )
    parser.add_argument(
        "--val_csv",
        nargs="*",
        default=None,
        help="CSV/XLSX files forced into the validation split.",
    )
    parser.add_argument(
        "--test_csv",
        nargs="*",
        default=None,
        help="CSV/XLSX files forced into the test split.",
    )
    parser.add_argument(
        "--output_dir",
        default="bert/artifacts/tangping_bert",
        help="Directory to save checkpoints, metrics, and predictions.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="hfl/chinese-roberta-wwm-ext",
        help="HF model name or local path for the base encoder.",
    )
    parser.add_argument("--text_col", default=None, help="Optional text column name.")
    parser.add_argument("--label_col", default=None, help="Optional label column name.")
    parser.add_argument(
        "--split_col",
        default=None,
        help="Optional split column with values train/val/test. If provided, skips random splitting.",
    )
    parser.add_argument(
        "--sheet_name",
        default=None,
        help="Optional Excel sheet name. Uses the first sheet if omitted.",
    )
    parser.add_argument("--max_length", type=int, default=128, help="Maximum token length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
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


def resolve_path_group(values: List[str] | None) -> List[Path]:
    return [Path(value) for value in (values or []) if str(value).strip()]


def normalize_single_label_frame(args: argparse.Namespace, path: Path) -> pd.DataFrame:
    working = load_text_collection_frame(
        path,
        sheet_name=args.sheet_name,
        text_col_hint=args.text_col,
    )
    working = drop_optional_training_metadata(working)
    label_source = detect_label_column(
        working,
        args.label_col,
        source_name=path.name,
        treat_two_as_negative=True,
    )
    working["label"] = working[label_source].map(
        lambda value: normalize_label_value(value, treat_two_as_negative=True)
    )
    usable = working[(working["__text"] != "") & working["label"].notna()].copy().reset_index(drop=True)
    if usable.empty:
        raise ValueError(f"No usable rows with valid text/label values in {path}")
    usable["label"] = usable["label"].astype(int)
    usable["__resolved_label_col"] = label_source
    return usable


def concat_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def resolve_input_groups(args: argparse.Namespace) -> Dict[str, List[Path]]:
    groups = {
        "pool": resolve_path_group(args.input_csv),
        "train": resolve_path_group(args.train_csv) + resolve_path_group(args.train_only_csv),
        "val": resolve_path_group(args.val_csv),
        "test": resolve_path_group(args.test_csv),
    }
    if any(groups.values()):
        return groups

    default_path = Path("bert/data/labeled_binary.csv")
    if default_path.exists():
        groups["pool"] = [default_path]
        return groups

    raise ValueError("Please provide at least one labeled CSV/XLSX via --input_csv/--train_csv/--val_csv/--test_csv.")


def prepare_collection_dataset(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Dict[str, Any], str | None]:
    from lib.splits import create_data_splits

    groups = resolve_input_groups(args)
    prepared_frames = {
        split_name: [normalize_single_label_frame(args, path) for path in paths]
        for split_name, paths in groups.items()
    }
    using_explicit_splits = any(prepared_frames[name] for name in ("train", "val", "test"))

    pool_df = concat_frames(prepared_frames["pool"])
    split_strategy: str | None = None
    if using_explicit_splits and not pool_df.empty:
        train_df, val_df, test_df = create_data_splits(
            df=pool_df,
            label_col="label",
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        split_strategy = "pool_random_split"
    else:
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()

    split_parts: List[pd.DataFrame] = []
    split_sizes: Dict[str, int] = {}
    source_breakdown: Dict[str, Dict[str, int]] = {}

    for split_name, base_frame in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        explicit_frame = concat_frames(prepared_frames[split_name])
        merged = concat_frames([frame for frame in [base_frame, explicit_frame] if not frame.empty])
        if merged.empty:
            continue
        tagged = merged.copy()
        tagged["__prepared_split"] = split_name
        split_parts.append(tagged)
        split_sizes[split_name] = int(len(tagged))
        source_breakdown[split_name] = {
            str(source): int(count)
            for source, count in tagged["__source_name"].value_counts().sort_index().to_dict().items()
        }

    if split_parts:
        combined = pd.concat(split_parts, ignore_index=True)
        combined["label"] = combined["label"].astype(int)
        combined["__prepared_row_id"] = range(len(combined))
        dataset_path = output_dir / "prepared_split_dataset.csv"
        combined.to_csv(dataset_path, index=False, encoding="utf-8-sig")
        manifest = {
            "mode": "explicit_split_collection",
            "dataset_path": str(dataset_path.resolve()),
            "text_col": "__text",
            "label_col": "label",
            "split_col": "__prepared_split",
            "split_strategy": split_strategy or "predefined_only",
            "split_sizes": split_sizes,
            "source_breakdown": source_breakdown,
            "input_groups": {name: [str(path.resolve()) for path in paths] for name, paths in groups.items()},
        }
        save_json(output_dir / "prepared_split_manifest.json", manifest)
        return dataset_path, manifest, "__prepared_split"

    combined = concat_frames(prepared_frames["pool"])
    if combined.empty:
        raise ValueError("No usable rows were found in the provided input files.")

    combined["label"] = combined["label"].astype(int)
    combined["__prepared_row_id"] = range(len(combined))
    dataset_path = output_dir / "prepared_dataset.csv"
    combined.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    manifest = {
        "mode": "pooled_random_split",
        "dataset_path": str(dataset_path.resolve()),
        "text_col": "__text",
        "label_col": "label",
        "split_col": None,
        "rows": int(len(combined)),
        "source_breakdown": {
            str(source): int(count)
            for source, count in combined["__source_name"].value_counts().sort_index().to_dict().items()
        },
        "input_groups": {name: [str(path.resolve()) for path in paths] for name, paths in groups.items()},
    }
    save_json(output_dir / "prepared_manifest.json", manifest)
    return dataset_path, manifest, None


def main() -> None:
    args = parse_args()
    from lib.training import TrainClassifierConfig, run_training

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_input_path, _, split_col = prepare_collection_dataset(args, output_dir)
    config = TrainClassifierConfig(
        input_csv=str(prepared_input_path),
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        text_col="__text",
        label_col="label",
        split_col=split_col or args.split_col,
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
    run_training(config, emit=emit)


if __name__ == "__main__":
    main()
