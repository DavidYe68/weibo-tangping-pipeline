#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from lib.collection_utils import drop_rows_overlapping_with_reference, load_text_collection_frame
from lib.data_utils import drop_optional_training_metadata
from lib.io_utils import save_json
from lib.labels import detect_label_column, normalize_label_value


def emit(message: str) -> None:
    print(f"[train] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "训练单标签 BERT 分类器。"
            "可直接读取一个或多个审核后的 CSV/XLSX，也可把部分文件固定到 train/val/test split。"
        )
    )
    parser.add_argument(
        "--input_csv",
        nargs="*",
        default=None,
        help=(
            "一个或多个待合并并随机切分的 CSV/XLSX。"
            "如果未提供，且 bert/data/labeled_binary.csv 存在，则回退到该默认文件。"
        ),
    )
    parser.add_argument(
        "--train_csv",
        nargs="*",
        default=None,
        help="强制放入训练集的 CSV/XLSX 文件。",
    )
    parser.add_argument(
        "--train_only_csv",
        nargs="*",
        default=None,
        help="--train_csv 的别名。",
    )
    parser.add_argument(
        "--val_csv",
        nargs="*",
        default=None,
        help="强制放入验证集的 CSV/XLSX 文件。",
    )
    parser.add_argument(
        "--test_csv",
        nargs="*",
        default=None,
        help="强制放入测试集的 CSV/XLSX 文件。",
    )
    parser.add_argument(
        "--output_dir",
        default="bert/artifacts/tangping_bert",
        help="保存 checkpoints、metrics 和 predictions 的目录；默认 bert/artifacts/tangping_bert。",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="hfl/chinese-roberta-wwm-ext",
        help="基础编码器的 Hugging Face 模型名或本地路径。",
    )
    parser.add_argument("--text_col", default=None, help="可选，显式指定文本列名。")
    parser.add_argument("--label_col", default=None, help="可选，显式指定标签列名。")
    parser.add_argument(
        "--split_col",
        default=None,
        help="可选，显式指定 split 列；取值应为 train/val/test，传入后跳过随机切分。",
    )
    parser.add_argument(
        "--sheet_name",
        default=None,
        help="可选 Excel sheet 名；不传时读取第一张表。",
    )
    parser.add_argument("--max_length", type=int, default=128, help="最大 token 长度；默认 128。")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size；默认 16。")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数；默认 3。")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率；默认 2e-5。")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay；默认 0.01。")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup 比例；默认 0.1。")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值；默认 1.0。")
    parser.add_argument("--val_size", type=float, default=0.1, help="随机切分时的验证集比例；默认 0.1。")
    parser.add_argument("--test_size", type=float, default=0.1, help="随机切分时的测试集比例；默认 0.1。")
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=0.5,
        help="正类判定阈值；默认 0.5。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子；默认 42。")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="训练设备；默认 auto。",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="只从本地加载模型和 tokenizer。",
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
    explicit_df = concat_frames(
        [
            frame
            for split_name in ("train", "val", "test")
            for frame in prepared_frames[split_name]
        ]
    )
    pool_overlap_removed = 0
    overlap_signature_columns: List[str] = []
    if using_explicit_splits and not pool_df.empty and not explicit_df.empty:
        pool_df, pool_overlap_removed, overlap_signature_columns = drop_rows_overlapping_with_reference(
            pool_df,
            explicit_df,
        )
        if pool_overlap_removed:
            emit(
                "Removed "
                f"{pool_overlap_removed} pooled rows that overlap with explicit train/val/test inputs "
                f"using signature columns={overlap_signature_columns}"
            )
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
            "pool_overlap_removed": pool_overlap_removed,
            "overlap_signature_columns": overlap_signature_columns,
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
