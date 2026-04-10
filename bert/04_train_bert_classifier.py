#!/usr/bin/env python3
import argparse
import sys

from lib.training import TrainClassifierConfig, run_training


def emit(message: str) -> None:
    print(f"[train] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, and test a BERT-style classifier for tangping relevance."
    )
    parser.add_argument(
        "--input_csv",
        default="bert/data/labeled_binary.csv",
        help="Training CSV/XLSX path.",
    )
    parser.add_argument(
        "--output_dir",
        default="bert/artifacts/tangping_bert",
        help="Directory to save checkpoints, metrics, and predictions.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-chinese",
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


def main() -> None:
    args = parse_args()
    config = TrainClassifierConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        text_col=args.text_col,
        label_col=args.label_col,
        split_col=args.split_col,
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
