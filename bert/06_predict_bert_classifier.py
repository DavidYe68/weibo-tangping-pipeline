#!/usr/bin/env python3
import argparse
import sys

from lib.prediction import BatchPredictionConfig, run_batch_prediction


def emit(message: str) -> None:
    print(f"[predict] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch prediction on parquet files with a fine-tuned BERT classifier."
    )
    parser.add_argument(
        "--model_dir",
        default="bert/artifacts/tangping_bert/best_model",
        help="Directory containing the fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--input_pattern",
        default="data/processed/text_dedup/*.parquet",
        help="Glob pattern for parquet files to classify.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/text_dedup_predicted",
        help="Directory for predicted parquet files.",
    )
    parser.add_argument("--text_col", default=None, help="Optional text column name.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for prediction.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum token length.")
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the positive class.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Prediction device.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load model/tokenizer from local files.",
    )
    parser.add_argument(
        "--only_positive",
        action="store_true",
        help="If set, only keep rows predicted as positive in the output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BatchPredictionConfig(
        model_dir=args.model_dir,
        input_pattern=args.input_pattern,
        output_dir=args.output_dir,
        text_col=args.text_col,
        batch_size=args.batch_size,
        max_length=args.max_length,
        positive_threshold=args.positive_threshold,
        device=args.device,
        local_files_only=bool(args.local_files_only),
        only_positive=bool(args.only_positive),
    )
    run_batch_prediction(config, emit=emit)


if __name__ == "__main__":
    main()
