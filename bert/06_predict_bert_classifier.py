#!/usr/bin/env python3
import argparse
import sys

from lib.prediction import BatchPredictionConfig, run_batch_prediction


def emit(message: str) -> None:
    print(f"[predict] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "用已训练好的 BERT 分类器对 parquet 批量预测。"
            "如果后续要进入 07-10 broad 分析链，建议把 --output_dir 显式设为 "
            "data/processed/text_dedup_predicted_broad。"
        )
    )
    parser.add_argument(
        "--model_dir",
        default="bert/artifacts/tangping_bert/best_model",
        help="已训练模型和 tokenizer 所在目录；默认 bert/artifacts/tangping_bert/best_model。",
    )
    parser.add_argument(
        "--input_pattern",
        default="data/processed/text_dedup/*.parquet",
        help="待预测 parquet 的 glob 模式；默认 data/processed/text_dedup/*.parquet。",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/text_dedup_predicted",
        help="预测结果输出目录；默认 data/processed/text_dedup_predicted。",
    )
    parser.add_argument("--text_col", default=None, help="可选，显式指定文本列名。")
    parser.add_argument("--batch_size", type=int, default=64, help="预测 batch size；默认 64。")
    parser.add_argument("--max_length", type=int, default=128, help="最大 token 长度；默认 128。")
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=0.5,
        help="正类判定阈值；默认 0.5。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="预测设备；默认 auto。",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="只从本地加载模型和 tokenizer。",
    )
    parser.add_argument(
        "--only_positive",
        action="store_true",
        help="如果传入，只保留预测为正类的行。",
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
