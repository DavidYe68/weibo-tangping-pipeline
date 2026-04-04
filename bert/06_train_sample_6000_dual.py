#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def emit(message: str) -> None:
    print(f"[dual-train] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train separate BERT classifiers for broad and strict labels from sample_6000_labeled.xlsx."
    )
    parser.add_argument(
        "--input_path",
        default="data/bert/sample_6000_labeled.xlsx",
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

    summary: Dict[str, Any] = {
        "input_path": str(input_path.resolve()),
        "base_output_dir": str(base_output_dir.resolve()),
        "runs": {},
    }

    for label_col in ("broad", "strict"):
        output_dir = base_output_dir / label_col
        command = build_train_command(args, label_col, output_dir)
        emit(f"Training {label_col} model -> {output_dir}")
        subprocess.run(command, check=True)
        metrics_path = output_dir / "metrics.json"
        summary["runs"][label_col] = {
            "output_dir": str(output_dir.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "metrics": load_metrics(metrics_path),
        }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(f"Dual-training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
