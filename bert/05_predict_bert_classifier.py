#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TEXT_CANDIDATES = [
    "cleaned_text",
    "cleaned_text_with_emoji",
    "text_raw",
    "微博正文",
    "text",
    "content",
    "body",
    "message",
    "post_text",
    "desc",
    "description",
    "title",
]
NON_TEXT_FALLBACK_EXCLUDE = {
    "发布时间",
    "created_at",
    "publish_time",
    "timestamp",
    "keyword",
    "hit_keyword",
    "query_keyword",
    "id",
    "mid",
    "话题",
    "转发数",
    "评论数",
    "点赞数",
    "ip",
    "source_file",
}


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


def resolve_input_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files and pattern.startswith("/"):
        files = sorted(glob.glob(pattern.lstrip("/")))
    if not files:
        raise FileNotFoundError(f"No parquet files matched input pattern: {pattern}")
    return files


def detect_text_column(df: pd.DataFrame, forced: Optional[str]) -> str:
    if forced:
        if forced not in df.columns:
            raise ValueError(f"Text column '{forced}' not found in parquet.")
        return forced

    for column in TEXT_CANDIDATES:
        if column in df.columns:
            return column

    object_like_cols = [
        column
        for column in df.columns
        if (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]))
        and column not in NON_TEXT_FALLBACK_EXCLUDE
    ]
    if not object_like_cols:
        raise ValueError("Could not auto-detect text column. Please pass --text_col.")

    scores: Dict[str, int] = {}
    for column in object_like_cols:
        series = df[column].astype("string")
        scores[column] = int(((series.notna()) & (series.str.strip() != "")).sum())

    best_column = max(scores, key=scores.get)
    if scores[best_column] <= 0:
        raise ValueError("Detected text candidates, but all values are empty.")
    return best_column


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_max_length(tokenizer: AutoTokenizer, requested_max_length: int) -> int:
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if tokenizer_max is None or tokenizer_max > 100000:
        return requested_max_length
    return min(requested_max_length, int(tokenizer_max))


class TextOnlyDataset(Dataset):
    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"text": self.texts[index]}


def build_collate_fn(
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return collate_fn


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def iterate_loader(loader: DataLoader, description: str) -> Any:
    if tqdm is None:
        return loader
    return tqdm(loader, desc=description, leave=False)


def predict_probabilities(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in iterate_loader(data_loader, "predict"):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1]
            all_probs.extend(probabilities.detach().cpu().tolist())

    return np.asarray(all_probs, dtype=np.float32)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "prediction_summary.json"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    input_files = resolve_input_files(args.input_pattern)
    output_dir.mkdir(parents=True, exist_ok=True)

    emit(f"Loading fine-tuned model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    max_length = resolve_max_length(tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=args.local_files_only,
    )

    device = resolve_device(args.device)
    model.to(device)
    emit(f"Using device: {device}")

    summary: Dict[str, Any] = {
        "model_dir": str(model_dir.resolve()),
        "input_pattern": args.input_pattern,
        "output_dir": str(output_dir.resolve()),
        "batch_size": args.batch_size,
        "max_length": max_length,
        "positive_threshold": args.positive_threshold,
        "device": str(device),
        "files": [],
        "totals": {
            "files": 0,
            "rows": 0,
            "positive_rows": 0,
            "negative_rows": 0,
        },
    }

    for input_path_str in input_files:
        input_path = Path(input_path_str)
        emit(f"Predicting {input_path}")

        df = pd.read_parquet(input_path)
        if df.empty:
            output_path = output_dir / input_path.name
            df.to_parquet(output_path, index=False)
            summary["files"].append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "rows": 0,
                    "positive_rows": 0,
                    "negative_rows": 0,
                    "text_col": None,
                }
            )
            summary["totals"]["files"] += 1
            continue

        text_col = detect_text_column(df, args.text_col)
        texts = df[text_col].fillna("").astype(str).tolist()

        data_loader = DataLoader(
            TextOnlyDataset(texts),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=build_collate_fn(tokenizer, max_length),
        )
        positive_probs = predict_probabilities(model, data_loader, device=device)
        preds = (positive_probs >= args.positive_threshold).astype(np.int64)

        result = df.copy()
        result["pred_label"] = preds
        result["pred_label_text"] = np.where(result["pred_label"] == 1, "相关", "无关")
        result["pred_prob_1"] = positive_probs.astype(float)
        result["pred_prob_0"] = 1.0 - result["pred_prob_1"]
        result["model_dir"] = str(model_dir.resolve())

        if args.only_positive:
            result = result[result["pred_label"] == 1].reset_index(drop=True)

        output_path = output_dir / input_path.name
        result.to_parquet(output_path, index=False)

        positive_rows = int((preds == 1).sum())
        negative_rows = int((preds == 0).sum())
        summary["files"].append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "rows": int(len(df)),
                "positive_rows": positive_rows,
                "negative_rows": negative_rows,
                "text_col": text_col,
            }
        )
        summary["totals"]["files"] += 1
        summary["totals"]["rows"] += int(len(df))
        summary["totals"]["positive_rows"] += positive_rows
        summary["totals"]["negative_rows"] += negative_rows

    save_json(summary_path, summary)
    emit(f"Prediction summary saved to {summary_path}")


if __name__ == "__main__":
    main()
