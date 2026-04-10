from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lib.data_utils import detect_text_column
from lib.io_utils import save_json
from lib.runtime import iterate_loader, resolve_device, resolve_max_length
from lib.text_modeling import TextOnlyDataset, build_text_collate_fn, move_batch_to_device


@dataclass
class BatchPredictionConfig:
    model_dir: str = "bert/artifacts/tangping_bert/best_model"
    input_pattern: str = "data/processed/text_dedup/*.parquet"
    output_dir: str = "data/processed/text_dedup_predicted"
    text_col: Optional[str] = None
    batch_size: int = 64
    max_length: int = 128
    positive_threshold: float = 0.5
    device: str = "auto"
    local_files_only: bool = False
    only_positive: bool = False


def resolve_input_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files and pattern.startswith("/"):
        files = sorted(glob.glob(pattern.lstrip("/")))
    if not files:
        raise FileNotFoundError(f"No parquet files matched input pattern: {pattern}")
    return files


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


def run_batch_prediction(
    config: BatchPredictionConfig,
    *,
    emit: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    emit_fn = emit or (lambda message: None)

    model_dir = Path(config.model_dir)
    output_dir = Path(config.output_dir)
    summary_path = output_dir / "prediction_summary.json"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    input_files = resolve_input_files(config.input_pattern)
    output_dir.mkdir(parents=True, exist_ok=True)

    emit_fn(f"Loading fine-tuned model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=config.local_files_only,
        use_fast=True,
    )
    max_length = resolve_max_length(tokenizer, config.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=config.local_files_only,
    )

    device = resolve_device(config.device)
    model.to(device)
    emit_fn(f"Using device: {device}")

    summary: Dict[str, Any] = {
        "model_dir": str(model_dir.resolve()),
        "input_pattern": config.input_pattern,
        "output_dir": str(output_dir.resolve()),
        "batch_size": config.batch_size,
        "max_length": max_length,
        "positive_threshold": config.positive_threshold,
        "device": str(device),
        "files": [],
        "totals": {
            "files": 0,
            "rows": 0,
            "positive_rows": 0,
            "negative_rows": 0,
        },
    }

    collate_fn = build_text_collate_fn(tokenizer, max_length)

    for input_path_str in input_files:
        input_path = Path(input_path_str)
        emit_fn(f"Predicting {input_path}")

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

        text_col = detect_text_column(df, config.text_col, source_name="parquet")
        texts = df[text_col].fillna("").astype(str).tolist()

        data_loader = DataLoader(
            TextOnlyDataset(texts),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        positive_probs = predict_probabilities(model, data_loader, device=device)
        preds = (positive_probs >= config.positive_threshold).astype(np.int64)

        result = df.copy()
        result["pred_label"] = preds
        result["pred_label_text"] = np.where(result["pred_label"] == 1, "相关", "无关")
        result["pred_prob_1"] = positive_probs.astype(float)
        result["pred_prob_0"] = 1.0 - result["pred_prob_1"]
        result["model_dir"] = str(model_dir.resolve())

        if config.only_positive:
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
    emit_fn(f"Prediction summary saved to {summary_path}")
    return summary
