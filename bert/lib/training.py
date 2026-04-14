from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from lib.data_utils import detect_text_column, load_training_dataframe
from lib.io_utils import save_json
from lib.labels import LABEL_CANDIDATES, detect_label_column, normalize_label_value
from lib.runtime import iterate_loader, resolve_device, resolve_max_length, set_seed
from lib.splits import create_data_splits, create_predefined_splits
from lib.text_modeling import TextLabelDataset, build_label_collate_fn, move_batch_to_device


@dataclass
class TrainClassifierConfig:
    input_csv: str = "bert/data/labeled_binary.csv"
    output_dir: str = "bert/artifacts/tangping_bert"
    model_name_or_path: str = "hfl/chinese-roberta-wwm-ext"
    text_col: Optional[str] = None
    label_col: Optional[str] = None
    split_col: Optional[str] = None
    sheet_name: Optional[str] = None
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    val_size: float = 0.1
    test_size: float = 0.1
    positive_threshold: float = 0.5
    seed: int = 42
    device: str = "auto"
    local_files_only: bool = False


def evaluate_model(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
    positive_threshold: float,
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for batch in iterate_loader(data_loader, "eval"):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            predictions = (probabilities >= positive_threshold).long()

            losses.append(float(outputs.loss.detach().cpu().item()))
            all_labels.extend(batch["labels"].detach().cpu().tolist())
            all_probs.extend(probabilities.detach().cpu().tolist())
            all_preds.extend(predictions.detach().cpu().tolist())

    labels_array = np.asarray(all_labels, dtype=np.int64)
    preds_array = np.asarray(all_preds, dtype=np.int64)
    probs_array = np.asarray(all_probs, dtype=np.float32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array,
        preds_array,
        average="binary",
        zero_division=0,
    )
    metrics: Dict[str, Any] = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(labels_array, preds_array)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(labels_array, preds_array, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels_array,
            preds_array,
            labels=[0, 1],
            target_names=["无关", "相关"],
            output_dict=True,
            zero_division=0,
        ),
        "support": int(len(labels_array)),
    }
    return metrics, probs_array, preds_array


def build_predictions_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    probs: np.ndarray,
    preds: np.ndarray,
) -> pd.DataFrame:
    result = df.copy()
    result["label_standard"] = label_col
    result["gold_label"] = result["label"].astype(int)
    result["gold_label_text"] = np.where(result["gold_label"] == 1, "相关", "无关")
    result["pred_label"] = preds.astype(int)
    result["pred_label_text"] = np.where(result["pred_label"] == 1, "相关", "无关")
    result["pred_prob_1"] = probs.astype(float)
    result["pred_prob_0"] = 1.0 - result["pred_prob_1"]
    result["pred_confidence"] = np.maximum(result["pred_prob_1"], result["pred_prob_0"])
    result["is_error"] = result["gold_label"] != result["pred_label"]
    result["error_type"] = np.where(
        result["is_error"],
        np.where(result["gold_label"] == 1, "FN", "FP"),
        "",
    )
    ordered_columns = [
        "label_standard",
        "__dual_row_id",
        "__dual_split",
        "id",
        text_col,
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
    existing_columns = [column for column in ordered_columns if column in result.columns]
    extra_columns = [column for column in result.columns if column not in existing_columns]
    return result[existing_columns + extra_columns]


def save_predictions(result: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_misclassified(result: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = result[result["is_error"]].copy()
    if not errors.empty and "pred_confidence" in errors.columns:
        errors = errors.sort_values(by="pred_confidence", ascending=False).reset_index(drop=True)
    errors.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_split(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def run_training(
    config: TrainClassifierConfig,
    *,
    emit: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    emit_fn = emit or (lambda message: None)
    set_seed(config.seed)

    input_csv = Path(config.input_csv)
    output_dir = Path(config.output_dir)
    best_model_dir = output_dir / "best_model"
    history_path = output_dir / "training_history.json"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "train_config.json"
    test_predictions_path = output_dir / "test_predictions.csv"
    test_misclassified_path = output_dir / "test_misclassified.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Training dataset not found: {input_csv}")

    emit_fn(f"Loading training data from {input_csv}")
    df = load_training_dataframe(input_csv, config.sheet_name)
    if df.empty:
        raise ValueError("Training dataset is empty.")

    text_col = detect_text_column(df, config.text_col, source_name="dataset")
    source_label_col = detect_label_column(
        df,
        config.label_col,
        candidates=LABEL_CANDIDATES,
        treat_two_as_negative=True,
        source_name="dataset",
    )

    working = df.copy()
    working["label"] = working[source_label_col].map(lambda value: normalize_label_value(value, treat_two_as_negative=True))
    working[text_col] = working[text_col].fillna("").astype(str).str.strip()
    working = working[(working["label"].notna()) & (working[text_col] != "")].reset_index(drop=True)
    working["label"] = working["label"].astype(int)

    if len(working) < 20:
        raise ValueError("Too few valid rows after cleaning to train a classifier.")

    label_counts = working["label"].value_counts().to_dict()
    for class_id in (0, 1):
        if label_counts.get(class_id, 0) < 2:
            raise ValueError("Each class needs at least 2 rows for stratified train/val/test splits.")

    if config.split_col:
        train_df, val_df, test_df = create_predefined_splits(df=working, split_col=config.split_col)
    else:
        train_df, val_df, test_df = create_data_splits(
            df=working,
            label_col="label",
            val_size=config.val_size,
            test_size=config.test_size,
            seed=config.seed,
        )

    train_labels = set(train_df["label"].astype(int).unique().tolist())
    if train_labels != {0, 1}:
        raise ValueError("Training split must contain both positive and negative samples.")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split(train_df, output_dir / "train_split.csv")
    save_split(val_df, output_dir / "val_split.csv")
    save_split(test_df, output_dir / "test_split.csv")

    # Avoid noisy background conversion attempts against Hub repos that only expose PyTorch weights.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

    emit_fn(f"Loading tokenizer and model from {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        local_files_only=config.local_files_only,
        use_fast=True,
    )
    max_length = resolve_max_length(tokenizer, config.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=2,
        local_files_only=config.local_files_only,
        use_safetensors=False,
    )
    model.config.id2label = {0: "无关", 1: "相关"}
    model.config.label2id = {"无关": 0, "相关": 1}

    device = resolve_device(config.device)
    model.to(device)
    emit_fn(f"Using device: {device}")

    collate_fn = build_label_collate_fn(tokenizer, max_length)
    train_loader = DataLoader(
        TextLabelDataset(train_df[text_col].tolist(), train_df["label"].tolist()),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TextLabelDataset(val_df[text_col].tolist(), val_df["label"].tolist()),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        TextLabelDataset(test_df[text_col].tolist(), test_df["label"].tolist()),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_train_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_train_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    training_history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in iterate_loader(train_loader, f"train epoch {epoch}"):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics, _, _ = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            positive_threshold=config.positive_threshold,
        )

        history_row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        training_history.append(history_row)
        emit_fn(
            f"Epoch {epoch}/{config.epochs} "
            f"train_loss={train_loss:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

    save_json(history_path, {"history": training_history})

    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    best_model.to(device)

    val_metrics, _, _ = evaluate_model(
        model=best_model,
        data_loader=val_loader,
        device=device,
        positive_threshold=config.positive_threshold,
    )
    test_metrics, test_probs, test_preds = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        device=device,
        positive_threshold=config.positive_threshold,
    )

    metrics_payload: Dict[str, Any] = {
        "model_name_or_path": config.model_name_or_path,
        "resolved_text_col": text_col,
        "resolved_label_col": source_label_col,
        "max_length": max_length,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "positive_threshold": config.positive_threshold,
        "seed": config.seed,
        "device": str(device),
        "split_col": config.split_col,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "label_distribution": {
            "train": {str(k): int(v) for k, v in train_df["label"].value_counts().sort_index().to_dict().items()},
            "val": {str(k): int(v) for k, v in val_df["label"].value_counts().sort_index().to_dict().items()},
            "test": {str(k): int(v) for k, v in test_df["label"].value_counts().sort_index().to_dict().items()},
        },
        "best_model_dir": str(best_model_dir.resolve()),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    save_json(metrics_path, metrics_payload)
    save_json(
        config_path,
        asdict(config) | {
            "input_csv": str(input_csv.resolve()),
            "output_dir": str(output_dir.resolve()),
            "text_col": text_col,
            "label_col": source_label_col,
            "device": str(device),
            "max_length": max_length,
        },
    )
    test_predictions = build_predictions_dataframe(
        df=test_df,
        text_col=text_col,
        label_col=source_label_col,
        probs=test_probs,
        preds=test_preds,
    )
    save_predictions(test_predictions, test_predictions_path)
    save_misclassified(test_predictions, test_misclassified_path)

    emit_fn(
        f"Finished {source_label_col}: "
        f"val_f1={val_metrics['f1']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} "
        f"artifacts={output_dir}"
    )

    return {
        "metrics_path": str(metrics_path.resolve()),
        "metrics": metrics_payload,
        "best_model_dir": str(best_model_dir.resolve()),
        "train_split_path": str((output_dir / 'train_split.csv').resolve()),
        "val_split_path": str((output_dir / 'val_split.csv').resolve()),
        "test_split_path": str((output_dir / 'test_split.csv').resolve()),
        "test_predictions_path": str(test_predictions_path.resolve()),
        "test_misclassified_path": str(test_misclassified_path.resolve()),
    }
