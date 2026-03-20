#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

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
    "label",
    "label_text",
    "label_raw",
    "tangping_related",
    "tangping_related_label",
}
LABEL_CANDIDATES = ["label", "tangping_related", "tangping_related_label"]
POSITIVE_ALIASES = {"1", "1.0", "true", "yes", "y", "relevant", "positive", "相关", "有关"}
NEGATIVE_ALIASES = {"0", "0.0", "false", "no", "n", "irrelevant", "negative", "无关", "不相关"}


def emit(message: str) -> None:
    print(f"[train] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, and test a BERT-style classifier for tangping relevance."
    )
    parser.add_argument(
        "--input_csv",
        default="data/bert/labeled_binary.csv",
        help="Training CSV path.",
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


def normalize_label_value(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if value in (0, 1):
            return int(value)

    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in POSITIVE_ALIASES:
        return 1
    if lowered in NEGATIVE_ALIASES:
        return 0
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_text_column(df: pd.DataFrame, forced: Optional[str]) -> str:
    if forced:
        if forced not in df.columns:
            raise ValueError(f"Text column '{forced}' not found in CSV.")
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


def detect_label_column(df: pd.DataFrame, forced: Optional[str]) -> str:
    if forced:
        if forced not in df.columns:
            raise ValueError(f"Label column '{forced}' not found in CSV.")
        return forced

    best_col = None
    best_valid = -1
    for column in LABEL_CANDIDATES:
        if column not in df.columns:
            continue
        normalized = df[column].map(normalize_label_value)
        valid_count = int(normalized.notna().sum())
        if valid_count > best_valid:
            best_valid = valid_count
            best_col = column

    if best_col is None or best_valid <= 0:
        raise ValueError("Could not auto-detect label column. Please pass --label_col.")
    return best_col


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


class TextLabelDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "text": self.texts[index],
            "label": int(self.labels[index]),
        }


def build_collate_fn(
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]:
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded

    return collate_fn


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def create_data_splits(
    df: pd.DataFrame,
    label_col: str,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0:
        raise ValueError("Both --val_size and --test_size must be positive.")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be smaller than 1.")

    labels = df[label_col]
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=seed,
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[label_col],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def iterate_loader(loader: DataLoader, description: str) -> Any:
    if tqdm is None:
        return loader
    return tqdm(loader, desc=description, leave=False)


def evaluate_model(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
    positive_threshold: float,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_predictions(
    df: pd.DataFrame,
    text_col: str,
    probs: np.ndarray,
    preds: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = df.copy()
    result["gold_label"] = result["label"].astype(int)
    result["pred_label"] = preds.astype(int)
    result["pred_label_text"] = np.where(result["pred_label"] == 1, "相关", "无关")
    result["pred_prob_1"] = probs.astype(float)
    result["pred_prob_0"] = 1.0 - result["pred_prob_1"]
    ordered_columns = [
        "id",
        text_col,
        "gold_label",
        "pred_label",
        "pred_label_text",
        "pred_prob_1",
        "pred_prob_0",
    ]
    existing_columns = [column for column in ordered_columns if column in result.columns]
    extra_columns = [column for column in result.columns if column not in existing_columns]
    result = result[existing_columns + extra_columns]
    result.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    best_model_dir = output_dir / "best_model"
    history_path = output_dir / "training_history.json"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "train_config.json"
    test_predictions_path = output_dir / "test_predictions.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {input_csv}")

    emit(f"Loading training data from {input_csv}")
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Training CSV is empty.")

    text_col = detect_text_column(df, args.text_col)
    source_label_col = detect_label_column(df, args.label_col)

    df = df.copy()
    df["label"] = df[source_label_col].map(normalize_label_value)
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[(df["label"].notna()) & (df[text_col] != "")].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    if len(df) < 20:
        raise ValueError("Too few valid rows after cleaning to train a classifier.")

    label_counts = df["label"].value_counts().to_dict()
    for class_id in (0, 1):
        if label_counts.get(class_id, 0) < 2:
            raise ValueError("Each class needs at least 2 rows for stratified train/val/test splits.")

    train_df, val_df, test_df = create_data_splits(
        df=df,
        label_col="label",
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    emit(f"Loading tokenizer and model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    max_length = resolve_max_length(tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        local_files_only=args.local_files_only,
    )
    model.config.id2label = {0: "无关", 1: "相关"}
    model.config.label2id = {"无关": 0, "相关": 1}

    device = resolve_device(args.device)
    model.to(device)
    emit(f"Using device: {device}")

    collate_fn = build_collate_fn(tokenizer, max_length)
    train_loader = DataLoader(
        TextLabelDataset(train_df[text_col].tolist(), train_df["label"].tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TextLabelDataset(val_df[text_col].tolist(), val_df["label"].tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        TextLabelDataset(test_df[text_col].tolist(), test_df["label"].tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_train_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    training_history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in iterate_loader(train_loader, f"train epoch {epoch}"):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics, _, _ = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            positive_threshold=args.positive_threshold,
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
        emit(
            f"Epoch {epoch}/{args.epochs} "
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
        positive_threshold=args.positive_threshold,
    )
    test_metrics, test_probs, test_preds = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        device=device,
        positive_threshold=args.positive_threshold,
    )

    metrics_payload: Dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "resolved_text_col": text_col,
        "resolved_label_col": source_label_col,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "positive_threshold": args.positive_threshold,
        "seed": args.seed,
        "device": str(device),
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
        {
            "input_csv": str(input_csv.resolve()),
            "output_dir": str(output_dir.resolve()),
            "text_col": text_col,
            "label_col": source_label_col,
            "max_length": max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "positive_threshold": args.positive_threshold,
            "seed": args.seed,
            "device": str(device),
            "local_files_only": bool(args.local_files_only),
        },
    )
    save_predictions(
        df=test_df,
        text_col=text_col,
        probs=test_probs,
        preds=test_preds,
        output_path=test_predictions_path,
    )

    emit(f"Best model saved to {best_model_dir}")
    emit(f"Metrics saved to {metrics_path}")
    emit(f"Test predictions saved to {test_predictions_path}")


if __name__ == "__main__":
    main()
