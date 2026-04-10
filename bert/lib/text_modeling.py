from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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


class TextOnlyDataset(Dataset):
    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"text": self.texts[index]}


def build_label_collate_fn(
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


def build_text_collate_fn(
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

