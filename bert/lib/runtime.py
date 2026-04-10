from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def iterate_loader(loader: Any, description: str) -> Any:
    if tqdm is None:
        return loader
    return tqdm(loader, desc=description, leave=False)

