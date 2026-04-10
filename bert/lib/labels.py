from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


LABEL_CANDIDATES = ["label", "tangping_related", "tangping_related_label", "broad", "strict"]
NORMALIZE_LABEL_CANDIDATES = ["tangping_related_label", "tangping_related", "label"]
POSITIVE_ALIASES = {"1", "1.0", "true", "yes", "y", "relevant", "positive", "相关", "有关"}
NEGATIVE_ALIASES = {"0", "0.0", "false", "no", "n", "irrelevant", "negative", "无关", "不相关"}
NEGATIVE_ALIASES_WITH_TWO = NEGATIVE_ALIASES | {"2", "2.0"}


def normalize_label_value(value: Any, *, treat_two_as_negative: bool = True) -> Optional[int]:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if value in (0, 1):
            return int(value)
        if treat_two_as_negative and value == 2:
            return 0

    if isinstance(value, np.integer):
        if int(value) in (0, 1):
            return int(value)
        if treat_two_as_negative and int(value) == 2:
            return 0

    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)
        if treat_two_as_negative and value == 2.0:
            return 0

    if isinstance(value, np.floating):
        numeric_value = float(value)
        if numeric_value in (0.0, 1.0):
            return int(numeric_value)
        if treat_two_as_negative and numeric_value == 2.0:
            return 0

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in POSITIVE_ALIASES:
        return 1
    negative_aliases = NEGATIVE_ALIASES_WITH_TWO if treat_two_as_negative else NEGATIVE_ALIASES
    if lowered in negative_aliases:
        return 0
    return None


def detect_label_column(
    df: pd.DataFrame,
    forced: Optional[str],
    *,
    candidates: Optional[Iterable[str]] = None,
    treat_two_as_negative: bool = True,
    source_name: str = "dataset",
) -> str:
    if forced:
        if forced not in df.columns:
            raise ValueError(f"Label column '{forced}' not found in {source_name}.")
        return forced

    best_col = None
    best_valid = -1
    resolved_candidates = list(candidates or LABEL_CANDIDATES)
    for column in resolved_candidates:
        if column not in df.columns:
            continue
        normalized = df[column].map(
            lambda value: normalize_label_value(value, treat_two_as_negative=treat_two_as_negative)
        )
        valid_count = int(normalized.notna().sum())
        if valid_count > best_valid:
            best_valid = valid_count
            best_col = column

    if best_col is None or best_valid <= 0:
        raise ValueError("Could not auto-detect label column. Please pass --label_col.")
    return best_col

