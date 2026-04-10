from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


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


def normalize_split_value(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None

    normalized = str(value).strip().lower()
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"val", "valid", "validation", "dev"}:
        return "val"
    if normalized in {"test", "testing"}:
        return "test"
    return None


def create_predefined_splits(df: pd.DataFrame, split_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_col not in df.columns:
        raise ValueError(f"Split column '{split_col}' not found in dataset.")

    resolved = df[split_col].map(normalize_split_value)
    invalid_mask = resolved.isna()
    if invalid_mask.any():
        invalid_values = (
            pd.Series(df.loc[invalid_mask, split_col].astype("string").unique()).dropna().astype(str).tolist()
        )
        preview = ", ".join(invalid_values[:5]) if invalid_values else "<empty>"
        raise ValueError(
            f"Split column '{split_col}' contains unsupported values. "
            f"Expected train/val/test, got: {preview}"
        )

    working = df.copy()
    working["__resolved_split"] = resolved
    train_df = working[working["__resolved_split"] == "train"].drop(columns=["__resolved_split"]).reset_index(drop=True)
    val_df = working[working["__resolved_split"] == "val"].drop(columns=["__resolved_split"]).reset_index(drop=True)
    test_df = working[working["__resolved_split"] == "test"].drop(columns=["__resolved_split"]).reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Predefined splits must all contain at least one row for train/val/test.")

    return train_df, val_df, test_df


def split_with_optional_stratify(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
    stratify_values: Optional[pd.Series],
) -> Dict[str, pd.DataFrame]:
    working = df.copy()
    key_column = "__dual_split_key"
    if stratify_values is not None:
        working[key_column] = stratify_values.astype(str)

    train_df, temp_df = train_test_split(
        working,
        test_size=val_size + test_size,
        stratify=working[key_column] if stratify_values is not None else None,
        random_state=seed,
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[key_column] if stratify_values is not None else None,
        random_state=seed,
    )

    frames = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    for name, frame in frames.items():
        if key_column in frame.columns:
            frames[name] = frame.drop(columns=[key_column])
    return frames


def create_shared_splits(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
    *,
    emit: Optional[Callable[[str], None]] = None,
) -> tuple[Dict[str, pd.DataFrame], str]:
    strategies = [
        ("pair", df["broad_norm"].astype(str) + "_" + df["strict_norm"].astype(str)),
        ("broad", df["broad_norm"]),
        ("strict", df["strict_norm"]),
        ("random", None),
    ]

    last_error: Exception | None = None
    for strategy_name, stratify_values in strategies:
        try:
            frames = split_with_optional_stratify(df, val_size, test_size, seed, stratify_values)
            return frames, strategy_name
        except ValueError as exc:
            last_error = exc
            if emit is not None:
                emit(f"Shared split fallback: strategy={strategy_name} unavailable ({exc})")

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to create shared splits.")
