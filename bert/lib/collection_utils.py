from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from lib.data_utils import detect_text_column, load_training_dataframe


def infer_source_name(path: Path) -> str:
    return path.stem.strip() or path.name


def load_text_collection_frame(
    path: Path,
    *,
    sheet_name: Optional[str],
    text_col_hint: Optional[str],
    source_name: Optional[str] = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = load_training_dataframe(path, sheet_name)
    if df.empty:
        raise ValueError(f"Input dataset is empty: {path}")

    resolved_text_col = detect_text_column(df, text_col_hint, source_name=path.name)

    working = df.copy()
    working["__text"] = working[resolved_text_col].fillna("").astype(str).str.strip()
    working["__source_name"] = source_name or infer_source_name(path)
    working["__source_path"] = str(path.resolve())
    working["__source_file"] = path.name
    return working


ROW_SIGNATURE_EXCLUDE_COLUMNS = {
    "__source_name",
    "__source_path",
    "__source_file",
    "__prepared_split",
    "__prepared_row_id",
    "__dual_split",
    "__dual_row_id",
    "__resolved_label_col",
    "__resolved_broad_col",
    "__resolved_strict_col",
}


def _normalize_signature_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return "|".join(f"{key}={value[key]}" for key in sorted(value))
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _resolve_signature_columns(pool_df: pd.DataFrame, reference_df: pd.DataFrame) -> list[str]:
    common_columns = [
        column
        for column in pool_df.columns
        if column in reference_df.columns and column not in ROW_SIGNATURE_EXCLUDE_COLUMNS
    ]
    if common_columns:
        return common_columns
    return [column for column in ("id", "mid", "__text") if column in pool_df.columns and column in reference_df.columns]


def _build_row_signatures(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    normalized = df.loc[:, columns].map(_normalize_signature_value)
    return pd.util.hash_pandas_object(normalized, index=False)


def drop_rows_overlapping_with_reference(
    pool_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, list[str]]:
    if pool_df.empty or reference_df.empty:
        return pool_df, 0, []

    signature_columns = _resolve_signature_columns(pool_df, reference_df)
    if not signature_columns:
        return pool_df, 0, []

    pool_signatures = _build_row_signatures(pool_df, signature_columns)
    reference_signatures = set(_build_row_signatures(reference_df, signature_columns).tolist())
    keep_mask = ~pool_signatures.isin(reference_signatures)
    removed = int((~keep_mask).sum())
    if removed <= 0:
        return pool_df, 0, signature_columns
    filtered = pool_df.loc[keep_mask].copy().reset_index(drop=True)
    return filtered, removed, signature_columns
