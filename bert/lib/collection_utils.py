from __future__ import annotations

from pathlib import Path
from typing import Optional

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
