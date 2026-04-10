from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import pandas as pd


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


def make_unique_columns(columns: Sequence[Any]) -> List[str]:
    counts: dict[str, int] = {}
    resolved: List[str] = []
    for index, column in enumerate(columns):
        base = str(column).strip() if pd.notna(column) else ""
        if not base:
            base = f"unnamed_{index}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        resolved.append(base if count == 0 else f"{base}__{count + 1}")
    return resolved


def row_contains_embedded_header(row: pd.Series) -> bool:
    normalized = {
        str(value).strip().lower()
        for value in row.tolist()
        if pd.notna(value) and str(value).strip()
    }
    expected_tokens = {
        "id",
        "cleaned_text",
        "cleaned_text_with_emoji",
        "text_raw",
        "broad",
        "strict",
        "label",
        "tangping_related",
        "tangping_related_label",
        "类型",
        "发布时间",
        "话题",
        "keyword",
    }
    return len(normalized & {token.lower() for token in expected_tokens}) >= 3


def load_training_dataframe(input_path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)

    if suffix not in {".xlsx", ".xls"}:
        raise ValueError(f"Unsupported training file format: {input_path.suffix}")

    resolved_sheet_name: Any = 0 if sheet_name in (None, "") else sheet_name
    df = pd.read_excel(input_path, sheet_name=resolved_sheet_name)
    if isinstance(df, dict):
        first_sheet_name = next(iter(df))
        df = df[first_sheet_name]
    df.columns = make_unique_columns(df.columns)

    if not df.empty and row_contains_embedded_header(df.iloc[0]):
        header = make_unique_columns(df.iloc[0].tolist())
        df = df.iloc[1:].copy()
        df.columns = header
    return df.reset_index(drop=True)


def detect_text_column(df: pd.DataFrame, forced: Optional[str], *, source_name: str = "dataset") -> str:
    if forced:
        if forced not in df.columns:
            raise ValueError(f"Text column '{forced}' not found in {source_name}.")
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

    scores: dict[str, int] = {}
    for column in object_like_cols:
        series = df[column].astype("string")
        scores[column] = int(((series.notna()) & (series.str.strip() != "")).sum())

    best_column = max(scores, key=scores.get)
    if scores[best_column] <= 0:
        raise ValueError("Detected text candidates, but all values are empty.")
    return best_column

